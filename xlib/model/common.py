import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, List, Union, Any, Dict
from xlib.model.position_embedding import fourier_embedding
from xlib.basetype import TensorDict


class PosedepGSHead(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, max_step: float = 0.02):
        super().__init__()
        self.max_step = max_step
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.GELU(),
            nn.Linear(hidden_channels, 3),
        )
        # Initialize to zero
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward_raw(self, x: torch.Tensor) -> TensorDict:
        # x: [B, 1, L, C]
        with torch.amp.autocast(device_type='cuda', enabled=False):
            x = x.float()
            return TensorDict({"offset_xyz": self.mlp(x)})

    def presum(self, dic: TensorDict) -> TensorDict:
        with torch.amp.autocast(device_type='cuda', enabled=False):
            return TensorDict({"offset_xyz": torch.tanh(dic["offset_xyz"]) * self.max_step})

    def sum_residuals(self, dics: List[TensorDict]) -> TensorDict:
        with torch.amp.autocast(device_type='cuda', enabled=False):
            ret = {}
            # Base offset from first dict (anchor)
            if "offset_xyz" in dics[0]:
                offset = dics[0]["offset_xyz"]
            else:
                offset = torch.zeros_like(dics[0]["scale_"][..., :3])
            # Add residuals from other dicts
            for i in range(1, len(dics)):
                if "offset_xyz" in dics[i]:
                    offset = offset + dics[i]["offset_xyz"]
            ret["offset_xyz"] = offset
            # Pass through other keys from anchor
            for k, v in dics[0].items():
                if k not in ret:
                    ret[k] = v
            return TensorDict(ret)

    def activate(self, dic: TensorDict) -> TensorDict:
        return dic

    def set_scale_base(self, scale_base):
        pass


class PointEmbed(nn.Module):
    """Adopted from LHM"""

    def __init__(self, hidden_dim: int = 48, dim: int = 128):
        super().__init__()
        assert hidden_dim % 6 == 0

        self.embedding_dim = hidden_dim
        e = torch.pow(2, torch.arange(self.embedding_dim // 6)).float() * np.pi
        e = torch.stack(
            [
                torch.cat(
                    [
                        e,
                        torch.zeros(self.embedding_dim // 6),
                        torch.zeros(self.embedding_dim // 6),
                    ]
                ),
                torch.cat(
                    [
                        torch.zeros(self.embedding_dim // 6),
                        e,
                        torch.zeros(self.embedding_dim // 6),
                    ]
                ),
                torch.cat(
                    [
                        torch.zeros(self.embedding_dim // 6),
                        torch.zeros(self.embedding_dim // 6),
                        e,
                    ]
                ),
            ]
        )
        self.register_buffer("basis", e)

        self.mlp = nn.Linear(self.embedding_dim + 3, dim)
        self.norm = nn.LayerNorm(dim)

    @staticmethod
    def embed(input: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
        projections = torch.einsum("bnd,de->bne", input, basis)
        embeddings = torch.cat([projections.sin(), projections.cos()], dim=2)
        return embeddings

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        embed = self.mlp(torch.cat([self.embed(input, self.basis), input], dim=2))
        embed = self.norm(embed)
        return embed


class Unpatchify(nn.Module):
    def __init__(
        self,
        embed_dim: int = 1024,
        out_channels: int = 14,
        patch_size: int = 8,
        hidden_dim: int = 4096,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.out_channels = out_channels
        self.patch_size = patch_size

        self.proj = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_channels * patch_size * patch_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.dim() == 4, f"Expected [B, C, H, W], got shape {x.shape}"
        B, C, H, W = x.shape
        assert C == self.embed_dim
        x = x.permute(0, 2, 3, 1).reshape(B, H * W, C)  # [B, H*W, C]
        x = self.proj(x)  # [B, H*W, out_ch*ps*ps]
        x = x.view(B, H, W, self.out_channels, self.patch_size, self.patch_size)
        x = x.permute(0, 3, 1, 4, 2, 5).contiguous()
        return x.view(B, self.out_channels, H * self.patch_size, W * self.patch_size)


class ConvUnpatchify(nn.Module):
    @staticmethod
    def _num_groups(channels: int, max_groups: int = 32) -> int:
        max_allowed = min(max_groups, max(1, channels // 8))
        for g in range(max_allowed, 0, -1):
            if channels % g == 0:
                return g
        return 1

    class _RefineBlock(nn.Module):
        def __init__(self, in_ch: int, out_ch: int, activate: bool):
            super().__init__()
            self.activate = activate
            self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False)
            self.norm = nn.GroupNorm(ConvUnpatchify._num_groups(out_ch), out_ch)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            y = self.conv(x)
            y = self.norm(y)
            if self.activate:
                y = F.gelu(y)
            return y

    def __init__(
        self,
        embed_dim: int = 1024,
        out_channels: int = 14,
        upsample_factor: int = 8,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.out_channels = out_channels
        self.upsample_factor = int(upsample_factor)

        self.token_proj = nn.Linear(embed_dim, embed_dim)
        n_bilinear_stages = int(np.log2(self.upsample_factor)) - 1
        self.conv_stages = nn.ModuleList()
        in_ch = embed_dim // 4
        self.post_shuffle_norm = nn.GroupNorm(self._num_groups(in_ch), in_ch)
        for i in range(n_bilinear_stages):
            out_ch = in_ch // 2
            self.conv_stages.append(self._RefineBlock(in_ch=in_ch, out_ch=out_ch, activate=True))
            in_ch = out_ch
        self.out_conv = nn.Conv2d(in_ch, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.dim() == 4, f"Expected [B, C, H, W], got shape {x.shape}"
        B, C, H, W = x.shape
        assert C == self.embed_dim
        x = x.permute(0, 2, 3, 1)  # [B, H, W, C]
        x = self.token_proj(x)  # [B, H, W, C]
        x = x.permute(0, 3, 1, 2).contiguous()  # [B, C, H, W]
        x = F.pixel_shuffle(x, upscale_factor=2)
        x = self.post_shuffle_norm(x)
        x = F.gelu(x)
        for conv_block in self.conv_stages:
            x = F.interpolate(x, scale_factor=2.0, mode="nearest")
            x = conv_block(x)
        x = self.out_conv(x)
        return x


class ResNetUnpatchify(nn.Module):
    _num_groups = staticmethod(ConvUnpatchify._num_groups)

    def __init__(
        self,
        embed_dim: int = 1024,
        out_channels: int = 14,
        upsample_factor: int = 8,
    ):
        super().__init__()
        from diffusers.models.resnet import ResnetBlock2D
        from diffusers.models.upsampling import Upsample2D

        self.embed_dim = embed_dim
        self.out_channels = out_channels
        self.upsample_factor = int(upsample_factor)

        if self.upsample_factor < 1 or (self.upsample_factor & (self.upsample_factor - 1)) != 0:
            raise ValueError(f"upsample_factor must be a positive power of 2, got {self.upsample_factor}")

        n_stages = int(np.log2(self.upsample_factor))
        if embed_dim % 4 != 0:
            raise ValueError(f"embed_dim must be divisible by 4, got {embed_dim}")

        self.token_proj = nn.Linear(embed_dim, embed_dim)
        base_channels = embed_dim // 4
        self.conv_in = nn.Conv2d(embed_dim, base_channels, kernel_size=1)

        self.resnet_stages = nn.ModuleList()
        self.upsample_stages = nn.ModuleList()
        in_ch = base_channels
        for stage_idx in range(n_stages):
            out_ch = base_channels // (2 ** stage_idx)
            if out_ch < 1:
                raise ValueError(
                    f"Decoder stage {stage_idx} would produce zero channels with embed_dim={embed_dim}"
                )
            self.resnet_stages.append(
                ResnetBlock2D(
                    in_channels=in_ch,
                    out_channels=out_ch,
                    temb_channels=None,
                    groups=self._num_groups(in_ch),
                    groups_out=self._num_groups(out_ch),
                    non_linearity="silu",
                )
            )
            self.upsample_stages.append(Upsample2D(out_ch, use_conv=True))
            in_ch = out_ch

        self.norm_out = nn.GroupNorm(self._num_groups(in_ch), in_ch)
        self.conv_out = nn.Conv2d(in_ch, out_channels, kernel_size=1)
        nn.init.normal_(self.conv_out.weight, mean=0.0, std=0.001)
        nn.init.zeros_(self.conv_out.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.dim() == 4, f"Expected [B, C, H, W], got shape {x.shape}"
        B, C, H, W = x.shape
        assert C == self.embed_dim
        x = x.permute(0, 2, 3, 1)  # [B, H, W, C]
        x = self.token_proj(x)  # [B, H, W, C]
        x = x.permute(0, 3, 1, 2).contiguous()  # [B, C, H, W]
        x = self.conv_in(x)
        for resnet_block, upsample in zip(self.resnet_stages, self.upsample_stages):
            x = resnet_block(x, temb=None)
            x = upsample(x)
        x = self.norm_out(x)
        x = F.silu(x)
        x = self.conv_out(x)
        return x


def build_uv_decoder(
    decoder_type: str,
    embed_dim: int,
    out_channels: int,
    patch_size: int,
    upsample_factor: Optional[int] = None,
) -> nn.Module:
    decoder_type = decoder_type.lower()
    if decoder_type == "unpatchify":
        return Unpatchify(
            embed_dim=embed_dim,
            out_channels=out_channels,
            patch_size=patch_size
        )
    if decoder_type == "conv_unpatchify":
        return ConvUnpatchify(
            embed_dim=embed_dim,
            out_channels=out_channels,
            upsample_factor=patch_size if upsample_factor is None else upsample_factor,
        )
    if decoder_type == "resnet_unpatchify":
        return ResNetUnpatchify(
            embed_dim=embed_dim,
            out_channels=out_channels,
            upsample_factor=patch_size if upsample_factor is None else upsample_factor,
        )
    raise ValueError(f"Unknown uv decoder_type: {decoder_type}")


def build_grid_coords(feat_map: torch.Tensor) -> torch.Tensor:
    B, _, H, W = feat_map.shape
    y, x = torch.meshgrid(
        torch.arange(H, device=feat_map.device, dtype=feat_map.dtype),
        torch.arange(W, device=feat_map.device, dtype=feat_map.dtype),
        indexing="ij",
    )
    z = torch.zeros_like(x)
    coords = torch.stack([x, y, z], dim=-1).view(1, H * W, 3)
    return coords.expand(B, -1, -1)


def spatial_tokens_to_feature_map(
    spatial_tokens: torch.Tensor,
    spatial_coords: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    B, N, C = spatial_tokens.shape
    H = W = None

    if spatial_coords is not None and spatial_coords.ndim == 2 and spatial_coords.shape[0] == N:
        x = spatial_coords[:, 0]
        y = spatial_coords[:, 1]
        H = int(y.max().item()) + 1
        W = int(x.max().item()) + 1
        if H * W != N:
            H = W = None

    if H is None or W is None:
        side = int(N ** 0.5)
        if side * side != N:
            raise ValueError(f"Cannot infer spatial feature-map shape from {N} tokens")
        H = W = side

    return spatial_tokens.permute(0, 2, 1).reshape(B, C, H, W)


class MLP(nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_channels=None,
        out_channels=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_channels = out_channels or in_channels
        hidden_channels = hidden_channels or in_channels
        self.fc1 = nn.Linear(in_channels, hidden_channels)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_channels, out_channels)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x



class PositionMLP(nn.Module):
    def __init__(
        self,
        in_dim: int,
        pos_embed_dim: int,
        h_dim: int,
        out_dim: int,
        n_layers: int,
        activation: str = "silu",
        bias: bool = True,
    ):
        super().__init__()

        self.dim_in = in_dim
        self.pos_embed_dim = pos_embed_dim // (2 * in_dim) * (2 * in_dim)
        total_in = in_dim + self.pos_embed_dim
        layers = [
            nn.Linear(total_in, h_dim, bias=bias),
            self.make_activation(activation),
        ]
        for _ in range(n_layers - 1):
            layers += [
                nn.Linear(h_dim, h_dim, bias=bias),
                self.make_activation(activation),
            ]
        layers.append(nn.Linear(h_dim, out_dim, bias=bias))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        pos_embedding = fourier_embedding(x, self.pos_embed_dim // self.dim_in)
        x = torch.cat([x, pos_embedding], dim=-1)
        x = self.layers(x)
        return x

    @staticmethod
    def make_activation(activation):
        if activation == "relu":
            return nn.ReLU(inplace=True)
        if activation == "silu":
            return nn.SiLU(inplace=True)
        if activation == "gelu":
            return nn.GELU()
        raise NotImplementedError
