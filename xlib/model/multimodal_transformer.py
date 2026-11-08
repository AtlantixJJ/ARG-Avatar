import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import List, Optional, Dict, Union

from functools import partial
from typing import Tuple, Type

from xlib.geometry.position_encoding import apply_rotary_enc, compute_axial_cis, compute_axial_cis_for_positions
from xlib.model.rope_3d import RoPE3DCrossAttention, compute_continuous_3d_cis, apply_rotary_enc_3d
from xlib.model.sam_3d_body.models.modules.drop_path import DropPath


class Attention(nn.Module):
    """
    An attention layer that allows for downscaling the size of the embedding
    after projection to queries, keys, and values.
    """

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        downsample_rate: int = 1,
        dropout: float = 0.0,
        kv_in_dim: int = None,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.kv_in_dim = kv_in_dim if kv_in_dim is not None else embedding_dim
        self.internal_dim = embedding_dim // downsample_rate
        self.num_heads = num_heads
        assert (
            self.internal_dim % num_heads == 0
        ), "num_heads must divide embedding_dim."

        self.q_proj = nn.Linear(embedding_dim, self.internal_dim, bias=False)
        self.k_proj = nn.Linear(self.kv_in_dim, self.internal_dim, bias=False)
        self.v_proj = nn.Linear(self.kv_in_dim, self.internal_dim, bias=False)
        self.out_proj = nn.Linear(self.internal_dim, embedding_dim)

        self.dropout_p = dropout

    def _separate_heads(self, x: Tensor, num_heads: int) -> Tensor:
        b, n, c = x.shape
        x = x.reshape(b, n, num_heads, c // num_heads)
        return x.transpose(1, 2)  # B x N_heads x N_tokens x C_per_head

    def _recombine_heads(self, x: Tensor) -> Tensor:
        b, n_heads, n_tokens, c_per_head = x.shape
        x = x.transpose(1, 2)
        return x.reshape(b, n_tokens, n_heads * c_per_head)  # B x N_tokens x C

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        # Input projections
        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)

        # Separate into heads
        q = self._separate_heads(q, self.num_heads)
        k = self._separate_heads(k, self.num_heads)
        v = self._separate_heads(v, self.num_heads)

        dropout_p = self.dropout_p if self.training else 0.0
        # Attention
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p)

        out = self._recombine_heads(out)
        out = self.out_proj(out)

        return out


class RoPE2DAttention(Attention):
    """Attention with rotary position encoding."""

    def __init__(
        self,
        *args,
        rope_theta=10000.0,
        # whether to repeat q rope to match k length
        # this is needed for cross-attention to memories
        rope_k_repeat=False,
        feat_sizes=(64, 64),  # [w, h] for stride 16 feats at 1024 resolution
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.rope_theta = rope_theta
        self.compute_cis = partial(
            compute_axial_cis, dim=self.internal_dim // self.num_heads, theta=rope_theta
        )
        freqs_cis = self.compute_cis(end_x=feat_sizes[0], end_y=feat_sizes[1])
        self.freqs_cis = (
            freqs_cis.to("cuda") if torch.cuda.is_available() else freqs_cis
        )
        self.rope_k_repeat = rope_k_repeat

    def forward(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        num_k_exclude_rope: int = 0,
        freqs_cis: Optional[Tensor] = None,
        patch_positions: Optional[Tensor] = None,
    ) -> Tensor:
        # Input projections
        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)

        # Separate into heads
        q = self._separate_heads(q, self.num_heads)
        k = self._separate_heads(k, self.num_heads)
        v = self._separate_heads(v, self.num_heads)

        # Apply rotary position encoding
        if freqs_cis is None:
            if patch_positions is not None:
                freqs_cis = compute_axial_cis_for_positions(
                    self.internal_dim // self.num_heads, patch_positions, self.rope_theta
                )
            else:
                w = h = math.sqrt(q.shape[-2])
                self.freqs_cis = self.freqs_cis.to(q.device)
                if self.freqs_cis.shape[0] != q.shape[-2]:
                    self.freqs_cis = self.compute_cis(end_x=w, end_y=h).to(q.device)
                freqs_cis = self.freqs_cis

        if q.shape[-2] != k.shape[-2]:
            assert self.rope_k_repeat

        num_k_rope = k.size(-2) - num_k_exclude_rope
        q, k[:, :, :num_k_rope] = apply_rotary_enc(
            q,
            k[:, :, :num_k_rope],
            freqs_cis=freqs_cis,
            repeat_freqs_k=self.rope_k_repeat,
        )

        dropout_p = self.dropout_p if self.training else 0.0
        # Attention
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p)

        out = self._recombine_heads(out)
        out = self.out_proj(out)

        return out


class RoPE3DAttention(Attention):
    """Self-attention with continuous 3D RoPE coordinates."""

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        rope_theta: float = 10000.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__(
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            downsample_rate=1,
            dropout=dropout,
            kv_in_dim=embedding_dim,
        )
        self.head_dim = self.internal_dim // self.num_heads
        self.rope_theta = rope_theta

    def forward(
        self,
        x: Tensor,
        coords: Tensor,
    ) -> Tensor:
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = self._separate_heads(q, self.num_heads)
        k = self._separate_heads(k, self.num_heads)
        v = self._separate_heads(v, self.num_heads)

        freqs = compute_continuous_3d_cis(
            coords=coords,
            dim=self.head_dim,
            theta=self.rope_theta,
        )
        q_rot, k_rot = apply_rotary_enc_3d(q, k, freqs, freqs)

        dropout_p = self.dropout_p if self.training else 0.0
        out = F.scaled_dot_product_attention(q_rot, k_rot, v, dropout_p=dropout_p)
        out = self._recombine_heads(out)
        return self.out_proj(out)


class MultiModalTransformerLayer(nn.Module):
    """
    Core transformer block for multi-modal fusion.
    
    This layer sequentially performs:
    1. Avatar Self-Attention: Avatar tokens attend to each other.
    2. Pose Cross-Attention (Optional): Avatar tokens attend to estimated pose features.
    3. Image Cross-Attention: Avatar tokens attend to multi-scale image features.
    
    The cross-attention can optionally use 3D Rotary Position Embeddings (RoPE3D)
    when avatar_coords and cond_coords are provided.
    """

    def __init__(
        self,
        avatar_dim: int = 1024,
        cond_dim: int = 1024,        # Dimension after projection for cross-attention
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        drop_path_rate: float = 0.0,
        use_rope_3d: bool = False,
        rope_theta: float = 10000.0,
    ):
        super().__init__()
        mlp_dim = int(avatar_dim * mlp_ratio)
        self.use_rope_3d = use_rope_3d

        # Self-attention for avatar tokens
        self.ln1 = nn.LayerNorm(avatar_dim)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=avatar_dim,
            num_heads=num_heads,
            batch_first=True
        )

        # Cross-attention: avatar tokens attend to conditioning (pose + rendered)
        self.ln2_q = nn.LayerNorm(avatar_dim)
        self.ln2_kv = nn.LayerNorm(cond_dim)
        if use_rope_3d:
            self.cross_attn = RoPE3DCrossAttention(
                avatar_dim=avatar_dim,
                cond_dim=cond_dim,
                num_heads=num_heads,
                rope_theta=rope_theta,
            )
            # Add pose cross-attention
            self.pose_cross_attn = nn.MultiheadAttention(
                embed_dim=avatar_dim,
                num_heads=num_heads,
                kdim=cond_dim,
                vdim=cond_dim,
                batch_first=True
            )
            self.ln2_pose_q = nn.LayerNorm(avatar_dim)
            self.ln2_pose_kv = nn.LayerNorm(cond_dim)
        else:
            self.cross_attn = nn.MultiheadAttention(
                embed_dim=avatar_dim,
                num_heads=num_heads,
                kdim=cond_dim,
                vdim=cond_dim,
                batch_first=True
            )

        # FFN
        self.ln3 = nn.LayerNorm(avatar_dim)
        self.ffn = nn.Sequential(
            nn.Linear(avatar_dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, avatar_dim),
        )
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()

    def forward(
        self,
        avatar_tokens: torch.Tensor,                           # [B, N_avatar, avatar_dim]
        spatial_cond_tokens: Optional[torch.Tensor] = None,     # [B, N_spatial, cond_dim]
        spatial_cond_coords: Optional[torch.Tensor] = None,     # [B, N_spatial, 3]
        nonspatial_cond_tokens: Optional[torch.Tensor] = None,  # [B, N_nonspatial, cond_dim]
        avatar_coords: Optional[torch.Tensor] = None,           # [B, N_avatar, 3]
    ) -> torch.Tensor:
        x = avatar_tokens

        # Self-attention (avatar tokens only)
        x_ln = self.ln1(x)
        attn_out, _ = self.self_attn(x_ln, x_ln, x_ln, need_weights=False)
        x = x + self.drop_path(attn_out)

        # Cross-attention (Q = avatar, KV = spatial + nonspatial)
        cond_parts = []
        if spatial_cond_tokens is not None:
            cond_parts.append(spatial_cond_tokens)
        if nonspatial_cond_tokens is not None:
            cond_parts.append(nonspatial_cond_tokens)
        assert cond_parts, "Cross-attention requires at least one of spatial_cond_tokens or nonspatial_cond_tokens."
        all_cond = torch.cat(cond_parts, dim=1) if len(cond_parts) > 1 else cond_parts[0]

        q = self.ln2_q(x)
        kv = self.ln2_kv(all_cond)

        if self.use_rope_3d:
            assert avatar_coords is not None and spatial_cond_coords is not None, \
                "RoPE3D cross-attention requires avatar_coords and spatial_cond_coords."
            if nonspatial_cond_tokens is not None:
                B = nonspatial_cond_tokens.shape[0]
                N_ns = nonspatial_cond_tokens.shape[1]
                zero_coords = spatial_cond_coords.new_zeros(B, N_ns, spatial_cond_coords.shape[-1])
                all_coords = torch.cat([spatial_cond_coords, zero_coords], dim=1)
            else:
                all_coords = spatial_cond_coords
            cross_out = self.cross_attn(
                q=q,
                kv=kv,
                q_coords=avatar_coords,
                kv_coords=all_coords,
            )
        else:
            cross_out, _ = self.cross_attn(q, kv, kv, need_weights=False)
        x = x + self.drop_path(cross_out)

        # FFN
        x = x + self.ffn(self.ln3(x))

        return x

class MultiModalTransformer(nn.Module):
    """
    Parallel transformer for multi-modal token refinement.
    
    Refines avatar tokens by cross-attending to image and pose information.
    The number of layers is typically matched to the SAM3D backbone decoder depth.
    
    Args:
        depth: Number of MultiModalTransformerLayers.
        avatar_dim: Input/output dimension of avatar tokens.
        cond_dim: Dimension of image features (keys/values).
        num_heads: Number of attention heads.
        mlp_ratio: FFN expansion factor.
        drop_path_rate: Stochastic depth rate.
        use_rendered: Whether to also attend to rendered images (deprecated).
        use_image: Whether to attend to input image features.
        use_rope_3d: Enable 3D RoPE in cross-attention.
        rope_theta: Theta for RoPE frequencies.
    """

    def __init__(
        self,
        depth: int = 6,               # Same as SAM3D decoder
        avatar_dim: int = 1024,
        cond_dim: int = 1024,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        drop_path_rate: float = 0.1,
        use_rendered: bool = False,
        use_image: bool = True,     # Flag to use image tokens
        use_rope_3d: bool = False,
        rope_theta: float = 10000.0,
    ):
        super().__init__()
        self.depth = depth
        self.use_rendered = use_rendered
        self.use_image = use_image
        self.cond_dim = cond_dim
        self.use_rope_3d = use_rope_3d
        self.rope_theta = rope_theta

        # Transformer layers
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.layers = nn.ModuleList([
            MultiModalTransformerLayer(
                avatar_dim=avatar_dim,
                cond_dim=cond_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                drop_path_rate=dpr[i],
                use_rope_3d=use_rope_3d,
                rope_theta=rope_theta,
            )
            for i in range(depth)
        ])
    
    def forward(
        self,
        avatar_tokens: torch.Tensor,                              # [B, N_avatar, avatar_dim]
        spatial_cond_tokens: Optional[torch.Tensor] = None,        # [B, N_spatial, cond_dim]
        spatial_cond_coords: Optional[torch.Tensor] = None,        # [B, N_spatial, 3]
        nonspatial_cond_tokens: Optional[torch.Tensor] = None,     # [B, N_nonspatial, cond_dim]
        avatar_coords: Optional[torch.Tensor] = None,              # [B, N_avatar, 3]
    ) -> torch.Tensor:
        x = avatar_tokens
        for layer in self.layers:
            x = layer(
                avatar_tokens=x,
                spatial_cond_tokens=spatial_cond_tokens,
                spatial_cond_coords=spatial_cond_coords,
                nonspatial_cond_tokens=nonspatial_cond_tokens,
                avatar_coords=avatar_coords,
            )

        return x


class AvatarBlock(nn.Module):
    """Unified avatar block: T2I CA -> optional SA3D -> SA2D -> FFN.
    """

    def __init__(
        self,
        embed_dim: int = 1024,
        cond_dim: int = 1024,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        drop_path_rate: float = 0.0,
        dropout: float = 0.0,
        rope_theta: float = 10000.0,
        feat_sizes: Tuple[int, int] = (64, 64),
        token_to_image_attn_type: str = "naive",
        sa3d_type: str = "none",
    ) -> None:
        super().__init__()
        mlp_dim = int(embed_dim * mlp_ratio)
        self.token_to_image_attn_type = token_to_image_attn_type
        self.sa3d_type = sa3d_type

        self.ln_t2i_q = nn.LayerNorm(embed_dim)
        self.ln_t2i_kv = nn.LayerNorm(cond_dim)
        if token_to_image_attn_type == "naive":
            self.t2i_attn = nn.MultiheadAttention(
                embed_dim=embed_dim,
                num_heads=num_heads,
                kdim=cond_dim,
                vdim=cond_dim,
                dropout=dropout,
                batch_first=True,
            )
        elif token_to_image_attn_type in {"fac-rope"}:
            self.t2i_attn = RoPE3DCrossAttention(
                avatar_dim=embed_dim,
                cond_dim=cond_dim,
                num_heads=num_heads,
                rope_theta=rope_theta,
            )
        elif token_to_image_attn_type == "projection":
            raise NotImplementedError("projection mode is not implemented")
        else:
            raise ValueError(f"Unknown token_to_image_attn_type: {token_to_image_attn_type}")

        
        self.ln_sa3d = None
        self.sa3d = None
        if sa3d_type != "none":
            self.ln_sa3d = nn.LayerNorm(embed_dim)
            self.sa3d = RoPE3DAttention(
                embedding_dim=embed_dim,
                num_heads=num_heads,
                rope_theta=rope_theta,
                dropout=dropout,
            )

        self.ln_sa2d = nn.LayerNorm(embed_dim)
        self.sa2d = RoPE2DAttention(
            embedding_dim=embed_dim,
            num_heads=num_heads,
            rope_theta=rope_theta,
            feat_sizes=feat_sizes,
            dropout=dropout,
        )

        self.ln_ffn = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, embed_dim),
            nn.Dropout(dropout),
        )
        self.proj_drop = nn.Dropout(dropout)
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()

    def _build_all_cond_and_coords(
        self,
        spatial_cond_tokens: Optional[torch.Tensor],
        spatial_cond_coords: Optional[torch.Tensor],
        nonspatial_cond_tokens: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        cond_parts = []
        if spatial_cond_tokens is not None:
            cond_parts.append(spatial_cond_tokens)
        if nonspatial_cond_tokens is not None:
            cond_parts.append(nonspatial_cond_tokens)
        if not cond_parts:
            raise ValueError("At least one conditioning source is required.")
        all_cond = torch.cat(cond_parts, dim=1) if len(cond_parts) > 1 else cond_parts[0]

        all_coords = None
        if self.token_to_image_attn_type in {"fac-rope"}:
            if spatial_cond_coords is None:
                raise ValueError(f"spatial_cond_coords is required for {self.token_to_image_attn_type}")
            sc = spatial_cond_coords
            if sc.ndim == 2:
                sc = sc.unsqueeze(0).expand(all_cond.shape[0], -1, -1)
            if sc.shape[-1] == 2:
                sc = F.pad(sc, (0, 1), value=0.0)
            if nonspatial_cond_tokens is not None:
                B, N_ns = nonspatial_cond_tokens.shape[:2]
                zeros = sc.new_zeros(B, N_ns, 3)
                all_coords = torch.cat([sc, zeros], dim=1)
            else:
                all_coords = sc
        return all_cond, all_coords

    def forward(
        self,
        avatar_tokens: torch.Tensor,
        spatial_cond_tokens: Optional[torch.Tensor] = None,
        spatial_cond_coords: Optional[torch.Tensor] = None,
        nonspatial_cond_tokens: Optional[torch.Tensor] = None,
        patch_uv_coords: Optional[torch.Tensor] = None,
        sa3d_coords: Optional[torch.Tensor] = None,
        projection_condition: Optional[Dict[str, torch.Tensor]] = None,
        token_coords: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = avatar_tokens

        if self.token_to_image_attn_type in {"naive", "fac-rope"}:
            all_cond, all_coords = self._build_all_cond_and_coords(
                spatial_cond_tokens=spatial_cond_tokens,
                spatial_cond_coords=spatial_cond_coords,
                nonspatial_cond_tokens=nonspatial_cond_tokens,
            )
            q = self.ln_t2i_q(x)
            kv = self.ln_t2i_kv(all_cond)
            if self.token_to_image_attn_type == "naive":
                t2i_out, _ = self.t2i_attn(q, kv, kv, need_weights=False)
            elif self.token_to_image_attn_type == "fac-rope":
                if token_coords is None:
                    raise ValueError("token_coords is required for fac-rope query coordinates")
                t2i_out = self.t2i_attn(
                    q=q,
                    kv=kv,
                    q_coords=token_coords,
                    kv_coords=all_coords,
                )
            else:
                if sa3d_coords is None:
                    raise ValueError("sa3d_coords is required for animation-rope query coordinates")
                t2i_out = self.t2i_attn(
                    q=q,
                    kv=kv,
                    q_coords=sa3d_coords,
                    kv_coords=all_coords,
                )
            x = x + self.drop_path(self.proj_drop(t2i_out))
        else:
            raise NotImplementedError("projection mode is not implemented")

        if self.sa3d is not None:
            if sa3d_coords is None:
                raise ValueError("sa3d_coords is required when sa3d_type != 'none'")
            x_ln = self.ln_sa3d(x)
            sa3d_out = self.sa3d(x_ln, coords=sa3d_coords)
            x = x + self.drop_path(self.proj_drop(sa3d_out))

        x_ln = self.ln_sa2d(x)
        sa2d_out = self.sa2d(
            x_ln,
            x_ln,
            x_ln,
            patch_positions=patch_uv_coords,
        )
        x = x + self.drop_path(self.proj_drop(sa2d_out))

        x = x + self.drop_path(self.ffn(self.ln_ffn(x)))
        return x


class MVAvatarBlock(AvatarBlock):
    """Multi-view avatar block with per-view cross-attention and shared UV mixing."""

    def __init__(
        self,
        embed_dim: int = 1024,
        cond_dim: int = 1024,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        drop_path_rate: float = 0.0,
        dropout: float = 0.0,
        rope_theta: float = 10000.0,
        feat_sizes: Tuple[int, int] = (64, 64),
        sa3d_type: str = "none",
        token_to_image_attn_type: str = "fac-rope",
    ) -> None:
        super().__init__(
            embed_dim=embed_dim,
            cond_dim=cond_dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            drop_path_rate=drop_path_rate,
            dropout=dropout,
            rope_theta=rope_theta,
            feat_sizes=feat_sizes,
            token_to_image_attn_type=token_to_image_attn_type,
            sa3d_type=sa3d_type,
        )

    def _forward_mv_stage(
        self,
        x: torch.Tensor,
        spatial_cond_tokens: torch.Tensor,
        spatial_cond_coords: torch.Tensor,
        token_coords: Optional[torch.Tensor] = None,
        sa3d_coords: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        q = self.ln_t2i_q(x)
        kv = self.ln_t2i_kv(spatial_cond_tokens)
        if self.token_to_image_attn_type == "naive":
            cross_out, _ = self.t2i_attn(q, kv, kv, need_weights=False)
        else:
            sc = spatial_cond_coords
            if sc.ndim == 2:
                if sc.shape[-1] == 2:
                    sc = F.pad(sc, (0, 1), value=0.0)
                sc = sc.unsqueeze(0).expand(x.shape[0], -1, -1)
            elif sc.shape[-1] == 2:
                sc = F.pad(sc, (0, 1), value=0.0)
            cross_out = self.t2i_attn(
                q=q,
                kv=kv,
                q_coords=token_coords,
                kv_coords=sc,
            )
        x = x + self.drop_path(self.proj_drop(cross_out))

        if self.sa3d is not None:
            if sa3d_coords is None:
                sa3d_coords = token_coords
            x_ln = self.ln_sa3d(x)
            sa3d_out = self.sa3d(x_ln, coords=sa3d_coords)
            x = x + self.drop_path(self.proj_drop(sa3d_out))
        return x

    def _forward_shared_stage(
        self,
        x: torch.Tensor,
        patch_uv_coords: torch.Tensor,
    ) -> torch.Tensor:
        x_ln = self.ln_sa2d(x)
        sa2d_out = self.sa2d(
            x_ln,
            x_ln,
            x_ln,
            patch_positions=patch_uv_coords,
        )
        x = x + self.drop_path(self.proj_drop(sa2d_out))
        x = x + self.drop_path(self.ffn(self.ln_ffn(x)))
        return x

    def forward(
        self,
        avatar_tokens: torch.Tensor,
        posed_token_additive: torch.Tensor,
        spatial_cond_tokens: torch.Tensor,
        token_coords: Union[torch.Tensor, List[torch.Tensor]],
        spatial_cond_coords: torch.Tensor,
        patch_uv_coords: torch.Tensor,
        multiview_fusion: nn.Module,
        sa3d_coords: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None,
    ) -> torch.Tensor:
        if spatial_cond_tokens.ndim != 4:
            raise ValueError("MVAvatarBlock.forward expects spatial_cond_tokens with shape [B, V, N, C]")
        if posed_token_additive.ndim != 4:
            raise ValueError("MVAvatarBlock.forward expects posed_token_additive with shape [B, V, L, C]")

        if isinstance(token_coords, (list, tuple)):
            token_coords = torch.stack(list(token_coords), dim=1)
        if token_coords is not None and token_coords.ndim != 4:
            raise ValueError("MVAvatarBlock.forward expects token_coords with shape [B, V, L, 3]")

        if isinstance(sa3d_coords, (list, tuple)):
            sa3d_coords = torch.stack(list(sa3d_coords), dim=1)
        elif sa3d_coords is None and token_coords is not None:
            sa3d_coords = token_coords
        if sa3d_coords is not None and sa3d_coords.ndim != 4:
            raise ValueError("MVAvatarBlock.forward expects sa3d_coords with shape [B, V, L, 3]")

        B, V, L, _ = posed_token_additive.shape
        x_query = avatar_tokens[:, None, :, :] + posed_token_additive

        if spatial_cond_coords.ndim == 4:
            sc_per_view = [spatial_cond_coords[:, v] for v in range(V)]
        else:
            sc_per_view = [spatial_cond_coords] * V

        per_view_tokens = torch.stack(
            [
                self._forward_mv_stage(
                    x_query[:, v],
                    spatial_cond_tokens[:, v],
                    sc_per_view[v],
                    token_coords[:, v] if token_coords is not None else None,
                    sa3d_coords[:, v] if sa3d_coords is not None else None,
                )
                for v in range(V)
            ],
            dim=1,
        )

        fused_tokens = multiview_fusion(avatar_tokens, per_view_tokens)
        return self._forward_shared_stage(fused_tokens, patch_uv_coords)


class MVAvatarBlockV2(MVAvatarBlock):
    """Multi-view avatar block V2 with per-view posed-token updates.

    Returns:
        avatar_tokens: [B, L, C] (canonical)
        posed_avatar_tokens: [B, V, L, C] (per-view posed)
    """

    def _forward_posed_branch(
        self,
        x: torch.Tensor,
        spatial_cond_tokens: torch.Tensor,
        spatial_cond_coords: torch.Tensor,
        token_coords: torch.Tensor,
        sa3d_coords: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # 1. 3D RoPE Cross-Attention from posed avatar tokens to per-view image tokens.
        q = self.ln_t2i_q(x)
        kv = self.ln_t2i_kv(spatial_cond_tokens)

        if self.token_to_image_attn_type != "fac-rope":
            raise ValueError("MVAvatarBlockV2 requires fac-rope cross attention")
        sc = spatial_cond_coords
        if sc.ndim == 2:
            if sc.shape[-1] == 2:
                sc = F.pad(sc, (0, 1), value=0.0)
            sc = sc.unsqueeze(0).expand(x.shape[0], -1, -1)
        elif sc.shape[-1] == 2:
            sc = F.pad(sc, (0, 1), value=0.0)

        cross_out = self.t2i_attn(
            q=q,
            kv=kv,
            q_coords=token_coords,
            kv_coords=sc,
        )

        x = x + self.drop_path(self.proj_drop(cross_out))

        # 2. 3D RoPE Self-Attention over the posed avatar tokens.
        if self.sa3d is None:
            raise ValueError("MVAvatarBlockV2 requires 3D RoPE self attention")
        if sa3d_coords is None:
            sa3d_coords = token_coords
        x_ln = self.ln_sa3d(x)
        sa3d_out = self.sa3d(x_ln, coords=sa3d_coords)
        x = x + self.drop_path(self.proj_drop(sa3d_out))

        # 3. FFN
        x = x + self.drop_path(self.ffn(self.ln_ffn(x)))
        return x

    def forward(
        self,
        avatar_tokens: torch.Tensor,
        posed_token_additive: torch.Tensor,
        spatial_cond_tokens: torch.Tensor,
        token_coords: Union[torch.Tensor, List[torch.Tensor]],
        spatial_cond_coords: torch.Tensor,
        patch_uv_coords: torch.Tensor,
        multiview_fusion: nn.Module,
        sa3d_coords: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if spatial_cond_tokens.ndim != 4:
            raise ValueError("MVAvatarBlockV2.forward expects spatial_cond_tokens with shape [B, V, N, C]")
        if posed_token_additive.ndim != 4:
            raise ValueError("MVAvatarBlockV2.forward expects posed_token_additive with shape [B, V, L, C]")

        if isinstance(token_coords, (list, tuple)):
            token_coords = torch.stack(list(token_coords), dim=1)
        if token_coords is not None and token_coords.ndim != 4:
            raise ValueError("MVAvatarBlockV2.forward expects token_coords with shape [B, V, L, 3]")

        if isinstance(sa3d_coords, (list, tuple)):
            sa3d_coords = torch.stack(list(sa3d_coords), dim=1)
        elif sa3d_coords is None and token_coords is not None:
            sa3d_coords = token_coords
        if sa3d_coords is not None and sa3d_coords.ndim != 4:
            raise ValueError("MVAvatarBlockV2.forward expects sa3d_coords with shape [B, V, L, 3]")

        B, V, L, _ = posed_token_additive.shape
        x_query = avatar_tokens[:, None, :, :] + posed_token_additive

        if spatial_cond_coords.ndim == 4:
            sc_per_view = [spatial_cond_coords[:, v] for v in range(V)]
        else:
            sc_per_view = [spatial_cond_coords] * V

        # Run per-view posed branch
        posed_avatar_tokens = torch.stack(
            [
                self._forward_posed_branch(
                    x_query[:, v],
                    spatial_cond_tokens[:, v],
                    sc_per_view[v],
                    token_coords[:, v] if token_coords is not None else None,
                    sa3d_coords[:, v] if sa3d_coords is not None else None,
                )
                for v in range(V)
            ],
            dim=1,
        )

        # Update canonical branch from posed branch outputs
        fused_tokens = multiview_fusion(avatar_tokens, posed_avatar_tokens)

        # Shared UV stage on fused tokens
        x_ln = self.ln_sa2d(fused_tokens)
        sa2d_out = self.sa2d(
            x_ln,
            x_ln,
            x_ln,
            patch_positions=patch_uv_coords,
        )
        avatar_tokens_new = fused_tokens + self.drop_path(self.proj_drop(sa2d_out))
        avatar_tokens_new = avatar_tokens_new + self.drop_path(self.ffn(self.ln_ffn(avatar_tokens_new)))

        return avatar_tokens_new, posed_avatar_tokens
