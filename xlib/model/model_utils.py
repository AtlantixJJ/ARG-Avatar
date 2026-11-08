"""Utility functions for models."""
import math
import os
import re
import torch
import torch.nn as nn
import logging
from typing import Dict, Optional


# ── SH-degree upgrade remapping ──────────────────────────────────────────────
# GSLayer n_layers=0 splits a flat vector by RAW_TO_ACTIVATED order:
#   rotation(4) | scale(3) | opacity(1) | sh_((sh+1)²×3) | offset_xyz(3)
# Total channels = 11 + (sh+1)²×3.  When sh_degree grows, offset_xyz shifts
# right and new SH bands are inserted as zeros before it.

_GS_FIXED_CH = 11   # rotation(4)+scale(3)+opacity(1)+offset_xyz(3)

def _gs_total_to_sh_degree(total: int) -> int:
    """Return sh_degree from total GS channels, or -1 if not a valid layout."""
    sh_ch = total - _GS_FIXED_CH
    if sh_ch <= 0:
        return -1
    deg_p1 = math.isqrt(sh_ch // 3)
    if deg_p1 * deg_p1 * 3 != sh_ch:
        return -1
    return deg_p1 - 1

_SH_LINEAR_RE = re.compile(r"token_gsheads\.\d+\.out_layers\.sh\.\d+\.(weight|bias)$")


def _try_adapt_sh_degree(
    key: str,
    src: torch.Tensor,
    dst_shape: torch.Size,
) -> Optional[torch.Tensor]:
    """
    Try to adapt *src* to *dst_shape* by upgrading SH degree.

    Handles two cases:
    1. conv_out.weight / conv_out.bias — full GS channel remap (offset_xyz shifts).
    2. token_gsheads[*].out_layers.sh — zero-pad the extra SH output channels.

    Returns the adapted tensor on success, None if this key/shape doesn't match.
    """
    # ── case 0: deform_shape output expansion 3 → 10 ──
    _DEFORM_SHAPE_KEYS = ("posedep_uv_decoder", "token_pose_offset_proj")
    if any(k in key for k in _DEFORM_SHAPE_KEYS):
        if key.endswith(".weight") or key.endswith(".bias"):
            src_ch, dst_ch = src.shape[0], dst_shape[0]
            if src_ch == 3 and dst_ch == 10:
                pad = torch.zeros(7, *src.shape[1:], dtype=src.dtype)
                if key.endswith(".bias"):
                    pad[3] = 1.0
                return torch.cat([src, pad], dim=0)
        return None

    # ── case 1: conv_out (dense UV decoder output) ──
    if key.endswith(".conv_out.weight") or key.endswith(".conv_out.bias"):
        src_ch = src.shape[0]
        dst_ch = dst_shape[0]
        if src_ch == dst_ch:
            return None  # no SH change, genuine mismatch
        src_sh = _gs_total_to_sh_degree(src_ch)
        dst_sh = _gs_total_to_sh_degree(dst_ch)
        if src_sh < 0 or dst_sh < 0 or dst_sh <= src_sh:
            return None  # not a GS layout or downgrade — skip

        src_sh_ch = (src_sh + 1) ** 2 * 3
        dst_sh_ch = (dst_sh + 1) ** 2 * 3
        # layout: [0:8] fixed-head | [8:8+src_sh_ch] sh | [8+src_sh_ch:] xyz
        HEAD = 8   # rotation(4)+scale(3)+opacity(1)

        out = torch.zeros(dst_ch, *src.shape[1:], dtype=src.dtype)
        out[:HEAD] = src[:HEAD]                         # rotation+scale+opacity
        out[HEAD:HEAD+src_sh_ch] = src[HEAD:HEAD+src_sh_ch]  # existing SH bands
        # new SH bands stay zero
        out[HEAD+dst_sh_ch:] = src[HEAD+src_sh_ch:]    # offset_xyz (shifted)
        return out

    # ── case 2: patch_embed Conv2d (input channels grow, SH appended last) ──
    # UVConditionCollector.total_dim includes neutral_avatar = gs_to_feature()
    # which orders: xyz|scale|rotation|opacity|sh_.  New SH bands land at the
    # end, so zero-pad dim=1 of the Conv2d weight.
    if key.endswith("patch_embed.weight") and src.ndim == 4:
        src_in, dst_in = src.shape[1], dst_shape[1]
        diff = dst_in - src_in
        if src.shape[0] == dst_shape[0] and diff > 0 and diff % 3 == 0:
            pad = torch.zeros(src.shape[0], diff, *src.shape[2:], dtype=src.dtype)
            return torch.cat([src, pad], dim=1)
        return None

    # ── case 3: token_gshead SH linear layer ────────────────────────────────
    if _SH_LINEAR_RE.search(key):
        src_out = src.shape[0]
        dst_out = dst_shape[0]
        if src_out == dst_out or src_out > dst_out:
            return None
        # src_out and dst_out must both be (sh+1)²×3
        src_sh_p1_sq = src_out // 3
        dst_sh_p1_sq = dst_out // 3
        if src_sh_p1_sq * 3 != src_out or dst_sh_p1_sq * 3 != dst_out:
            return None

        extra = dst_out - src_out
        if src.ndim == 2:   # weight [out, in]
            pad = torch.zeros(extra, src.shape[1], dtype=src.dtype)
        else:               # bias [out]
            pad = torch.zeros(extra, dtype=src.dtype)
        return torch.cat([src, pad], dim=0)

    return None


def _repair_deform_shape_bias(key: str, tensor: torch.Tensor) -> torch.Tensor:
    """Restore identity quaternion bias for old deform_shape-expanded checkpoints."""
    if not key.endswith(".bias") or tensor.shape != (10,):
        return tensor
    if not any(k in key for k in ("posedep_uv_decoder", "token_pose_offset_proj")):
        return tensor
    if tensor[6:10].detach().abs().max() > 1e-8:
        return tensor
    repaired = tensor.clone()
    repaired[6] = 1.0
    return repaired


class ModelUtils(nn.Module):
    """Utility functions for models."""

    def __init__(self):
        super().__init__()
        self.trainable_modules = []

    def _load_checkpoint_state_dict(self, state_dict: Dict[str, torch.Tensor], ckpt_path: Optional[str] = None):
        model_state = self.state_dict()
        filtered_state = {}
        skipped_shape = []
        unexpected_in_ckpt = []
        trainable_prefixes = tuple(self.trainable_modules)

        def _is_trainable_key(key: str) -> bool:
            for prefix in trainable_prefixes:
                if key == prefix or key.startswith(prefix + "."):
                    return True
            return False

        ckpt_trainable_count = 0

        for k, v in state_dict.items():
            if not _is_trainable_key(k):
                continue
            ckpt_trainable_count += 1
            if k not in model_state:
                unexpected_in_ckpt.append(k)
                continue
            target = model_state[k]
            if isinstance(v, torch.Tensor) and v.shape != target.shape:
                adapted = _try_adapt_sh_degree(k, v, target.shape)
                if adapted is not None and adapted.shape == target.shape:
                    filtered_state[k] = _repair_deform_shape_bias(k, adapted)
                else:
                    skipped_shape.append((k, tuple(v.shape), tuple(target.shape)))
                continue
            filtered_state[k] = _repair_deform_shape_bias(k, v) if isinstance(v, torch.Tensor) else v

        # Load with a full state_dict so non-trainable/frozen keys are never reported as missing.
        merged_state = dict(model_state)
        merged_state.update(filtered_state)
        self.load_state_dict(merged_state, strict=True)

        trainable_model_keys = {k for k in model_state.keys() if _is_trainable_key(k)}
        loaded_trainable_keys = set(filtered_state.keys())
        missing = sorted(trainable_model_keys - loaded_trainable_keys)
        unexpected = list(dict.fromkeys(unexpected_in_ckpt))

        prefix = f"[{ckpt_path}] " if ckpt_path is not None else ""
        print(f"{prefix}Loaded compatible trainable checkpoint parameters: {len(filtered_state)}/{ckpt_trainable_count}")
        if skipped_shape:
            print(f"{prefix}Skipped shape-mismatch keys: {len(skipped_shape)}")
            for k, src_shape, dst_shape in skipped_shape[:20]:
                print(f"  - {k}: ckpt{src_shape} != model{dst_shape}")
            if len(skipped_shape) > 20:
                print(f"  ... ({len(skipped_shape) - 20} more)")
        self._print_key_summary(f"{prefix}Missing keys after load", missing)
        self._print_key_summary(f"{prefix}Unexpected keys in checkpoint", unexpected)
        logging.info(
            "Checkpoint load summary (%s): loaded=%d, skipped_shape=%d, missing=%d, unexpected=%d",
            ckpt_path if ckpt_path is not None else "in-memory",
            len(filtered_state),
            len(skipped_shape),
            len(missing),
            len(unexpected),
        )
        return missing, unexpected, skipped_shape

    def load_checkpoint(self, ckpt_path: str, **kwargs):
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        state_dict = ckpt.get("state_dict", ckpt)
        missing, unexpected, _ = self._load_checkpoint_state_dict(state_dict, ckpt_path=ckpt_path)
        return missing, unexpected

    def get_trainable_parameters(self):
        """Return all parameters with requires_grad=True."""
        return [p for p in self.parameters() if p.requires_grad]

    def trainable_state_dict(self):
        """Return the state dict of the model with only trainable parameters."""
        return {k: v for k, v in self.state_dict().items() if k.split('.')[0] in self.trainable_modules}

    def to(self, device):
        """Move model and MHR animatable Gaussians to device."""
        super().to(device)
        self.mhr_animatable_gs = self.mhr_animatable_gs.to(device)
        return self

    @staticmethod
    def _print_key_summary(title: str, keys, max_items: int = 20) -> None:
        if not keys:
            return
        print(f"{title}: {len(keys)}")
        for k in keys[:max_items]:
            print(f"  - {k}")
        if len(keys) > max_items:
            print(f"  ... ({len(keys) - max_items} more)")