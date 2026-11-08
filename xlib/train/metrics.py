"""Image quality metrics for ARGAvatar training, evaluation, and inference."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional


def _ssim_gaussian_kernel(window_size: int = 11, sigma: float = 1.5) -> torch.Tensor:
    coords = torch.arange(window_size, dtype=torch.float32) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    kernel = g.unsqueeze(1) * g.unsqueeze(0)
    return kernel / kernel.sum()


def compute_ssim(
    pred: torch.Tensor,
    gt: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    window_size: int = 11,
    data_range: float = 1.0,
) -> torch.Tensor:
    """SSIM for [C,H,W] (returns scalar) or [N,C,H,W] (returns [N])."""
    single = pred.ndim == 3
    if single:
        pred = pred.unsqueeze(0)
        gt = gt.unsqueeze(0)
        if mask is not None:
            mask = mask.unsqueeze(0)

    if mask is not None:
        if mask.ndim == 3:
            mask = mask.unsqueeze(1)
        mask = mask.to(device=pred.device, dtype=pred.dtype)
        pred = pred * mask
        gt = gt * mask

    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2

    kernel = _ssim_gaussian_kernel(window_size).to(device=pred.device, dtype=pred.dtype)
    C = pred.shape[1]
    kernel = kernel.expand(C, 1, window_size, window_size)
    pad = window_size // 2

    mu1 = F.conv2d(pred, kernel, padding=pad, groups=C)
    mu2 = F.conv2d(gt, kernel, padding=pad, groups=C)
    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(pred * pred, kernel, padding=pad, groups=C) - mu1_sq
    sigma2_sq = F.conv2d(gt * gt, kernel, padding=pad, groups=C) - mu2_sq
    sigma12 = F.conv2d(pred * gt, kernel, padding=pad, groups=C) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    )
    result = ssim_map.view(ssim_map.shape[0], -1).mean(dim=1)
    return result[0] if single else result


def compute_psnr(
    pred: torch.Tensor, gt: torch.Tensor, mask: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """PSNR for [C,H,W] (returns scalar) or [N,C,H,W] (returns [N])."""
    single = pred.ndim == 3
    if single:
        pred = pred.unsqueeze(0)
        gt = gt.unsqueeze(0)
        if mask is not None:
            mask = mask.unsqueeze(0)

    if mask is not None:
        if mask.ndim == 3:
            mask = mask.unsqueeze(1)
        mask = mask.to(device=pred.device, dtype=pred.dtype)
        pred = pred * mask
        gt = gt * mask

    mse = F.mse_loss(pred, gt, reduction="none")
    mse = mse.reshape(mse.shape[0], -1).mean(dim=1)

    psnr = -10.0 * torch.log10(mse + 1e-10)
    psnr[mse == 0] = 100.0
    return psnr[0] if single else psnr


def compute_lpips_masked(
    preds: torch.Tensor,
    gts: torch.Tensor,
    masks: torch.Tensor,
    bg_colors: torch.Tensor,
    lpips_model: nn.Module,
) -> torch.Tensor:
    """LPIPS on bg-filled composites for [C,H,W] (scalar) or [N,C,H,W] ([N])."""
    single = preds.ndim == 3
    if single:
        preds = preds.unsqueeze(0)
        gts = gts.unsqueeze(0)
        masks = masks.unsqueeze(0)

    with torch.autocast(device_type="cuda", enabled=False):
        preds = preds.float()
        gts = gts.float()
        m = masks.float()
        if m.ndim == 3:
            m = m.unsqueeze(1)

        bg = bg_colors.to(device=preds.device, dtype=preds.dtype)
        if bg.ndim == 1:
            bg = bg.reshape(-1)[:3].reshape(1, 3, 1, 1)
        elif bg.ndim == 2:
            bg = bg[:, :3].reshape(bg.shape[0], 3, 1, 1)
        else:
            bg = bg.reshape(preds.shape[0], 3, 1, 1)

        pred_comp = preds * m + bg * (1 - m)
        gt_comp = gts * m + bg * (1 - m)
        result = lpips_model(pred_comp, gt_comp, normalize=True).flatten()
    return result[0] if single else result


def compute_overall_eval_metrics(all_results: Dict[str, Dict]) -> Dict[str, float]:
    """Weighted-mean aggregate of per-dataset eval metrics."""
    ds_keys = list(all_results.keys())
    total_n_in = sum(all_results[ds].get("n_input", 0) for ds in ds_keys)
    total_n_cross = sum(all_results[ds].get("n_cross", 0) for ds in ds_keys)

    def _wavg(key: str, n_key: str) -> float:
        total = sum(all_results[ds].get(n_key, 0) for ds in ds_keys)
        if total == 0:
            return 0.0
        return sum(all_results[ds].get(key, 0.0) * all_results[ds].get(n_key, 0) for ds in ds_keys) / total

    return {
        "psnr_input": _wavg("psnr_input", "n_input"),
        "psnr_cross": _wavg("psnr_cross", "n_cross"),
        "lpips_input": _wavg("lpips_input", "n_input"),
        "lpips_cross": _wavg("lpips_cross", "n_cross"),
        "ssim_input": _wavg("ssim_input", "n_input"),
        "ssim_cross": _wavg("ssim_cross", "n_cross"),
        "n_input": total_n_in,
        "n_cross": total_n_cross,
    }
