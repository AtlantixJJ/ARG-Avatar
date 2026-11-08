"""StageTimer: wall-clock + peak GPU memory tracking across samples."""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Any, Dict, List

import numpy as np
import torch


class StageTimer:
    """Accumulates per-stage wall-clock times and peak GPU memory across samples."""

    def __init__(self, sync_cuda: bool = False):
        self._stage_times: Dict[str, List[float]] = {}
        self._peak_gpu_gbs: List[float] = []
        self._sync_cuda = sync_cuda and torch.cuda.is_available()

    def begin_sample(self):
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    def end_sample(self) -> Dict[str, Any]:
        """Record peak GPU for this sample; return per-sample timing dict."""
        peak_gb = None
        if torch.cuda.is_available():
            peak_gb = torch.cuda.max_memory_allocated() / 1024**3
            self._peak_gpu_gbs.append(peak_gb)
        sample_timing: Dict[str, Any] = {
            f"{stage}_s": round(times[-1], 4) for stage, times in self._stage_times.items() if times
        }
        sample_timing["peak_gpu_mem_gb"] = round(peak_gb, 3) if peak_gb is not None else None
        return sample_timing

    @contextmanager
    def stage(self, name: str):
        if self._sync_cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        try:
            yield
        finally:
            if self._sync_cuda:
                torch.cuda.synchronize()
            self._stage_times.setdefault(name, []).append(time.perf_counter() - t0)

    def to_dict(self) -> Dict[str, Any]:
        """Aggregate stats across all samples for the dataset-level summary."""
        stages = {}
        total_s = 0.0
        n = 0
        for stage, times in self._stage_times.items():
            arr = np.array(times)
            stages[stage] = {
                "total_s": round(float(arr.sum()), 4),
                "mean_s": round(float(arr.mean()), 4),
                "std_s": round(float(arr.std()), 4),
            }
            total_s += arr.sum()
            n = max(n, len(arr))
        return {
            "n_samples": n,
            "stages": stages,
            "total_s": round(total_s, 4),
            "throughput_samples_per_s": round(n / total_s, 4) if total_s > 0 else 0.0,
            "peak_gpu_mem_gb": round(max(self._peak_gpu_gbs), 3) if self._peak_gpu_gbs else None,
        }
