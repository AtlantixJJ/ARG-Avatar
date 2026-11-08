"""Worker-side evaluation: types, metrics, ModelEvalHost, ModelHostPool."""

import argparse
import gc
import contextlib
import faulthandler
import os
import random
import sys
import traceback
import yaml
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Must be set before any CUDA-initializing import.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ["MKL_THREADING_LAYER"] = "GNU"
faulthandler.enable(all_threads=True)

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import lpips
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import multiprocessing as mp
import signal
from omegaconf import OmegaConf

torch.multiprocessing.set_sharing_strategy("file_system")

from xlib.utils import AsyncExecutor, load_config
from xlib.train.train_helper import setup_model_sam3d, get_unwrapped_model, resolve_sam3d_cfg
from xlib.basetype import TensorDict
from xlib.train.metrics import (
    compute_psnr,
    compute_ssim,
    compute_lpips_masked,
)
from xlib.render.gaussian_splatting import set_gaussian_type
from evaluation.common.results import (
    chw_tensor_to_uint8,
    mask_tensor_to_uint8,
    save_rgb,
    save_rgba,
    encoded_artifact_name,
)
from evaluation.argavatar.eval_argavatar import (
    load_cached_dataset_results,
    _get_val_ds_keys,
    render_in_raw_space,
)
from evaluation.argavatar.eval_profile import make_profile_event

# IPC message tag constants
MSG_REQUEST_JOB  = "request_job"
MSG_JOB          = "job"
MSG_JOB_DONE     = "job_done"
MSG_LOAD_CPU     = "load_cpu_model"
MSG_END_DATASET  = "end_dataset"
MSG_DATASET_DONE = "dataset_done"
MSG_WORKER_DONE  = "worker_done"
MSG_WORKER_ERROR = "worker_error"
MSG_PROFILE      = "profile"

_ORIGINAL_CUDA_VISIBLE_DEVICES = os.environ.get("CUDA_VISIBLE_DEVICES")

# Per-process cache: image_list_path -> items dict (or None on failure).
_val_list_cache: Dict[str, Optional[Dict]] = {}


def _load_val_list_items(image_list_path: str) -> Optional[Dict]:
    if image_list_path in _val_list_cache:
        return _val_list_cache[image_list_path]
    try:
        with open(image_list_path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        items = data.get("items") if isinstance(data, dict) else None
    except Exception:
        items = None
    _val_list_cache[image_list_path] = items
    return items


def _expected_view_counts(train_args, ds_key: str, n_views: int, n_samples: int) -> Optional[Tuple[int, int]]:
    """Return (expected_n_input, expected_n_cross) from the val list YAML, or None if unavailable."""
    val_cfg = train_args.data.val
    if not (hasattr(val_cfg, "keys") and "ds_list" in val_cfg and "ds_keys" in val_cfg):
        return None
    ds_keys = list(val_cfg.ds_keys)
    if ds_key not in ds_keys:
        return None
    ds_cfg = dict(list(val_cfg.ds_list)[ds_keys.index(ds_key)])
    image_list_path = ds_cfg.get("image_list")
    if not image_list_path or not os.path.isfile(image_list_path):
        return None
    items = _load_val_list_items(image_list_path)
    if not items:
        return None
    sample_list = list(items.values())
    if n_samples > 0:
        sample_list = sample_list[:n_samples]
    num_sup_views = int(ds_cfg.get("num_sup_views", -1))
    n_input_total = 0
    n_cross_total = 0
    for item in sample_list:
        n_input_views = len(item.get("input_images", []))
        n_support_views = len(item.get("images", []))
        if num_sup_views >= 0:
            n_support_views = min(num_sup_views, n_support_views)

        # The evaluator renders input_images followed by support images, then
        # classifies every view not selected by input_view_mask as "cross".
        n_rendered = n_input_views + n_support_views
        n_selected = n_rendered if n_views < 0 else min(n_views, n_rendered)
        n_input_total += n_selected
        n_cross_total += n_rendered - n_selected
    return n_input_total, n_cross_total


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

@dataclass
class PendingEvalJob:
    ds_key: str
    batch_id: int
    sample_name: str
    count: int
    batch_dict: dict
    model_name: str


class ModelMeta:
    def __init__(self, model_name: str, checkpoint_path: str, train_args: Any, config_path: str):
        self.model_name = model_name
        self.checkpoint_path = checkpoint_path
        self.train_args = train_args
        self.config_path = config_path
        self.project_name = train_args.project_name
        self.ckpt_stem = os.path.splitext(os.path.basename(checkpoint_path))[0]
        self.val_ds_keys = _get_val_ds_keys(train_args)

    def results_dir_for_views(self, n_views: int) -> str:
        results_subdir = self.ckpt_stem
        if n_views is not None:
            results_subdir += f"_n{int(n_views)}"
        return os.path.join("expr", "results", self.project_name, results_subdir)


# ---------------------------------------------------------------------------
# Metrics accumulator
# ---------------------------------------------------------------------------

class MetricsAccumulator:
    KEYS = ("psnr_input", "psnr_cross", "lpips_input", "lpips_cross", "ssim_input", "ssim_cross")

    def __init__(self):
        self.sums: Dict[str, float] = {k: 0.0 for k in self.KEYS}
        self.counts: Dict[str, int] = {"input": 0, "cross": 0}

    def add_view(self, psnr: float, lpips_val: float, ssim: float, is_input: bool):
        if is_input:
            self.sums["psnr_input"]  += float(psnr)
            self.sums["lpips_input"] += float(lpips_val)
            self.sums["ssim_input"]  += float(ssim)
            self.counts["input"] += 1
        else:
            self.sums["psnr_cross"]  += float(psnr)
            self.sums["lpips_cross"] += float(lpips_val)
            self.sums["ssim_cross"]  += float(ssim)
            self.counts["cross"] += 1

    def merge_payload(self, payload: dict):
        for mk, value in payload["sums"].items():
            self.sums[mk] += value
        self.counts["input"] += payload["counts"]["input"]
        self.counts["cross"] += payload["counts"]["cross"]

    def to_payload(self) -> dict:
        return {"sums": dict(self.sums), "counts": dict(self.counts)}

    def to_metrics(self) -> Dict[str, Any]:
        s, c = self.sums, self.counts
        return {
            "psnr_input":  s["psnr_input"]  / max(1, c["input"]),
            "psnr_cross":  s["psnr_cross"]  / max(1, c["cross"]),
            "lpips_input": s["lpips_input"] / max(1, c["input"]),
            "lpips_cross": s["lpips_cross"] / max(1, c["cross"]),
            "ssim_input":  s["ssim_input"]  / max(1, c["input"]),
            "ssim_cross":  s["ssim_cross"]  / max(1, c["cross"]),
            "n_input": c["input"],
            "n_cross": c["cross"],
        }

    def is_empty(self) -> bool:
        return self.counts["input"] == 0 and self.counts["cross"] == 0


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def resolve_worker_device_index(gpu_id: str) -> int:
    gpu_id = str(gpu_id)
    if _ORIGINAL_CUDA_VISIBLE_DEVICES:
        visible = [x.strip() for x in _ORIGINAL_CUDA_VISIBLE_DEVICES.split(",") if x.strip()]
        # gpu_id is a physical GPU ID: find its 0-based position in CUDA_VISIBLE_DEVICES
        if gpu_id in visible:
            return visible.index(gpu_id)
        # gpu_id is already a 0-based device index
        if gpu_id.isdigit():
            idx = int(gpu_id)
            if idx < len(visible):
                return idx
    return int(gpu_id) if gpu_id.isdigit() else 0


def should_save_vis_for_count(n_vis: int, count: int) -> bool:
    return n_vis < 0 or count < n_vis


def apply_view_mask(batch_dict: dict, n_input_views: int, device) -> TensorDict:
    batch = TensorDict.from_dict(batch_dict).to(device)
    B, V = batch["images"].shape[:2]
    n = min(n_input_views, V)
    ivm = torch.zeros((B, V), dtype=torch.bool, device=device)
    ivm[:, :n] = True
    batch["input_view_mask"] = ivm
    # G4: masking composite on the worker GPU instead of the manager CPU.
    bg = batch["bg_color"].view(B, V, 3, 1, 1)
    batch["images"] = batch["images"] * batch["masks"] + (1 - batch["masks"]) * bg
    if "hr_images" in batch:
        _, _, _, H, W = batch["masks"].shape
        H_r, W_r = batch["hr_images"].shape[-2:]
        sr_mask = F.interpolate(
            batch["masks"].view(B * V, 1, H, W), size=(H_r, W_r),
            mode="bilinear", align_corners=False,
        ).view(B, V, 1, H_r, W_r)
        batch["hr_images"] = batch["hr_images"] * sr_mask + (1 - sr_mask) * bg
    return batch


def _preprocess_batch_cpu(batch_raw) -> dict:
    if isinstance(batch_raw, list) and len(batch_raw) == 2:
        _, batch_raw = batch_raw
    batch = dict(batch_raw)
    # Set bg to white; masking composite is deferred to apply_view_mask on the worker GPU (G4).
    batch["bg_color"] = torch.ones_like(batch["bg_color"])
    for v in batch.values():
        if isinstance(v, torch.Tensor):
            v.share_memory_()
    return batch


def _move_tensors_to_cpu(obj):
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu()
    if isinstance(obj, dict):
        return {k: _move_tensors_to_cpu(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_move_tensors_to_cpu(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_move_tensors_to_cpu(v) for v in obj)
    return obj


def _autocast_dtype(train_args) -> Optional[torch.dtype]:
    precision = str(getattr(train_args.training, "mixed_precision", "no")).lower()
    if precision == "bf16":
        return torch.bfloat16
    if precision == "fp16":
        return torch.float16
    return None


def _save_vis_from_snapshots(vis_dir: str, sample_name: str, snapshots: List[Dict[str, Any]]) -> None:
    sample_dir = os.path.join(vis_dir, sample_name)
    render_subdir = os.path.join(sample_dir, "render")
    render_human_input_subdir = os.path.join(sample_dir, "render_human_input")
    os.makedirs(render_subdir, exist_ok=True)
    os.makedirs(render_human_input_subdir, exist_ok=True)
    for snap in snapshots:
        image_path = snap["image_path"]
        assert image_path is not None, "image_path is required"
        fname = encoded_artifact_name(image_path, suffix=".png")
        if snap.get("render_mask_np") is not None:
            save_rgba(os.path.join(render_subdir, fname), snap["render_np"], snap["render_mask_np"])
        else:
            save_rgb(os.path.join(render_subdir, fname), snap["render_np"])
        save_rgb(os.path.join(render_human_input_subdir, fname), snap["gt_np"])


# ---------------------------------------------------------------------------
# ModelEvalHost
# ---------------------------------------------------------------------------

class ModelEvalHost:
    def __init__(self, gpu_id, device_index, job_data_list, eval_views, input_queue, output_queue, args_dict):
        self.gpu_id = gpu_id
        self.device = torch.device(f"cuda:{device_index}")
        self.eval_views: List[int] = eval_views
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.args = argparse.Namespace(**args_dict)

        self.cpu_models: Dict[str, nn.Module] = {}
        self.gpu_model_name: Optional[str] = None
        self.gpu_model: Optional[nn.Module] = None

        self.model_names: List[str] = []
        self.model_meta: Dict[str, ModelMeta] = {}
        for jd in job_data_list:
            self.model_names.append(jd["model_name"])
            train_args = OmegaConf.create(jd["train_args"])
            meta = ModelMeta(jd["model_name"], jd["checkpoint_path"], train_args, jd["config_path"])
            self.model_meta[jd["model_name"]] = meta

        self.lpips_fn = None
        self._vis_async_executor = AsyncExecutor()
        self._cached_ds_views: set = set()  # populated by _precompute_cached_views in run()

    def _precompute_cached_views(self) -> set:
        """Return the set of (model_name, ds_key, n_views) triples whose cached results are valid."""
        if self.args.ignore_cache:
            return set()
        cached = set()
        for model_name, meta in self.model_meta.items():
            for n_views in self.eval_views:
                out_json = os.path.join(meta.results_dir_for_views(n_views), "eval_results.json")
                ds_results = load_cached_dataset_results(out_json)
                for ds_key, row in ds_results.items():
                    expected = _expected_view_counts(meta.train_args, ds_key, n_views, self.args.n_samples)
                    if expected is None or (row["n_input"] == expected[0] and row["n_cross"] == expected[1]):
                        cached.add((model_name, ds_key, n_views))
        return cached

    @contextlib.contextmanager
    def profile_span(self, name: str, **kwargs):
        self.emit_profile(f"start_{name}", **kwargs)
        try:
            yield
        finally:
            self.emit_profile(f"end_{name}", **kwargs)

    def emit_profile(self, state: str, **payload):
        event = make_profile_event(self.args, "worker", state, gpu_id=self.gpu_id, **payload)
        if event is not None:
            self.output_queue.put((MSG_PROFILE, event))

    def cuda_sync(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize(self.device)

    # Cap on models held in CPU RAM per worker. The cache is a switching
    # optimisation, not a correctness requirement -- letting it grow unbounded
    # is what OOM-killed a 6-model sweep. With static pinning (the usual case)
    # a worker only ever sees one model and this never trips.
    MAX_CPU_MODELS = 2

    def ensure_cpu_model(self, model_name: str):
        if model_name in self.cpu_models:
            return
        while len(self.cpu_models) >= self.MAX_CPU_MODELS:
            # Never evict the model currently resident on the GPU.
            victim = next(
                (name for name in self.cpu_models if name != self.gpu_model_name),
                None,
            )
            if victim is None:
                break
            print(f"[GPU {self.gpu_id}] Evicting {victim} from CPU RAM (cap {self.MAX_CPU_MODELS})")
            del self.cpu_models[victim]
            gc.collect()
        meta = self.model_meta[model_name]
        print(f"[GPU {self.gpu_id}] Loading {model_name} to CPU RAM...")
        seed = int(meta.train_args.training.get("seed", 42))
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        cfg = resolve_sam3d_cfg(meta.train_args, meta.config_path)
        set_gaussian_type(meta.train_args.model.get("gaussian_type", "3d"))
        meta.train_args.model["load_checkpoint"] = meta.checkpoint_path
        model = setup_model_sam3d(meta.train_args, cfg, device=torch.device("cpu"))
        model.eval()
        self.cpu_models[model_name] = model

    def ensure_gpu_model(self, model_name: str):
        if self.gpu_model_name == model_name:
            return
        self.ensure_cpu_model(model_name)
        if self.gpu_model is not None:
            self.gpu_model.cpu()
            self.gpu_model = None
            torch.cuda.empty_cache()
        self.gpu_model = self.cpu_models[model_name].to(self.device)
        self.gpu_model_name = model_name

    def _job_view_steps(self, ds_key, batch_dict, model_name, do_vis):
        V_batch = batch_dict["images"].shape[1]
        steps = []
        for n_views in self.eval_views:
            if n_views > V_batch:
                continue
            if (model_name, ds_key, n_views) in self._cached_ds_views and not do_vis:
                continue
            steps.append(n_views)
        return steps

    def _forward_and_offload(self, model, job_batch, autocast_dtype) -> Tuple[Any, Any]:
        with torch.autocast(
            device_type="cuda",
            dtype=autocast_dtype or torch.float32,
            enabled=autocast_dtype is not None,
        ):
            result = get_unwrapped_model(model).two_phase_model_forward(job_batch, output_on_cpu=True)
        _pass = result["passes"][-1]
        gs_tpose = _move_tensors_to_cpu(_pass["output_canonical_gs_dic"])
        offset_pose = _move_tensors_to_cpu(_pass.get("offset_pose"))
        del result, _pass
        return gs_tpose, offset_pose

    def _render_views(self, model, job_batch, gs_tpose, offset_pose) -> List[Dict]:
        return render_in_raw_space(
            model, job_batch, gs_tpose, offset_pose, self.device, self.args.max_render_height
        )

    def _score_render_entries(self, render_entries, ivm_1d, V_batch) -> MetricsAccumulator:
        acc = MetricsAccumulator()
        if not render_entries:
            return acc

        with torch.no_grad():
            for i, e in enumerate(render_entries):
                pred = e["render_result"]["body"]["image"][0].to(self.device)
                gt = e["gt_image"].to(self.device)
                mask = e["gt_mask"].to(self.device)
                bg = e["bg_color"].to(self.device)
                psnr = compute_psnr(pred, gt, mask).item()
                ssim = compute_ssim(pred, gt, mask).item()
                lp = compute_lpips_masked(pred, gt, mask, bg, self.lpips_fn).item()
                is_input = bool(ivm_1d[i % V_batch].item())
                acc.add_view(float(psnr), float(lp), float(ssim), is_input)
        return acc

    def _snapshot_vis_render_entries(
        self, model_name: str, n_views: int, ds_key: str,
        sample_name: str, count: int, render_entries: List[Dict[str, Any]],
    ) -> None:
        if self.args.n_vis >= 0 and count >= self.args.n_vis:
            return
        vis_dir = os.path.join(
            self.model_meta[model_name].results_dir_for_views(n_views), "vis", ds_key
        )
        os.makedirs(vis_dir, exist_ok=True)
        snapshots = [
            {
                "render_np": chw_tensor_to_uint8(e["render_result"]["body"]["image"][0]),
                "render_mask_np": (
                    mask_tensor_to_uint8(e["render_result"]["body"]["mask"][0])
                    if e["render_result"]["body"].get("mask") is not None
                    else None
                ),
                "gt_np": chw_tensor_to_uint8(e["gt_image"]),
                "image_path": e.get("image_path"),
            }
            for e in render_entries
        ]
        self._vis_async_executor(_save_vis_from_snapshots, vis_dir, sample_name, snapshots)

    def process_job(self, ds_key, sample_name, count, batch_dict, model_name):
        previous_model_name = self.gpu_model_name
        self.cuda_sync()
        with self.profile_span(
            "model_switch",
            model_name=model_name, ds_key=ds_key, batch_id=count,
            sample_name=sample_name, previous_model_name=previous_model_name,
        ):
            self.ensure_gpu_model(model_name)
            self.cuda_sync()

        model = self.gpu_model
        meta = self.model_meta[model_name]
        autocast_dtype = _autocast_dtype(meta.train_args)
        V_batch = batch_dict["images"].shape[1]
        job_results = {}
        do_vis = should_save_vis_for_count(self.args.n_vis, count)
        view_steps = self._job_view_steps(ds_key, batch_dict, model_name, do_vis)

        for n_views in view_steps:
            metrics_cached = (model_name, ds_key, n_views) in self._cached_ds_views
            job_batch = apply_view_mask(batch_dict, n_views, self.device)
            prof_kw = dict(
                model_name=model_name, ds_key=ds_key, batch_id=count,
                sample_name=sample_name, n_views=n_views,
            )

            self.cuda_sync()
            with self.profile_span("forward", **prof_kw):
                gs_tpose, offset_pose = self._forward_and_offload(model, job_batch, autocast_dtype)
                self.cuda_sync()

            self.cuda_sync()
            with self.profile_span("render", **prof_kw):
                entries = self._render_views(model, job_batch, gs_tpose, offset_pose)
                self.cuda_sync()

            ivm_1d = job_batch["input_view_mask"][0].bool()

            with self.profile_span("visualize", **prof_kw):
                self._snapshot_vis_render_entries(model_name, n_views, ds_key, sample_name, count, entries)

            self.cuda_sync()
            with self.profile_span("metrics", **prof_kw):
                acc = self._score_render_entries(entries, ivm_1d, V_batch)
                self.cuda_sync()

            if not metrics_cached:
                job_results[n_views] = acc.to_payload()

            del entries, gs_tpose, offset_pose, job_batch

        return job_results

    def run(self):
        torch.cuda.set_device(self.device)
        torch.set_grad_enabled(False)
        self.lpips_fn = lpips.LPIPS(net='alex').to(self.device)
        self.lpips_fn.requires_grad_(False)
        self._cached_ds_views = self._precompute_cached_views()

        try:
            while True:
                with self.profile_span("request_job", loaded_model_name=self.gpu_model_name):
                    self.output_queue.put((MSG_REQUEST_JOB, self.gpu_id, self.gpu_model_name, list(self.cpu_models.keys())))
                with self.profile_span("queue_get", loaded_model_name=self.gpu_model_name):
                    msg = self.input_queue.get()
                if msg is None:
                    break

                with self.profile_span("job_unpack", loaded_model_name=self.gpu_model_name):
                    tag = msg[0]
                    if tag == MSG_LOAD_CPU:
                        _, model_name = msg
                    elif tag == MSG_JOB:
                        _, ds_key, batch_id, sample_name, count, batch_dict, model_name = msg
                    elif tag == MSG_END_DATASET:
                        _, ds_key = msg

                if tag == MSG_LOAD_CPU:
                    self.emit_profile("start_cpu_model_preload", model_name=model_name)
                    self.ensure_cpu_model(model_name)
                    self.emit_profile(
                        "end_cpu_model_preload",
                        model_name=model_name,
                        loaded_models=list(self.cpu_models.keys()),
                    )
                elif tag == MSG_JOB:
                    partial_results = self.process_job(ds_key, sample_name, count, batch_dict, model_name)
                    self.output_queue.put((MSG_JOB_DONE, self.gpu_id, ds_key, batch_id, model_name, partial_results))
                elif tag == MSG_END_DATASET:
                    self.output_queue.put((MSG_DATASET_DONE, self.gpu_id, ds_key))
        finally:
            if self._vis_async_executor.thread is not None:
                self._vis_async_executor.clean_sync()

        self.output_queue.put((MSG_WORKER_DONE, self.gpu_id))


# ---------------------------------------------------------------------------
# ModelHostPool
# ---------------------------------------------------------------------------

class ModelHostPool:
    """Owns all worker subprocesses and inter-process communication queues."""

    def __init__(
        self,
        gpu_ids: List[str],
        full_job_data_list: List[dict],
        eval_views: List[int],
        args_dict: dict,
    ):
        unique_gpu_ids = list(dict.fromkeys(gpu_ids))
        if len(unique_gpu_ids) < len(gpu_ids):
            print(f"[ModelHostPool] Warning: duplicate gpu_ids removed. Using: {unique_gpu_ids}")
        if not unique_gpu_ids:
            raise ValueError("At least one GPU id must be provided.")
        self.gpu_ids: List[str] = unique_gpu_ids

        self.worker_loaded_model: Dict[str, Optional[str]] = {gid: None for gid in unique_gpu_ids}
        self.worker_cpu_models: Dict[str, List[str]] = {gid: [] for gid in unique_gpu_ids}

        ctx = mp.get_context("spawn")
        self.request_queue = ctx.Queue()
        self._response_queues: Dict[str, Any] = {}
        self._processes: List[Any] = []
        self._gpu_id_to_proc: Dict[str, Any] = {}

        for gpu_id in unique_gpu_ids:
            device_index = resolve_worker_device_index(gpu_id)
            print(f"[ModelHostPool] Spawning worker on cuda:{device_index}")
            q = ctx.Queue(maxsize=8)
            self._response_queues[gpu_id] = q
            p = ctx.Process(
                target=model_worker_entry,
                args=(gpu_id, device_index, full_job_data_list, eval_views, q, self.request_queue, args_dict),
                daemon=False,
            )
            p.start()
            self._processes.append(p)
            self._gpu_id_to_proc[gpu_id] = p

        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    def send_job(self, gpu_id, ds_key, batch_id, sample_name, count, batch_dict, model_name):
        self._response_queues[gpu_id].put(
            (MSG_JOB, ds_key, batch_id, sample_name, count, batch_dict, model_name)
        )

    def send_end_dataset(self, gpu_id: str, ds_key: str):
        self._response_queues[gpu_id].put((MSG_END_DATASET, ds_key))

    def send_load_cpu_model(self, gpu_id: str, model_name: str):
        self._response_queues[gpu_id].put((MSG_LOAD_CPU, model_name))

    def check_liveness(self):
        dead = [gid for gid, p in self._gpu_id_to_proc.items() if not p.is_alive()]
        if dead:
            raise RuntimeError(
                "Worker(s) died unexpectedly before sending a worker_error message: "
                + self.describe_processes(dead)
            )

    def describe_processes(self, gpu_ids: Optional[List[str]] = None) -> str:
        gpu_ids = self.gpu_ids if gpu_ids is None else gpu_ids
        details = []
        for gid in gpu_ids:
            p = self._gpu_id_to_proc[gid]
            code = p.exitcode
            if code is not None and code < 0:
                sig = signal.Signals(-code).name if -code in signal.Signals._value2member_map_ else f"SIG{-code}"
                details.append(f"{gid}: pid={p.pid} alive={p.is_alive()} exitcode={code} ({sig})")
            else:
                details.append(f"{gid}: pid={p.pid} alive={p.is_alive()} exitcode={code}")
        return "; ".join(details)

    def shutdown(self):
        for q in self._response_queues.values():
            try:
                q.put(None)
            except Exception:
                raise
        for p in self._processes:
            p.join(timeout=120)
            if p.is_alive():
                p.terminate()
                p.join()

    def terminate(self):
        for p in self._processes:
            p.terminate()
        for p in self._processes:
            p.join()


# ---------------------------------------------------------------------------
# Worker entry point
# ---------------------------------------------------------------------------

def model_worker_entry(gpu_id, device_index, job_data_list, eval_views, input_queue, output_queue, args_dict):
    try:
        host = ModelEvalHost(gpu_id, device_index, job_data_list, eval_views, input_queue, output_queue, args_dict)
        host.run()
    except KeyboardInterrupt:
        return
    except torch.cuda.OutOfMemoryError:
        try:
            torch.cuda.set_device(device_index)
            alloc = torch.cuda.memory_allocated(device_index) / 1024 ** 3
            reserved = torch.cuda.memory_reserved(device_index) / 1024 ** 3
            peak = torch.cuda.max_memory_allocated(device_index) / 1024 ** 3
            mem = f"\nCUDA memory at OOM: alloc={alloc:.2f}GiB reserved={reserved:.2f}GiB peak={peak:.2f}GiB"
        except Exception:
            mem = "\nCUDA memory at OOM: unavailable"
        output_queue.put((MSG_WORKER_ERROR, gpu_id, traceback.format_exc() + mem))
        raise
    except Exception:
        output_queue.put((MSG_WORKER_ERROR, gpu_id, traceback.format_exc()))
        raise
