"""Manager-side evaluation: job config, scheduling, DDPEvalManager."""

import argparse
import contextlib
import glob
import os
import signal
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import yaml
from omegaconf import OmegaConf
from tqdm import tqdm
import queue as _queue

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from xlib.utils import load_config
from evaluation.argavatar.eval_profile import ProfileWriter, make_profile_event
from evaluation.argavatar.eval_argavatar import (
    save_results_json,
    load_cached_dataset_results,
    results_keys_in_eval_order,
    build_val_dataloaders,
    _to_plain_dict,
)
from evaluation.common.results import print_eval_table
from evaluation.argavatar.worker import (
    PendingEvalJob,
    ModelMeta,
    MetricsAccumulator,
    ModelHostPool,
    _preprocess_batch_cpu,
    _expected_view_counts,
    should_save_vis_for_count,
    MSG_PROFILE,
    MSG_REQUEST_JOB,
    MSG_JOB_DONE,
    MSG_DATASET_DONE,
    MSG_WORKER_ERROR,
)

_TERMINATING = False


# ---------------------------------------------------------------------------
# Job config
# ---------------------------------------------------------------------------

def resolve_jobs(jobs_raw: List[Any], args: argparse.Namespace) -> List[Dict[str, Any]]:
    resolved = []
    for entry in jobs_raw:
        if isinstance(entry, str):
            model_name = entry.split("|")[0]
            checkpoint_path = None
            data_config = None
        elif isinstance(entry, dict):
            model_name = entry["model_name"]
            checkpoint_path = entry.get("checkpoint_path")
            data_config = entry.get("data_config")
        else:
            continue

        config_path = os.path.join(args.config_dir, f"{model_name}.yaml")
        if not os.path.exists(config_path):
            print(f"[SKIP] {model_name} - missing config {config_path}")
            continue
        train_args = load_config(config_path)

        if not checkpoint_path:
            expr_model_dir = os.path.join(args.expr_dir, model_name)
            ckpts = sorted(glob.glob(os.path.join(expr_model_dir, "PT_*.pth")), key=os.path.getmtime, reverse=True)
            if ckpts:
                checkpoint_path = ckpts[0]
            else:
                # Fall back to the canonical release path. download_weights.sh mirrors each
                # checkpoint into expr/<name>/PT_<step>.pth, but a tree where only
                # checkpoints/ was populated is still a valid release -- resolve it rather
                # than silently skipping the model.
                canonical = os.path.join("checkpoints", f"{model_name}.pth")
                if not os.path.exists(canonical):
                    print(f"[SKIP] {model_name} - no checkpoints in {expr_model_dir} nor {canonical}")
                    continue
                checkpoint_path = canonical

        data_cfg = OmegaConf.load(data_config or args.data_config)
        OmegaConf.resolve(data_cfg)
        train_args.data = data_cfg.data

        resolved.append({
            "model_name": model_name,
            "config_path": config_path,
            "checkpoint_path": checkpoint_path,
            "train_args": train_args,
        })
    return resolved


# ---------------------------------------------------------------------------
# Scheduling
# ---------------------------------------------------------------------------

def choose_job(
    pending_jobs: List[PendingEvalJob],
    gpu_model_name: Optional[str],
    cpu_model_names: List[str],
    globally_loaded_cpu_models: Set[str],
    pinned_model: Optional[str] = None,
) -> Optional[PendingEvalJob]:
    """Choose the next job from pending_jobs based on scheduling priority.

    When `pinned_model` is set the worker only ever takes that model's jobs, so
    `worker.ensure_cpu_model` caches exactly one model instead of accumulating
    every model it is handed (that cache is never evicted).
    """
    if not pending_jobs:
        return None

    if pinned_model is not None:
        candidates = [idx for idx, job in enumerate(pending_jobs) if job.model_name == pinned_model]
        if not candidates:
            return None
        oldest = min(pending_jobs[idx].batch_id for idx in candidates)
        for idx in candidates:
            if pending_jobs[idx].batch_id == oldest:
                return pending_jobs.pop(idx)
        return None

    oldest_batch_id = min(job.batch_id for job in pending_jobs)
    candidate_indices = [
        idx for idx, job in enumerate(pending_jobs)
        if job.batch_id == oldest_batch_id
    ]

    if gpu_model_name is not None:
        for idx in candidate_indices:
            if pending_jobs[idx].model_name == gpu_model_name:
                return pending_jobs.pop(idx)

    cpu_model_set = set(cpu_model_names)
    for idx in candidate_indices:
        if pending_jobs[idx].model_name in cpu_model_set:
            return pending_jobs.pop(idx)

    for idx in candidate_indices:
        if pending_jobs[idx].model_name not in globally_loaded_cpu_models:
            return pending_jobs.pop(idx)

    return pending_jobs.pop(candidate_indices[0])


def model_needs_dataset(meta: ModelMeta, ds_key: str, eval_views: List[int], ignore_cache: bool = False, n_samples: int = -1) -> bool:
    if ignore_cache:
        return True
    for n_views in eval_views:
        out_json = os.path.join(meta.results_dir_for_views(n_views), "eval_results.json")
        cached = load_cached_dataset_results(out_json)
        if ds_key not in cached:
            return True
        expected = _expected_view_counts(meta.train_args, ds_key, n_views, n_samples)
        if expected is not None:
            row = cached[ds_key]
            if row["n_input"] != expected[0] or row["n_cross"] != expected[1]:
                print(
                    f"[cache invalid] {meta.model_name} {ds_key} n={n_views}: "
                    f"cached n_input={row['n_input']} n_cross={row['n_cross']} "
                    f"expected n_input={expected[0]} n_cross={expected[1]}"
                )
                return True
    return False


# ---------------------------------------------------------------------------
# DDPEvalManager
# ---------------------------------------------------------------------------

class DDPEvalManager:
    """Coordinates validation data loading and ModelHostPool for distributed evaluation."""

    def __init__(
        self,
        jobs: List[dict],
        gpu_ids: List[str],
        eval_views: List[int],
        args: argparse.Namespace,
    ):
        self.eval_views = eval_views
        self.args = args
        self.profile_writer = ProfileWriter(args)

        self.model_meta_by_name: Dict[str, ModelMeta] = {
            j["model_name"]: ModelMeta(j["model_name"], j["checkpoint_path"], j["train_args"], j["config_path"])
            for j in jobs
        }
        self.all_model_names: List[str] = list(dict.fromkeys(j["model_name"] for j in jobs))
        self._proto_train_args = jobs[0]["train_args"]

        # Pin one model per GPU when they fit, so each worker holds exactly one
        # model in CPU RAM. Without this, workers switch models freely and
        # `ensure_cpu_model` accumulates every model it sees with no eviction,
        # which on a 6-model sweep grew past 200 GB RSS and got OOM-killed.
        self.pinned_model_by_gpu: Dict[str, str] = {}
        pin = getattr(args, "disable_dynamic_switching", False) or len(self.all_model_names) <= len(gpu_ids)
        if pin and len(self.all_model_names) <= len(gpu_ids):
            self.pinned_model_by_gpu = {
                gpu_id: model_name
                for gpu_id, model_name in zip(gpu_ids, self.all_model_names)
            }
            print(
                "[DDPEvalManager] Static model pinning enabled "
                f"({len(self.all_model_names)} models <= {len(gpu_ids)} GPUs):"
            )
            for gpu_id, model_name in self.pinned_model_by_gpu.items():
                print(f"  GPU {gpu_id} -> {model_name}")
            idle = [g for g in gpu_ids if g not in self.pinned_model_by_gpu]
            if idle:
                print(f"  unused GPUs (more GPUs than models): {', '.join(idle)}")
        elif getattr(args, "disable_dynamic_switching", False):
            print(
                f"[DDPEvalManager] --disable_dynamic_switching ignored: "
                f"{len(self.all_model_names)} models > {len(gpu_ids)} GPUs, so each worker "
                "must handle more than one model."
            )

        full_job_data_list = [
            {
                "model_name": j["model_name"],
                "train_args": _to_plain_dict(j["train_args"]),
                "config_path": j["config_path"],
                "checkpoint_path": j["checkpoint_path"],
            }
            for j in jobs
        ]
        self.pool = ModelHostPool(gpu_ids, full_job_data_list, eval_views, vars(args))

        self._accumulators: Dict = defaultdict(lambda: defaultdict(dict))
        self._active_pbar = None
        self._active_val_iter = None
        self._msg_handlers = {
            MSG_PROFILE:      self._on_profile,
            MSG_REQUEST_JOB:  self._on_request_job,
            MSG_JOB_DONE:     self._on_job_done,
            MSG_DATASET_DONE: self._on_dataset_done,
            MSG_WORKER_ERROR: self._on_worker_error,
        }

    @contextlib.contextmanager
    def profile_span(self, name: str, **kwargs):
        self.emit_profile(f"start_{name}", **kwargs)
        try:
            yield
        finally:
            self.emit_profile(f"end_{name}", **kwargs)

    def emit_profile(self, state: str, **payload):
        self.profile_writer.event(make_profile_event(self.args, "manager", state, **payload))

    def build_val_loaders(self, num_workers: int):
        proto = OmegaConf.to_container(self._proto_train_args, resolve=True)
        if "training" in proto:
            for _key in ("num_sup_views", "n_val_sup_views", "num_input_views"):
                proto["training"].pop(_key, None)
        return build_val_dataloaders(OmegaConf.create(proto), num_workers=num_workers)

    # ------------------------------------------------------------------
    # Message handlers (dispatch table)
    # ------------------------------------------------------------------

    def _on_profile(self, msg, **_):
        self.profile_writer.event(msg[1])

    def _on_request_job(self, msg, waiting_workers, **_):
        self.pool.worker_loaded_model[msg[1]] = msg[2]
        self.pool.worker_cpu_models[msg[1]] = msg[3]
        waiting_workers.append(msg)

    def _on_job_done(self, msg, ds_key, inflight_jobs, remaining_jobs_by_batch, pbar, **_):
        gpu_id, d_key, b_id, m_name, partial_res = msg[1], msg[2], msg[3], msg[4], msg[5]
        inflight_key = (gpu_id, d_key, b_id, m_name)
        if inflight_key in inflight_jobs:
            del inflight_jobs[inflight_key]
        for nv, payload in partial_res.items():
            if d_key not in self._accumulators[m_name][nv]:
                self._accumulators[m_name][nv][d_key] = MetricsAccumulator()
            self._accumulators[m_name][nv][d_key].merge_payload(payload)
        if d_key == ds_key and pbar is not None:
            remaining_jobs_by_batch[b_id] -= 1
            if remaining_jobs_by_batch[b_id] <= 0:
                del remaining_jobs_by_batch[b_id]
                pbar.update(1)

    def _on_dataset_done(self, msg, ds_key, workers_done_with_ds, **_):
        if msg[2] == ds_key:
            workers_done_with_ds.add(msg[1])

    def _on_worker_error(self, msg, **_):
        raise RuntimeError(f"Worker {msg[1]} failed:\n{msg[2]}")

    def _handle_worker_message(
        self,
        msg,
        ds_key: str,
        inflight_jobs: dict,
        remaining_jobs_by_batch: dict,
        waiting_workers: list,
        workers_done_with_ds: set,
        pbar,
    ):
        handler = self._msg_handlers.get(msg[0])
        if handler:
            handler(
                msg,
                ds_key=ds_key,
                inflight_jobs=inflight_jobs,
                remaining_jobs_by_batch=remaining_jobs_by_batch,
                waiting_workers=waiting_workers,
                workers_done_with_ds=workers_done_with_ds,
                pbar=pbar,
            )

    # ------------------------------------------------------------------
    # Job dispatching
    # ------------------------------------------------------------------

    def _dispatch_job(self, gpu_id: str, job: PendingEvalJob, inflight_jobs: dict):
        payload_bytes = self._estimate_payload_nbytes(job.batch_dict)
        with self.profile_span(
            "dispatch",
            gpu_id=gpu_id, ds_key=job.ds_key, batch_id=job.batch_id,
            sample_name=job.sample_name, model_name=job.model_name,
            payload_mb=round(payload_bytes / 1024 ** 2, 2),
        ):
            self.pool.send_job(
                gpu_id, job.ds_key, job.batch_id, job.sample_name,
                job.count, job.batch_dict, job.model_name,
            )
            inflight_jobs[(gpu_id, job.ds_key, job.batch_id, job.model_name)] = job

    def _estimate_payload_nbytes(self, obj) -> int:
        if hasattr(obj, "nelement") and hasattr(obj, "element_size"):
            return int(obj.nelement() * obj.element_size())
        if isinstance(obj, dict):
            return sum(self._estimate_payload_nbytes(v) for v in obj.values())
        if isinstance(obj, (list, tuple)):
            return sum(self._estimate_payload_nbytes(v) for v in obj)
        return 0

    def _num_pending_batches(self, pending_jobs, inflight_jobs) -> int:
        return len(set(j.batch_id for j in pending_jobs) | set(k[2] for k in inflight_jobs))

    def _update_pbar_postfix(
        self,
        pbar,
        pending_jobs,
        inflight_jobs,
        max_pending_batches,
        last_pbar_postfix,
        last_pbar_postfix_time,
    ):
        if pbar is None:
            return last_pbar_postfix, last_pbar_postfix_time
        num_pending = self._num_pending_batches(pending_jobs, inflight_jobs)
        postfix = f"{num_pending}/{max_pending_batches}"
        now = time.monotonic()
        if postfix != last_pbar_postfix or now - last_pbar_postfix_time >= 5.0:
            pbar.set_postfix(buf=postfix)
            return postfix, now
        return last_pbar_postfix, last_pbar_postfix_time

    # ------------------------------------------------------------------
    # _run_dataset helpers
    # ------------------------------------------------------------------

    def _produce_next_batch(self, val_iter, batches_produced: int, ds_key: str):
        try:
            self.emit_profile("start_data_fetch", ds_key=ds_key, batch_id=batches_produced)
            batch_raw = next(val_iter)
            self.emit_profile("end_data_fetch", ds_key=ds_key, batch_id=batches_produced)
        except StopIteration:
            self.emit_profile("end_data_fetch", ds_key=ds_key, batch_id=batches_produced)
            return None
        except BaseException as exc:
            self.emit_profile("end_data_fetch", ds_key=ds_key, batch_id=batches_produced)
            raise RuntimeError(
                "DataLoader fetch failed while receiving the next validation batch.\n"
                f"dataset={ds_key} batch_id={batches_produced}\n"
                f"exception={type(exc).__name__}: {exc}\n"
                f"DataLoader workers: {self._describe_val_iter_workers(val_iter)}\n"
                f"Model workers: {self.pool.describe_processes()}\n"
                f"Original traceback:\n{traceback.format_exc()}"
            ) from exc
        with self.profile_span("data_preprocess", ds_key=ds_key, batch_id=batches_produced):
            try:
                batch_dict = _preprocess_batch_cpu(batch_raw)
            except BaseException as exc:
                raise RuntimeError(
                    "CPU batch preprocessing failed after DataLoader fetch.\n"
                    f"dataset={ds_key} batch_id={batches_produced}\n"
                    f"exception={type(exc).__name__}: {exc}\n"
                    f"DataLoader workers: {self._describe_val_iter_workers(val_iter)}\n"
                    f"Model workers: {self.pool.describe_processes()}\n"
                    f"Original traceback:\n{traceback.format_exc()}"
                ) from exc
        raw_id = batch_dict.get("sample_identifier", [None])[0]
        sample_name = (
            str(raw_id).replace("/", "_").replace(os.sep, "_")
            if raw_id is not None
            else f"sample{batches_produced:04d}"
        )
        return batch_dict, sample_name

    def _describe_val_iter_workers(self, val_iter) -> str:
        if val_iter is None:
            return "no active iterator"
        workers = getattr(val_iter, "_workers", None)
        if not workers:
            return "num_workers=0 or worker list unavailable"
        statuses = getattr(val_iter, "_workers_status", None)
        details = []
        for idx, worker in enumerate(workers):
            status = None
            if statuses is not None and idx < len(statuses):
                status = statuses[idx]
            details.append(
                f"idx={idx} pid={worker.pid} alive={worker.is_alive()} "
                f"exitcode={worker.exitcode} status={status}"
            )
        return "; ".join(details)

    def _enqueue_batch_as_jobs(
        self, batch_dict, sample_name, batch_id, ds_key,
        active_model_names, metrics_needed_by_model,
        pending_jobs, remaining_jobs_by_batch, pbar,
    ):
        do_vis_for_batch = should_save_vis_for_count(self.args.n_vis, batch_id)
        jobs_added = 0
        for model_name in active_model_names:
            if not metrics_needed_by_model[model_name] and not do_vis_for_batch:
                continue
            pending_jobs.append(PendingEvalJob(
                ds_key=ds_key, batch_id=batch_id, sample_name=sample_name,
                count=batch_id, batch_dict=batch_dict, model_name=model_name,
            ))
            jobs_added += 1
        if jobs_added == 0:
            if pbar is not None:
                pbar.update(1)
        else:
            remaining_jobs_by_batch[batch_id] = jobs_added

    def _drain_worker_queue(
        self, ds_key, inflight_jobs, remaining_jobs_by_batch,
        waiting_workers, workers_done_with_ds, pbar,
    ):
        while True:
            try:
                msg = self.pool.request_queue.get_nowait()
                self._handle_worker_message(
                    msg, ds_key, inflight_jobs, remaining_jobs_by_batch,
                    waiting_workers, workers_done_with_ds, pbar,
                )
            except _queue.Empty:
                break
            except BaseException as exc:
                raise RuntimeError(
                    "Failed while receiving a model-worker IPC message.\n"
                    f"dataset={ds_key}\n"
                    f"exception={type(exc).__name__}: {exc}\n"
                    f"Model workers: {self.pool.describe_processes()}\n"
                    f"Original traceback:\n{traceback.format_exc()}"
                ) from exc

    def _dispatch_to_waiting_workers(
        self, waiting_workers, pending_jobs, dataset_finished,
        inflight_jobs, workers_told_end, next_ds_requests, ds_key,
    ) -> list:
        with self.profile_span("scheduler", ds_key=ds_key):
            globally_loaded_cpu_models = set()
            for cpu_list in self.pool.worker_cpu_models.values():
                globally_loaded_cpu_models.update(cpu_list)

            still_waiting = []
            for w_msg in waiting_workers:
                gpu_id, gpu_model_name, cpu_model_names = w_msg[1], w_msg[2], w_msg[3]
                job = choose_job(
                    pending_jobs, gpu_model_name, cpu_model_names, globally_loaded_cpu_models,
                    pinned_model=self.pinned_model_by_gpu.get(gpu_id),
                )
                if job:
                    self._dispatch_job(gpu_id, job, inflight_jobs)
                elif dataset_finished and not pending_jobs:
                    if gpu_id not in workers_told_end:
                        self.pool.send_end_dataset(gpu_id, ds_key)
                        workers_told_end.add(gpu_id)
                    else:
                        next_ds_requests.append(w_msg)
                else:
                    still_waiting.append(w_msg)
        return still_waiting

    def _blocking_wait_one_message(
        self, ds_key, inflight_jobs, remaining_jobs_by_batch,
        waiting_workers, workers_done_with_ds, pbar,
    ):
        try:
            msg = self.pool.request_queue.get(timeout=0.1)
            self._handle_worker_message(
                msg, ds_key, inflight_jobs, remaining_jobs_by_batch,
                waiting_workers, workers_done_with_ds, pbar,
            )
        except _queue.Empty:
            pass
        except BaseException as exc:
            raise RuntimeError(
                "Failed while waiting for a model-worker IPC message.\n"
                f"dataset={ds_key}\n"
                f"exception={type(exc).__name__}: {exc}\n"
                f"Model workers: {self.pool.describe_processes()}\n"
                f"Original traceback:\n{traceback.format_exc()}"
            ) from exc

    # ------------------------------------------------------------------
    # Result writing
    # ------------------------------------------------------------------

    def _write_dataset_results(self, ds_key: str, active_model_names: List[str]):
        for mname in active_model_names:
            meta = self.model_meta_by_name[mname]
            for nv in self.eval_views:
                acc = self._accumulators[mname][nv].get(ds_key)
                if acc is None or acc.is_empty():
                    continue
                metrics = acc.to_metrics()
                res_dir = meta.results_dir_for_views(nv)
                out_json = os.path.join(res_dir, "eval_results.json")
                cached = load_cached_dataset_results(out_json)
                cached[ds_key] = metrics
                save_results_json(out_json, cached, results_keys_in_eval_order(meta.val_ds_keys, cached))
                cfg_f = os.path.join(res_dir, "dataset_config.yaml")
                if not os.path.exists(cfg_f):
                    os.makedirs(res_dir, exist_ok=True)
                    with open(cfg_f, "w", encoding="utf-8") as _f:
                        yaml.dump(_to_plain_dict(meta.train_args.data), _f, default_flow_style=False, allow_unicode=True)
                print(f"\n[DDPEvalManager] {mname} n={nv} on {ds_key}:")
                print_eval_table({ds_key: metrics})

    def _print_final_summary(self):
        for mname in self.all_model_names:
            for nv in self.eval_views:
                res = {}
                for dk, acc in self._accumulators[mname][nv].items():
                    if not acc.is_empty():
                        res[dk] = acc.to_metrics()
                if res:
                    print(f"\nFinal results for {mname} n={nv}:")
                    print_eval_table(res)

    def _terminate_pool_after_interrupt(self):
        if self._active_pbar is not None:
            try:
                self._active_pbar.close()
            except Exception:
                pass
            self._active_pbar = None
        if self._active_val_iter is not None:
            try:
                shutdown_workers = getattr(self._active_val_iter, "_shutdown_workers", None)
                if shutdown_workers is not None:
                    shutdown_workers()
            except Exception:
                pass
            self._active_val_iter = None
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        try:
            self.pool.terminate()
        except Exception as exc:
            print(f"[DDPEvalManager] Ignoring cleanup error during interrupt: {exc}")

    # ------------------------------------------------------------------
    # Per-dataset evaluation loop
    # ------------------------------------------------------------------

    def _run_dataset(self, ds_key: str, val_dl, pending_from_prev_ds: List) -> List:
        """Evaluate one dataset across all models."""
        n_total = len(val_dl)
        if self.args.n_samples > 0:
            n_total = min(n_total, self.args.n_samples)

        metrics_needed_by_model = {
            name: model_needs_dataset(
                self.model_meta_by_name[name], ds_key, self.eval_views,
                ignore_cache=self.args.ignore_cache,
                n_samples=self.args.n_samples,
            )
            for name in self.all_model_names
        }
        active_model_names = [name for name in self.all_model_names
            if metrics_needed_by_model[name]]

        if active_model_names:
            print(f"\nEvaluating {ds_key} ({n_total} samples) for {len(active_model_names)} model(s)...")
        else:
            print(f"\nSkipping {ds_key}: all requested model/view results are cached.")

        pending_jobs: List[PendingEvalJob] = []
        inflight_jobs: Dict[Tuple, PendingEvalJob] = {}
        remaining_jobs_by_batch: Dict[int, int] = {}
        pbar = (
            tqdm(
                total=n_total,
                desc=f"Evaluating {ds_key}",
                position=0,
                leave=True,
                dynamic_ncols=True,
            )
            if active_model_names and n_total > 0
            else None
        )
        self._active_pbar = pbar

        max_pending_batches = max(1, int(self.args.max_pending_batches))
        batches_produced = 0
        dataset_finished = not active_model_names or n_total <= 0
        val_iter = None if dataset_finished else iter(val_dl)
        self._active_val_iter = val_iter

        workers_done_with_ds: Set[str] = set()
        workers_told_end: Set[str] = set()
        next_ds_requests: List = []
        waiting_workers: List = pending_from_prev_ds
        last_pbar_postfix = None
        last_pbar_postfix_time = 0.0

        while len(workers_done_with_ds) < len(self.pool.gpu_ids):
            self._drain_worker_queue(
                ds_key, inflight_jobs, remaining_jobs_by_batch,
                waiting_workers, workers_done_with_ds, pbar,
            )

            waiting_workers = self._dispatch_to_waiting_workers(
                waiting_workers, pending_jobs, dataset_finished,
                inflight_jobs, workers_told_end, next_ds_requests, ds_key,
            )

            last_pbar_postfix, last_pbar_postfix_time = self._update_pbar_postfix(
                pbar, pending_jobs, inflight_jobs, max_pending_batches,
                last_pbar_postfix, last_pbar_postfix_time,
            )

            num_pending = self._num_pending_batches(pending_jobs, inflight_jobs)
            can_fetch = (
                not dataset_finished
                and num_pending < max_pending_batches
                and batches_produced < n_total
            )
            if can_fetch:
                result = self._produce_next_batch(val_iter, batches_produced, ds_key)
                if result is None:
                    dataset_finished = True
                else:
                    batch_dict, sample_name = result
                    self._enqueue_batch_as_jobs(
                        batch_dict, sample_name, batches_produced, ds_key,
                        active_model_names, metrics_needed_by_model,
                        pending_jobs, remaining_jobs_by_batch, pbar,
                    )
                    batches_produced += 1
                if batches_produced >= n_total:
                    dataset_finished = True

            waiting_workers = self._dispatch_to_waiting_workers(
                waiting_workers, pending_jobs, dataset_finished,
                inflight_jobs, workers_told_end, next_ds_requests, ds_key,
            )

            self._blocking_wait_one_message(
                ds_key, inflight_jobs, remaining_jobs_by_batch,
                waiting_workers, workers_done_with_ds, pbar,
            )

            self.pool.check_liveness()
            self.profile_writer.maybe_flush_trace()

        if pbar is not None:
            pbar.close()
            self._active_pbar = None
        self._active_val_iter = None

        self._write_dataset_results(ds_key, active_model_names)
        return next_ds_requests

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run(self):
        val_loaders = list(self.build_val_loaders(self.args.num_workers))
        ds_keys = [dk for dk, _ in val_loaders]

        # Skip models whose every (dataset, view-count) pair is already cached.
        if not self.args.ignore_cache and ds_keys:
            fully_cached = [
                name for name in self.all_model_names
                if all(
                    not model_needs_dataset(self.model_meta_by_name[name], dk, self.eval_views, n_samples=self.args.n_samples)
                    for dk in ds_keys
                )
            ]
            for name in fully_cached:
                print(
                    f"[SKIP] {name} — all {len(ds_keys)} dataset(s) already cached "
                    f"for views {self.eval_views}"
                )
                self.all_model_names.remove(name)

        if not self.all_model_names:
            print("[DDPEvalManager] All models fully cached — nothing to evaluate.")
            self.pool.shutdown()
            self.profile_writer.close()
            return

        n_models = len(self.all_model_names)
        for i, gpu_id in enumerate(self.pool.gpu_ids):
            model_name = self.all_model_names[i % n_models]
            print(f"[DDPEvalManager] Worker {gpu_id} initial load: {model_name}")
            self.pool.send_load_cpu_model(gpu_id, model_name)
        pending: List = []
        workers_terminated = False
        try:
            for ds_key, val_dl in val_loaders:
                pending = self._run_dataset(ds_key, val_dl, pending)
        except KeyboardInterrupt:
            print("\nInterrupted. Terminating...")
            self._terminate_pool_after_interrupt()
            workers_terminated = True
            sys.exit(1)
        except Exception:
            if not _TERMINATING:
                raise
            print("\nInterrupted. Terminating...")
            self._terminate_pool_after_interrupt()
            workers_terminated = True
            sys.exit(1)
        finally:
            if not workers_terminated:
                self.pool.shutdown()
            self.profile_writer.close()
        self._print_final_summary()
