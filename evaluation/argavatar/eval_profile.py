"""Timeline profiling utilities for multi-process eval (Chrome/Perfetto trace)."""

from __future__ import annotations

import json
import os
import time
from typing import Optional, Tuple

TRACE_FLUSH_INTERVAL_S = 300.0


def _profile_now_us(profile_t0_ns: int) -> float:
    return (time.perf_counter_ns() - profile_t0_ns) / 1000.0


def profile_enabled(args) -> bool:
    return bool(getattr(args, "profile_trace_json", None))


def make_profile_event(
    args,
    process: str,
    state: str,
    *,
    gpu_id: Optional[str] = None,
    **payload,
) -> Optional[dict]:
    if not profile_enabled(args):
        return None
    profile_t0_ns = int(getattr(args, "profile_t0_ns", 0))
    event = {
        "ts_us": _profile_now_us(profile_t0_ns),
        "process": process,
        "pid": os.getpid(),
        "state": state,
    }
    if gpu_id is not None:
        event["gpu_id"] = str(gpu_id)
    event.update({k: v for k, v in payload.items() if v is not None})
    return event


def configure_profile_args(args) -> None:
    """Record the profiling start time on *args* (mutates in place)."""
    args.profile_t0_ns = time.perf_counter_ns()


class ProfileWriter:
    """Parent-process writer for Chrome/Perfetto trace spans."""

    def __init__(self, args):
        self.args = args
        self.enabled = profile_enabled(args)
        self.trace_events = []
        self.open_spans = {}
        self._last_trace_flush_time = time.monotonic()

    def event(self, event: Optional[dict]):
        if not event:
            return
        self._update_trace(event)
        self.maybe_flush_trace()

    def _span_key(self, event: dict, phase: str) -> Tuple:
        return (
            event.get("process"),
            event.get("gpu_id"),
            phase,
            event.get("model_name"),
            event.get("ds_key"),
            event.get("batch_id"),
            event.get("n_views"),
            event.get("sample_name"),
        )

    @staticmethod
    def _trace_pid(start: dict) -> str:
        pid = start["process"]
        if start.get("process") == "worker" and start.get("gpu_id") is not None:
            pid = f"{pid}-gpu{start['gpu_id']}"
        return pid

    @staticmethod
    def _trace_tid(start: dict) -> str:
        if start.get("process") == "worker":
            return start.get("model_name") or start.get("ds_key") or "main"
        return start.get("ds_key") or "main"

    def _update_trace(self, event: dict):
        state = str(event.get("state", ""))
        if state.startswith("start_"):
            phase = state[len("start_"):]
            self.open_spans[self._span_key(event, phase)] = dict(event)
        elif state.startswith("end_"):
            phase = state[len("end_"):]
            key = self._span_key(event, phase)
            start = self.open_spans.pop(key, None)
            if start is None:
                return
            dur_us = max(0.0, float(event["ts_us"]) - float(start["ts_us"]))
            self.trace_events.append({
                "name": phase,
                "cat": "eval_profile",
                "ph": "X",
                "ts": float(start["ts_us"]),
                "dur": dur_us,
                "pid": self._trace_pid(start),
                "tid": self._trace_tid(start),
                "args": {
                    k: v for k, v in start.items()
                    if k not in {"ts_us", "state", "process", "pid"}
                },
            })

    def _write_trace_json(self, *, checkpoint: bool = False):
        os.makedirs(os.path.dirname(self.args.profile_trace_json) or ".", exist_ok=True)
        with open(self.args.profile_trace_json, "w", encoding="utf-8") as f:
            json.dump({"traceEvents": self.trace_events}, f)
        label = "trace checkpoint" if checkpoint else "trace"
        print(
            f"[ProfileWriter] Wrote {label} ({len(self.trace_events)} events): "
            f"{self.args.profile_trace_json}"
        )

    def maybe_flush_trace(self):
        if not self.enabled:
            return
        now = time.monotonic()
        if now - self._last_trace_flush_time < TRACE_FLUSH_INTERVAL_S:
            return
        self._last_trace_flush_time = now
        self._write_trace_json(checkpoint=True)

    def close(self):
        if self.enabled:
            self._write_trace_json()
