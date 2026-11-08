# Evaluation

The ARGAvatar evaluation harness runs out of the box in the `argavatar`
environment.

**NeuMan is the only benchmark.** Everything below evaluates on it alone. Set it
up first — see [Benchmark data](../README.md#21-benchmark-data):

```bash
python scripts/download_evaluation/prepare_neuman.py --all
```

Every command is run from the repository root with `PYTHONPATH` empty.

## ARGAvatar

Runs in `argavatar` with no extra setup.

| Path | What it does |
|---|---|
| `argavatar/eval_argavatar.py` | Evaluate one config + checkpoint. |
| `argavatar/manager.py` | Job resolution, GPU worker pool, results writing. |
| `argavatar/worker.py` | Per-GPU model host and metric accumulation. |
| `argavatar/eval_profile.py` | Timing/throughput profiling. |
| `common/` | Metric plumbing shared here: results I/O, caching, timing, val-list parsing. |

The release sweep is this same command, over the one model listed in
`configs/_shared/release_eval_jobs.yaml`:

```bash
PYTHONPATH= python evaluation/argavatar/eval_argavatar.py \
  --config configs/ARGAvatar-Final.yaml \
  --checkpoint_path checkpoints/ARGAvatar-Final.pth \
  --data_config configs/_shared/data_val_list.yaml \
  --eval_views 1 2 3 4 6
```

Results land in `expr/results/<project_name>/<ckpt_stem>_n<N>/eval_results.json`.

Every cell is recomputed by default. `--skip_existing` is the opt-in that serves
already-computed samples from disk; do **not** pass it when the numbers must be
comparable, or a table can end up mixing runs.

> `--checkpoint_path` is optional: without it a bare model name resolves through
> `manager.py::resolve_jobs`, which globs `expr/<name>/PT_*.pth` and falls back to
> `checkpoints/<name>.pth`. Passing it explicitly is unambiguous, which is why the
> sweep above does.

### Collecting results

| Path | What it does |
|---|---|
| `scripts/collect_eval_results.py` | Merge `expr/results/` into `expr/download/*.json`. |
