import json
import re
from collections import defaultdict
from pathlib import Path


def parse_result_key(key):
    """Parse '{model_name}_{step}_n{views}', '{model_name}_n{views}', '{model_name}_{step}', or '{model_name}'."""
    # 1. model_step_nviews, e.g. ARGAvatar-Final_80000_n6, MV-UV_35850_n6
    m = re.match(r"^(.+)_(\d+)_n(\d+)$", key)
    if m:
        return m.group(1), int(m.group(2)), int(m.group(3))
    # 2. model_nviews, e.g. ARGAvatar-Final_n6
    m = re.match(r"^(.+)_n(\d+)$", key)
    if m:
        return m.group(1), 0, int(m.group(2))
    # 3. model_step, e.g. MV-UV_35850
    m = re.match(r"^(.+)_(\d+)$", key)
    if m:
        return m.group(1), int(m.group(2)), None
    # 4. plain model name
    return key, 0, None


def build_view_grouped_list(
    all_results,
    model_filter,
    require_views=True,
    n_views_only=None,
    include_step_in_key=True,
):
    """Group results by view count, keeping every checkpoint step per model."""
    grouped = defaultdict(dict)

    for key, result in all_results.items():
        model_name, step, n_views = parse_result_key(key)
        if not model_filter(model_name):
            continue
        if require_views and n_views is None:
            continue
        if n_views_only is not None and n_views != n_views_only:
            continue

        result_key = f"{model_name}-{step}" if (include_step_in_key and step is not None and step > 0) else model_name
        grouped[n_views][result_key] = result

    return {str(n_views): grouped[n_views] for n_views in sorted(grouped)}


def build_best_model_list(all_results):
    """Best models (ARGAvatar / MV-UV, excluding ablations), grouped by view count."""

    def is_best_model(model_name):
        return (
            model_name.startswith("ARGAvatar")
            or model_name.startswith("MV-UV-")
        ) and not model_name.startswith("MV-UVSmall") and "ablation" not in model_name.lower()

    return build_view_grouped_list(all_results, is_best_model, require_views=True)


def load_eval_payload(eval_json_path):
    with open(eval_json_path, "r") as f:
        return json.load(f)


def build_ablation_list(all_results):
    """MV-UVSmall ablation models at n6 input views."""

    def is_ablation(model_name):
        return model_name.startswith("MV-UVSmall") or "ablation" in model_name.lower()

    return build_view_grouped_list(
        all_results,
        is_ablation,
        require_views=True,
        n_views_only=6,
        include_step_in_key=True,
    )


def collect_argavatar_results(all_results):
    results_root = Path("expr/results")
    if not results_root.exists():
        print(f"Warning: {results_root} does not exist, skipping ARGAvatar results.")
        return

    for model_dir in sorted(results_root.iterdir()):
        if not model_dir.is_dir():
            continue

        model_name = model_dir.name
        eval_json_paths = sorted(model_dir.rglob("eval_results.json"))
        if not eval_json_paths:
            print(f"Warning: No eval_results.json found in {model_dir}, skipping.")
            continue

        for eval_json_path in eval_json_paths:
            rel_to_model = eval_json_path.parent.relative_to(model_dir)
            if str(rel_to_model) == ".":
                key = model_name
            else:
                folder_name = rel_to_model.parts[0]
                if folder_name.startswith("PT_"):
                    run_id = folder_name.replace("PT_", "")
                    key = f"{model_name}_{run_id}"
                elif folder_name.startswith(f"{model_name}_") or folder_name == model_name:
                    key = folder_name
                else:
                    key = f"{model_name}_{folder_name}"

            try:
                all_results[key] = load_eval_payload(eval_json_path)
                print(f"Collected ARGAvatar: {key}")
            except Exception as e:
                print(f"Error reading {eval_json_path}: {e}")


def collect_results():
    download_dir = Path("expr/download")
    download_dir.mkdir(parents=True, exist_ok=True)

    all_results = {}

    collect_argavatar_results(all_results)

    outputs = {
        "all_eval_results.json": all_results,
        "ablation_eval_results.json": build_ablation_list(all_results),
        "best_model_eval_results.json": build_best_model_list(all_results),
    }

    for filename, payload in outputs.items():
        output_path = download_dir / filename
        with open(output_path, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"Wrote {len(payload)} entries to {output_path}")

    print(f"\nSuccessfully collected {len(all_results)} results into {download_dir}/")


if __name__ == "__main__":
    collect_results()

