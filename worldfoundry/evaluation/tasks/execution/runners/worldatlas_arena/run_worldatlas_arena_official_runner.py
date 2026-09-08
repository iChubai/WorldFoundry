#!/usr/bin/env python3
"""Run the in-tree OpenEnvision WorldArena evaluator or normalize its reports.

The upstream project calls the benchmark both ``WorldArena`` and
``WorldAtlas Arena``.  WorldFoundry uses the latter as the benchmark id so it
does not collide with the unrelated Tsinghua WorldArena benchmark already in
the catalog.
"""

from __future__ import annotations

import ast
import os
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[6]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from worldfoundry.evaluation.tasks.execution.framework import official_runner as ors  # noqa: E402
from worldfoundry.evaluation.tasks.execution.framework.io import mean_numeric, scalar_number  # noqa: E402

RUNNER_ROOT = Path(__file__).resolve().parent
DEFAULT_RUNTIME_ROOT = RUNNER_ROOT / "runtime" / "worldatlas_arena"
DEFAULT_CONFIG_PATH = DEFAULT_RUNTIME_ROOT / "config" / "benchmark.yaml"
DIMENSION_IDS = (
    "long_sequence",
    "action_control",
    "consistency_3d_4d",
    "physics",
    "quality",
    "real_time",
)

CONFIG = ors.BenchRunnerConfig(
    benchmark_id="worldatlas-arena",
    display_name="WorldArena (WorldAtlas Arena)",
    root_env="WORLDFOUNDRY_WORLDATLAS_ARENA_ROOT",
    results_path_env="WORLDFOUNDRY_WORLDATLAS_ARENA_RESULTS_PATH",
    default_repo_subdir=("worldfoundry/evaluation/tasks/execution/runners/worldatlas_arena/runtime/worldatlas_arena"),
    metric_order=DIMENSION_IDS,
    metric_specs={
        metric_id: {
            "name": metric_id.replace("_", " ").title(),
            "group": "official_dimension",
            "higher_is_better": True,
        }
        for metric_id in DIMENSION_IDS
    },
    metric_aliases={metric_id: metric_id for metric_id in DIMENSION_IDS},
    # Upstream deliberately publishes a vector of dimensions, not a single
    # benchmark-wide score.  Keep the shared runner from inventing one.
    average_metric_id="__no_derived_average__",
    official_entry="worldarena.cli",
    official_output_globs=("upstream/per_dimension_summary.json", "**/per_dimension_summary.json"),
    usage_epilog=(
        "Normalize a report bundle:\n"
        "  python run_worldatlas_arena_official_runner.py --official-results-path <report-dir> "
        "--output-dir <out>\n\n"
        "Run the vendored evaluator:\n"
        "  python run_worldatlas_arena_official_runner.py --run-official "
        "--manifest <worldarena_manifest.jsonl> --generated-artifact-dir <predictions> "
        "--output-dir <out>"
    ),
)


@lru_cache(maxsize=1)
def _upstream_metric_dimensions() -> dict[str, str]:
    """Read the vendored upstream mapping without importing its heavy stack."""
    catalog_path = DEFAULT_RUNTIME_ROOT / "worldarena" / "benchmark" / "metrics" / "catalog.py"
    module = ast.parse(catalog_path.read_text(encoding="utf-8"), filename=str(catalog_path))
    for node in module.body:
        target_name: str | None = None
        value: ast.expr | None = None
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            target_name, value = node.target.id, node.value
        elif isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            target_name, value = node.targets[0].id, node.value
        if target_name == "METRIC_DIMENSIONS" and value is not None:
            payload = ast.literal_eval(value)
            if isinstance(payload, dict):
                return {str(key): str(dimension) for key, dimension in payload.items()}
    raise RuntimeError(f"vendored WorldAtlas Arena metric mapping is missing: {catalog_path}")


def _add_score(buckets: dict[str, list[float]], metric_id: Any, value: Any) -> None:
    key = str(metric_id).strip().lower().replace("-", "_").replace(" ", "_")
    dimension = key if key in DIMENSION_IDS else _upstream_metric_dimensions().get(key)
    score = scalar_number(value)
    if dimension in buckets and score is not None and 0.0 <= score <= 1.0:
        buckets[dimension].append(float(score))


def _collect_scores(payload: Any, buckets: dict[str, list[float]]) -> None:
    if isinstance(payload, list):
        for row in payload:
            if not isinstance(row, dict):
                continue
            metric_id = row.get("dimension") or row.get("metric_id") or row.get("metric")
            value = row.get("normalized_score", row.get("normalized", row.get("score")))
            _add_score(buckets, metric_id, value)
        return
    if not isinstance(payload, dict):
        return

    for container in ("per_dimension_summary", "per_metric_summary"):
        nested = payload.get(container)
        if isinstance(nested, (dict, list)):
            _collect_scores(nested, buckets)

    for key, value in payload.items():
        if key in {"per_dimension_summary", "per_metric_summary"}:
            continue
        if isinstance(value, dict):
            # Official summaries are {suite: {dimension-or-metric: score}}.
            for nested_key, nested_value in value.items():
                _add_score(buckets, nested_key, nested_value)
        else:
            # Also accept a flat dimension/metric mapping for exported snippets.
            _add_score(buckets, key, value)


def extract_metrics(payload: Any, results_path: Path) -> dict[str, dict[str, Any]]:
    """Aggregate upstream normalized suite scores into its six public axes."""
    buckets = {dimension: [] for dimension in DIMENSION_IDS}
    _collect_scores(payload, buckets)
    extracted: dict[str, dict[str, Any]] = {}
    for dimension in DIMENSION_IDS:
        score = mean_numeric(buckets[dimension])
        if score is None:
            continue
        extracted[dimension] = {
            "metric_id": dimension,
            "raw_score": score,
            "normalized_score": score,
            "source": str(results_path),
            "sample_count": len(buckets[dimension]),
        }
    return extracted


def prepare_upstream_results(
    config: ors.BenchRunnerConfig,
    results_path: Path,
    args: Any,
    output_dir: Path,
) -> Path:
    """Resolve a report bundle to the canonical dimension summary."""
    del config, args, output_dir
    resolved = results_path.expanduser().resolve()
    if resolved.is_dir():
        direct = resolved / "per_dimension_summary.json"
        if direct.is_file():
            return direct
        matches = sorted(resolved.rglob("per_dimension_summary.json"))
        if matches:
            return matches[-1]
    if resolved.name in {"per_metric_summary.json", "run_manifest.json"}:
        sibling = resolved.with_name("per_dimension_summary.json")
        if sibling.is_file():
            return sibling
    return resolved


def discover_official_results(output_dir: Path, repo_root: Path | None) -> Path | None:
    del repo_root
    return ors.discover_by_globs(
        [output_dir],
        ("upstream/per_dimension_summary.json", "**/per_dimension_summary.json"),
    )


def _absolute_from_project(project_root: Path, value: Any) -> Any:
    if not isinstance(value, str) or not value.strip():
        return value
    path = Path(value).expanduser()
    return str(path if path.is_absolute() else (project_root / path).resolve())


def _materialize_runtime_config(source_path: Path, output_dir: Path) -> Path:
    """Keep every mutable upstream path below the WorldFoundry output root."""
    source_path = source_path.expanduser().resolve()
    payload = yaml.safe_load(source_path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"WorldAtlas Arena config must be a mapping: {source_path}")
    project_root = source_path.parent.parent
    state_root = output_dir / "upstream_runtime"
    paths = payload.setdefault("paths", {})
    if not isinstance(paths, dict):
        raise ValueError(f"WorldAtlas Arena config paths must be a mapping: {source_path}")
    for key in ("image_root", "video_root"):
        if key in paths:
            paths[key] = _absolute_from_project(project_root, paths[key])
    for key in ("inventory_dir", "manifest_dir", "predictions_dir", "reports_dir", "cache_dir"):
        paths[key] = str((state_root / key).resolve())

    benchmark = payload.get("benchmark")
    if isinstance(benchmark, dict):
        reference = benchmark.get("reference_corpus")
        if isinstance(reference, dict):
            for key in ("root", "index_path"):
                if reference.get(key):
                    reference[key] = _absolute_from_project(project_root, reference[key])
            reference["cache_dir"] = str((state_root / "reference_corpus_cache").resolve())

    destination = state_root / "config" / "benchmark.yaml"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return destination


def build_official_command(
    *,
    config: ors.BenchRunnerConfig,
    repo_root: Path,
    generated_video_dir: Path,
    output_dir: Path,
    args: Any,
) -> list[str] | None:
    del config
    manifest = args.manifest or os.environ.get("WORLDFOUNDRY_WORLDATLAS_ARENA_MANIFEST")
    if not manifest:
        raise ValueError("--manifest or WORLDFOUNDRY_WORLDATLAS_ARENA_MANIFEST is required")
    manifest_path = Path(manifest).expanduser().resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(f"WorldAtlas Arena manifest not found: {manifest_path}")
    config_path = Path(args.config).expanduser().resolve() if args.config else repo_root / "config" / "benchmark.yaml"
    if not config_path.is_file():
        raise FileNotFoundError(f"WorldAtlas Arena config not found: {config_path}")
    runtime_config = _materialize_runtime_config(config_path, output_dir)
    upstream_output = output_dir / "upstream"
    command = [
        args.python,
        "-m",
        "worldarena.cli",
        "evaluate",
        "--config",
        str(runtime_config),
        "--manifest",
        str(manifest_path),
        "--predictions-root",
        str(generated_video_dir),
        "--output-dir",
        str(upstream_output),
        "--model-name",
        args.model_name or generated_video_dir.name or "world-model",
    ]
    if args.suites:
        command.extend(["--suites", args.suites])
    if args.limit is not None:
        command.extend(["--limit", str(args.limit)])
    return command


def extend_parser(parser) -> None:
    parser.add_argument(
        "--generated-artifact-dir",
        dest="generated_video_dir",
        type=Path,
        help="Alias for --generated-video-dir used by the unified WorldFoundry CLI.",
    )
    parser.add_argument("--manifest", type=Path, help="Official WorldAtlas Arena benchmark manifest JSONL.")
    parser.add_argument("--config", type=Path, help="Upstream benchmark YAML; defaults to the vendored config.")
    parser.add_argument("--suites", help="Comma-separated upstream suite selection.")
    parser.add_argument("--limit", type=int, help="Bound the number of manifest samples.")
    parser.add_argument("--model-name", help="Model label recorded in the upstream report bundle.")


def main(argv: list[str] | None = None) -> int:
    return ors.run_main(
        CONFIG,
        ors.RunnerHooks(
            build_official_command=build_official_command,
            discover_official_results=discover_official_results,
            extract_metrics=extract_metrics,
            prepare_upstream_results=prepare_upstream_results,
            extend_parser=extend_parser,
        ),
        argv,
    )


if __name__ == "__main__":
    raise SystemExit(main())
