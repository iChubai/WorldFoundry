"""Shared scorecard / manifest layout for in-process evaluation runs.

``contract`` and ``existing_results`` write the same artifact names.  Generation
vs offline scoring stay in those modules; this module owns paths, coercion, and
directory setup.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from worldfoundry.evaluation.utils import jsonable, reset_jsonl

JsonRow = dict[str, Any]

SESSION_JSONL_KEYS = ("requests", "results", "artifacts", "sample_ledger", "per_sample")


def utcnow_iso() -> str:
    """UTC timestamp with ``Z`` suffix and no microseconds."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def coerce_mapping(value: Any) -> JsonRow:
    payload = jsonable(value)
    if isinstance(payload, Mapping):
        return dict(payload)
    return {"value": payload}


def coerce_optional_mapping(value: Mapping[str, Any] | Any | None, default: Mapping[str, Any]) -> JsonRow:
    if value is None:
        return dict(default)
    return coerce_mapping(value)


def prepare_run_output_dir(output_dir: str | Path) -> Path:
    root = Path(output_dir).resolve()
    (root / "metrics").mkdir(parents=True, exist_ok=True)
    return root


def session_paths(output_dir: Path) -> dict[str, Path]:
    """Working paths used while a run is in progress."""
    metrics_dir = output_dir / "metrics"
    return {
        "manifest": output_dir / "run_manifest.json",
        "environment": output_dir / "environment.json",
        "env_requirements": output_dir / "env_requirements.json",
        "execution_plan": output_dir / "execution_plan.json",
        "requests": output_dir / "requests.jsonl",
        "results": output_dir / "results.jsonl",
        "artifacts": output_dir / "artifacts.jsonl",
        "sample_ledger": output_dir / "sample_ledger.jsonl",
        "per_sample": metrics_dir / "per_sample.jsonl",
        "summary": metrics_dir / "summary.json",
        "run_summary": output_dir / "summary.json",
        "report": output_dir / "report.md",
        "scorecard": output_dir / "scorecard.json",
    }


def reset_session_jsonl(paths: Mapping[str, Path]) -> None:
    for key in SESSION_JSONL_KEYS:
        reset_jsonl(paths[key])


def artifact_report_paths(output_dir: Path, artifact_count: int) -> dict[str, str]:
    """Absolute path map written into the finished scorecard."""
    paths = {
        "run_manifest": output_dir / "run_manifest.json",
        "environment": output_dir / "environment.json",
        "env_requirements": output_dir / "env_requirements.json",
        "execution_plan": output_dir / "execution_plan.json",
        "requests": output_dir / "requests.jsonl",
        "results": output_dir / "results.jsonl",
        "sample_ledger": output_dir / "sample_ledger.jsonl",
        "per_sample_metrics": output_dir / "metrics" / "per_sample.jsonl",
        "summary": output_dir / "metrics" / "summary.json",
        "run_summary": output_dir / "summary.json",
        "report": output_dir / "report.md",
        "scorecard": output_dir / "scorecard.json",
    }
    if artifact_count:
        paths["artifacts"] = output_dir / "artifacts.jsonl"
    return {name: str(path.resolve()) for name, path in paths.items()}
