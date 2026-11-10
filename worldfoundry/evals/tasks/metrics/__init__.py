"""Registration and compute APIs for reusable evaluation metrics.

The package root stays lightweight so CLI discovery and existing-results
evaluation do not require optional metric dependencies such as PIL, torch, or
model-specific scorer packages. Heavy metric implementations are imported when
their public symbols are first accessed.
"""

from __future__ import annotations

import sys
from importlib import import_module
from types import ModuleType
from typing import Any

from .registry import (
    BuiltinExistingResultsMetric,
    DuplicateMetricRegistryKeyError,
    MetricRegistry,
    MetricRegistryEntry,
    MetricRegistryError,
    UnknownMetricRegistryKeyError,
    create_existing_results_metric,
    default_metric_registry,
    list_metric_registry_entries,
    missing_artifacts,
    normalize_artifact_records,
    validate_metric_ids,
)

_PACKAGE = __name__
_SCORERS_PACKAGE = "worldfoundry.evals.tasks.execution.runners._scorers"

_LAZY_MODULES = {
    "fid": f"{_PACKAGE}.fid",
    "fvd": f"{_PACKAGE}.fvd",
    "jedi": f"{_PACKAGE}.jedi",
    "clip_score": f"{_SCORERS_PACKAGE}.clip_score",
    "itm_score": f"{_SCORERS_PACKAGE}.itm_score",
    "scoring": f"{_SCORERS_PACKAGE}.scoring",
    "vqa_score": f"{_SCORERS_PACKAGE}.vqa_score",
}

_LAZY_ATTRS = {
    "CLIPScore": f"{_SCORERS_PACKAGE}.clip_score",
    "list_all_clipscore_models": f"{_SCORERS_PACKAGE}.clip_score",
    "compute_clip_fid": f"{_PACKAGE}.fid",
    "compute_distribution_metrics": f"{_PACKAGE}.fid",
    "compute_fid": f"{_PACKAGE}.fid",
    "compute_fid_vid": f"{_PACKAGE}.fid",
    "compute_scene_fid": f"{_PACKAGE}.fid",
    "compute_swav_fid": f"{_PACKAGE}.fid",
    "extract_object_crops": f"{_PACKAGE}.fid",
    "summarize_distribution_metrics": f"{_PACKAGE}.fid",
    "compute_kid": f"{_PACKAGE}.kid",
    "compute_inception_score": f"{_PACKAGE}.inception_score",
    "ITMScore": f"{_SCORERS_PACKAGE}.itm_score",
    "list_all_itmscore_models": f"{_SCORERS_PACKAGE}.itm_score",
    "compute_jedi_from_features": f"{_PACKAGE}.jedi",
    "compute_mock_jedi": f"{_PACKAGE}.jedi",
    "ensure_ffmpeg": f"{_SCORERS_PACKAGE}.scoring",
    "get_score_model": f"{_SCORERS_PACKAGE}.scoring",
    "list_all_models": f"{_SCORERS_PACKAGE}.scoring",
    "package_root": f"{_SCORERS_PACKAGE}.scoring",
    "resolve_cache_dir": f"{_SCORERS_PACKAGE}.scoring",
    "compute_fvd_from_frame_dirs": f"{_PACKAGE}.fvd",
    "compute_fvd_from_numpy": f"{_PACKAGE}.fvd",
    "VQAScore": f"{_SCORERS_PACKAGE}.vqa_score",
    "list_all_vqascore_models": f"{_SCORERS_PACKAGE}.vqa_score",
}

_BENCHMARK_METRIC_EXPORTS = {
    "BenchmarkMetricInput",
    "BenchmarkMetricProtocol",
    "BLOCKED_EVALUATION_KINDS",
    "ExternalMetricEvaluationRequest",
    "ExternalMetricEvaluatorEntry",
    "ExternalMetricEvaluatorRegistry",
    "FORMULA_EVALUATOR_BINDINGS",
    "LOCAL_EVALUATORS",
    "MetricEvaluationCallable",
    "MetricEvaluatorBinding",
    "SUCCESS_METRIC_IDS_BY_BENCHMARK",
    "boolean_accuracy",
    "camera_binary_classification_metrics",
    "camera_retrieval_metrics",
    "camera_vqa_metrics",
    "chronomagic_average_scores",
    "default_external_metric_evaluator_registry",
    "evaluate_external_metric",
    "get_external_metric_evaluator",
    "jedi_mmd_score",
    "list_external_metric_evaluators",
    "multiple_choice_accuracy",
    "pairwise_preference_accuracy",
    "parse_worldmodelbench_score",
    "payload_from_request_parts",
    "records_from_payload",
    "score_vector_spearman",
    "success_rate",
    "success_metric_bindings",
    "vbench_final_score",
    "videoverse_subquestion_metrics",
    "worldmodelbench_score",
}


def _load_module_alias(name: str) -> ModuleType:
    module_path = _LAZY_MODULES[name]
    module = import_module(module_path)
    sys.modules[f"{_PACKAGE}.{name}"] = module
    globals()[name] = module
    return module


def __getattr__(name: str) -> Any:
    if name in _LAZY_MODULES:
        return _load_module_alias(name)
    if name in _LAZY_ATTRS:
        module = import_module(_LAZY_ATTRS[name])
        value = getattr(module, name)
        globals()[name] = value
        return value
    if name in _BENCHMARK_METRIC_EXPORTS:
        module = import_module("worldfoundry.evals.tasks.execution.runners._benchmark_metrics")
        value = getattr(module, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {_PACKAGE!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))


__all__ = [
    "BLOCKED_EVALUATION_KINDS",
    "FORMULA_EVALUATOR_BINDINGS",
    "LOCAL_EVALUATORS",
    "SUCCESS_METRIC_IDS_BY_BENCHMARK",
    "BenchmarkMetricInput",
    "BenchmarkMetricProtocol",
    "BuiltinExistingResultsMetric",
    "CLIPScore",
    "DuplicateMetricRegistryKeyError",
    "ExternalMetricEvaluationRequest",
    "ExternalMetricEvaluatorEntry",
    "ExternalMetricEvaluatorRegistry",
    "ITMScore",
    "MetricEvaluationCallable",
    "MetricEvaluatorBinding",
    "MetricRegistry",
    "MetricRegistryEntry",
    "MetricRegistryError",
    "UnknownMetricRegistryKeyError",
    "VQAScore",
    "boolean_accuracy",
    "camera_binary_classification_metrics",
    "camera_retrieval_metrics",
    "camera_vqa_metrics",
    "chronomagic_average_scores",
    "clip_score",
    "compute_clip_fid",
    "compute_distribution_metrics",
    "compute_fid",
    "compute_fid_vid",
    "compute_fvd_from_frame_dirs",
    "compute_fvd_from_numpy",
    "compute_inception_score",
    "compute_jedi_from_features",
    "compute_kid",
    "compute_mock_jedi",
    "compute_scene_fid",
    "compute_swav_fid",
    "create_existing_results_metric",
    "default_external_metric_evaluator_registry",
    "default_metric_registry",
    "ensure_ffmpeg",
    "evaluate_external_metric",
    "extract_object_crops",
    "fid",
    "fvd",
    "get_external_metric_evaluator",
    "get_score_model",
    "itm_score",
    "jedi",
    "jedi_mmd_score",
    "list_all_clipscore_models",
    "list_all_itmscore_models",
    "list_all_models",
    "list_all_vqascore_models",
    "list_external_metric_evaluators",
    "list_metric_registry_entries",
    "missing_artifacts",
    "multiple_choice_accuracy",
    "normalize_artifact_records",
    "package_root",
    "pairwise_preference_accuracy",
    "parse_worldmodelbench_score",
    "payload_from_request_parts",
    "records_from_payload",
    "resolve_cache_dir",
    "score_vector_spearman",
    "scoring",
    "success_metric_bindings",
    "success_rate",
    "validate_metric_ids",
    "vbench_final_score",
    "videoverse_subquestion_metrics",
    "vqa_score",
    "worldmodelbench_score",
]
