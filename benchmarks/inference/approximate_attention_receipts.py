"""Strict, framework-independent audit for sparse-attention request receipts."""

from __future__ import annotations

import math
import re
from collections.abc import Collection, Mapping, Sequence
from typing import Any

_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _counter(values: Mapping[str, Any], name: str) -> int | None:
    value = values.get(name)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


def _shape(value: object) -> tuple[int, ...] | None:
    if not isinstance(value, (list, tuple)):
        return None
    if not value or any(
        isinstance(item, bool) or not isinstance(item, int) or item <= 0
        for item in value
    ):
        return None
    return tuple(value)


def _tensor_contract_matches(event: Mapping[str, Any]) -> bool:
    input_contract = event.get("input")
    output_contract = event.get("output")
    if not isinstance(input_contract, Mapping) or not isinstance(
        output_contract,
        Mapping,
    ):
        return False
    input_shape = _shape(input_contract.get("shape"))
    output_shape = _shape(output_contract.get("shape"))
    input_device = input_contract.get("device")
    output_device = output_contract.get("device")
    input_dtype = input_contract.get("dtype")
    output_dtype = output_contract.get("dtype")
    return bool(
        input_shape is not None
        and input_shape == output_shape
        and isinstance(input_device, str)
        and input_device
        and input_device == output_device
        and isinstance(input_dtype, str)
        and input_dtype
        and input_dtype == output_dtype
        and list(input_shape) == event.get("input_shape")
        and input_device == event.get("input_device")
        and input_dtype == event.get("input_dtype")
        and list(output_shape) == event.get("output_shape")
        and output_device == event.get("output_device")
        and output_dtype == event.get("output_dtype")
    )


def _vmoba_route(
    profile: Mapping[str, Any],
    layer_idx: int,
) -> tuple[str, int | tuple[int, int] | tuple[int, int, int], int]:
    first_full_layer = int(profile["first_full_layer"])
    relative_layer = layer_idx - first_full_layer
    cycle = (
        ("temporal", int(profile["temporal_layer"])),
        ("spatial", int(profile["spatial_layer"])),
        ("spatiotemporal", int(profile["st_layer"])),
    )
    position = relative_layer % sum(count for _, count in cycle)
    for route, count in cycle:
        if position < count:
            if route == "temporal":
                return route, int(profile["temporal_chunk_size"]), int(
                    profile["temporal_topk"]
                )
            if route == "spatial":
                return route, tuple(int(v) for v in profile["spatial_chunk_size"]), int(
                    profile["spatial_topk"]
                )
            return route, tuple(int(v) for v in profile["st_chunk_size"]), int(
                profile["st_topk"]
            )
        position -= count
    raise AssertionError("unreachable VMoBA route")


def _expected_execution(
    *,
    kind: str,
    profile: Mapping[str, Any],
    step: int,
    total_steps: int,
    layer_idx: int,
) -> str:
    if kind == "vmoba":
        if step < int(profile["first_full_step"]) or layer_idx < int(
            profile["first_full_layer"]
        ):
            return "scheduled_dense"
        return "sparse"
    dense_steps = int(profile.get("dense_steps", 0))
    if step < dense_steps or step >= total_steps - dense_steps:
        return "scheduled_dense"
    if kind == "draft_attn" and layer_idx == 0:
        return "provider_dense"
    if kind == "rainfusion_attn" and step < int(
        profile.get("lightx2v_skip_timesteps", -1)
    ):
        return "provider_dense"
    return "sparse"


def audit_approximate_attention_receipt(
    report: Mapping[str, Any],
    *,
    expected_kind: str,
    allowed_provider_paths: Collection[str],
    expected_grid: Sequence[int],
    expected_wrapped_blocks: int,
    expected_total_steps: int,
    expected_routed_steps: bool,
    expected_branch_steps: Mapping[str, Sequence[int]],
    expected_profile: Mapping[str, Any],
    expected_provider_symbols: Sequence[str] = (),
    expected_lightx2v: Mapping[str, Any] | None = None,
    expected_fastvideo_sla: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate one finalized request down to every branch/step/layer event."""

    checks: dict[str, dict[str, Any]] = {}
    issues: list[str] = []

    def require(name: str, passed: bool, actual: Any, expected: Any) -> None:
        checks[name] = {
            "actual": actual,
            "expected": expected,
            "passed": bool(passed),
        }
        if not passed:
            issues.append(f"{name}: actual={actual!r}, expected={expected!r}")

    kind = str(expected_kind).strip().casefold()
    if expected_lightx2v is not None and expected_fastvideo_sla is not None:
        raise ValueError(
            "one receipt cannot be audited as both LightX2V and FastVideo SLA"
        )
    providers = frozenset(str(path) for path in allowed_provider_paths)
    grid = tuple(int(value) for value in expected_grid)
    if (
        isinstance(expected_total_steps, bool)
        or not isinstance(expected_total_steps, int)
        or expected_total_steps <= 0
    ):
        raise ValueError("expected_total_steps must be a positive integer")
    if not isinstance(expected_routed_steps, bool):
        raise ValueError("expected_routed_steps must be a boolean")
    total_steps = int(expected_total_steps)
    branches: dict[str, tuple[int, ...]] = {}
    for branch, raw_steps in expected_branch_steps.items():
        steps = tuple(int(step) for step in raw_steps)
        if (
            not branch
            or not steps
            or len(steps) != len(set(steps))
            or list(steps) != sorted(steps)
            or steps[0] < 0
            or steps[-1] >= total_steps
        ):
            raise ValueError(
                "expected sparse-attention branch steps must be non-empty, unique, "
                "sorted, and within expected_total_steps; "
                f"got {branch!r}: {raw_steps!r} for {total_steps} steps"
            )
        branches[str(branch)] = steps
    if not branches:
        raise ValueError("expected sparse-attention branches cannot be empty")
    expected_layers = tuple(range(int(expected_wrapped_blocks)))
    expected_keys = {
        (branch, step, layer)
        for branch, steps in branches.items()
        for step in steps
        for layer in expected_layers
    }
    expected_calls = len(expected_keys)

    request_id = report.get("request_id")
    request_epoch = _counter(report, "request_epoch")
    require("receipt.kind", report.get("kind") == kind, report.get("kind"), kind)
    require(
        "receipt.finalized",
        isinstance(request_id, str)
        and bool(request_id.strip())
        and request_epoch is not None
        and request_epoch > 0
        and report.get("request_local") is True
        and report.get("finalized") is True
        and report.get("completed") is True
        and report.get("release_reason") == "completed",
        {
            "request_id": request_id,
            "request_epoch": request_epoch,
            "request_local": report.get("request_local"),
            "finalized": report.get("finalized"),
            "completed": report.get("completed"),
            "release_reason": report.get("release_reason"),
        },
        "finalized completed request-local receipt",
    )
    require(
        "receipt.workload",
        _counter(report, "wrapped_blocks") == expected_wrapped_blocks
        and tuple(report.get("expected_layers") or ()) == expected_layers
        and tuple(report.get("grid_size") or ()) == grid
        and _counter(report, "expected_calls") == expected_calls,
        {
            "wrapped_blocks": report.get("wrapped_blocks"),
            "expected_layers": report.get("expected_layers"),
            "grid_size": report.get("grid_size"),
            "expected_calls": report.get("expected_calls"),
        },
        {
            "wrapped_blocks": expected_wrapped_blocks,
            "expected_layers": list(expected_layers),
            "grid_size": list(grid),
            "expected_calls": expected_calls,
        },
    )
    require(
        "receipt.routed_steps",
        report.get("routed_steps") is expected_routed_steps,
        report.get("routed_steps"),
        expected_routed_steps,
    )

    provider_path = report.get("provider_path")
    provider_paths = report.get("provider_paths")
    require(
        "receipt.provider",
        isinstance(provider_path, str)
        and provider_path in providers
        and isinstance(provider_paths, (list, tuple))
        and tuple(provider_paths) == (provider_path,),
        {"provider_path": provider_path, "provider_paths": provider_paths},
        {"allowed": sorted(providers), "one_request_provider": True},
    )

    sparse_calls = _counter(report, "sparse_calls")
    provider_dense_calls = _counter(report, "provider_dense_calls")
    attempts = _counter(report, "kernel_attempts")
    scheduled_dense_calls = _counter(report, "scheduled_dense_calls")
    dense_fallback_calls = _counter(report, "dense_fallback_calls")
    kernel_fallbacks = _counter(report, "kernel_fallbacks")
    require(
        "receipt.execution_counters",
        sparse_calls is not None
        and sparse_calls > 0
        and provider_dense_calls is not None
        and attempts == sparse_calls + provider_dense_calls
        and scheduled_dense_calls is not None
        and sparse_calls + provider_dense_calls + scheduled_dense_calls
        == expected_calls
        and dense_fallback_calls == 0
        and kernel_fallbacks == 0
        and report.get("runtime_effective") is True
        and report.get("effective_kernel") == kind,
        {
            "expected_calls": expected_calls,
            "kernel_attempts": attempts,
            "sparse_calls": sparse_calls,
            "provider_dense_calls": provider_dense_calls,
            "scheduled_dense_calls": scheduled_dense_calls,
            "dense_fallback_calls": dense_fallback_calls,
            "kernel_fallbacks": kernel_fallbacks,
            "runtime_effective": report.get("runtime_effective"),
            "effective_kernel": report.get("effective_kernel"),
        },
        "attempts == sparse + provider_dense; sparse > 0; provider executions + "
        "scheduled_dense == expected; zero fallback",
    )
    notes = report.get("notes")
    require(
        "receipt.notes",
        isinstance(notes, (list, tuple)) and not notes,
        notes,
        [],
    )

    events_value = report.get("events")
    events = list(events_value) if isinstance(events_value, (list, tuple)) else []
    require(
        "receipt.event_count",
        isinstance(events_value, (list, tuple))
        and _counter(report, "event_count") == expected_calls
        and len(events) == expected_calls,
        {"event_count": report.get("event_count"), "events": len(events)},
        expected_calls,
    )

    malformed: dict[str, list[int]] = {
        "mapping": [],
        "algorithm": [],
        "ownership": [],
        "schedule": [],
        "module": [],
        "execution": [],
        "provider": [],
        "tensor": [],
        "grid": [],
    }
    observed_keys: list[tuple[str, int, int]] = []
    sparse_events: list[Mapping[str, Any]] = []
    provider_dense_events: list[Mapping[str, Any]] = []
    execution_counts: dict[str, int] = {}
    for index, value in enumerate(events):
        if not isinstance(value, Mapping):
            malformed["mapping"].append(index)
            continue
        event = value
        branch = event.get("branch")
        step = event.get("step")
        step_index = event.get("step_index")
        layer = event.get("layer_idx")
        if (
            not isinstance(branch, str)
            or isinstance(step, bool)
            or not isinstance(step, int)
            or isinstance(step_index, bool)
            or not isinstance(step_index, int)
            or isinstance(layer, bool)
            or not isinstance(layer, int)
        ):
            malformed["schedule"].append(index)
            continue
        observed_keys.append((branch, step, layer))
        if event.get("algorithm") != kind:
            malformed["algorithm"].append(index)
        if (
            event.get("request_id") != request_id
            or event.get("request_epoch") != request_epoch
            or event.get("request_local") is not True
        ):
            malformed["ownership"].append(index)
        if (
            (branch, step, layer) not in expected_keys
            or step_index != step
            or event.get("total_steps") != total_steps
        ):
            malformed["schedule"].append(index)
        if event.get("module_path") != f"blocks.{layer}.self_attn":
            malformed["module"].append(index)
        execution = event.get("execution")
        expected_execution = _expected_execution(
            kind=kind,
            profile=expected_profile,
            step=step,
            total_steps=total_steps,
            layer_idx=layer,
        )
        if execution != expected_execution:
            malformed["execution"].append(index)
        if isinstance(execution, str):
            execution_counts[execution] = execution_counts.get(execution, 0) + 1
        if execution == "sparse":
            sparse_events.append(event)
            if (
                event.get("provider_attempted") is not True
                or event.get("provider_path") != provider_path
                or event.get("fallback_reason") is not None
                or event.get("dense_fallback_executed") is not False
            ):
                malformed["provider"].append(index)
        elif execution == "provider_dense":
            provider_dense_events.append(event)
            if (
                event.get("provider_attempted") is not True
                or not isinstance(event.get("provider_path"), str)
                or not str(event.get("provider_path")).strip()
                or event.get("fallback_reason") is not None
                or event.get("dense_fallback_executed") is not False
            ):
                malformed["provider"].append(index)
        elif execution == "scheduled_dense":
            if (
                event.get("provider_attempted") is not False
                or event.get("provider_path") is not None
                or event.get("fallback_reason") is not None
                or event.get("dense_fallback_executed") is not False
            ):
                malformed["provider"].append(index)
        else:
            malformed["execution"].append(index)
        if tuple(event.get("grid_size") or ()) != grid:
            malformed["grid"].append(index)
        if not _tensor_contract_matches(event):
            malformed["tensor"].append(index)

    require(
        "receipt.event_schema",
        not any(malformed.values()),
        {name: indices for name, indices in malformed.items() if indices},
        {},
    )
    require(
        "receipt.event_coverage",
        len(observed_keys) == len(set(observed_keys))
        and set(observed_keys) == expected_keys,
        {
            "observed": len(observed_keys),
            "unique": len(set(observed_keys)),
            "missing": len(expected_keys - set(observed_keys)),
            "unexpected": len(set(observed_keys) - expected_keys),
        },
        {
            "observed": expected_calls,
            "unique": expected_calls,
            "missing": 0,
            "unexpected": 0,
        },
    )
    actual_sparse_calls = execution_counts.get("sparse", 0)
    actual_provider_dense_calls = execution_counts.get("provider_dense", 0)
    actual_scheduled_dense_calls = execution_counts.get("scheduled_dense", 0)
    actual_provider_attempts = sum(
        bool(event.get("provider_attempted"))
        for event in events
        if isinstance(event, Mapping)
    )
    actual_dense_fallbacks = sum(
        bool(event.get("dense_fallback_executed"))
        for event in events
        if isinstance(event, Mapping)
    )
    require(
        "receipt.event_counter_reconciliation",
        set(execution_counts).issubset(
            {"sparse", "provider_dense", "scheduled_dense"}
        )
        and actual_sparse_calls == sparse_calls
        and actual_provider_dense_calls == provider_dense_calls
        and actual_scheduled_dense_calls == scheduled_dense_calls
        and actual_provider_attempts == attempts
        and actual_dense_fallbacks == dense_fallback_calls,
        {
            "execution_counts": execution_counts,
            "provider_attempts": actual_provider_attempts,
            "dense_fallbacks": actual_dense_fallbacks,
        },
        {
            "execution_counts": {
                "sparse": sparse_calls,
                **(
                    {"provider_dense": provider_dense_calls}
                    if provider_dense_calls
                    else {}
                ),
                "scheduled_dense": scheduled_dense_calls,
            },
            "provider_attempts": attempts,
            "dense_fallbacks": dense_fallback_calls,
        },
    )

    coverage = report.get("coverage")
    if not isinstance(coverage, Mapping):
        coverage = {}
    expected_execution_counts = {
        name: count
        for name, count in execution_counts.items()
        if count > 0
    }
    coverage_branches = coverage.get("branches")
    require(
        "receipt.coverage_summary",
        coverage.get("complete") is True
        and tuple(coverage.get("expected_layers") or ()) == expected_layers
        and _counter(coverage, "expected_layer_count") == expected_wrapped_blocks
        and _counter(coverage, "expected_calls") == expected_calls
        and _counter(coverage, "observed_calls") == expected_calls
        and coverage.get("event_totals_match_counters") is True
        and coverage.get("provider_contract_complete") is True
        and coverage.get("request_local") is True
        and coverage.get("routed_steps") is expected_routed_steps
        and coverage.get("execution_counts") == expected_execution_counts
        and isinstance(coverage_branches, Mapping)
        and set(coverage_branches) == set(branches),
        coverage,
        "complete request/branch/step/layer coverage",
    )
    malformed_branches: dict[str, Any] = {}
    if isinstance(coverage_branches, Mapping):
        for branch, steps in branches.items():
            branch_report = coverage_branches.get(branch)
            if not isinstance(branch_report, Mapping):
                malformed_branches[branch] = branch_report
                continue
            expected_layer_events = len(steps) * expected_wrapped_blocks
            reported_expected_steps = branch_report.get("expected_steps")
            if not (
                isinstance(reported_expected_steps, (list, tuple))
                and tuple(reported_expected_steps) == steps
                and tuple(branch_report.get("observed_steps") or ()) == steps
                and tuple(branch_report.get("completed_steps") or ()) == steps
                and branch_report.get("steps_contiguous") is True
                and branch_report.get("routed_steps") is expected_routed_steps
                and branch_report.get("all_steps_complete") is True
                and _counter(branch_report, "layer_events") == expected_layer_events
                and branch_report.get("missing_layers") == {}
                and _counter(branch_report, "last_step") == steps[-1]
            ):
                malformed_branches[branch] = branch_report
    require(
        "receipt.branch_coverage",
        not malformed_branches
        and isinstance(report.get("branches"), Mapping)
        and report.get("branches") == coverage_branches,
        {
            "malformed": malformed_branches,
            "top_level_matches_coverage": report.get("branches")
            == coverage_branches,
        },
        {"malformed": {}, "top_level_matches_coverage": True},
    )

    lifecycle = report.get("lifecycle")
    if not isinstance(lifecycle, Mapping):
        lifecycle = {}
    snapshots = _counter(lifecycle, "receipt_snapshots")
    max_snapshots = _counter(lifecycle, "max_receipt_snapshots")
    require(
        "receipt.lifecycle",
        _counter(lifecycle, "live_requests") == 0
        and snapshots is not None
        and snapshots > 0
        and max_snapshots is not None
        and max_snapshots > 0
        and snapshots <= max_snapshots,
        lifecycle,
        "zero live requests and one bounded immutable snapshot",
    )

    if kind == "vmoba":
        vmoba = report.get("vmoba")
        if not isinstance(vmoba, Mapping):
            vmoba = {}
        cycle = {
            "temporal": int(expected_profile["temporal_layer"]),
            "spatial": int(expected_profile["spatial_layer"]),
            "spatiotemporal": int(expected_profile["st_layer"]),
        }
        require(
            "receipt.vmoba.configuration",
            _counter(vmoba, "first_full_step")
            == int(expected_profile["first_full_step"])
            and _counter(vmoba, "first_full_layer")
            == int(expected_profile["first_full_layer"])
            and vmoba.get("layer_cycle") == cycle
            and tuple(vmoba.get("provider_symbols") or ())
            == tuple(expected_provider_symbols),
            vmoba,
            {
                "first_full_step": expected_profile["first_full_step"],
                "first_full_layer": expected_profile["first_full_layer"],
                "layer_cycle": cycle,
                "provider_symbols": list(expected_provider_symbols),
            },
        )
        vmoba_malformed: list[int] = []
        expected_chunk_calls = {
            "temporal": 0,
            "spatial": 0,
            "spatiotemporal": 0,
        }
        tokens = math.prod(grid)
        for index, event in enumerate(sparse_events):
            layer = int(event["layer_idx"])
            route, chunk_layout, requested_topk = _vmoba_route(
                expected_profile,
                layer,
            )
            expected_chunk_calls[route] += 1
            if route == "temporal":
                provider_chunk_size = int(chunk_layout) * grid[1] * grid[2]
                normalized_layout: int | list[int] = int(chunk_layout)
            elif route == "spatial":
                chunk_h, chunk_w = chunk_layout
                provider_chunk_size = grid[0] * chunk_h * chunk_w
                normalized_layout = list(chunk_layout)
            else:
                provider_chunk_size = math.prod(chunk_layout)
                normalized_layout = list(chunk_layout)
            effective_topk = min(
                requested_topk,
                math.ceil(tokens / provider_chunk_size),
            )
            packed_input = event.get("packed_input")
            restored_output = event.get("restored_output")
            packed_shape = (
                _shape(packed_input.get("shape"))
                if isinstance(packed_input, Mapping)
                else None
            )
            restored_shape = (
                _shape(restored_output.get("shape"))
                if isinstance(restored_output, Mapping)
                else None
            )
            flat_shape = _shape(event.get("input_shape"))
            valid_boundaries = bool(
                packed_shape is not None
                and len(packed_shape) == 4
                and packed_shape[1] == tokens
                and packed_shape == restored_shape
                and flat_shape is not None
                and len(flat_shape) == 3
                and flat_shape[0] == packed_shape[0] * tokens
                and flat_shape[1:] == packed_shape[2:]
                and isinstance(packed_input, Mapping)
                and isinstance(restored_output, Mapping)
                and packed_input.get("device") == restored_output.get("device")
                and packed_input.get("dtype") == restored_output.get("dtype")
            )
            if not (
                event.get("chunk_kind") == route
                and event.get("chunk_layout") == normalized_layout
                and _counter(event, "provider_chunk_size") == provider_chunk_size
                and _counter(event, "requested_topk") == requested_topk
                and _counter(event, "effective_topk") == effective_topk
                and event.get("select_mode")
                == expected_profile["moba_select_mode"]
                and event.get("threshold") == expected_profile["moba_threshold"]
                and event.get("threshold_type")
                == expected_profile["moba_threshold_type"]
                and tuple(event.get("provider_symbols") or ())
                == tuple(expected_provider_symbols)
                and tuple(event.get("validated_grid") or ()) == grid
                and event.get("validated_profile")
                == {
                    "temporal_chunk_size": expected_profile[
                        "temporal_chunk_size"
                    ],
                    "spatial_chunk_size": list(
                        expected_profile["spatial_chunk_size"]
                    ),
                    "st_chunk_size": list(expected_profile["st_chunk_size"]),
                }
                and valid_boundaries
            ):
                vmoba_malformed.append(index)
        vmoba_events = vmoba.get("events")
        chunk_calls = vmoba.get("chunk_calls")
        require(
            "receipt.vmoba.events",
            not vmoba_malformed
            and isinstance(vmoba_events, (list, tuple))
            and list(vmoba_events) == sparse_events
            and chunk_calls == expected_chunk_calls
            and all(count > 0 for count in expected_chunk_calls.values()),
            {
                "malformed": vmoba_malformed,
                "events": len(vmoba_events)
                if isinstance(vmoba_events, (list, tuple))
                else None,
                "chunk_calls": chunk_calls,
            },
            {
                "events": len(sparse_events),
                "chunk_calls": expected_chunk_calls,
            },
        )
    elif expected_lightx2v is not None:
        expected_operator = expected_lightx2v.get("operator")
        expected_symbol = expected_lightx2v.get("provider_symbol")
        expected_family = expected_lightx2v.get("provider_family")
        expected_commit = expected_lightx2v.get("commit")
        expected_fingerprint = expected_lightx2v.get("source_fingerprint")
        if not all(
            isinstance(value, str) and bool(value)
            for value in (
                expected_operator,
                expected_symbol,
                expected_family,
                expected_commit,
                expected_fingerprint,
            )
        ):
            raise ValueError(
                "expected_lightx2v requires non-empty operator, provider_symbol, "
                "provider_family, commit, and source_fingerprint strings"
            )
        if providers != frozenset({expected_symbol}):
            raise ValueError(
                "LightX2V receipt audit requires exactly the configured canonical "
                f"provider symbol; got {sorted(providers)!r}"
            )

        lightx2v_malformed: list[int] = []
        provider_events = [*sparse_events, *provider_dense_events]
        expected_provider_config = {
            "sparsity_ratio": expected_profile["sparsity"],
            "operator": expected_operator,
            "nbhd_coefficient": list(
                expected_profile.get("nbhd_coefficient", (1.0, 0.5, 0.056))
            ),
            "nbhd_min_width": expected_profile.get("nbhd_min_width", 1.0),
            "attnmap_frame_num": (
                expected_profile.get("attnmap_frame_num") or grid[0]
            ),
            "per_block_mean": expected_profile.get(
                "lightx2v_per_block_mean", False
            ),
            "pool_size": expected_profile.get("lightx2v_pool_size", 128),
            "skip_timesteps": expected_profile.get(
                "lightx2v_skip_timesteps", -1
            ),
            "dense_attn_type": expected_profile.get(
                "lightx2v_dense_attn_type", "flash_attn3"
            ),
            "svg_sample_mse_max_row": expected_profile.get(
                "svg_sample_mse_max_row", 10000
            ),
            "svg_num_sampled_rows": expected_profile.get(
                "svg_num_sampled_rows", 64
            ),
            "svg_context_length": expected_profile.get(
                "svg_context_length", 0
            ),
        }
        event_provider_calls = 0
        for index, event in enumerate(provider_events):
            provider_calls = _counter(event, "provider_calls")
            adapter_path = event.get("adapter_path")
            source_root = event.get("provider_source_root")
            execution = event.get("execution")
            event_provider_path = event.get("provider_path")
            canonical_provider_path = event.get(
                "canonical_sparse_provider_path"
            )
            expected_attnmap_frames = expected_profile.get("attnmap_frame_num")
            if expected_attnmap_frames is None:
                expected_attnmap_frames = grid[0]
            nbhd_valid = True
            if kind == "nbhd":
                nbhd_valid = bool(
                    _counter(event, "attnmap_frame_num")
                    == int(expected_attnmap_frames)
                    and tuple(event.get("nbhd_coefficient") or ())
                    == tuple(expected_profile["nbhd_coefficient"])
                    and event.get("nbhd_min_width")
                    == expected_profile["nbhd_min_width"]
                )
            if not (
                event.get("algorithm") == kind
                and event.get("operator") == expected_operator
                and event.get("provider_family") == expected_family
                and canonical_provider_path == expected_symbol
                and (
                    event_provider_path == expected_symbol
                    if execution == "sparse"
                    else isinstance(event_provider_path, str)
                    and bool(event_provider_path)
                    if execution == "provider_dense"
                    else False
                )
                and event.get("sparse_kernel_executed")
                is (execution == "sparse")
                and event.get("provider_dense_executed")
                is (execution == "provider_dense")
                and isinstance(adapter_path, str)
                and adapter_path.startswith("lightx2v.")
                and provider_calls is not None
                and provider_calls > 0
                and event.get("reference_lightx2v_commit") == expected_commit
                and event.get("provider_source_commit") == expected_commit
                and event.get("provider_source_clean") is True
                and event.get("provider_source_fingerprint")
                == expected_fingerprint
                and isinstance(source_root, str)
                and source_root
                and event.get("reference_parity_verified") is True
                and event.get("runtime_effective") is True
                and tuple(event.get("grid_size") or ()) == grid
                and event.get("sparsity_ratio")
                == expected_profile["sparsity"]
                and event.get("provider_config") == expected_provider_config
                # Runtime normalizes the provider receipt into this event. A
                # second nested copy is an obsolete synthetic-test schema and
                # must not be accepted as evidence from a real request.
                and "provider_receipt" not in event
                and nbhd_valid
            ):
                lightx2v_malformed.append(index)
            event_provider_calls += provider_calls or 0

        lightx2v = report.get("lightx2v")
        if not isinstance(lightx2v, Mapping):
            lightx2v = {}
        lightx2v_events = lightx2v.get("events")
        lightx2v_provider_dense_events = lightx2v.get("provider_dense_events")
        lightx2v_summary = {
            key: lightx2v.get(key)
            for key in (
                "algorithm",
                "provider_family",
                "provider_families",
                "operator",
                "canonical_provider_symbol",
                "canonical_provider_executed",
                "reference_lightx2v_commit",
                "provider_source_commit",
                "provider_source_commits",
                "provider_source_fingerprints",
                "provider_commit_complete",
                "provider_source_clean",
                "reference_parity_verified",
                "provider_calls",
                "receipt_count",
                "sparse_event_count",
                "provider_dense_event_count",
            )
        }
        lightx2v_summary.update(
            events=(
                len(lightx2v_events)
                if isinstance(lightx2v_events, (list, tuple))
                else None
            ),
            events_match=(
                isinstance(lightx2v_events, (list, tuple))
                and list(lightx2v_events) == sparse_events
            ),
            provider_dense_events=(
                len(lightx2v_provider_dense_events)
                if isinstance(lightx2v_provider_dense_events, (list, tuple))
                else None
            ),
            provider_dense_events_match=(
                isinstance(lightx2v_provider_dense_events, (list, tuple))
                and list(lightx2v_provider_dense_events) == provider_dense_events
            ),
        )
        require(
            "receipt.lightx2v.events",
            not lightx2v_malformed,
            lightx2v_malformed,
            [],
        )
        require(
            "receipt.lightx2v.summary",
            lightx2v.get("algorithm") == kind
            and lightx2v.get("provider_family") == expected_family
            and tuple(lightx2v.get("provider_families") or ())
            == (expected_family,)
            and lightx2v.get("operator") == expected_operator
            and lightx2v.get("canonical_provider_symbol") == expected_symbol
            and lightx2v.get("canonical_provider_executed") is True
            and lightx2v.get("reference_lightx2v_commit") == expected_commit
            and lightx2v.get("provider_source_commit") == expected_commit
            and tuple(lightx2v.get("provider_source_commits") or ())
            == (expected_commit,)
            and tuple(lightx2v.get("provider_source_fingerprints") or ())
            == (expected_fingerprint,)
            and lightx2v.get("provider_commit_complete") is True
            and lightx2v.get("provider_source_clean") is True
            and lightx2v.get("reference_parity_verified") is True
            and _counter(lightx2v, "provider_calls") == event_provider_calls
            and _counter(lightx2v, "receipt_count") == len(provider_events)
            and _counter(lightx2v, "sparse_event_count") == len(sparse_events)
            and _counter(lightx2v, "provider_dense_event_count")
            == len(provider_dense_events)
            and isinstance(
                lightx2v.get("provider_dense_events"), (list, tuple)
            )
            and list(lightx2v["provider_dense_events"])
            == provider_dense_events
            and isinstance(lightx2v.get("events"), (list, tuple))
            and list(lightx2v["events"]) == sparse_events,
            lightx2v_summary,
            "canonical pinned LightX2V provider with one complete receipt per "
            "sparse event",
        )
    elif expected_fastvideo_sla is not None:
        expected_symbol = expected_fastvideo_sla.get("provider_path")
        expected_family = expected_fastvideo_sla.get("provider_family")
        expected_commit = expected_fastvideo_sla.get("commit")
        expected_fingerprint = expected_fastvideo_sla.get("source_fingerprint")
        expected_layout = expected_fastvideo_sla.get("checkpoint_layout")
        expected_projection_fingerprint = expected_fastvideo_sla.get(
            "projection_fingerprint"
        )
        if not all(
            isinstance(value, str) and bool(value)
            for value in (
                expected_symbol,
                expected_family,
                expected_commit,
                expected_fingerprint,
            )
        ):
            raise ValueError(
                "expected_fastvideo_sla requires non-empty provider_path, "
                "provider_family, commit, and source_fingerprint strings"
            )
        if _SHA256.fullmatch(str(expected_fingerprint)) is None:
            raise ValueError(
                "expected_fastvideo_sla source_fingerprint must be SHA-256"
            )
        if expected_layout is not None and expected_layout not in {
            "fastvideo_diffusers",
            "turbodiffusion_original",
        }:
            raise ValueError(
                "expected_fastvideo_sla checkpoint_layout must name one audited layout"
            )
        if (
            expected_projection_fingerprint is not None
            and (
                not isinstance(expected_projection_fingerprint, str)
                or _SHA256.fullmatch(expected_projection_fingerprint) is None
            )
        ):
            raise ValueError(
                "expected_fastvideo_sla projection_fingerprint must be SHA-256"
            )
        if providers != frozenset({expected_symbol}):
            raise ValueError(
                "FastVideo SLA receipt audit requires exactly the configured "
                f"canonical provider class; got {sorted(providers)!r}"
            )

        event_provider_calls = 0
        malformed_fastvideo: list[int] = []
        provider_fingerprints: set[str] = set()
        layouts: set[str] = set()
        global_projection_fingerprints: set[str] = set()
        source_roots: set[str] = set()
        source_files: set[str] = set()
        layer_projection_fingerprints: dict[int, set[str]] = {}
        layer_source_keys: dict[int, set[tuple[str, ...]]] = {}
        global_call_indices: list[int] = []
        layer_call_indices: dict[int, list[int]] = {}
        for index, event in enumerate(sparse_events):
            layer = int(event["layer_idx"])
            provider_calls = _counter(event, "provider_calls")
            call_index = _counter(event, "call_index")
            layer_call_index = _counter(event, "layer_call_index")
            provider_fingerprint = event.get("provider_fingerprint")
            source_fingerprint = event.get("provider_source_fingerprint")
            projection_fingerprint = event.get(
                "projection_weight_fingerprint"
            )
            global_projection_fingerprint = event.get(
                "all_projection_weights_fingerprint"
            )
            layout = event.get("checkpoint_layout")
            source_root = event.get("provider_source_root")
            source_file = event.get("provider_source_file")
            source_keys_value = event.get("projection_source_keys")
            source_keys = (
                tuple(source_keys_value)
                if isinstance(source_keys_value, (list, tuple))
                and all(isinstance(key, str) for key in source_keys_value)
                else ()
            )
            if layout == "fastvideo_diffusers":
                prefix = f"blocks.{layer}.attn1.attn_impl.proj_l"
            elif layout == "turbodiffusion_original":
                prefix = (
                    f"blocks.{layer}.self_attn.attn_op.local_attn.proj_l"
                )
            else:
                prefix = ""
            expected_source_keys = (
                f"{prefix}.weight",
                f"{prefix}.bias",
            )
            valid = bool(
                event.get("algorithm") == kind
                and event.get("provider_family") == expected_family
                and event.get("provider_path") == expected_symbol
                and event.get("reference_provider_path") == expected_symbol
                and event.get("injected_test_provider") is False
                and provider_calls == 1
                and call_index is not None
                and call_index > 0
                and layer_call_index is not None
                and layer_call_index > 0
                and isinstance(provider_fingerprint, str)
                and _SHA256.fullmatch(provider_fingerprint) is not None
                and event.get("reference_fastvideo_commit") == expected_commit
                and event.get("provider_source_commit") == expected_commit
                and event.get("provider_source_clean") is True
                and source_fingerprint == expected_fingerprint
                and isinstance(source_root, str)
                and bool(source_root)
                and isinstance(source_file, str)
                and bool(source_file)
                and event.get("reference_parity_verified") is True
                and layout
                in {"fastvideo_diffusers", "turbodiffusion_original"}
                and (
                    expected_layout is None or layout == expected_layout
                )
                and source_keys == expected_source_keys
                and isinstance(projection_fingerprint, str)
                and _SHA256.fullmatch(projection_fingerprint) is not None
                and isinstance(global_projection_fingerprint, str)
                and _SHA256.fullmatch(global_projection_fingerprint) is not None
                and (
                    expected_projection_fingerprint is None
                    or global_projection_fingerprint
                    == expected_projection_fingerprint
                )
                and event.get("layer_prefix") == f"blocks.{layer}.attn1"
                and _counter(event, "current_timestep") == int(event["step"])
                and event.get("topk_ratio")
                == expected_profile["fastvideo_topk_ratio"]
                and event.get("feature_map")
                == expected_profile["fastvideo_feature_map"]
                and event.get("runtime_effective") is True
                and "provider_receipt" not in event
            )
            if not valid:
                malformed_fastvideo.append(index)
            event_provider_calls += provider_calls or 0
            if isinstance(provider_fingerprint, str):
                provider_fingerprints.add(provider_fingerprint)
            if isinstance(layout, str):
                layouts.add(layout)
            if isinstance(global_projection_fingerprint, str):
                global_projection_fingerprints.add(
                    global_projection_fingerprint
                )
            if isinstance(source_root, str):
                source_roots.add(source_root)
            if isinstance(source_file, str):
                source_files.add(source_file)
            if isinstance(projection_fingerprint, str):
                layer_projection_fingerprints.setdefault(layer, set()).add(
                    projection_fingerprint
                )
            if source_keys:
                layer_source_keys.setdefault(layer, set()).add(source_keys)
            if call_index is not None:
                global_call_indices.append(call_index)
            if layer_call_index is not None:
                layer_call_indices.setdefault(layer, []).append(layer_call_index)

        expected_layer_set = set(expected_layers)
        provider_counter_complete = bool(global_call_indices) and len(
            global_call_indices
        ) == len(set(global_call_indices))
        layer_counter_complete = set(layer_call_indices) == expected_layer_set and all(
            len(indices) == len(set(indices)) for indices in layer_call_indices.values()
        )
        projection_coverage_complete = bool(expected_layer_set) and (
            set(layer_projection_fingerprints) == expected_layer_set
            and all(
                len(values) == 1
                for values in layer_projection_fingerprints.values()
            )
            and set(layer_source_keys) == expected_layer_set
            and all(len(values) == 1 for values in layer_source_keys.values())
        )
        require(
            "receipt.fastvideo_sla.events",
            not malformed_fastvideo
            and len(provider_fingerprints) == 1
            and len(layouts) == 1
            and len(global_projection_fingerprints) == 1
            and len(source_roots) == 1
            and len(source_files) == 1
            and provider_counter_complete
            and layer_counter_complete
            and projection_coverage_complete,
            {
                "malformed": malformed_fastvideo,
                "provider_fingerprints": sorted(provider_fingerprints),
                "layouts": sorted(layouts),
                "global_projection_fingerprints": sorted(
                    global_projection_fingerprints
                ),
                "source_roots": sorted(source_roots),
                "source_files": sorted(source_files),
                "provider_counter_complete": provider_counter_complete,
                "layer_counter_complete": layer_counter_complete,
                "projection_coverage_complete": projection_coverage_complete,
            },
            "one pinned provider/source/layout/global projection and complete "
            "per-layer fingerprints/counters",
        )

        fastvideo = report.get("fastvideo_sla")
        if not isinstance(fastvideo, Mapping):
            fastvideo = {}
        normalized_layer_fingerprints = {
            str(layer): next(iter(values))
            for layer, values in sorted(layer_projection_fingerprints.items())
            if len(values) == 1
        }
        normalized_source_keys = {
            str(layer): list(next(iter(values)))
            for layer, values in sorted(layer_source_keys.items())
            if len(values) == 1
        }
        require(
            "receipt.fastvideo_sla.summary",
            fastvideo.get("algorithm") == kind
            and fastvideo.get("provider_family") == expected_family
            and tuple(fastvideo.get("provider_families") or ())
            == (expected_family,)
            and tuple(fastvideo.get("expected_layers") or ()) == expected_layers
            and fastvideo.get("canonical_provider_path") == expected_symbol
            and fastvideo.get("canonical_provider_executed") is True
            and fastvideo.get("reference_fastvideo_commit") == expected_commit
            and fastvideo.get("provider_source_commit") == expected_commit
            and tuple(fastvideo.get("provider_source_commits") or ())
            == (expected_commit,)
            and tuple(fastvideo.get("provider_source_fingerprints") or ())
            == (expected_fingerprint,)
            and tuple(fastvideo.get("provider_fingerprints") or ())
            == tuple(sorted(provider_fingerprints))
            and fastvideo.get("provider_source_root")
            == next(iter(source_roots), None)
            and tuple(fastvideo.get("provider_source_roots") or ())
            == tuple(sorted(source_roots))
            and fastvideo.get("provider_source_file")
            == next(iter(source_files), None)
            and tuple(fastvideo.get("provider_source_files") or ())
            == tuple(sorted(source_files))
            and fastvideo.get("provider_commit_complete") is True
            and fastvideo.get("provider_source_clean") is True
            and fastvideo.get("reference_parity_verified") is True
            and fastvideo.get("checkpoint_layout") == next(iter(layouts), None)
            and tuple(fastvideo.get("checkpoint_layouts") or ())
            == tuple(sorted(layouts))
            and fastvideo.get("all_projection_weights_fingerprint")
            == next(iter(global_projection_fingerprints), None)
            and tuple(fastvideo.get("projection_fingerprints") or ())
            == tuple(sorted(global_projection_fingerprints))
            and fastvideo.get("layer_projection_fingerprints")
            == normalized_layer_fingerprints
            and fastvideo.get("projection_source_keys")
            == normalized_source_keys
            and fastvideo.get("topk_ratio")
            == expected_profile["fastvideo_topk_ratio"]
            and fastvideo.get("feature_map")
            == expected_profile["fastvideo_feature_map"]
            and _counter(fastvideo, "provider_calls") == event_provider_calls
            and _counter(fastvideo, "receipt_count") == len(sparse_events)
            and _counter(fastvideo, "sparse_event_count") == len(sparse_events)
            and isinstance(fastvideo.get("events"), (list, tuple))
            and list(fastvideo["events"]) == sparse_events,
            fastvideo,
            "canonical pinned FastVideo learned-SLA provider with exact "
            "checkpoint/source/projection evidence",
        )

    return {"passed": not issues, "checks": checks, "issues": issues}


__all__ = ["audit_approximate_attention_receipt"]
