"""Bind the standalone runner's calls to its real pipeline signatures without CUDA imports."""

from __future__ import annotations

import ast
import inspect
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from omegaconf import OmegaConf

from worldfoundry.core.io.paths import resolve_data_path
from worldfoundry.synthesis.visual_generation.forcing import runtime

_RUNTIME_ROOT = Path(runtime.__file__).parent / "causal_forcing_runtime"


def _inference_signature(module_name: str) -> inspect.Signature:
    """Load only the real signature; importing the runner initializes CUDA."""
    path = _RUNTIME_ROOT / "pipeline" / f"{module_name}.py"
    source = ast.parse(path.read_text(encoding="utf-8"))
    owner = next(node for node in source.body if isinstance(node, ast.ClassDef))
    method = next(node for node in owner.body if isinstance(node, ast.FunctionDef) and node.name == "inference")
    method.body = [ast.Pass()]
    method.decorator_list = []
    module = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), method],
        type_ignores=[],
    )
    namespace: dict[str, Any] = {}
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    return inspect.signature(namespace["inference"])


@pytest.mark.parametrize(
    "config_name,module_name",
    [
        ("ar_diffusion_tf_framewise.yaml", "causal_diffusion_inference"),
        ("causal_forcing_dmd_framewise.yaml", "causal_inference"),
    ],
)
@pytest.mark.parametrize("report_timing", [False, True])
@pytest.mark.parametrize("sample_index", [0, 1])
def test_causal_runner_dispatch_matches_pipeline_signature(
    config_name: str, module_name: str, report_timing: bool, sample_index: int
) -> None:
    config_dir = resolve_data_path("models", "runtime", "configs", "causal_forcing")
    config = OmegaConf.merge(
        OmegaConf.load(config_dir / "default_config.yaml"), OmegaConf.load(config_dir / config_name)
    )
    few_step = hasattr(config, "denoising_step_list")
    signature = _inference_signature(module_name)
    calls: list[dict[str, Any]] = []

    def inference(**kwargs: Any) -> tuple[object, object]:
        signature.bind(None, **kwargs)
        calls.append(kwargs)
        return object(), object()

    runner = _RUNTIME_ROOT / "inference.py"
    source = ast.parse(runner.read_text(encoding="utf-8"))
    loop = next(node for node in source.body if isinstance(node, ast.For))
    start = next(
        index
        for index, node in enumerate(loop.body)
        if isinstance(node, ast.Assign)
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id == "sample_report_timing"
    )
    end = next(
        index
        for index, node in enumerate(loop.body)
        if isinstance(node, ast.Assign)
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Attribute)
        and node.value.func.attr == "inference"
    )
    namespace = {
        "few_step_inference": few_step,
        "args": SimpleNamespace(report_timing=report_timing),
        "i": sample_index,
        "pipeline": SimpleNamespace(inference=inference),
        "sampled_noise": object(),
        "prompts": ["A landscape"],
        "initial_latent": None,
    }
    dispatch = ast.Module(body=loop.body[start : end + 1], type_ignores=[])
    exec(compile(dispatch, str(runner), "exec"), namespace)

    assert len(calls) == 1
    assert calls[0]["return_latents"] is True
    assert ("report_timing" in calls[0]) is few_step
    assert namespace["sample_report_timing"] is (few_step and report_timing and sample_index >= 1)
    if few_step:
        assert calls[0]["report_timing"] is namespace["sample_report_timing"]
