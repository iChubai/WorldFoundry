from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    ("mode", "model_config"),
    (
        ("static", "3dgs_res_704_1280_views_121_multi_6_prune.yaml"),
        ("dynamic", "3dgs_res_704_1280_views_121_multi_6_dynamic_prune.yaml"),
    ),
)
def test_lyra1_packaged_demo_configs_resolve_to_the_model_definition(mode, model_config) -> None:
    omegaconf = pytest.importorskip("omegaconf")
    from worldfoundry.base_models.three_dimensions.point_clouds.lyra import runtime_v1

    root = runtime_v1.LYRA1_CONFIG_ROOT
    demo = omegaconf.OmegaConf.merge(
        omegaconf.OmegaConf.load(root / "inference" / "default.yaml"),
        omegaconf.OmegaConf.load(root / "demo" / f"lyra_{mode}.yaml"),
    )
    model_paths = runtime_v1.Lyra1Representation._resolve_model_config_paths(demo.config_path, None)

    assert model_paths == [str(root / "model" / "default.yaml"), str(root / "model" / model_config)]
    model = omegaconf.OmegaConf.merge(*(omegaconf.OmegaConf.load(path) for path in model_paths))
    assert model.vae_backbone == "cosmos1"
    assert list(model.img_size) == [704, 1280]
    assert model.num_input_multi_views == 6
    assert bool(model.get("use_time_embedding")) is (mode == "dynamic")


def test_lyra1_reconstruction_runs_sample_with_packaged_configs(monkeypatch, tmp_path) -> None:
    omegaconf = pytest.importorskip("omegaconf")
    from worldfoundry.base_models.three_dimensions.point_clouds.lyra import runtime_v1

    checkpoints = tmp_path / "checkpoints"
    (checkpoints / "Lyra").mkdir(parents=True)
    for name in ("lyra_static.pt", "lyra_dynamic.pt"):
        (checkpoints / "Lyra" / name).touch()
    generated_root = tmp_path / "generated"
    generated_root.mkdir()
    registry: dict[str, dict[str, Any]] = {
        "lyra_static_template_generated": {"cls": object, "kwargs": {"root_path": ""}}
    }
    seen = {}

    def run_sample(config) -> None:
        seen["config"] = config
        seen["registered"] = dict(registry[config.dataset_name]["kwargs"])
        seen["cwd"] = Path.cwd()

    def load_and_merge_configs(paths):
        return omegaconf.OmegaConf.merge(*(omegaconf.OmegaConf.load(path) for path in paths))

    modules = {
        "worldfoundry.synthesis.visual_generation.lyra_1.lyra1_runtime.sample": SimpleNamespace(main=run_sample),
        "worldfoundry.synthesis.visual_generation.lyra_1.lyra1_runtime.src.models.eval_inputs.registry": SimpleNamespace(
            dataset_registry=registry
        ),
        "worldfoundry.synthesis.visual_generation.lyra_1.lyra1_runtime.src.models.utils.misc": SimpleNamespace(
            load_and_merge_configs=load_and_merge_configs
        ),
    }
    monkeypatch.setattr(runtime_v1, "resolve_lyra1_repo_root", lambda path: str(tmp_path))
    monkeypatch.setattr(runtime_v1.importlib, "import_module", modules.__getitem__)

    model = runtime_v1.Lyra1Representation.from_pretrained("lyra-1", device="cpu", checkpoint_dir=str(checkpoints))
    paths = list(sys.path)
    previous_cwd = Path.cwd()
    result = model.get_representation(
        {"generated_root": str(generated_root), "mode": "static", "output_dir": str(tmp_path / "out")}
    )

    assert sys.path == paths
    assert Path.cwd() == previous_cwd
    config = seen["config"]
    root = runtime_v1.LYRA1_CONFIG_ROOT
    assert config.ckpt_path == str(checkpoints / "Lyra" / "lyra_static.pt")
    assert config.vae_path == str(checkpoints / "Cosmos-Tokenize1-CV8x8x8-720p")
    assert list(config.config_path) == [
        str(root / "model" / "default.yaml"),
        str(root / "model" / "3dgs_res_704_1280_views_121_multi_6_prune.yaml"),
    ]
    assert seen["registered"]["root_path"] == str(generated_root)
    assert seen["cwd"] == tmp_path
    assert config.dataset_name not in registry
    render_root = tmp_path / "out" / "static_view_indices_fixed_5_0_1_2_3_4" / config.dataset_name
    assert result["render_roots"] == [str(render_root)]
    assert result["main_render_dir"] == str(render_root / "main_gaussians_renderings")
