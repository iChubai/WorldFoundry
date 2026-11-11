# SPDX-FileCopyrightText: 2026 OpenEnvision
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType

import pytest
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import OmegaConf

from worldfoundry.synthesis.visual_generation.diamond import worldfoundry_runtime as runtime
from worldfoundry.synthesis.visual_generation.diamond.agent import AgentConfig
from worldfoundry.synthesis.visual_generation.diamond.models.actor_critic import ActorCriticConfig
from worldfoundry.synthesis.visual_generation.diamond.models.diffusion import DenoiserConfig, DiffusionSamplerConfig
from worldfoundry.synthesis.visual_generation.diamond.models.rew_end_model import RewEndModelConfig
from worldfoundry.synthesis.visual_generation.diamond.worldfoundry_runtime import CONFIG_DIR

pytestmark = pytest.mark.unit


def test_packaged_agent_config_constructs_owned_dataclasses() -> None:
    with initialize_config_dir(version_base="1.3", config_dir=str(CONFIG_DIR)):
        config = compose(config_name="trainer")
    agent = instantiate(config.agent, num_actions=6)

    assert isinstance(agent, AgentConfig)
    assert isinstance(agent.denoiser, DenoiserConfig)
    assert isinstance(agent.rew_end_model, RewEndModelConfig)
    assert isinstance(agent.actor_critic, ActorCriticConfig)
    assert agent.denoiser.inner_model.num_actions == 6
    assert agent.rew_end_model.num_actions == 6
    assert agent.actor_critic.num_actions == 6
    assert agent.denoiser.inner_model.num_steps_conditioning == 4
    assert agent.actor_critic.img_size == agent.rew_end_model.img_size == 64


def test_packaged_diffusion_sampler_config_constructs_owned_dataclass() -> None:
    config = OmegaConf.load(CONFIG_DIR / "trainer.yaml")
    config.world_model_env.diffusion_sampler.s_tmax = float("inf")
    sampler = instantiate(config.world_model_env.diffusion_sampler)

    assert (
        config.world_model_env._target_ == "worldfoundry.synthesis.visual_generation.diamond.envs.WorldModelEnvConfig"
    )
    assert isinstance(sampler, DiffusionSamplerConfig)
    assert config.world_model_env.horizon == 15
    assert sampler.num_steps_denoising == 3
    assert sampler.order == 1


def test_released_config_targets_resolve_without_global_aliases(monkeypatch: pytest.MonkeyPatch) -> None:
    with initialize_config_dir(version_base="1.3", config_dir=str(CONFIG_DIR)):
        config = compose(config_name="trainer")
    expected = OmegaConf.to_container(config, resolve=False)
    for node in (
        "agent",
        "agent.denoiser",
        "agent.denoiser.inner_model",
        "agent.rew_end_model",
        "agent.actor_critic",
        "world_model_env",
        "world_model_env.diffusion_sampler",
    ):
        key = f"{node}._target_"
        target = OmegaConf.select(config, key)
        OmegaConf.update(config, key, target.removeprefix("worldfoundry.synthesis.visual_generation.diamond."))
    foreign = ModuleType("models")
    monkeypatch.setitem(sys.modules, "models", foreign)
    search_path = sys.path.copy()

    runtime.use_package_config_targets(config)

    assert OmegaConf.to_container(config, resolve=False) == expected
    assert sys.path == search_path
    assert sys.modules["models"] is foreign
    assert isinstance(instantiate(config.agent, num_actions=6), AgentConfig)
    runtime.use_package_config_targets(config)
    assert OmegaConf.to_container(config, resolve=False) == expected


def test_unused_training_config_values_remain_without_dangling_targets() -> None:
    config = OmegaConf.load(CONFIG_DIR / "trainer.yaml")

    assert OmegaConf.to_container(config.denoiser.sigma_distribution) == {
        "loc": -0.4,
        "scale": 1.2,
        "sigma_min": 0.002,
        "sigma_max": 20,
    }
    assert OmegaConf.to_container(config.actor_critic.actor_critic_loss) == {
        "backup_every": 15,
        "gamma": 0.985,
        "lambda_": 0.95,
        "weight_value_loss": 1.0,
        "weight_entropy_loss": 0.001,
    }


@pytest.mark.parametrize("pretrained", [False, True])
def test_atari_module_launch_keeps_runtime_options(pretrained: bool) -> None:
    context = {
        "python": "/env/python",
        "entrypoint": "/unused/play.py",
        "output_dir": "/output",
        "output_path": "/output/video.mp4",
        "options": {
            "pretrained": pretrained,
            "pretrained_dir": "/assets",
            "pretrained_game": "Pong",
            "model_path": "/model.pt",
            "config_dir": "/config",
            "fps": 12,
            "size": 320,
            "num_steps_initial_collect": 20,
            "frames": 4,
            "record": True,
            "store_denoising_trajectory": True,
            "store_original_obs": True,
        },
    }
    checkpoint_args = (
        ["--pretrained", "--pretrained-game", "Pong", "--pretrained-dir", "/assets"]
        if pretrained
        else [
            "--checkpoint",
            "/model.pt",
        ]
    )

    assert runtime.build_command(context) == [
        "/env/python",
        "-m",
        "worldfoundry.synthesis.visual_generation.diamond.play",
        "--config-dir",
        "/config",
        "--fps",
        "12",
        "--size",
        "320",
        "--num-steps-initial-collect",
        "20",
        "--no-header",
        *checkpoint_args,
        "--record",
        "--store-denoising-trajectory",
        "--store-original-obs",
        "--headless-steps",
        "4",
        "--output-path",
        "/output/video.mp4",
        "--dataset-dir",
        "/output/initialization_dataset",
    ]


@pytest.mark.parametrize("sampler", ["fast", "higher_quality"])
def test_csgo_module_launch_keeps_runtime_options(sampler: str) -> None:
    context = {
        "python": "/env/python",
        "output_path": "/output/video.mp4",
        "options": {
            "variant": "diamond-csgo",
            "pretrained_dir": "/assets",
            "spawn_id": 2,
            "action": "backward",
            "sampler": sampler,
            "headless_steps": 7,
            "frames": 99,
            "fps": 10,
            "seed": 19,
            "device": "cuda:1",
            "checkpoint_path": "/model.pt",
            "spawn_dir": "/spawn",
            "actions_path": "/actions.npy",
        },
    }

    assert runtime.build_command(context) == [
        "/env/python",
        "-m",
        "worldfoundry.synthesis.visual_generation.diamond.csgo_inference",
        "--pretrained-dir",
        "/assets",
        "--spawn-id",
        "2",
        "--action",
        "backward",
        "--sampler",
        sampler,
        "--headless-steps",
        "7",
        "--fps",
        "10",
        "--seed",
        "19",
        "--device",
        "cuda:1",
        "--output-path",
        "/output/video.mp4",
        "--checkpoint",
        "/model.pt",
        "--spawn-dir",
        "/spawn",
        "--actions-path",
        "/actions.npy",
    ]


def test_csgo_requirements_remain_variant_specific(tmp_path: Path) -> None:
    for filename in (
        "csgo/config/agent/csgo.yaml",
        "csgo/model/csgo.pt",
        "csgo/spawn/0/low_res.npy",
        "csgo/spawn/0/full_res.npy",
        "csgo/spawn/0/act.npy",
    ):
        path = tmp_path / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    options = {"variant": "csgo", "pretrained_dir": str(tmp_path)}
    kwargs = {"options": options, "runtime_root": None, "entrypoint": None, "profile": None}

    assert runtime.missing_requirements(**kwargs) == []
    options["actions_path"] = str(tmp_path / "missing.npy")
    missing = runtime.missing_requirements(**kwargs)
    assert len(missing) == 1
    assert missing[0]["path"] == options["actions_path"]
