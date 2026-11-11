# SPDX-FileCopyrightText: 2026 OpenEnvision
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import OmegaConf

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
