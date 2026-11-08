from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch

from worldfoundry.base_models.diffusion_model.components import (
    ComponentBuildContext,
    ComponentKey,
    ComponentKind,
)
from worldfoundry.base_models.diffusion_model.contracts import (
    Conditioning,
    DenoiserInput,
    DiffusionOutput,
    DiffusionRequest,
    SamplingConfig,
)
from worldfoundry.base_models.diffusion_model.extensions import DiffusionExtension
from worldfoundry.base_models.diffusion_model.loaders import CheckpointSpec
from worldfoundry.base_models.diffusion_model.models.denoisers import wan as wan_denoisers
from worldfoundry.base_models.diffusion_model.models.denoisers.wan import (
    SelfForcingWanDenoiser,
    build_self_forcing_wan21_denoiser,
    convert_self_forcing_generator_state_dict,
)
from worldfoundry.base_models.diffusion_model.models.initializers.wan.component import (
    WanTextToVideoLatentInitializer,
)
from worldfoundry.base_models.diffusion_model.models.networks.wan.variants.forcing.self_forcing import (
    CausalWanModel,
)
from worldfoundry.base_models.diffusion_model.optimizations import RuntimePolicy
from worldfoundry.base_models.diffusion_model.recipes.registry import default_native_diffusion_registry
from worldfoundry.base_models.diffusion_model.recipes.wan import (
    SELF_FORCING_MODEL_ID,
    SELF_FORCING_REPO_ID,
    SELF_FORCING_REVISION,
    WAN21_T2V_1P3B_REPO_ID,
    self_forcing_wan21_t2v_1p3b_recipe,
)
from worldfoundry.base_models.diffusion_model.runners.autoregressive import AutoregressiveWindowRunner
from worldfoundry.base_models.diffusion_model.runners.strategies import (
    ExecutionBuildContext,
    build_autoregressive_window_strategy,
)
from worldfoundry.base_models.diffusion_model.schedulers.wan import WanFlowMatchEulerScheduler
from worldfoundry.core.nn import FlowMatchScheduler
from worldfoundry.pipelines.forcing import NativeSelfForcingPipeline

TINY_CONFIG = {
    "model_type": "t2v",
    "patch_size": (1, 2, 2),
    "text_len": 8,
    "in_dim": 16,
    "dim": 24,
    "ffn_dim": 32,
    "freq_dim": 8,
    "text_dim": 16,
    "out_dim": 16,
    "num_heads": 2,
    "num_layers": 2,
    "local_attn_size": -1,
    "sink_size": 0,
    "qk_norm": True,
    "cross_attn_norm": True,
    "eps": 1e-6,
}


def _official_training_scheduler() -> FlowMatchScheduler:
    """The scheduler built by the official ``WanDiffusionWrapper(timestep_shift=5.0)``."""

    scheduler = FlowMatchScheduler(shift=5.0, sigma_min=0.0, extra_one_step=True)
    scheduler.set_timesteps(1000, training=True)
    return scheduler


def _tiny_model(seed: int = 0) -> CausalWanModel:
    torch.manual_seed(seed)
    model = CausalWanModel(**TINY_CONFIG).eval()
    # The released head is zero-initialised; random weights make x0 depend on the flow.
    torch.nn.init.normal_(model.head.head.weight, std=0.02)
    return model


def test_self_forcing_converter_prefers_ema_and_strips_wrapper_prefixes() -> None:
    weight = torch.ones(1)
    ema = torch.zeros(1)
    converted = convert_self_forcing_generator_state_dict(
        {
            "generator": {"model._fsdp_wrapped_module.blocks.0.ffn.0.weight": weight},
            "generator_ema": {"model.blocks.0.ffn.0.weight": ema},
        }
    )
    assert converted == {"blocks.0.ffn.0.weight": ema}

    fsdp = convert_self_forcing_generator_state_dict(
        {
            "generator": {
                "model._fsdp_wrapped_module.patch_embedding.weight": weight,
                "_fsdp_wrapped_module.model.blocks.1._fsdp_wrapped_module.self_attn.q.bias": weight,
            }
        }
    )
    assert set(fsdp) == {"patch_embedding.weight", "blocks.1.self_attn.q.bias"}

    with pytest.raises(KeyError, match="generator_ema"):
        convert_self_forcing_generator_state_dict({"critic": {}})


@pytest.mark.parametrize("role", ["generator_ema", "generator"])
def test_self_forcing_builder_loads_both_layouts_strictly(
    role: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reference = _tiny_model(seed=3)
    prefix = "model." if role == "generator_ema" else "model._fsdp_wrapped_module."
    checkpoint = tmp_path / "self_forcing_dmd.pt"
    torch.save({role: {prefix + name: value for name, value in reference.state_dict().items()}}, checkpoint)
    monkeypatch.setattr(wan_denoisers, "SELF_FORCING_WAN21_T2V_1P3B_CONFIG", TINY_CONFIG)

    denoiser = build_self_forcing_wan21_denoiser(
        ComponentBuildContext(
            model_id=SELF_FORCING_MODEL_ID,
            key=ComponentKey(ComponentKind.DENOISER),
            policy=RuntimePolicy(device="cpu", dtype=torch.float32),
            checkpoints={"weights": CheckpointSpec(source=checkpoint)},
            component_options={"block_size": 3, "cache_frames": 6},
        )
    )

    assert isinstance(denoiser.model, CausalWanModel)
    assert denoiser.model.num_frame_per_block == 3
    loaded = denoiser.model.state_dict()
    assert loaded.keys() == reference.state_dict().keys()
    for name, value in reference.state_dict().items():
        torch.testing.assert_close(loaded[name], value, rtol=0, atol=0)

    extra = tmp_path / "unexpected.pt"
    state = {prefix + name: value for name, value in reference.state_dict().items()}
    state[prefix + "blocks.0.unused.weight"] = torch.zeros(1)
    torch.save({role: state}, extra)
    with pytest.raises(RuntimeError, match="Unexpected key"):
        build_self_forcing_wan21_denoiser(
            ComponentBuildContext(
                model_id=SELF_FORCING_MODEL_ID,
                key=ComponentKey(ComponentKind.DENOISER),
                policy=RuntimePolicy(device="cpu", dtype=torch.float32),
                checkpoints={"weights": CheckpointSpec(source=extra)},
            )
        )


def test_self_forcing_training_calendar_matches_official_scheduler() -> None:
    official = _official_training_scheduler()
    denoiser = SelfForcingWanDenoiser(torch.nn.Identity(), compute_dtype=torch.float32)
    torch.testing.assert_close(denoiser.training_schedule.sigmas, official.sigmas, rtol=0, atol=0)
    torch.testing.assert_close(denoiser.training_schedule.timesteps, official.timesteps, rtol=0, atol=0)


@pytest.mark.parametrize("compute_dtype", [torch.float32, torch.bfloat16])
def test_self_forcing_x0_matches_official_wrapper_arithmetic(compute_dtype: torch.dtype) -> None:
    flow = torch.randn(1, 16, 3, 4, 4, generator=torch.Generator().manual_seed(1))

    class FlowNetwork(torch.nn.Module):
        num_layers = 1
        num_heads = 1
        dim = 2

        def forward(self, x: torch.Tensor, **_: Any) -> torch.Tensor:
            return flow.to(dtype=x.dtype)

    denoiser = SelfForcingWanDenoiser(FlowNetwork(), cache_frames=3, compute_dtype=compute_dtype)
    official = _official_training_scheduler()
    timestep = torch.cat((official.timesteps, torch.tensor([0.0])))[1000 - 750]
    latents = torch.randn(1, 16, 3, 4, 4, generator=torch.Generator().manual_seed(2))
    cache = denoiser.create_kv_cache(
        batch_size=1,
        n_views=1,
        latent_frames_per_view=3,
        frame_sequence_length=4,
        device=torch.device("cpu"),
        dtype=compute_dtype,
    )
    output = denoiser(
        DenoiserInput(
            latents=latents,
            timestep=timestep,
            next_timestep=torch.tensor(0.0),
            conditioning={"context": torch.zeros(1, 1, 2), "kv_cache": cache, "current_start": 0},
            step_index=1,
            total_steps=4,
        )
    )

    # WanDiffusionWrapper._convert_flow_pred_to_x0 on [B*F, C, H, W] tensors.
    flow_btchw = flow.to(compute_dtype).transpose(1, 2).flatten(0, 1)
    noisy_btchw = latents.to(compute_dtype).transpose(1, 2).flatten(0, 1)
    timestep_id = torch.argmin(
        (official.timesteps.double().unsqueeze(0) - timestep.expand(3).unsqueeze(1)).abs(), dim=1
    )
    sigma_t = official.sigmas.double()[timestep_id].reshape(-1, 1, 1, 1)
    expected = (noisy_btchw.double() - sigma_t * flow_btchw.double()).to(compute_dtype)
    expected = expected.unflatten(0, (1, 3)).transpose(1, 2).to(torch.float32)
    assert output.sample.dtype == torch.float32
    torch.testing.assert_close(output.sample, expected, rtol=0, atol=0)


def test_self_forcing_cache_matches_official_global_attention_layout() -> None:
    model = CausalWanModel(**{**TINY_CONFIG, "num_layers": 3})
    denoiser = SelfForcingWanDenoiser(model, compute_dtype=torch.bfloat16)
    cache = denoiser.create_kv_cache(
        batch_size=1,
        n_views=1,
        latent_frames_per_view=21,
        frame_sequence_length=1560,
        device=torch.device("cpu"),
        dtype=torch.bfloat16,
    )
    assert len(cache["kv_cache"]) == 3
    assert cache["kv_cache"][0]["k"].shape == (1, 32760, 2, 12)
    assert cache["kv_cache"][0]["k"].dtype == torch.bfloat16
    assert [layer["is_init"] for layer in cache["crossattn_cache"]] == [False, False, False]
    with pytest.raises(ValueError, match="at most 21 latent frames"):
        denoiser.create_kv_cache(
            batch_size=1,
            n_views=1,
            latent_frames_per_view=24,
            frame_sequence_length=1560,
            device=torch.device("cpu"),
            dtype=torch.bfloat16,
        )


def test_self_forcing_recipe_registry_and_checkpoints() -> None:
    recipe = self_forcing_wan21_t2v_1p3b_recipe()
    registry = default_native_diffusion_registry()
    for name in ("self-forcing", "self_forcing", "gdhe17/Self-Forcing", "self-forcing-dmd"):
        assert registry.resolve(name).model_id == SELF_FORCING_MODEL_ID

    assert recipe.execution.strategy == "autoregressive-window"
    assert dict(recipe.execution.options) == {
        "prediction_mode": "distilled-x0",
        "fixed_timesteps": (1000, 750, 500, 250),
        "context_timestep": 0,
    }
    components = {component.key: component for component in recipe.components}
    denoiser = components[recipe.execution.bindings["denoiser"]]
    assert denoiser.factory is build_self_forcing_wan21_denoiser
    assert dict(denoiser.options) == {"block_size": 3, "cache_frames": 21, "timestep_shift": 5.0}
    assert components[recipe.execution.bindings["scheduler"]].options["shift"] == 5.0
    assert "latent_encoder" not in recipe.execution.bindings

    dit = recipe.checkpoints["dit"]
    assert (dit.repo_id, dit.revision, dit.files) == (
        SELF_FORCING_REPO_ID,
        SELF_FORCING_REVISION,
        ("checkpoints/self_forcing_dmd.pt",),
    )
    assert recipe.checkpoints["text-encoder"].files == ("models_t5_umt5-xxl-enc-bf16.pth",)
    assert recipe.checkpoints["vae"].files == ("Wan2.1_VAE.pth",)
    assert {recipe.checkpoints[name].repo_id for name in ("text-encoder", "tokenizer", "vae")} == {
        WAN21_T2V_1P3B_REPO_ID
    }
    assert recipe.options["default_num_frames"] == 81
    assert (recipe.options["default_height"], recipe.options["default_width"]) == (480, 832)
    assert recipe.options["default_guidance_scale"] == 1.0
    assert recipe.options["default_fps"] == 16


class _Conditioner:
    def encode(self, request: DiffusionRequest, **_: object) -> Conditioning:
        del request
        context = torch.randn(1, 8, 16, generator=torch.Generator().manual_seed(5))
        return Conditioning(positive={"context": context})


class _Decoder:
    def decode(self, latents: torch.Tensor, request: DiffusionRequest) -> torch.Tensor:
        del request
        return latents


class _CallRecorder(DiffusionExtension):
    extension_id = "self-forcing-call-recorder"

    def __init__(self) -> None:
        self.calls: list[tuple[str, float, int]] = []
        self.caches: list[dict[str, Any]] = []

    def before_denoiser(self, context, model_input: DenoiserInput) -> DenoiserInput:
        del context
        current_start = model_input.conditioning["current_start"]
        cache = model_input.conditioning["kv_cache"]
        assert isinstance(current_start, int)
        assert isinstance(cache, dict)
        self.calls.append((model_input.branch, float(model_input.timestep), current_start))
        self.caches.append(cache)
        return model_input


def test_self_forcing_window_runner_rolls_out_two_blocks_end_to_end() -> None:
    recipe = self_forcing_wan21_t2v_1p3b_recipe()
    denoiser = SelfForcingWanDenoiser(_tiny_model(), block_size=3, cache_frames=6, compute_dtype=torch.float32)
    by_role = {
        "denoiser": denoiser,
        "conditioner": _Conditioner(),
        "latent_initializer": WanTextToVideoLatentInitializer(),
        "scheduler": WanFlowMatchEulerScheduler(shift=5.0),
        "decoder": _Decoder(),
    }
    recorder = _CallRecorder()
    runner = build_autoregressive_window_strategy(
        ExecutionBuildContext(
            recipe=recipe,
            components={recipe.execution.bindings[role]: value for role, value in by_role.items()},
            policy=RuntimePolicy(device="cpu", dtype=torch.float32),
            extensions=(recorder,),
        )
    )
    assert isinstance(runner, AutoregressiveWindowRunner)

    request = DiffusionRequest(
        prompt="a red fox running through snow",
        height=32,
        width=32,
        num_frames=21,
        sampling=SamplingConfig(
            num_inference_steps=4,
            guidance_scale=1.0,
            seed=42,
            scheduler_options={"shift": 5.0},
        ),
    )
    output = runner.run(request)

    assert output.latents.shape == (1, 16, 6, 4, 4)
    assert output.sample.shape == output.latents.shape
    assert bool(torch.isfinite(output.latents).all())
    assert output.metadata["prediction_mode"] == "distilled-x0"
    assert output.metadata["block_size"] == 3

    official = _official_training_scheduler()
    official_steps = torch.cat((official.timesteps, torch.tensor([0.0])))[
        1000 - torch.tensor([1000, 750, 500, 250])
    ].tolist()
    frame_tokens = 4 * 4 // 4
    expected_calls = []
    for block_start in (0, 3):
        current_start = block_start * frame_tokens
        expected_calls.extend(("positive", step, current_start) for step in official_steps)
        expected_calls.append(("positive-cache-commit", 0.0, current_start))
    assert recorder.calls == expected_calls

    cache = recorder.caches[0]
    assert all(value is cache for value in recorder.caches)
    assert int(cache["kv_cache"][0]["global_end_index"]) == 6 * frame_tokens
    assert all(layer["is_init"] for layer in cache["crossattn_cache"])

    rerun = runner.run(request)
    torch.testing.assert_close(rerun.latents, output.latents, rtol=0, atol=0)


def test_native_self_forcing_pipeline_forwards_release_defaults() -> None:
    requests: list[DiffusionRequest] = []

    class _NativePipeline:
        model_id = SELF_FORCING_MODEL_ID

        def __call__(self, request: DiffusionRequest) -> DiffusionOutput:
            requests.append(request)
            return DiffusionOutput(sample=torch.zeros(1, 3, 81, 8, 8), latents=torch.zeros(1, 16, 21, 1, 1))

    pipeline = NativeSelfForcingPipeline(native_pipeline=_NativePipeline(), device="cpu")
    result = pipeline(prompt="a red fox running through snow", seed=42, return_dict=True)

    request = requests[0]
    assert (request.height, request.width, request.num_frames) == (480, 832, 81)
    assert request.sampling.num_inference_steps == 4
    assert request.sampling.guidance_scale == 1.0
    assert request.sampling.seed == 42
    assert dict(request.sampling.scheduler_options) == {"shift": 5.0}
    assert request.inputs["fps"] == 16
    assert result["generation_type"] == "t2v"
    assert NativeSelfForcingPipeline.PRIMARY_CHECKPOINT_ROLE == "dit"
    assert NativeSelfForcingPipeline._checkpoint_overrides(None, {"checkpoint_path": "/ckpt/self_forcing_dmd.pt"}) == {
        "dit": "/ckpt/self_forcing_dmd.pt"
    }
