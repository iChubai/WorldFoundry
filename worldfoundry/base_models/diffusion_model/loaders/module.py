"""Native PyTorch module loading shared by every diffusion family."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Mapping

import torch

from ..optimizations import OffloadMode, QuantizationMode, RuntimePolicy
from .checkpoints import CheckpointSpec
from .materialize import MaterializedCheckpoint, NativeCheckpointResolver

StateDictConverter = Callable[[Mapping[str, object]], Mapping[str, object]]
CheckpointConfigResolver = Callable[[MaterializedCheckpoint], Mapping[str, object]]
PostLoadHook = Callable[[torch.nn.Module], None]


@dataclass(frozen=True, slots=True)
class ModuleLoadSpec:
    """Construction and weight-mapping rules for one PyTorch component."""

    module_class: type[torch.nn.Module]
    config: Mapping[str, object] = field(default_factory=dict)
    config_resolver: CheckpointConfigResolver | None = None
    state_dict_converter: StateDictConverter | None = None
    vram_module_map: Mapping[type[torch.nn.Module], type[torch.nn.Module]] | None = None
    layer_container: str | None = None
    vram_limit_gib: float | None = None
    post_load_hook: PostLoadHook | None = None


class NativeModuleLoader:
    """Thin diffusion-facing adapter over WorldFoundry's shared model loader."""

    def load(
        self,
        spec: ModuleLoadSpec,
        checkpoint: CheckpointSpec,
        policy: RuntimePolicy,
    ) -> torch.nn.Module:
        materialized = NativeCheckpointResolver().materialize(checkpoint)
        sources = tuple(str(path) for path in materialized.paths)
        config = dict(spec.config)
        if spec.config_resolver is not None:
            resolved_config = dict(spec.config_resolver(materialized))
            overlap = sorted(set(config) & set(resolved_config))
            if overlap:
                raise ValueError(f"static and checkpoint-derived module config overlap: {overlap}")
            config.update(resolved_config)

        from worldfoundry.core.model_loading.optimize import AppliedOptimizations

        applied = AppliedOptimizations()
        quantize = policy.quantization.mode is not QuantizationMode.NONE
        if quantize and policy.offload.mode is OffloadMode.DISK:
            # Disk offload replaces weights with placeholders before this loader
            # returns; an in-place weight transform would see no real tensors.
            raise NotImplementedError("quantization is not supported together with disk offload")

        # Lazy imports keep the canonical package importable without optional
        # Transformers/VRAM dependencies until a real module is constructed.
        from worldfoundry.core.model_loading import (
            load_model,
            load_model_with_disk_offload,
        )

        source: str | list[str]
        source = sources[0] if len(sources) == 1 else list(sources)
        if policy.offload.mode is OffloadMode.DISK:
            if spec.post_load_hook is not None:
                raise ValueError("post-load mutations are not supported with disk offload")
            if spec.vram_module_map is None:
                raise ValueError("disk offload requires ModuleLoadSpec.vram_module_map")
            module = load_model_with_disk_offload(
                spec.module_class,
                source,
                config=config,
                torch_dtype=policy.dtype,
                device=policy.device,
                state_dict_converter=spec.state_dict_converter,
                module_map=dict(spec.vram_module_map),
            )
        else:
            vram_config = None
            module_map = None
            if spec.vram_module_map is not None and policy.offload.mode in {
                OffloadMode.COMPONENT,
                OffloadMode.BLOCK,
            }:
                module_map = dict(spec.vram_module_map)
                vram_config = {
                    "offload_dtype": policy.dtype,
                    "offload_device": policy.offload.target,
                    "onload_dtype": policy.dtype,
                    "onload_device": policy.device,
                    "preparing_dtype": policy.dtype,
                    "preparing_device": policy.device,
                    "computation_dtype": policy.dtype,
                    "computation_device": policy.device,
                }
            module = load_model(
                spec.module_class,
                source,
                config=config,
                torch_dtype=policy.dtype,
                device=policy.device,
                state_dict_converter=spec.state_dict_converter,
                module_map=module_map,
                vram_config=vram_config,
                vram_limit=spec.vram_limit_gib,
            )
            # Adapter fusion and other in-place mutations must run while the
            # checkpoint weights are still materialized. Layerwise CPU
            # offload replaces inactive parameters with zero-sized CUDA
            # placeholders, so applying a LoRA after installing its hooks
            # corrupts shape-based traversal and fails during the merge.
            if spec.post_load_hook is not None:
                spec.post_load_hook(module)
            # Approximate (LOSSY) self-attention: opt-in STA/VSA sparse kernels.
            # OFF by default and never auto-selected; the user picks the kind +
            # params via ``options["approximate_attention"]``. Wrapping the
            # self-attention processors is numerically lossy on the sparse steps,
            # so engaging it marks the audit quality_tier "approximate". The
            # processor falls back to exact attention when no kernel is present.
            # Installed BEFORE fusion: fusion skips blocks whose processor is
            # non-default, so an approximate block keeps its separate q/k/v and
            # its processor is not bypassed by the fused fast-path forward.
            approx_opt = policy.options.get("approximate_attention")
            if approx_opt:
                from ..optimizations.approximate_attention import (
                    install_approximate_attention,
                    parse_approximate_attention,
                )

                approx_config = parse_approximate_attention(approx_opt)
                approx_state = install_approximate_attention(module, approx_config)
                applied.record_approximate_attention(
                    requested=True,
                    kind=approx_config.kind,
                    wrapped_blocks=approx_state.wrapped_blocks,
                    effective_kernel=approx_state.effective_kernel,
                )
                try:
                    module._worldfoundry_approximate_attention = approx_state
                except (AttributeError, TypeError):
                    pass
            # Merged-QKV fusion runs before quantization so the fused
            # ``qkv`` projection (a single [dim, 3*dim] Linear) is what gets
            # quantized: it is a larger GEMM than the separate q/k/v layers and
            # therefore more likely to clear the FP8 work-size gate. Opt-in via
            # ``options["fuse_qkv"]``; it is a no-op on already-merged models and
            # skips blocks running an approximate (non-default) processor.
            fuse_requested = bool(policy.options.get("fuse_qkv", False))
            if fuse_requested:
                from ..optimizations.qkv_fusion import fuse_qkv_projections

                applied.record_fusion(requested=True, fused_blocks=fuse_qkv_projections(module))
            # Static cross-attention K/V cache: conditioning-derived K/V do not
            # change across denoise steps, so wrapping the cross-attention
            # processors lets a runner reuse them. Runner must call the attached
            # cache's invalidate() per request. Numerically identical (same
            # projection reused); opt-in via ``options["static_cross_kv"]``.
            if bool(policy.options.get("static_cross_kv", False)):
                from ..optimizations.static_cross_kv import install_static_cross_kv_cache

                cross_cache = install_static_cross_kv_cache(module)
                applied.record_cross_kv(requested=True, wrapped_blocks=cross_cache.wrapped_blocks)
                try:
                    module._worldfoundry_static_cross_kv = cross_cache
                except (AttributeError, TypeError):
                    pass
            # Quantization runs after the dense checkpoint is materialized (and
            # after any post-load fusion) but BEFORE offload hooks, per
            # inference_operator_optimization_plan.md §4.1: layerwise offload
            # swaps parameters for placeholders, so an in-place weight transform
            # must precede it. FP8 replaces eligible Linear layers in place and
            # keeps a dense fallback; unsupported modes are reported as skipped.
            if quantize:
                from worldfoundry.core.model_loading.optimize import apply_quantization_policy

                applied.record_quantization(apply_quantization_policy(module, policy.quantization))
            if policy.offload.mode is OffloadMode.BLOCK and module_map is None:
                from worldfoundry.core.vram import enable_layerwise_cpu_offload

                enable_layerwise_cpu_offload(
                    module,
                    layer_container=spec.layer_container,
                    device=policy.device,
                    pin_memory=policy.offload.pin_memory,
                )
            elif policy.offload.mode is OffloadMode.COMPONENT and module_map is None:
                raise ValueError("component offload requires ModuleLoadSpec.vram_module_map")

        if policy.compile:
            # Route through the persistent compile cache instead of a bare
            # torch.compile: this reuses the process-wide Inductor/Triton disk
            # cache (keyed by device/CUDA/Triton fingerprint) and an
            # instance-level variant cache, so a reload or a second module with
            # the same options does not recompile from scratch. Lazy import
            # keeps CPU-only / doc import paths free of the compile runtime.
            from worldfoundry.runtime.compile_cache import CompilePolicy, compile_module_cached

            module = compile_module_cached(
                module,
                policy=CompilePolicy(enabled=True),
                namespace="diffusion-modules",
                options=dict(policy.options) or None,
            )
            # compile_module_cached swallows tracing failures (strict=False) and
            # returns the original module. _orig_mod is the torch.compile wrapper
            # marker, so its presence is the ground truth of whether the island
            # actually compiled rather than silently falling back to eager.
            applied.record_compile(requested=True, compiled=hasattr(module, "_orig_mod"))
        else:
            applied.record_compile(requested=False)
        if not isinstance(module, torch.nn.Module):
            raise TypeError(f"module loader returned {type(module).__name__}, expected torch.nn.Module")
        # Attach the auditable optimization record so a caller can fold it into
        # a PerformanceManifest without changing this method's return type.
        try:
            module._worldfoundry_applied_optimizations = applied
        except (AttributeError, TypeError):
            pass
        return module


__all__ = [
    "CheckpointConfigResolver",
    "ModuleLoadSpec",
    "NativeModuleLoader",
    "PostLoadHook",
    "StateDictConverter",
]
