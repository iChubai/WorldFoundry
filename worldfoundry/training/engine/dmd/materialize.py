"""Family-neutral native DMD run materialization.

This is the single body that constructs a three-role, two-optimizer DMD run:
open the immutable cache, load student/real-score/fake-score adapters, apply
tuning and FSDP2, build the model-neutral DMD stack, and wire exact-resume
checkpoint state, metrics, and export.  Every model-family-specific decision is
delegated to a :class:`DMDFamilyStrategy`; the Wan family provides one that
reproduces the historical ``materialize_wan_dmd_training_run`` behavior exactly.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path
from typing import Literal

import torch

from worldfoundry.core.io.integrity import append_jsonl_durable
from worldfoundry.core.time import utc_now_iso
from worldfoundry.training.checkpoint.checkpointer import TrainingCheckpointer
from worldfoundry.training.checkpoint.state import TrainingProgress, TrainingState
from worldfoundry.training.data.dataset import TrainingManifestDataset
from worldfoundry.training.distributed.fsdp import (
    FSDP2Application,
    apply_fsdp2,
    apply_fsdp2_frozen_reference,
)
from worldfoundry.training.distributed.parallel import (
    DistributedTrainingContext,
    ParallelPlan,
)
from worldfoundry.training.engine.artifacts import create_run_directory
from worldfoundry.training.engine.dmd.strategy import DMDFamilyStrategy, DMDTrainableRoles
from worldfoundry.training.post_training.distillation.dmd.builder import (
    build_native_dmd_training_stack,
)
from worldfoundry.training.post_training.distillation.dmd.session import (
    NativeDMDTrainingSession,
)
from worldfoundry.training.post_training.shared.distributed import (
    PostTrainingParallelContext,
)
from worldfoundry.training.post_training.shared.prediction import NativeFlowPredictionAdapter
from worldfoundry.training.post_training.shared.role_checkpoints import resolve_role_checkpoint
from worldfoundry.training.recipes.post_training.recipe import PostTrainingRecipe


def _resolve_paths(
    recipe: PostTrainingRecipe,
    *,
    base_dir: str | Path,
    output_dir: str | Path | None,
) -> tuple[Path, Path, Path]:
    root = Path(base_dir).expanduser().resolve()
    cache_path = Path(recipe.data.cache or "")
    manifest_path = Path(recipe.data.manifest)
    destination = Path(output_dir or recipe.run.output_dir)
    if not cache_path.is_absolute():
        cache_path = root / cache_path
    if not manifest_path.is_absolute():
        manifest_path = root / manifest_path
    if not destination.is_absolute():
        destination = root / destination
    return cache_path, manifest_path, destination.resolve()


def materialize_dmd_training_run(
    recipe: PostTrainingRecipe,
    *,
    strategy: DMDFamilyStrategy,
    base_dir: str | Path = ".",
    device: str | torch.device = "cuda",
    output_dir: str | Path | None = None,
    resume_checkpoint: str | Path | None = None,
    audited_role_overrides: Mapping[str, object] | None = None,
    verify_media_hashes: bool = True,
    audit_cache_on_open: bool = True,
    verify_cache_on_read: bool = True,
    force_torch_attention: bool = True,
    fused_adamw: bool | Literal["auto"] = "auto",
    initialization_seed: int | None = None,
) -> object:
    """Materialize an independently owned native three-role DMD run for any family."""

    if not isinstance(strategy, DMDFamilyStrategy):
        raise TypeError("strategy must implement DMDFamilyStrategy")
    algorithm = strategy.validate_recipe(recipe)
    cache_path, manifest_path, destination = _resolve_paths(
        recipe,
        base_dir=base_dir,
        output_dir=output_dir,
    )

    resolved_device = torch.device(device)
    if resolved_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    distributed_context: DistributedTrainingContext | None = None
    if recipe.distributed.backend == "fsdp2":
        if resolved_device.type != "cuda":
            raise ValueError("native DMD FSDP2 materialization requires CUDA")
        distributed_context = DistributedTrainingContext(device_type="cuda")
        resolved_device = distributed_context.device
    try:
        world_size = 1 if distributed_context is None else distributed_context.world_size
        rank = 0 if distributed_context is None else distributed_context.rank
        plan = ParallelPlan.resolve(recipe.distributed, world_size=world_size)
        create_run_directory(destination, distributed_context)
        if force_torch_attention:
            for name in (
                "WORLDFOUNDRY_ATTENTION_IMPLEMENTATION",
                "WORLDFOUNDRY_ATTENTION_BACKEND",
            ):
                configured = os.environ.get(name)
                if configured not in {None, "torch"}:
                    raise ValueError(f"correctness-first DMD requires {name}=torch; got {configured!r}")
                os.environ[name] = "torch"

        manifest = TrainingManifestDataset.from_file(
            manifest_path,
            split=recipe.data.split,
            verify_files=True,
            verify_hashes=verify_media_hashes,
        )
        cache = strategy.open_cache(
            cache_path=cache_path,
            manifest=manifest,
            audit_on_open=audit_cache_on_open,
            verify_on_read=verify_cache_on_read,
        )
        unconditional = strategy.read_shared_unconditional(cache_path)

        from worldfoundry.base_models.diffusion_model.assembly import (
            NativeDiffusionAssembler,
        )
        from worldfoundry.base_models.diffusion_model.loaders import CheckpointSpec
        from worldfoundry.base_models.diffusion_model.recipes.registry import (
            default_native_diffusion_registry,
        )

        native_recipe = default_native_diffusion_registry().resolve(recipe.model.recipe)
        default_dit = native_recipe.checkpoints["dit"]
        raw_overrides = dict(audited_role_overrides or {})
        unknown_overrides = sorted(set(raw_overrides) - {"student", "real-score", "fake-score"})
        if unknown_overrides:
            raise ValueError(f"unknown DMD role overrides: {unknown_overrides}")
        for name, value in raw_overrides.items():
            if not isinstance(value, CheckpointSpec):
                raise TypeError(f"audited role override {name!r} must be CheckpointSpec")
        student_checkpoint = resolve_role_checkpoint(
            role="student",
            reference=recipe.model.checkpoint,
            native_default=default_dit,
            audited_local_override=raw_overrides.get("student"),
        )
        real_checkpoint = resolve_role_checkpoint(
            role="real-score",
            reference=algorithm.real_score_checkpoint,
            native_default=default_dit,
            audited_local_override=raw_overrides.get("real-score"),
        )
        fake_checkpoint = resolve_role_checkpoint(
            role="fake-score",
            reference=algorithm.fake_score_checkpoint,
            native_default=default_dit,
            audited_local_override=raw_overrides.get("fake-score"),
        )
        assembler = NativeDiffusionAssembler()
        dtype = _torch_dtype(recipe.runtime.param_dtype)
        adapter_options = strategy.role_adapter_options(
            assembler=assembler,
            native_recipe=native_recipe,
            device=resolved_device,
            dtype=dtype,
            num_train_timesteps=algorithm.num_train_timesteps,
            gradient_checkpointing=recipe.runtime.activation_checkpoint == "full",
            force_torch_attention=force_torch_attention,
        )
        strategy.seed_initialization(initialization_seed)
        student = strategy.load_role_adapter(role="student", checkpoint=student_checkpoint, **adapter_options)
        strategy.validate_model_dtype(student, dtype)
        student_peft = strategy.apply_tuning(recipe, student)
        student_fsdp: FSDP2Application | None = None
        mesh = None
        if distributed_context is not None:
            mesh = plan.build_device_mesh(resolved_device.type)
            student_fsdp = apply_fsdp2(
                student,
                plan=plan,
                mesh=mesh,
                param_dtype=dtype,
                reduce_dtype=_torch_dtype(recipe.runtime.reduce_dtype),
            )

        real_score = strategy.load_role_adapter(role="real-score", checkpoint=real_checkpoint, **adapter_options)
        strategy.validate_model_dtype(real_score, dtype)
        real_score.trainable_module.requires_grad_(False)
        real_score.trainable_module.eval()
        real_score_fsdp: FSDP2Application | None = None
        if distributed_context is not None:
            assert mesh is not None
            real_score_fsdp = apply_fsdp2_frozen_reference(
                real_score,
                plan=plan,
                mesh=mesh,
                param_dtype=dtype,
                reduce_dtype=_torch_dtype(recipe.runtime.reduce_dtype),
            )

        fake_score = strategy.load_role_adapter(role="fake-score", checkpoint=fake_checkpoint, **adapter_options)
        strategy.validate_model_dtype(fake_score, dtype)
        fake_score_peft = strategy.apply_tuning(recipe, fake_score)
        fake_score_fsdp: FSDP2Application | None = None
        if distributed_context is not None:
            assert mesh is not None
            fake_score_fsdp = apply_fsdp2(
                fake_score,
                plan=plan,
                mesh=mesh,
                param_dtype=dtype,
                reduce_dtype=_torch_dtype(recipe.runtime.reduce_dtype),
            )

        expected_contract = strategy.validate_cache_contract(recipe, student, cache)
        if unconditional is not None:
            strategy.validate_shared_unconditional_cache(
                unconditional,
                cache,
                student,
                expected_contract=expected_contract,
            )
        source_loader, token_sampler = strategy.build_cache_loader(
            recipe=recipe,
            dataset=cache,
            rank=rank,
            world_size=world_size,
            default_pin_memory=resolved_device.type == "cuda",
        )
        shared_unconditional = None if unconditional is None else unconditional.tensors
        dmd_loader = strategy.build_dmd_loader(
            source_loader,
            student,
            shared_unconditional=shared_unconditional,
        )
        autocast_dtype = None if dtype is torch.float32 else dtype
        parallel_context = PostTrainingParallelContext.current()
        stack = build_native_dmd_training_stack(
            recipe,
            student=NativeFlowPredictionAdapter(
                student,
                autocast_dtype=autocast_dtype,
                checkpoint_identity=recipe.model.checkpoint,
            ),
            real_score=NativeFlowPredictionAdapter(
                real_score,
                autocast_dtype=autocast_dtype,
                checkpoint_identity=algorithm.real_score_checkpoint,
            ),
            fake_score=NativeFlowPredictionAdapter(
                fake_score,
                autocast_dtype=autocast_dtype,
                checkpoint_identity=algorithm.fake_score_checkpoint,
            ),
            parallel_context=parallel_context,
            fused_adamw=fused_adamw,
        )
        roles = strategy.make_role_bundle(
            student=student,
            real_score=real_score,
            fake_score=fake_score,
            student_checkpoint=student_checkpoint,
            real_score_checkpoint=real_checkpoint,
            fake_score_checkpoint=fake_checkpoint,
            student_peft=student_peft,
            fake_score_peft=fake_score_peft,
            student_fsdp=student_fsdp,
            real_score_fsdp=real_score_fsdp,
            fake_score_fsdp=fake_score_fsdp,
        )
        progress = TrainingProgress(optimizer_steps=stack.engine.global_step)
        generator = torch.Generator(device=resolved_device)
        generator.manual_seed((int(initialization_seed or recipe.data.shuffle_seed) + rank) % (2**63 - 1))
        data_identity = _data_identity(
            cache=cache,
            expected_contract=expected_contract,
            unconditional=unconditional,
            token_sampler=token_sampler,
            plan=plan,
        )
        identity = {
            "schema": f"worldfoundry-{strategy.family}-dmd-resume-identity",
            "recipe_digest": recipe.digest,
            "roles": roles.runtime_identity(),
            "data": data_identity,
            "runtime": recipe.to_dict()["runtime"],
            "distributed": recipe.to_dict()["distributed"],
            "tuning": recipe.to_dict()["tuning"],
            "initialization_seed": initialization_seed,
            "rank_seed_derivation": "base-seed-plus-rank",
        }
        checkpoint_model = DMDTrainableRoles(
            student.trainable_module,
            fake_score.trainable_module,
        )
        checkpoint_state = TrainingState(
            model=checkpoint_model,
            optimizer=(stack.student_optimizer, stack.fake_score_optimizer),
            engine=stack.engine,
            dataloader=dmd_loader,
            objective_generator=generator,
            progress=progress,
            identity=identity,
            ignore_frozen_parameters=recipe.tuning.mode == "lora",
            **stack.checkpoint_state_kwargs(),
        )
        checkpointer = TrainingCheckpointer(destination / "checkpoints")
        resume_artifact = None
        if resume_checkpoint is not None:
            resume_artifact = checkpointer.load(checkpoint_state, resume_checkpoint)

        def event_sink(event: Mapping[str, object]) -> None:
            if rank != 0:
                return
            append_jsonl_durable(
                destination / "metrics.jsonl",
                {
                    **dict(event),
                    "run_id": recipe.run.id,
                    "recipe_digest": recipe.digest,
                    "recorded_at": utc_now_iso(),
                },
                root=destination,
            )

        session = NativeDMDTrainingSession(
            stack.engine,
            dmd_loader,
            progress,
            checkpoint_state=checkpoint_state,
            checkpointer=checkpointer,
            save_every_steps=recipe.checkpoint.save_every_steps,
            asynchronous_checkpoints=recipe.checkpoint.async_save,
            event_sink=event_sink,
        )
        return strategy.make_training_run(
            recipe=recipe,
            session=session,
            checkpoint_state=checkpoint_state,
            checkpointer=checkpointer,
            roles=roles,
            output_dir=destination,
            data_identity=data_identity,
            resume_artifact=resume_artifact,
            distributed_context=distributed_context,
        )
    except Exception:
        if distributed_context is not None:
            distributed_context.close()
        raise


def _torch_dtype(value: str) -> torch.dtype:
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[value]


def _data_identity(
    *,
    cache: object,
    expected_contract: str,
    unconditional: object | None,
    token_sampler: object,
    plan: object,
) -> dict[str, object]:
    identity: dict[str, object] = {
        "cache_schema": cache.index.schema,
        "cache_index_sha256": cache.index_sha256,
        "cache_contract_sha256": expected_contract,
        "dataset_digest": cache.dataset_digest,
        "sample_count": len(cache),
        "latent_token_budget": token_sampler.max_latent_tokens,
        "token_sampler_config_sha256": token_sampler.config_digest,
        "parallel_plan": plan.to_dict(),
    }
    if unconditional is not None:
        identity["unconditional_identity_sha256"] = unconditional.artifact.identity_sha256
        identity["unconditional_object_sha256"] = unconditional.artifact.object_sha256
    return identity


__all__ = ["materialize_dmd_training_run"]
