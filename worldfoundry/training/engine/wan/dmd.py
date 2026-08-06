"""Native Wan DMD family strategy and run entry point.

The DMD run lifecycle is family-neutral and lives in
:mod:`worldfoundry.training.engine.dmd.materialize`.  This module provides the
Wan-specific strategy (recipe validation, role loading, cache contract, and the
Wan run/bundle containers) and keeps the historical
``materialize_wan_dmd_training_run`` entry point intact.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Literal

import torch

from worldfoundry.training.data.dataset import TrainingManifestDataset
from worldfoundry.training.data.shared_conditioning import (
    SharedConditioningSample,
    SharedConditioningStore,
)
from worldfoundry.training.data.video_cache import VideoCachedDataset
from worldfoundry.training.engine.dmd.materialize import materialize_dmd_training_run
from worldfoundry.training.engine.dmd.strategy import DMDTrainableRoles
from worldfoundry.training.models.wan import WanTrainAdapter
from worldfoundry.training.post_training.distillation.dmd.batching import NativeDMDDataLoader
from worldfoundry.training.recipes.post_training.algorithms.dmd import DMDAlgorithmSpec
from worldfoundry.training.recipes.post_training.recipe import PostTrainingRecipe

from .cache import (
    audit_wan_cache_against_manifest,
    build_wan_cache_loader,
    validate_wan_cache_contract,
)
from .dmd_run import WAN_DMD_RUN_SCHEMA, WanDMDTrainingRun
from .roles import (
    WanDMDRoleBundle,
    apply_wan_tuning,
    load_wan_role_adapter,
    seed_initialization,
    torch_dtype,
    validate_model_dtype,
)

WAN_DMD_SUPPORTED_RECIPES = frozenset({"wan2.1-t2v-1.3b"})


def _validate_recipe(recipe: PostTrainingRecipe) -> DMDAlgorithmSpec:
    if not isinstance(recipe, PostTrainingRecipe):
        raise TypeError("recipe must be PostTrainingRecipe")
    if not isinstance(recipe.algorithm, DMDAlgorithmSpec):
        raise TypeError("Wan DMD materialization requires algorithm.type='dmd'")
    if recipe.model.recipe not in WAN_DMD_SUPPORTED_RECIPES:
        raise ValueError("native Wan DMD currently requires wan2.1-t2v-1.3b")
    if recipe.data.cache is None:
        raise ValueError("Wan DMD requires an immutable data.cache")
    if recipe.data.max_latent_tokens_per_microbatch is None:
        raise ValueError("Wan DMD requires data.max_latent_tokens_per_microbatch")
    if recipe.data.tail_policy not in {"drop", "pad"}:
        raise ValueError("Wan DMD token batching requires data.tail_policy='drop' or 'pad'")
    if recipe.distributed.backend not in {"single", "fsdp2"}:
        raise ValueError("Wan DMD currently supports single or FSDP2 execution")
    if recipe.runtime.activation_checkpoint not in {"none", "full"}:
        raise ValueError("Wan DMD activation_checkpoint must be 'none' or 'full'")
    if recipe.tuning.mode == "partial":
        raise ValueError("partial Wan DMD tuning needs an explicit parameter policy")
    return recipe.algorithm


def _validate_unconditional_cache(
    sample: SharedConditioningSample,
    dataset: VideoCachedDataset,
    adapter: WanTrainAdapter,
    *,
    expected_contract: str,
) -> None:
    identity = sample.artifact.identity
    if identity.branch != "unconditional":
        raise ValueError("Wan DMD shared conditioning must use the unconditional branch")
    if identity.model_recipe_digest != expected_contract:
        raise ValueError("Wan DMD unconditional context belongs to another model contract")
    asset_identities = {
        (entry.provenance.conditioner_digest, entry.provenance.tokenizer_digest) for entry in dataset.index.entries
    }
    if asset_identities != {(identity.conditioner_digest, identity.tokenizer_digest)}:
        raise ValueError("Wan DMD unconditional context encoder identity differs from sample contexts")
    if set(sample.tensors) != {"context"}:
        raise ValueError("Wan DMD unconditional conditioning must contain only context")
    descriptor = identity.tensors["context"]
    expected_shape = (
        adapter.expected_text_length,
        adapter.expected_context_features,
    )
    if descriptor.shape != expected_shape or descriptor.layout != "sequence-features":
        raise ValueError("Wan DMD unconditional context tensor contract is incompatible")


class WanDMDStrategy:
    """Wan-specific hooks consumed by the shared DMD materializer."""

    family = "wan"

    def validate_recipe(self, recipe: object) -> DMDAlgorithmSpec:
        assert isinstance(recipe, PostTrainingRecipe)
        return _validate_recipe(recipe)

    def role_adapter_options(
        self,
        *,
        assembler: object,
        native_recipe: object,
        device: object,
        dtype: object,
        num_train_timesteps: int,
        gradient_checkpointing: bool,
        force_torch_attention: bool,
    ) -> dict[str, object]:
        return {
            "assembler": assembler,
            "native_recipe": native_recipe,
            "device": device,
            "dtype": dtype,
            "num_train_timesteps": num_train_timesteps,
            "gradient_checkpointing": gradient_checkpointing,
            "force_torch_attention": force_torch_attention,
        }

    def load_role_adapter(self, *, role: str, checkpoint: object, **options: object) -> WanTrainAdapter:
        del role  # Wan loads every role through the same materializer.
        return load_wan_role_adapter(checkpoint=checkpoint, **options)

    def validate_model_dtype(self, adapter: object, expected_dtype: object) -> None:
        assert isinstance(adapter, WanTrainAdapter)
        assert isinstance(expected_dtype, torch.dtype)
        validate_model_dtype(adapter, expected_dtype)

    def apply_tuning(self, recipe: object, adapter: object) -> object | None:
        assert isinstance(recipe, PostTrainingRecipe)
        assert isinstance(adapter, WanTrainAdapter)
        return apply_wan_tuning(recipe, adapter)

    def seed_initialization(self, seed: int | None) -> None:
        seed_initialization(seed)

    def open_cache(
        self,
        *,
        cache_path: object,
        manifest: object,
        audit_on_open: bool,
        verify_on_read: bool,
    ) -> VideoCachedDataset:
        assert isinstance(manifest, TrainingManifestDataset)
        cache = VideoCachedDataset(
            cache_path,
            expected_dataset_digest=manifest.dataset_digest,
            audit_on_open=audit_on_open,
            verify_on_read=verify_on_read,
        )
        audit_wan_cache_against_manifest(cache, manifest)
        return cache

    def read_shared_unconditional(self, cache_path: object) -> SharedConditioningSample:
        return SharedConditioningStore(cache_path).read("unconditional")

    def validate_cache_contract(self, recipe: object, adapter: object, dataset: object) -> str:
        assert isinstance(recipe, PostTrainingRecipe)
        assert isinstance(adapter, WanTrainAdapter)
        assert isinstance(dataset, VideoCachedDataset)
        return validate_wan_cache_contract(recipe, adapter, dataset)

    def validate_shared_unconditional_cache(
        self,
        sample: object,
        dataset: object,
        adapter: object,
        *,
        expected_contract: str,
    ) -> None:
        assert isinstance(sample, SharedConditioningSample)
        assert isinstance(dataset, VideoCachedDataset)
        assert isinstance(adapter, WanTrainAdapter)
        _validate_unconditional_cache(sample, dataset, adapter, expected_contract=expected_contract)

    def build_cache_loader(
        self,
        *,
        recipe: object,
        dataset: object,
        rank: int,
        world_size: int,
        default_pin_memory: bool,
    ) -> tuple[object, object]:
        assert isinstance(recipe, PostTrainingRecipe)
        assert isinstance(dataset, VideoCachedDataset)
        return build_wan_cache_loader(
            recipe=recipe,
            dataset=dataset,
            rank=rank,
            world_size=world_size,
            default_pin_memory=default_pin_memory,
        )

    def build_dmd_loader(
        self,
        source_loader: object,
        adapter: object,
        *,
        shared_unconditional: object,
    ) -> NativeDMDDataLoader:
        assert isinstance(adapter, WanTrainAdapter)
        return NativeDMDDataLoader(
            source_loader,
            adapter,
            shared_unconditional_conditioning=shared_unconditional,
        )

    def make_role_bundle(self, **roles: object) -> WanDMDRoleBundle:
        return WanDMDRoleBundle(**roles)

    def make_training_run(self, **kwargs: object) -> WanDMDTrainingRun:
        return WanDMDTrainingRun(**kwargs)


def materialize_wan_dmd_training_run(
    recipe: PostTrainingRecipe,
    *,
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
) -> WanDMDTrainingRun:
    """Materialize an independently owned native Wan three-role DMD run."""

    run = materialize_dmd_training_run(
        recipe,
        strategy=WanDMDStrategy(),
        base_dir=base_dir,
        device=device,
        output_dir=output_dir,
        resume_checkpoint=resume_checkpoint,
        audited_role_overrides=audited_role_overrides,
        verify_media_hashes=verify_media_hashes,
        audit_cache_on_open=audit_cache_on_open,
        verify_cache_on_read=verify_cache_on_read,
        force_torch_attention=force_torch_attention,
        fused_adamw=fused_adamw,
        initialization_seed=initialization_seed,
    )
    assert isinstance(run, WanDMDTrainingRun)
    return run


__all__ = [
    "DMDTrainableRoles",
    "WAN_DMD_RUN_SCHEMA",
    "WanDMDRoleBundle",
    "WanDMDStrategy",
    "WanDMDTrainingRun",
    "materialize_wan_dmd_training_run",
]
