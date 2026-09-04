"""Strategy seam that lets the shared DMD body drive any model family.

A family strategy owns exactly the model-specific parts of DMD materialization:
recipe validation, role adapter loading, tuning policy, cache contract audit,
data loading, and the family's role-bundle/run container types.  Everything
else — three-role construction, optimizer engine, DCP-backed exact resume,
metrics, and export — lives once in :mod:`worldfoundry.training.engine.dmd.materialize`.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from torch import nn


class DMDTrainableRoles(nn.Module):
    """One DCP model tree containing only the mutable DMD roles.

    DMD checkpoints exactly the student and fake-score parameters; the frozen
    real-score teacher is reloaded from its own audited checkpoint on resume.
    This container is family-neutral: it only requires two distinct modules.
    """

    def __init__(self, student: nn.Module, fake_score: nn.Module) -> None:
        super().__init__()
        if not isinstance(student, nn.Module) or not isinstance(fake_score, nn.Module):
            raise TypeError("DMD trainable roles must be nn.Module values")
        if student is fake_score:
            raise ValueError("DMD student and fake-score modules must be distinct")
        self.student = student
        self.fake_score = fake_score


@runtime_checkable
class DMDFamilyStrategy(Protocol):
    """Model-family-specific hooks consumed by the shared DMD materializer."""

    family: str

    def validate_recipe(self, recipe: object) -> object:
        """Validate the recipe for this family and return its DMDAlgorithmSpec."""

    def load_role_adapter(self, *, role: str, checkpoint: object, **options: object) -> object:
        """Build one independently checkpointed denoiser role adapter."""

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
        """Return the shared keyword options passed to every ``load_role_adapter`` call."""

    def validate_model_dtype(self, adapter: object, expected_dtype: object) -> None:
        """Reject silently mixed base-model precision before tuning or sharding."""

    def apply_tuning(self, recipe: object, adapter: object) -> object | None:
        """Apply the full-parameter or LoRA policy to one mutable role."""

    def seed_initialization(self, seed: int | None) -> None:
        """Seed role construction deterministically on CPU and every CUDA device."""

    def validate_cache_contract(self, recipe: object, adapter: object, dataset: object) -> str:
        """Audit tensor geometry/preprocessing/encoder identity; return the contract digest."""

    def validate_shared_unconditional_cache(
        self,
        sample: object,
        dataset: object,
        adapter: object,
        *,
        expected_contract: str,
    ) -> None:
        """Validate the shared unconditional conditioning entry, if the family uses one."""

    def open_cache(self, *, cache_path: object, manifest: object, audit_on_open: bool, verify_on_read: bool) -> object:
        """Open and audit the immutable feature cache against its manifest."""

    def read_shared_unconditional(self, cache_path: object) -> object | None:
        """Read the shared unconditional conditioning sample, or ``None`` if unused."""

    def build_cache_loader(
        self,
        *,
        recipe: object,
        dataset: object,
        rank: int,
        world_size: int,
        default_pin_memory: bool,
    ) -> tuple[object, object]:
        """Build the family's stateful loader and latent-token sampler."""

    def build_dmd_loader(self, source_loader: object, adapter: object, *, shared_unconditional: object) -> object:
        """Wrap the source loader into the family's DMD data loader."""

    def make_role_bundle(self, **roles: object) -> object:
        """Construct the family's frozen role bundle (identities for the manifest)."""

    def make_training_run(self, **kwargs: object) -> object:
        """Construct the family's training-run container."""


__all__ = ["DMDFamilyStrategy", "DMDTrainableRoles"]
