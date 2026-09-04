"""EDM / rectified-flow x0-prediction training objective.

Cosmos-Predict families (cosmos1 EDM, cosmos2 rectified-flow) wrap their DiT in
preconditioning that makes the denoiser return a *clean* ``x0`` estimate rather
than a flow velocity.  Their training corruption is additive Karras noise
``x = x0 + sigma * noise`` with ``sigma`` sampled log-normally over a wide range
(not the ``[0, 1]`` flow-matching sigma), and the loss is a sigma-weighted MSE
between the predicted and true clean latents.

This objective is deliberately independent of :mod:`flow_matching`: it exposes a
distinct ``prediction_type`` so the model-neutral ``flow_velocity`` seam (and the
distillation/RL stack built on it) is never widened.  The single-device SFT
engine matches ``adapter.prediction_type == objective.prediction_type`` and is
otherwise objective-agnostic, so an x0 adapter + this objective train through
the existing engine untouched.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from math import isfinite, prod

from worldfoundry.training.api.contracts import ObjectiveBatch, PreparedBatch, TrainStepResult
from worldfoundry.training.objectives.flow_matching import flow_matching_mse

X0_PREDICTION_TYPE = "x0"


def _require_torch():
    try:
        import torch
    except ModuleNotFoundError as error:
        raise RuntimeError("x0 training requires the 'train-core' extra (PyTorch)") from error
    return torch


def _sigma_for_reference(sigmas: object, reference: object) -> object:
    reference_ndim = getattr(reference, "ndim", 0)
    return sigmas.reshape((int(sigmas.shape[0]),) + (1,) * (int(reference_ndim) - 1))


@dataclass(frozen=True, slots=True)
class X0DiffusionConfig:
    """Log-normal sigma sampling plus the preconditioning loss weight family."""

    weighting: str = "edm"  # "edm" (cosmos1) or "rectified_flow" (cosmos2)
    sigma_data: float = 0.5
    p_mean: float = -1.2
    p_std: float = 1.2
    sigma_min: float = 0.002
    sigma_max: float = 80.0

    def __post_init__(self) -> None:
        weighting = str(self.weighting).strip().lower().replace("-", "_")
        if weighting not in {"edm", "rectified_flow", "uniform"}:
            raise ValueError(f"unsupported x0 loss weighting: {weighting!r}")
        for name, value in (
            ("sigma_data", self.sigma_data),
            ("p_std", self.p_std),
            ("sigma_min", self.sigma_min),
            ("sigma_max", self.sigma_max),
        ):
            if not isfinite(float(value)) or float(value) <= 0:
                raise ValueError(f"x0 {name} must be finite and positive")
        if not isfinite(float(self.p_mean)):
            raise ValueError("x0 p_mean must be finite")
        if self.sigma_min >= self.sigma_max:
            raise ValueError("x0 sigma_min must be smaller than sigma_max")
        object.__setattr__(self, "weighting", weighting)
        object.__setattr__(self, "sigma_data", float(self.sigma_data))
        object.__setattr__(self, "p_mean", float(self.p_mean))
        object.__setattr__(self, "p_std", float(self.p_std))
        object.__setattr__(self, "sigma_min", float(self.sigma_min))
        object.__setattr__(self, "sigma_max", float(self.sigma_max))

    def loss_weight(self, sigma: object) -> object:
        if self.weighting == "uniform":
            return sigma.new_ones(sigma.shape)
        if self.weighting == "rectified_flow":
            # (sigma + 1)^2 / sigma^2 — the rectified-flow x0 weight.
            return (sigma + 1.0).square() / sigma.square()
        # EDM: (sigma^2 + sigma_data^2) / (sigma * sigma_data)^2.
        return (sigma.square() + self.sigma_data**2) / (sigma * self.sigma_data).square()


class X0DiffusionObjective:
    """Additive-noise x0 corruption and sigma-weighted MSE reduction."""

    prediction_type = X0_PREDICTION_TYPE

    def __init__(self, config: X0DiffusionConfig | None = None) -> None:
        self.config = config or X0DiffusionConfig()

    def _sample_sigmas(self, batch_size: int, *, device: object, generator: object | None):
        torch = _require_torch()
        if isinstance(batch_size, bool) or int(batch_size) <= 0:
            raise ValueError("batch_size must be a positive integer")
        logits = torch.randn(batch_size, device=device, dtype=torch.float32, generator=generator)
        sigma = (logits * self.config.p_std + self.config.p_mean).exp()
        return sigma.clamp(self.config.sigma_min, self.config.sigma_max)

    def corrupt(self, batch: PreparedBatch, *, generator: object | None = None) -> ObjectiveBatch:
        torch = _require_torch()
        clean_tree = batch.clean_latents
        first = next(iter(clean_tree.values())) if isinstance(clean_tree, Mapping) else clean_tree
        if not torch.is_tensor(first):
            raise TypeError("clean_latents must contain torch.Tensor values")
        sigmas = self._sample_sigmas(batch.batch_size, device=first.device, generator=generator)

        def corrupt_one(clean):
            if not torch.is_tensor(clean):
                raise TypeError("clean_latents must contain torch.Tensor values")
            noise = torch.randn(clean.shape, device=clean.device, dtype=clean.dtype, generator=generator)
            sigma = _sigma_for_reference(sigmas.to(device=clean.device, dtype=clean.dtype), clean)
            return clean + sigma * noise, clean, noise

        if isinstance(clean_tree, Mapping):
            triples = {key: corrupt_one(clean) for key, clean in clean_tree.items()}
            model_input = {key: value[0] for key, value in triples.items()}
            target = {key: value[1] for key, value in triples.items()}
            noise = {key: value[2] for key, value in triples.items()}
        else:
            model_input, target, noise = corrupt_one(clean_tree)

        metadata = dict(batch.metadata)
        metadata["objective"] = "x0_diffusion"
        metadata["prediction_type"] = self.prediction_type
        metadata["x0_weighting"] = self.config.weighting
        return ObjectiveBatch(
            sample_ids=batch.sample_ids,
            model_input=model_input,
            target=target,
            sigmas=sigmas,
            timesteps=sigmas,
            conditioning=batch.conditioning,
            noise=noise,
            loss_mask=batch.loss_mask,
            sample_weights=batch.sample_weights,
            metadata=metadata,
        )

    def _sigma_weighted_sample_weights(self, batch: ObjectiveBatch):
        torch = _require_torch()
        sigma = batch.sigmas.to(dtype=torch.float32).reshape(-1)
        weight = self.config.loss_weight(sigma)
        base = batch.sample_weights
        if base is None:
            return weight
        base = base.to(device=weight.device, dtype=torch.float32).reshape(-1)
        return weight * base

    def compute_loss(self, prediction: object, batch: ObjectiveBatch) -> TrainStepResult:
        torch = _require_torch()
        target_tree = batch.target
        weighted = self._sigma_weighted_sample_weights(batch)
        if isinstance(target_tree, Mapping):
            if not isinstance(prediction, Mapping) or set(prediction) != set(target_tree):
                raise ValueError("prediction keys must match target keys")
            mask_tree = batch.loss_mask
            results = {}
            for key, target in target_tree.items():
                mask = mask_tree.get(key) if isinstance(mask_tree, Mapping) else mask_tree
                results[key] = flow_matching_mse(
                    prediction[key], target, loss_mask=mask, sample_weights=weighted
                )
        else:
            if isinstance(prediction, Mapping):
                raise ValueError("a tensor target requires a tensor prediction")
            results = {
                "main": flow_matching_mse(
                    prediction, target_tree, loss_mask=batch.loss_mask, sample_weights=weighted
                )
            }

        numerator = sum(
            (result.numerator for result in results.values()),
            torch.zeros((), device=next(iter(results.values())).loss.device),
        )
        denominator = sum(
            (result.denominator for result in results.values()),
            torch.zeros((), device=next(iter(results.values())).loss.device),
        )
        loss = numerator / denominator
        component_losses = {f"x0/{key}": result.loss for key, result in results.items()}
        losses = {"x0": loss, **component_losses}
        target_values = target_tree.values() if isinstance(target_tree, Mapping) else (target_tree,)
        latent_tokens = sum(
            int(value.shape[0]) * prod(int(size) for size in value.shape[2:]) for value in target_values
        )
        return TrainStepResult(
            loss=loss,
            losses=losses,
            metrics={
                "loss_numerator": numerator.detach(),
                "loss_denominator": denominator.detach(),
                "sigma_mean": batch.sigmas.float().mean().detach(),
                "sigma_min": batch.sigmas.float().min().detach(),
                "sigma_max": batch.sigmas.float().max().detach(),
            },
            sample_count=batch.batch_size,
            latent_token_count=latent_tokens,
            diagnostics={
                "prediction_type": self.prediction_type,
                "weighting": self.config.weighting,
                "sigma_data": self.config.sigma_data,
            },
        )


__all__ = [
    "X0_PREDICTION_TYPE",
    "X0DiffusionConfig",
    "X0DiffusionObjective",
]
