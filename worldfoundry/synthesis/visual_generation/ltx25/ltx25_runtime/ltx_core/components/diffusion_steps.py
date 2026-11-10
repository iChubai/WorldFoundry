import torch

from worldfoundry.base_models.diffusion_model.models.representations.ltx.protocols import DiffusionStepProtocol
from worldfoundry.core.nn.transformer.dit import denoised_to_velocity


class EulerDiffusionStep(DiffusionStepProtocol):
    """
    First-order Euler method for diffusion sampling.
    Takes a single step from the current noise level (sigma) to the next by
    computing velocity from the denoised prediction and applying: sample + velocity * dt.
    """

    def step(
        self, sample: torch.Tensor, denoised_sample: torch.Tensor, sigmas: torch.Tensor, step_index: int, **_kwargs
    ) -> torch.Tensor:
        sigma = sigmas[step_index]
        sigma_next = sigmas[step_index + 1]
        dt = sigma_next - sigma
        velocity = denoised_to_velocity(sample, sigma, denoised_sample)

        return (sample.to(torch.float32) + velocity.to(torch.float32) * dt).to(sample.dtype)


class EulerAncestralDiffusionStep(DiffusionStepProtocol):
    """Ancestral (SDE) Euler step for rectified-flow models.
    Each step advances deterministically to an intermediate noise level
    ``sigma_down <= sigma_next`` and then renoises back up to ``sigma_next``,
    rescaling the signal component by ``alpha_next / alpha_down`` so the
    transition stays variance-preserving. ``eta`` interpolates between a plain
    Euler step (``eta=0``, ``sigma_down == sigma_next``, no noise added) and a
    fully ancestral step (``eta=1``).
    This is the rectified-flow parameterization (``alpha = 1 - sigma``), which
    is the one that applies to LTX-2. It deliberately does **not** reuse
    :func:`_get_ancestral_step`: that helper implements the DDIM/variance-
    exploding ancestral coefficients, which give a different ``sigma_down`` and
    a different amount of injected noise for the same ``eta``. The two agree
    only at ``eta=0``.
    """

    def __init__(self, eta: float = 1.0, s_noise: float = 1.0) -> None:
        self.eta = eta
        self.s_noise = s_noise

    def step(
        self,
        sample: torch.Tensor,
        denoised_sample: torch.Tensor,
        sigmas: torch.Tensor,
        step_index: int,
        noise: torch.Tensor | None = None,
        **_kwargs,
    ) -> torch.Tensor:
        """Advance one ancestral Euler step.
        Args:
            sample: Current noisy latent x_t.
            denoised_sample: Denoised prediction x_0.
            sigmas: Full sigma schedule tensor.
            step_index: Current step index.
            noise: Noise tensor for the renoise term. Required when ``eta > 0``;
                unused (and may be ``None``) when ``eta == 0``.
        Returns:
            Updated latent x_{t-1}, or ``denoised_sample`` when the next sigma is 0.
        """
        sigma = sigmas[step_index].to(torch.float32)
        sigma_next = sigmas[step_index + 1].to(torch.float32)
        if sigma_next == 0:
            return denoised_sample.to(sample.dtype)
        if self.eta > 0 and noise is None:
            raise ValueError("EulerAncestralDiffusionStep requires a noise tensor when eta > 0")

        x = sample.to(torch.float32)
        denoised = denoised_sample.to(torch.float32)

        downstep_ratio = 1.0 + (sigma_next / sigma - 1.0) * self.eta
        sigma_down = sigma_next * downstep_ratio

        # Euler step to sigma_down, expressed as an interpolation between x and x_0.
        sigma_down_ratio = sigma_down / sigma
        x_next = sigma_down_ratio * x + (1.0 - sigma_down_ratio) * denoised

        if self.eta > 0:
            # Renoise from sigma_down back up to sigma_next.
            alpha_next = 1.0 - sigma_next
            alpha_down = 1.0 - sigma_down
            renoise_coeff = (sigma_next**2 - sigma_down**2 * alpha_next**2 / alpha_down**2).clamp(min=0) ** 0.5
            x_next = (alpha_next / alpha_down) * x_next + noise.to(torch.float32) * self.s_noise * renoise_coeff
        return x_next.to(sample.dtype)
