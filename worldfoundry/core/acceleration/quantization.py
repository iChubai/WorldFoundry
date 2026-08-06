"""In-tree low-precision linear layers with portable dense fallback."""

from __future__ import annotations

import os

import torch
import torch.nn.functional as F
from torch import nn

from worldfoundry.core.kernels.capabilities import kernel_device_profile

_FP8_DTYPES = (torch.float8_e4m3fn, torch.float8_e5m2)


def _quantize_tensorwise_fp8(
    value: torch.Tensor,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    if dtype not in _FP8_DTYPES:
        raise ValueError(f"unsupported FP8 dtype: {dtype}")
    fp8_max = float(torch.finfo(dtype).max)
    scale = (value.detach().float().abs().amax() / fp8_max).clamp_min(torch.finfo(torch.float32).tiny)
    quantized = (value.float() / scale).clamp(-fp8_max, fp8_max).to(dtype)
    return quantized, scale.reshape(1).float()


def _quantize_rowwise_fp8(
    value: torch.Tensor,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-row (last-axis) symmetric FP8 quantization.

    Returns row-major ``[rows, K]`` codes and an FP32 ``[rows, 1]`` decode
    scale. Used for per-token activation and per-output-channel weight scaling,
    which tracks per-row dynamic range better than one tensor-wide scale.
    """

    if dtype not in _FP8_DTYPES:
        raise ValueError(f"unsupported FP8 dtype: {dtype}")
    fp8_max = float(torch.finfo(dtype).max)
    amax = value.detach().float().abs().amax(dim=-1, keepdim=True)
    scale = (amax / fp8_max).clamp_min(torch.finfo(torch.float32).tiny)
    quantized = (value.float() / scale).clamp(-fp8_max, fp8_max).to(dtype)
    return quantized.contiguous(), scale.float()


def _quantize_rowwise_fp8_fused(
    value: torch.Tensor,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Row-wise FP8 quantization using the fused Triton kernel when available.

    The pure-PyTorch path makes several HBM passes and, for large ``K``, costs
    more than the FP8 GEMM it feeds. The Triton kernel does it in one pass and
    is numerically within +/-1 FP8 ULP of the reference. Falls back to the
    reference on any Triton import/compile/launch problem so the layer stays
    correct on GPUs or runtimes without a working Triton.
    """

    if value.is_cuda:
        try:
            from worldfoundry.core.acceleration.triton_fp8 import quantize_rowwise_fp8_triton

            return quantize_rowwise_fp8_triton(value, dtype)
        except Exception:  # noqa: BLE001 - any Triton failure must degrade safely
            pass
    return _quantize_rowwise_fp8(value, dtype)


def _scaled_mm_supports_rowwise() -> bool:
    """Whether ``torch._scaled_mm`` accepts per-row (``[M,1]``/``[1,N]``) scales.

    Row-wise scaled-MM landed in PyTorch 2.5. On 2.4 and earlier ``_scaled_mm``
    only accepts scalar (tensor-wise) scales, so callers must fall back to the
    tensor-wise path. Probed once and cached.
    """

    cached = getattr(_scaled_mm_supports_rowwise, "_cached", None)
    if cached is not None:
        return cached
    supported = False
    if callable(getattr(torch, "_scaled_mm", None)) and torch.cuda.is_available():
        try:
            dtype = torch.float8_e4m3fn
            a = torch.zeros(16, 16, device="cuda", dtype=dtype)
            b = torch.zeros(16, 16, device="cuda", dtype=dtype).t()
            scale_a = torch.ones(16, 1, device="cuda", dtype=torch.float32)
            scale_b = torch.ones(1, 16, device="cuda", dtype=torch.float32)
            torch._scaled_mm(a, b, scale_a=scale_a, scale_b=scale_b, out_dtype=torch.bfloat16)
            supported = True
        except (RuntimeError, TypeError):
            supported = False
    _scaled_mm_supports_rowwise._cached = supported
    return supported


def _scaled_mm_modern() -> bool:
    """Whether ``torch._scaled_mm`` uses the PyTorch >= 2.5 tensor-wise API.

    PyTorch 2.4 exposed a different, incompatible ``_scaled_mm`` (keyword-only
    scales, ``(out, amax)`` tuple return). WorldFoundry's FP8 fast path targets
    the 2.5+ single-tensor API; on 2.4 the layer must use its dense fallback
    instead of crashing. Probed once and cached.
    """

    cached = getattr(_scaled_mm_modern, "_cached", None)
    if cached is not None:
        return cached
    supported = False
    if callable(getattr(torch, "_scaled_mm", None)) and torch.cuda.is_available():
        try:
            dtype = torch.float8_e4m3fn
            a = torch.zeros(16, 16, device="cuda", dtype=dtype)
            b = torch.zeros(16, 16, device="cuda", dtype=dtype).t()
            scale = torch.ones(1, device="cuda", dtype=torch.float32)
            out = torch._scaled_mm(a, b, scale_a=scale, scale_b=scale, out_dtype=torch.bfloat16)
            supported = isinstance(out, torch.Tensor)
        except (RuntimeError, TypeError):
            supported = False
    _scaled_mm_modern._cached = supported
    return supported


def _fp8_hardware_eligible(device: torch.device) -> bool:
    if device.type != "cuda":
        return False
    profile = kernel_device_profile(device)
    return profile.supports_fp8 and _scaled_mm_modern()


# Dynamic per-token quantization only pays off once the GEMM is large enough to
# amortize the quant pass and cuBLASLt setup. H100 microbenchmarks (row-wise FP8
# vs BF16, quant + GEMM) put the crossover around M*K*N ~= 1.2e10: below it BF16
# wins (small GEMMs regress to 0.2-0.7x), above it FP8 wins (1.3-1.7x). Gating on
# the work size keeps small projections/short sequences on the faster dense path.
# Override with WORLDFOUNDRY_FP8_MIN_GEMM_FLOP (M*K*N units; 0 disables the gate).
_FP8_MIN_GEMM_WORK = 1.2e10


def _fp8_min_gemm_work() -> float:
    # Read the env override once and cache it: this runs on every eligible
    # forward, and a per-call getenv measurably taxes small GEMMs.
    cached = getattr(_fp8_min_gemm_work, "_cached", None)
    if cached is not None:
        return cached
    raw = os.getenv("WORLDFOUNDRY_FP8_MIN_GEMM_FLOP")
    value = _FP8_MIN_GEMM_WORK
    if raw is not None:
        try:
            value = max(float(raw), 0.0)
        except ValueError:
            value = _FP8_MIN_GEMM_WORK
    _fp8_min_gemm_work._cached = value
    return value


def _fp8_linear_eligible(
    input: torch.Tensor,
    out_features: int,
    hardware_eligible: bool,
) -> bool:
    if input.device.type != "cuda" or torch.is_grad_enabled():
        return False
    if input.dtype not in {torch.float16, torch.bfloat16}:
        return False
    in_features = input.shape[-1]
    if in_features % 16 or out_features % 16:
        return False
    # Skip FP8 for GEMMs too small to beat BF16 (launch-bound / low-arithmetic).
    threshold = _fp8_min_gemm_work()
    if threshold > 0.0:
        tokens = input.numel() // in_features if in_features else 0
        if float(tokens) * in_features * out_features < threshold:
            return False
    return hardware_eligible


class Float8Linear(nn.Module):
    """Inference-only dynamic-FP8 linear with an exact dense fallback.

    Two scaling modes share one storage format (row-major ``[out, in]`` FP8
    codes viewed as the column-major right GEMM operand):

    - ``"rowwise"``: per-token activation scale ``[M,1]`` and per-output-channel
      weight scale ``[N,1]``. Tracks per-row dynamic range, so activation
      outliers in one token do not inflate the scale of every other row. This
      is the preferred path on Hopper and requires ``torch._scaled_mm`` row-wise
      support (PyTorch >= 2.5) plus ``use_fast_accum`` for competitive speed.
    - ``"tensorwise"``: one scalar scale per operand. The only option on
      PyTorch <= 2.4, kept as an automatic fallback.

    ``scaling="auto"`` (default) picks row-wise when the runtime supports it and
    otherwise tensor-wise. A dense weight copy is retained by default so the
    same checkpoint stays runnable on V100/T4/A100, ROCm, CPU, and any workload
    that is not FP8-eligible (fp32 input, misaligned features, autograd).
    """

    def __init__(
        self,
        weight: torch.Tensor,
        bias: torch.Tensor | None = None,
        *,
        fp8_dtype: torch.dtype = torch.float8_e4m3fn,
        keep_dense_fallback: bool = True,
        scaling: str = "auto",
    ) -> None:
        super().__init__()
        if weight.ndim != 2:
            raise ValueError("weight must have [out_features, in_features] shape")
        if fp8_dtype not in _FP8_DTYPES:
            raise ValueError(f"unsupported FP8 dtype: {fp8_dtype}")
        if scaling not in {"auto", "rowwise", "tensorwise"}:
            raise ValueError(f"unsupported scaling mode: {scaling!r}")
        rowwise = scaling == "rowwise" or (scaling == "auto" and _scaled_mm_supports_rowwise())
        if rowwise:
            quantized, scale = _quantize_rowwise_fp8(weight.detach(), fp8_dtype)
        else:
            quantized, scale = _quantize_tensorwise_fp8(weight.detach(), fp8_dtype)
        self.in_features = int(weight.shape[1])
        self.out_features = int(weight.shape[0])
        self.fp8_dtype = fp8_dtype
        self.scaling = "rowwise" if rowwise else "tensorwise"
        self.low_precision_enabled = True
        self.register_buffer("weight_fp8", quantized.contiguous())
        self.register_buffer("weight_scale", scale.to(device=weight.device))
        dense = weight.detach().clone() if keep_dense_fallback else None
        self.register_buffer("weight", dense)
        self.register_buffer("bias", None if bias is None else bias.detach().clone())
        self._hardware_eligible = _fp8_hardware_eligible(weight.device)

    @classmethod
    def from_linear(
        cls,
        layer: nn.Linear,
        *,
        fp8_dtype: torch.dtype = torch.float8_e4m3fn,
        keep_dense_fallback: bool = True,
        scaling: str = "auto",
    ) -> "Float8Linear":
        return cls(
            layer.weight,
            layer.bias,
            fp8_dtype=fp8_dtype,
            keep_dense_fallback=keep_dense_fallback,
            scaling=scaling,
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if input.shape[-1] != self.in_features:
            raise ValueError(f"expected input width {self.in_features}, got {input.shape[-1]}")
        if input.numel() == 0:
            return input.new_empty((*input.shape[:-1], self.out_features))
        if not self.low_precision_enabled or not _fp8_linear_eligible(
            input,
            self.out_features,
            self._hardware_eligible,
        ):
            if self.weight is None:
                raise RuntimeError("FP8 is unavailable for this workload and no dense fallback was retained")
            bias = None if self.bias is None else self.bias.to(dtype=input.dtype)
            return F.linear(input, self.weight.to(dtype=input.dtype), bias)

        original_shape = input.shape
        flattened = input.reshape(-1, self.in_features).contiguous()
        bias = None if self.bias is None else self.bias.to(dtype=input.dtype)
        # A contiguous [N, K] weight transposes to the column-major [K, N]
        # right operand for torch._scaled_mm/cuBLASLt without another copy.
        weight_mat = self.weight_fp8.t()
        if self.scaling == "rowwise":
            input_fp8, input_scale = _quantize_rowwise_fp8_fused(flattened, self.fp8_dtype)
            # scale_a is per-token [M,1]; scale_b is per-output-channel [1,N].
            # use_fast_accum is required for row-wise scaled-MM to beat BF16 on
            # Hopper; FP8 quantization error dominates the accumulation error,
            # so fast accumulation does not measurably change parity here.
            output = torch._scaled_mm(
                input_fp8,
                weight_mat,
                scale_a=input_scale,
                scale_b=self.weight_scale.reshape(1, -1),
                bias=bias,
                out_dtype=input.dtype,
                use_fast_accum=True,
            )
        else:
            input_fp8, input_scale = _quantize_tensorwise_fp8(flattened, self.fp8_dtype)
            output = torch._scaled_mm(
                input_fp8,
                weight_mat,
                scale_a=input_scale,
                scale_b=self.weight_scale,
                bias=bias,
                out_dtype=input.dtype,
                use_fast_accum=False,
            )
        return output.reshape(*original_shape[:-1], self.out_features)

    def _apply(self, fn, recurse: bool = True):
        # ``Module.to(dtype=...)`` applies its dtype conversion to every
        # floating buffer.  FP8 payloads and their FP32 decode scale are
        # storage-format metadata, not model-compute tensors, so preserve the
        # original values/dtypes while still following the requested device.
        weight_fp8 = self.weight_fp8
        weight_scale = self.weight_scale
        result = super()._apply(fn, recurse=recurse)
        target_device = self.weight_fp8.device
        self.weight_fp8 = weight_fp8.to(device=target_device)
        self.weight_scale = weight_scale.to(device=target_device)
        self._hardware_eligible = _fp8_hardware_eligible(self.weight_fp8.device)
        return result

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bias={self.bias is not None}, fp8_dtype={self.fp8_dtype}, "
            f"scaling={self.scaling}, dense_fallback={self.weight is not None}"
        )


def replace_linear_with_float8(
    module: nn.Module,
    *,
    min_features: int = 1024,
    fp8_dtype: torch.dtype = torch.float8_e4m3fn,
    keep_dense_fallback: bool = True,
    exclude: tuple[str, ...] = (),
    scaling: str = "auto",
) -> int:
    """Replace eligible child ``nn.Linear`` modules in-place.

    Apply it after loading the dense checkpoint and placing its modules.  This
    explicit inference transform turns quantized weights into buffers, avoids
    hidden runtime monkey-patching, and returns the number of replacements.
    ``scaling`` selects the FP8 scaling mode (see :class:`Float8Linear`).
    """

    replaced = 0
    for name, child in tuple(module.named_children()):
        if any(pattern and pattern in name for pattern in exclude):
            continue
        if isinstance(child, nn.Linear):
            if (
                child.in_features >= min_features
                and child.out_features >= min_features
                and child.in_features % 16 == 0
                and child.out_features % 16 == 0
            ):
                setattr(
                    module,
                    name,
                    Float8Linear.from_linear(
                        child,
                        fp8_dtype=fp8_dtype,
                        keep_dense_fallback=keep_dense_fallback,
                        scaling=scaling,
                    ),
                )
                replaced += 1
            continue
        replaced += replace_linear_with_float8(
            child,
            min_features=min_features,
            fp8_dtype=fp8_dtype,
            keep_dense_fallback=keep_dense_fallback,
            exclude=exclude,
            scaling=scaling,
        )
    return replaced


def set_low_precision_enabled(module: nn.Module, enabled: bool) -> int:
    """Toggle in-tree FP8/NVFP4 modules for dense boundary denoising steps."""

    updated = 0
    for child in module.modules():
        if hasattr(child, "low_precision_enabled"):
            child.low_precision_enabled = bool(enabled)
            updated += 1
    return updated


__all__ = ["Float8Linear", "replace_linear_with_float8", "set_low_precision_enabled"]
