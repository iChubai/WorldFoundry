"""Layerwise CPU offload with asynchronous CUDA prefetch.

Each transformer block keeps a CPU master copy; a pre-forward hook copies
the next layer onto GPU while the current layer runs. That overlaps H2D
with compute, unlike :class:`~worldfoundry.core.vram.memory.DynamicSwapInstaller`
which moves on attribute access (no prefetch).

:class:`LayerwiseOffloadHandle` retains hook handles so :meth:`disable`
can restore parameters (CC-27). The mutation scope blocks LoRA / optimizer
writes from racing a swap. Do not stack this with TP/CP on the same
module — those expect resident shards.
"""

from __future__ import annotations

from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, field
from typing import Any, Iterator

import torch
from torch import nn

_compiler_disable = getattr(getattr(torch, "compiler", None), "disable", lambda fn: fn)

# ──────────────────────────────────────────────────────────────────────────
# Public handle — retain RemovableHandle objects so disable() can undo offload
# ──────────────────────────────────────────────────────────────────────────


@dataclass
class LayerwiseOffloadHandle:
    """Handle returned by ``enable_layerwise_cpu_offload``.

    Holds the registered hook handles so the offload can be undone with
    :meth:`disable` (CC-27); previously the ``RemovableHandle`` objects were
    dropped and a model could never leave offload mode without a rebuild.
    """

    enabled: bool
    layer_count: int
    reason: str = ""
    _model: nn.Module | None = field(default=None, repr=False, compare=False)
    _layer_hooks: list = field(default_factory=list, repr=False, compare=False)
    _metrics: _LayerwiseOffloadMetrics | None = field(
        default=None,
        repr=False,
        compare=False,
    )

    def reset_request_window(self) -> None:
        """Reset request-local counters without discarding a prefetched layer."""

        if self._metrics is not None:
            self._metrics.reset_request_window()

    def report(self) -> dict[str, Any]:
        """Return auditable lifetime and request-local execution evidence.

        ``enabled`` only proves that hooks were installed.  ``effective`` is
        true after a forward actually used the two-layer CUDA window with
        asynchronous H2D copies and no emergency synchronous materialization.
        Pageable host tensors are surfaced explicitly because
        ``non_blocking=True`` alone does not prove copy/compute overlap.
        """

        metrics = self._metrics
        if metrics is None:
            return {
                "requested": self.enabled,
                "enabled": self.enabled,
                "effective": False,
                "mode": "disabled",
                "layer_count": self.layer_count,
                "reason": self.reason or "no layerwise offload metrics",
            }
        return metrics.report(
            enabled=self.enabled,
            layer_count=self.layer_count,
            reason=self.reason,
        )

    def disable(self) -> bool:
        """Remove offload hooks and restore real parameters onto each layer.

        Parameters are restored from the CPU master copies (on CPU); move the
        model to the desired device afterwards. Returns True when hooks were
        removed. Handles created for an already-enabled model
        (``reason="already enabled"``) do not own hooks and return False.
        """

        if not self.enabled or self._model is None or not self._layer_hooks:
            self.enabled = False
            return False
        if self._metrics is not None:
            # A primed/prefetched tensor may still be owned by the copy stream.
            # Synchronize that stream before dropping its final Python
            # reference so allocator reuse cannot race the in-flight H2D.
            self._metrics.copy_stream.synchronize()
        for layer, state, hook_handles in self._layer_hooks:
            for hook_handle in hook_handles:
                hook_handle.remove()
            for name, param in layer.named_parameters(recurse=True):
                cpu_tensor = state.cpu_named_parameters.get(name)
                if cpu_tensor is not None:
                    param.data = cpu_tensor
            state.gpu_named_parameters.clear()
            state.cpu_named_parameters.clear()
            state.next_state = None
            for attribute in ("_worldfoundry_layerwise_cpu_offload", "_worldfoundry_layerwise_cpu_offload_state"):
                layer.__dict__.pop(attribute, None)
        self._model.__dict__.pop("_worldfoundry_layerwise_cpu_offload", None)
        self._layer_hooks.clear()
        self._model = None
        self._metrics = None
        self.enabled = False
        return True


@dataclass
class _LayerwiseOffloadMetrics:
    """Shared counters for one two-layer CUDA prefetch window."""

    copy_stream: torch.cuda.Stream = field(repr=False, compare=False)
    pin_memory_requested: bool = True
    primed: bool = False
    active_layers: set[int] = field(default_factory=set, repr=False)
    lifetime: dict[str, int] = field(default_factory=dict, repr=False)
    request: dict[str, int] = field(default_factory=dict, repr=False)

    _COUNTERS = (
        "forward_calls",
        "prefetch_calls",
        "async_copy_tensors",
        "async_copy_bytes",
        "synchronous_copy_tensors",
        "synchronous_copy_bytes",
        "wait_calls",
        "release_calls",
        "pinned_cpu_tensors",
        "pageable_cpu_tensors",
        "peak_active_layers",
    )

    def __post_init__(self) -> None:
        self.lifetime = {name: 0 for name in self._COUNTERS}
        self.request = {name: 0 for name in self._COUNTERS}

    def count(self, name: str, amount: int = 1, *, request: bool = True) -> None:
        self.lifetime[name] = int(self.lifetime.get(name, 0)) + int(amount)
        if request:
            self.request[name] = int(self.request.get(name, 0)) + int(amount)

    def set_layer_active(self, state_id: int, active: bool) -> None:
        if active:
            self.active_layers.add(state_id)
            active_count = len(self.active_layers)
            self.lifetime["peak_active_layers"] = max(
                self.lifetime["peak_active_layers"],
                active_count,
            )
            self.request["peak_active_layers"] = max(
                self.request["peak_active_layers"],
                active_count,
            )
        else:
            self.active_layers.discard(state_id)

    def reset_request_window(self) -> None:
        self.request = {name: 0 for name in self._COUNTERS}
        # One layer is deliberately kept warm across denoising calls.  The
        # request peak starts at the physical state rather than pretending
        # the cache is empty.
        self.request["peak_active_layers"] = len(self.active_layers)

    def report(
        self,
        *,
        enabled: bool,
        layer_count: int,
        reason: str,
    ) -> dict[str, Any]:
        request = dict(self.request)
        used = request["forward_calls"] > 0
        async_h2d = request["async_copy_tensors"] > 0
        no_sync_h2d = request["synchronous_copy_tensors"] == 0
        pinned = request["pageable_cpu_tensors"] == 0
        bounded_window = request["peak_active_layers"] <= min(2, layer_count)
        effective = bool(
            enabled
            and used
            and async_h2d
            and no_sync_h2d
            and pinned
            and bounded_window
        )
        issues: list[str] = []
        if used and not async_h2d:
            issues.append("no asynchronous H2D copy executed in the request window")
        if request["synchronous_copy_tensors"]:
            issues.append("emergency synchronous H2D materialization executed")
        if request["pageable_cpu_tensors"]:
            issues.append("one or more CPU masters were not pinned")
        if not bounded_window:
            issues.append("more than two transformer layers were CUDA-resident")
        return {
            "requested": True,
            "enabled": bool(enabled),
            "effective": effective,
            "mode": "async-double-buffer",
            "layer_count": int(layer_count),
            "pin_memory_requested": bool(self.pin_memory_requested),
            "primed": bool(self.primed),
            "active_layers": len(self.active_layers),
            "request": request,
            "lifetime": dict(self.lifetime),
            "issues": issues,
            "reason": reason or None,
        }


# ──────────────────────────────────────────────────────────────────────────
# Enable / mutation scope — opt-in; disabled handle on non-CUDA or no ModuleList
# ──────────────────────────────────────────────────────────────────────────


def enable_layerwise_cpu_offload(
    model: nn.Module,
    *,
    layer_container: str | None = None,
    device: torch.device | str | None = None,
    pin_memory: bool = True,
) -> LayerwiseOffloadHandle:
    """Attach layerwise CPU offload hooks to the first or named ``ModuleList``.

    The helper keeps parameters on CPU and moves one layer at a time to CUDA.
    The next layer is prefetched on a separate CUDA stream while the current
    layer runs. It is intentionally opt-in and returns a disabled handle on
    non-CUDA systems. Keep the returned handle if you need to switch the
    model back to full-device residency later via ``handle.disable()``.
    """

    if not torch.cuda.is_available():
        return LayerwiseOffloadHandle(enabled=False, layer_count=0, reason="CUDA is not available")
    if getattr(model, "_worldfoundry_layerwise_cpu_offload", False):
        return LayerwiseOffloadHandle(enabled=True, layer_count=0, reason="already enabled")

    target_device = torch.device(device or f"cuda:{torch.cuda.current_device()}")
    if target_device.type != "cuda":
        return LayerwiseOffloadHandle(enabled=False, layer_count=0, reason=f"target device is {target_device.type!r}")

    layers = _find_layer_container(model, layer_container)
    if layers is None or len(layers) == 0:
        return LayerwiseOffloadHandle(enabled=False, layer_count=0, reason="no nn.ModuleList layer container found")

    stream = torch.cuda.Stream(device=target_device)
    metrics = _LayerwiseOffloadMetrics(
        copy_stream=stream,
        pin_memory_requested=pin_memory,
    )
    states = [
        _LayerwiseOffloadState(
            layer,
            target_device,
            stream,
            metrics=metrics,
            state_id=index,
            pin_memory=pin_memory,
        )
        for index, layer in enumerate(layers)
    ]
    layer_hooks: list[tuple[nn.Module, _LayerwiseOffloadState, list]] = []
    for index, state in enumerate(states):
        # Wrap around so the last layer prefetches the first for the next step.
        state.next_state = states[(index + 1) % len(states)]
        state.offload_to_cpu()
        layer = layers[index]
        if getattr(layer, "_worldfoundry_layerwise_cpu_offload", False):
            continue
        pre_handle = layer.register_forward_pre_hook(_make_pre_hook(state), with_kwargs=True)
        post_handle = layer.register_forward_hook(
            _make_post_hook(state),
            with_kwargs=True,
            always_call=True,
        )
        layer_hooks.append((layer, state, [pre_handle, post_handle]))
        setattr(layer, "_worldfoundry_layerwise_cpu_offload", True)
        setattr(layer, "_worldfoundry_layerwise_cpu_offload_state", state)
    setattr(model, "_worldfoundry_layerwise_cpu_offload", True)
    # Match the upstream double-buffer lifecycle: the first CUDA buffer is
    # populated before layer 0 is entered, so the first timed forward does not
    # fall back to a synchronous H2D copy.  The pre-hook waits on this stream,
    # then starts layer 1 while layer 0 computes.
    states[0].prefetch(request_counter=False)
    metrics.primed = True
    return LayerwiseOffloadHandle(
        enabled=True,
        layer_count=len(states),
        _model=model,
        _layer_hooks=layer_hooks,
        _metrics=metrics,
    )


def layerwise_offload_mutation_scope(module: nn.Module) -> Iterator[None]:
    """Temporarily materialize offloaded parameters for in-place mutations."""

    state = getattr(module, "_worldfoundry_layerwise_cpu_offload_state", None)
    if state is None:
        return nullcontext()
    return state.mutate_params_scope()


# ──────────────────────────────────────────────────────────────────────────
# Per-layer CPU master + GPU cache — prefetch overlaps H2D with compute
# ──────────────────────────────────────────────────────────────────────────


class _LayerwiseOffloadState:
    """CPU master copy, GPU cache, and next-layer prefetch for one block.

    Placeholders replace ``param.data`` so Parameter identity (optimizer /
    hook targets) stays stable while bytes live on CPU. Do not stack with
    TP/CP: those expect resident shards, not ping-pong storage.
    """

    def __init__(
        self,
        module: nn.Module,
        device: torch.device,
        async_copy_stream: torch.cuda.Stream,
        *,
        metrics: _LayerwiseOffloadMetrics,
        state_id: int,
        pin_memory: bool,
    ) -> None:
        """Bind one module to a compute device and a shared prefetch stream."""
        self.module = module
        self.device = device
        self.async_copy_stream = async_copy_stream
        self.metrics = metrics
        self.state_id = int(state_id)
        self.pin_memory = pin_memory
        self.cpu_named_parameters: dict[str, torch.Tensor] = {}
        self.gpu_named_parameters: dict[str, torch.Tensor] = {}
        self.next_state: _LayerwiseOffloadState | None = None

    @_compiler_disable
    def offload_to_cpu(self) -> None:
        """Snapshot weights to CPU (optionally pinned) and drop GPU storage.

        ``pin_memory`` can raise on some allocators; the copy still lives on
        pageable CPU RAM so offload continues rather than aborting setup.
        """
        for name, param in self.module.named_parameters(recurse=True):
            cpu_tensor = param.detach().to("cpu")
            if self.pin_memory:
                try:
                    cpu_tensor = cpu_tensor.pin_memory()
                except RuntimeError:
                    pass
            counter = (
                "pinned_cpu_tensors"
                if cpu_tensor.is_pinned()
                else "pageable_cpu_tensors"
            )
            # Setup inventory is lifetime evidence.  Request counters are
            # incremented when a copy actually consumes the CPU master.
            self.metrics.count(counter, request=False)
            self.cpu_named_parameters[name] = cpu_tensor
            param.data = _tensor_placeholder(param.data, self.device)

    @_compiler_disable
    def wait_and_materialize(self) -> None:
        """Block the compute stream on prefetch, then bind GPU copies to params."""
        torch.cuda.current_stream(self.device).wait_stream(self.async_copy_stream)
        self.metrics.count("wait_calls")
        named_parameters = dict(self.module.named_parameters(recurse=True))
        for name, param in named_parameters.items():
            if name not in self.cpu_named_parameters:
                continue
            if name not in self.gpu_named_parameters:
                cpu_tensor = self.cpu_named_parameters[name]
                gpu_tensor = cpu_tensor.to(self.device, non_blocking=False)
                self.gpu_named_parameters[name] = gpu_tensor
                self.metrics.count("synchronous_copy_tensors")
                self.metrics.count(
                    "synchronous_copy_bytes",
                    cpu_tensor.numel() * cpu_tensor.element_size(),
                )
                self.metrics.count(
                    "pinned_cpu_tensors"
                    if cpu_tensor.is_pinned()
                    else "pageable_cpu_tensors"
                )
            param.data = self.gpu_named_parameters[name]
        if self.gpu_named_parameters:
            self.metrics.set_layer_active(self.state_id, True)

    @_compiler_disable
    def prefetch(self, *, request_counter: bool = True) -> None:
        """Copy missing GPU weights on the async stream; skip already-resident names.

        ``record_stream`` lets the compute stream reuse the copy after
        :meth:`wait_and_materialize` without a full device sync.
        """
        self.metrics.count("prefetch_calls", request=request_counter)
        compute_stream = torch.cuda.current_stream(self.device)
        copied = False
        with torch.cuda.stream(self.async_copy_stream):
            for name, cpu_tensor in self.cpu_named_parameters.items():
                if name in self.gpu_named_parameters:
                    continue
                gpu_tensor = cpu_tensor.to(self.device, non_blocking=True)
                gpu_tensor.record_stream(compute_stream)
                self.gpu_named_parameters[name] = gpu_tensor
                copied = True
                self.metrics.count(
                    "async_copy_tensors",
                    request=request_counter,
                )
                self.metrics.count(
                    "async_copy_bytes",
                    cpu_tensor.numel() * cpu_tensor.element_size(),
                    request=request_counter,
                )
                self.metrics.count(
                    "pinned_cpu_tensors"
                    if cpu_tensor.is_pinned()
                    else "pageable_cpu_tensors",
                    request=request_counter,
                )
        if copied:
            self.metrics.set_layer_active(self.state_id, True)

    @_compiler_disable
    def release_gpu_params(self) -> None:
        """Drop GPU copies after forward so only the CPU master remains."""
        named_parameters = dict(self.module.named_parameters(recurse=True))
        for name, param in named_parameters.items():
            if name not in self.cpu_named_parameters:
                continue
            if name in self.gpu_named_parameters:
                param.data = _tensor_placeholder(param.data, self.device)
                del self.gpu_named_parameters[name]
        if not self.gpu_named_parameters:
            self.metrics.set_layer_active(self.state_id, False)
        self.metrics.count("release_calls")

    @contextmanager
    def mutate_params_scope(self) -> Iterator[None]:
        """Materialize, yield for in-place writes, then refresh the CPU master.

        LoRA / optimizer updates must not race a swap. After the body the GPU
        cache is discarded and :meth:`offload_to_cpu` snapshots the mutated
        weights so later prefetch sees the write.
        """
        self.wait_and_materialize()
        try:
            yield
        finally:
            self.cpu_named_parameters.clear()
            self.gpu_named_parameters.clear()
            self.offload_to_cpu()


# ──────────────────────────────────────────────────────────────────────────
# Hooks and container lookup — first ModuleList unless a dotted path is given
# ──────────────────────────────────────────────────────────────────────────


def _find_layer_container(model: nn.Module, layer_container: str | None) -> nn.ModuleList | None:
    """Resolve the decoder ``ModuleList``; return ``None`` when the path is not one.

    A dotted ``layer_container`` is required when the first ``ModuleList`` is
    not the transformer stack (e.g. a vision tower). Unresolved or non-list
    attributes disable offload rather than wrapping the wrong children.
    """
    if layer_container:
        current: object = model
        for part in layer_container.split("."):
            current = getattr(current, part)
        return current if isinstance(current, nn.ModuleList) else None
    for module in model.modules():
        for child in module.children():
            if isinstance(child, nn.ModuleList):
                return child
    return None


def _make_pre_hook(state: _LayerwiseOffloadState):
    """Build a pre-forward hook that materializes this layer and prefetches the next."""

    @_compiler_disable
    def pre_hook(module: nn.Module, args, kwargs):
        """Wait for this layer's copy, then start H2D for ``next_state``."""
        state.metrics.count("forward_calls")
        state.wait_and_materialize()
        if state.next_state is not None:
            state.next_state.prefetch()
        return args, kwargs

    return pre_hook


def _make_post_hook(state: _LayerwiseOffloadState):
    """Build a post-forward hook that frees this layer's GPU copies."""

    @_compiler_disable
    def post_hook(module: nn.Module, args, kwargs, output):
        """Release GPU params; the output tensor is returned unchanged."""
        state.release_gpu_params()
        return output

    return post_hook


def _tensor_placeholder(tensor: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Zero-size storage with the same rank/dtype so Parameter identity survives.

    A scalar uses ``(0,)`` because ``empty(())`` is still one element. The
    placeholder lives on ``device`` so later ``.to(device)`` is a no-op.
    """
    shape = (0,) if tensor.ndim <= 0 else (0,) * tensor.ndim
    return torch.empty(shape, dtype=tensor.dtype, device=device)


__all__ = [
    "LayerwiseOffloadHandle",
    "enable_layerwise_cpu_offload",
    "layerwise_offload_mutation_scope",
]
