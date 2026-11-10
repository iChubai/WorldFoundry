"""Checkpoint loading rules for the LTX-2.5 diffusion video VAE.

The encoder and :class:`DiffusionVideoDecoder` modules are the canonical
``autoencoders.ltx.video`` implementations. This module maps released
safetensors keys onto them for the staged builders and installs the selected
decode preset after loading.
"""

import dataclasses
from pathlib import Path

import safetensors
import torch

from worldfoundry.base_models.diffusion_model.models.autoencoders.ltx.video.diffusion_video_decoder import (
    DiffusionVideoDecoder,
)
from worldfoundry.base_models.diffusion_model.models.autoencoders.ltx.video.transformer.apply import apply_diffvae_mode
from worldfoundry.base_models.diffusion_model.models.autoencoders.ltx.video.transformer.config import DiffVAEMode
from worldfoundry.base_models.diffusion_model.models.autoencoders.ltx.video.transformer.dsl_kernels import (
    FUSED_CTX_BIAS_BUFFER,
    FUSED_CTX_WEIGHT_BUFFER,
    NA_SOFTMAX_BOUND_BUFFER,
)
from worldfoundry.base_models.diffusion_model.optimizations.ltx.compile import CompilationConfig

from ..loader.module_ops import ModuleOps
from ..loader.sd_ops import KeyValueOperationResult, SDOps
from ..loader.sft_loader import SafetensorsModelStateDictLoader

_DIFFUSION_VAE_CLASS_NAME = "CausalDiffusionVAE"


def require_diffusion_video_vae(checkpoint_path: str) -> None:
    """Reject video VAEs other than the LTX-2.5 ``CausalDiffusionVAE``.

    LTX-2.5 latents are defined by its diffusion VAE; earlier convolutional LTX
    VAEs encode a different latent space.
    """
    metadata = SafetensorsModelStateDictLoader().metadata(checkpoint_path)
    class_name = metadata.get("config", {}).get("vae", {}).get("_class_name")
    if class_name != _DIFFUSION_VAE_CLASS_NAME:
        raise ValueError(
            f"LTX-2.5 requires a {_DIFFUSION_VAE_CLASS_NAME} video VAE; {checkpoint_path} declares {class_name!r}."
        )


def estimate_diffusion_decoder_weight_bytes(checkpoint_path: str | Path) -> int:
    """Sum on-disk bf16-sized nbytes for ``vae.decoder`` / ``decoder`` tensors (header only)."""
    path = Path(checkpoint_path)
    total = 0
    with safetensors.safe_open(path, framework="pt", device="cpu") as handle:
        for key in handle.keys():  # noqa: SIM118  # safetensors SafeOpen is not a Mapping
            if not (key.startswith(("vae.decoder.", "decoder.")) or "per_channel_statistics" in key):
                continue
            shape = handle.get_slice(key).get_shape()
            n = 1
            for dim in shape:
                n *= int(dim)
            total += n * 2  # bf16
    return total


VAE_ENCODER_COMFY_KEYS_FILTER = (
    SDOps("VAE_ENCODER_COMFY_KEYS_FILTER")
    .with_matching(prefix="vae.encoder.")
    .with_matching(prefix="vae.per_channel_statistics.")
    .with_matching(prefix="encoder.")
    .with_matching(prefix="per_channel_statistics.")
    .with_replacement("vae.encoder.", "")
    .with_replacement("vae.per_channel_statistics.", "per_channel_statistics.")
    .with_replacement("encoder.", "")
)


def _split_fused_qkv_param(key: str, value: torch.Tensor) -> list[KeyValueOperationResult]:
    """Split fused ``...qkv.weight`` / ``...qkv.bias`` into ``to_q`` / ``to_k`` / ``to_v``.
    Checkpoints store a single Linear(dim, 3*dim) under ``qkv.{weight,bias}``;
    ``QKVProjections`` owns three separate linears, so the loader rewrites keys here.
    """
    if value.shape[0] % 3 != 0:
        msg = f"fused QKV param {key!r} leading dim {value.shape[0]} is not divisible by 3"
        raise ValueError(msg)
    d = value.shape[0] // 3
    # key is "...qkv.weight" or "...qkv.bias"
    leaf = "weight" if key.endswith(".weight") else "bias"
    prefix = key[: -len(leaf)]  # "...qkv."
    return [
        KeyValueOperationResult(f"{prefix}to_q.{leaf}", value[:d].detach().clone()),
        KeyValueOperationResult(f"{prefix}to_k.{leaf}", value[d : 2 * d].detach().clone()),
        KeyValueOperationResult(f"{prefix}to_v.{leaf}", value[2 * d :].detach().clone()),
    ]


_RAW_DIFFUSION_DECODER_PREFIX = "vae.decoder."
_BARE_DIFFUSION_DECODER_PREFIX = "decoder."
_K_NORM_SUFFIX = ".attn.k_norm.weight"
_CONTEXT_PROJ_WEIGHT_SUFFIX = ".context_proj.weight"
_CONTEXT_PROJ_BIAS_SUFFIX = ".context_proj.bias"
_TYPE_EMB_KEY = "type_emb"
_CONV_IN_WEIGHT_KEY = "conv_in.weight"
_UPSAMPLE3_WEIGHT_KEY = "upsamples.3.proj.weight"
_UPSAMPLE3_BIAS_KEY = "upsamples.3.proj.bias"
_GATE_PARAM_SUFFIXES = (".gate_msa", ".gate_mlp", ".gate_ctx")
# Post-rename Linear leaf → sibling gate key suffix (folded W←g·W, b←g·b).
_GATE_FOLD_TARGETS: tuple[tuple[str, str], ...] = (
    (".attn.proj.weight", ".gate_msa"),
    (".attn.proj.bias", ".gate_msa"),
    (".mlp.w_down.weight", ".gate_mlp"),
    (".mlp.w_down.bias", ".gate_mlp"),
    (".context_proj.weight", ".gate_ctx"),
    (".context_proj.bias", ".gate_ctx"),
)


def _strip_diffusion_decoder_prefix(key: str) -> str | None:
    """Strip ``vae.decoder.`` or bare ``decoder.``; return None if neither matches."""
    if key.startswith(_RAW_DIFFUSION_DECODER_PREFIX):
        return key.removeprefix(_RAW_DIFFUSION_DECODER_PREFIX)
    if key.startswith(_BARE_DIFFUSION_DECODER_PREFIX):
        return key.removeprefix(_BARE_DIFFUSION_DECODER_PREFIX)
    return None


def _read_diff_vae_gates(checkpoint_path: str | Path) -> dict[str, torch.Tensor]:
    """Return ``{post_rename_gate_key: tensor}`` for every DiffVAE ``gate_*``.
    Keys are stripped of ``vae.decoder.`` / bare ``decoder.`` so they match the
    names the loader passes to kv-ops after SDOps rename. Streaming load is
    one-key-at-a-time, so siblings must be pre-read before fold ops run (same
    pattern as ``_read_scales`` / ``_build_prequant_fold_sd_ops`` in fp8_cast).
    """
    out: dict[str, torch.Tensor] = {}
    with safetensors.safe_open(str(checkpoint_path), framework="pt", device="cpu") as handle:
        for key in handle.keys():  # noqa: SIM118
            stripped = _strip_diffusion_decoder_prefix(key)
            if stripped is None or not key.endswith(_GATE_PARAM_SUFFIXES):
                continue
            out[stripped] = handle.get_tensor(key)
    return out


def _drop_coarse_param(_key: str, _value: torch.Tensor) -> list[KeyValueOperationResult]:
    """Drop bundled DiffVAE preview/coarse head weights (``coarse_*``)."""
    return []


def _gate_key_for_fold_target(param_key: str) -> str | None:
    for leaf, gate_suffix in _GATE_FOLD_TARGETS:
        if param_key.endswith(leaf):
            return param_key[: -len(leaf)] + gate_suffix
    return None


def _fold_gate_into_linear(
    param_key: str, value: torch.Tensor, gates: dict[str, torch.Tensor]
) -> list[KeyValueOperationResult]:
    """Fold a static gate into a Linear weight/bias when present; else pass through."""
    gate_key = _gate_key_for_fold_target(param_key)
    if gate_key is None:
        return [KeyValueOperationResult(param_key, value)]
    gate = gates.get(gate_key)
    if gate is None:
        return [KeyValueOperationResult(param_key, value)]
    gate_f = gate.to(device=value.device, dtype=torch.float32)
    value_f = value.to(dtype=torch.float32)
    if value.ndim == 2:
        folded = gate_f.unsqueeze(1) * value_f
    elif value.ndim == 1:
        folded = gate_f * value_f
    else:
        raise ValueError(f"Unsupported param rank {value.ndim} for gate fold on {param_key}")
    return [KeyValueOperationResult(param_key, folded.to(dtype=value.dtype))]


def _build_diffusion_vae_decoder_sd_ops(
    gates: dict[str, torch.Tensor],
    *,
    stage4_hop: "_Stage4Hop | None" = None,
) -> SDOps:
    """DiffVAE decoder SDOps: rename, QKV split, coarse drop, gate fold/drop.
    *gates* is keyed by post-rename gate param names (see ``_read_diff_vae_gates``).
    Empty map (bundled n1 / QKV unit tests): fold is a no-op; drop never fires.
    *stage4_hop* additionally packs the stage-4 upsample into every block's
    ``context_proj``, which is what lets the fused CuTe DSL block read stage-4 directly.
    """

    def _drop_gate(gate_key: str, _value: torch.Tensor) -> list[KeyValueOperationResult]:
        if gate_key not in gates:
            raise ValueError(
                f"Gate key {gate_key!r} has no matching entry in the pre-read gates dict; "
                f"_read_diff_vae_gates and the loader's rename map have drifted"
            )
        return []

    def _on_attn_proj(param_key: str, value: torch.Tensor) -> list[KeyValueOperationResult]:
        return _fold_gate_into_linear(param_key, value, gates)

    def _on_mlp_w_down(param_key: str, value: torch.Tensor) -> list[KeyValueOperationResult]:
        return _fold_gate_into_linear(param_key, value, gates)

    def _on_context_proj(param_key: str, value: torch.Tensor) -> list[KeyValueOperationResult]:
        return _fold_gate_into_linear(param_key, value, gates)

    def _on_context_proj_weight(param_key: str, value: torch.Tensor) -> list[KeyValueOperationResult]:
        folded = _fold_gate_into_linear(param_key, value, gates)
        if stage4_hop is None:
            return folded
        return folded + stage4_hop.fused_results(param_key, folded[0].new_value, gates)

    # Drop/fold ops before QKV split so first-match kv-ops never mis-route.
    # Match both monolith ``vae.decoder.*`` and Comfy-split bare ``decoder.*``.
    return (
        SDOps("DIFFUSION_VAE_DECODER_COMFY_KEYS_FILTER")
        .with_matching(prefix="vae.decoder.")
        .with_matching(prefix="decoder.")
        .with_replacement("vae.decoder.", "")
        .with_replacement("decoder.", "")
        .with_replacement("t_embedder.mlp.0.", "t_embedder.timestep_embedder.linear_1.")
        .with_replacement("t_embedder.mlp.2.", "t_embedder.timestep_embedder.linear_2.")
        .with_matching(prefix="vae.per_channel_statistics.")
        .with_matching(prefix="per_channel_statistics.")
        .with_replacement("vae.per_channel_statistics.", "per_channel_statistics.")
        .with_kv_operation(operation=_drop_coarse_param, key_prefix="coarse_")
        .with_kv_operation(operation=_drop_gate, key_suffix=".gate_msa")
        .with_kv_operation(operation=_drop_gate, key_suffix=".gate_mlp")
        .with_kv_operation(operation=_drop_gate, key_suffix=".gate_ctx")
        .with_kv_operation(operation=_on_attn_proj, key_suffix=".attn.proj.weight")
        .with_kv_operation(operation=_on_attn_proj, key_suffix=".attn.proj.bias")
        .with_kv_operation(operation=_on_mlp_w_down, key_suffix=".mlp.w_down.weight")
        .with_kv_operation(operation=_on_mlp_w_down, key_suffix=".mlp.w_down.bias")
        .with_kv_operation(operation=_on_context_proj_weight, key_suffix=".context_proj.weight")
        .with_kv_operation(operation=_on_context_proj, key_suffix=".context_proj.bias")
        .with_kv_operation(operation=_split_fused_qkv_param, key_suffix=".qkv.weight")
        .with_kv_operation(operation=_split_fused_qkv_param, key_suffix=".qkv.bias")
    )


def _checkpoint_has_type_emb(checkpoint_path: str | Path) -> bool:
    """Whether the checkpoint carries the keyframe ``type_emb`` tag."""
    with safetensors.safe_open(str(checkpoint_path), framework="pt", device="cpu") as handle:
        return any(_strip_diffusion_decoder_prefix(key) == _TYPE_EMB_KEY for key in handle.keys())  # noqa: SIM118


def _emit_zero_type_emb(param_key: str, value: torch.Tensor) -> list[KeyValueOperationResult]:
    """Pass ``conv_in.weight`` through and synthesize a zero ``type_emb`` beside it.
    ``type_emb`` is registered unconditionally on the decoder but only exists in
    keyframe-trained checkpoints. The loader does a non-strict ``assign=True`` load onto a
    meta model, so a missing parameter would stay on the meta device -- and
    ``_check_uninitialized`` then returns the whole decoder still on meta, breaking every
    pre-keyframe checkpoint. Synthesizing zeros here keeps those checkpoints loadable and
    is exactly the no-op the tag has when untrained (it is added to the keyframe latents,
    and zero means "no tag"). Attached to ``conv_in.weight`` because it always exists and
    its ``(out, in)`` shape carries the latent width.
    """
    return [
        KeyValueOperationResult(param_key, value),
        KeyValueOperationResult(_TYPE_EMB_KEY, torch.zeros(value.shape[1], dtype=value.dtype)),
    ]


def _emit_na_softmax_bound(param_key: str, value: torch.Tensor) -> list[KeyValueOperationResult]:
    """Pass ``attn.k_norm.weight`` through and emit the DSL kernel's softmax bound beside it.
    The CuTe DSL neighborhood-attention kernel replaces the online row max with a fixed
    per-row offset built from ``sqrt(head_dim) * max|k_norm.weight|``. Deriving it here
    means it is computed once, during load, into the buffer
    ``DiffVAEMode.BLACKWELL_DSL`` registered -- rather than per forward, where reading it
    off the weight would be a device-to-host sync inside a compiled region.
    The extra key is harmless when the kernel is not in use: loading is not strict, so
    a model without the buffer ignores it.
    """
    from ltx_kernels.vae import softmax_row_bound

    bound_key = param_key.removesuffix(_K_NORM_SUFFIX) + f".attn.{NA_SOFTMAX_BOUND_BUFFER}"
    return [
        KeyValueOperationResult(param_key, value),
        KeyValueOperationResult(bound_key, softmax_row_bound(value)),
    ]


def pack_fused_upsample_ctx_weights(
    w_upsample: torch.Tensor,
    b_upsample: torch.Tensor,
    w_ctx: torch.Tensor,
    b_ctx: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compose ``context_proj(pixel_shuffle(upsample(.)))`` into packed Linear weights.
    Thin wrapper over :func:`ltx_kernels.vae.pack_fused_ctx_weights` so SDOps tests
    can import from ``model_configurator`` without a hard kernels dependency at
    module import time.
    """
    from ltx_kernels.vae import pack_fused_ctx_weights

    return pack_fused_ctx_weights(w_upsample, b_upsample, w_ctx, b_ctx)


@dataclasses.dataclass(frozen=True)
class _Stage4Hop:
    """``upsamples[3].proj`` and every ``context_proj.bias``, pre-read from a checkpoint.
    The fused weight a block needs is built from three tensors that arrive at three
    different keys, and a streaming load sees one key at a time -- so the two the
    ``context_proj.weight`` op cannot wait for are read up front. The upsample is one
    tensor for the whole decoder; the biases are one small vector per block. (Same
    pattern as :func:`_read_diff_vae_gates`, which pre-reads the gates for the fold.)
    """

    upsample_weight: torch.Tensor
    upsample_bias: torch.Tensor
    context_biases: dict[str, torch.Tensor]  # post-rename ``....context_proj.bias``

    @classmethod
    def read(cls, checkpoint_path: str | Path) -> "_Stage4Hop":
        upsample_weight = upsample_bias = None
        context_biases: dict[str, torch.Tensor] = {}
        with safetensors.safe_open(str(checkpoint_path), framework="pt", device="cpu") as handle:
            for key in handle.keys():  # noqa: SIM118
                short = _strip_diffusion_decoder_prefix(key)
                if short is None:
                    continue
                if short == _UPSAMPLE3_WEIGHT_KEY:
                    upsample_weight = handle.get_tensor(key)
                elif short == _UPSAMPLE3_BIAS_KEY:
                    upsample_bias = handle.get_tensor(key)
                elif short.endswith(_CONTEXT_PROJ_BIAS_SUFFIX):
                    context_biases[short] = handle.get_tensor(key)
        if upsample_weight is None or upsample_bias is None:
            raise ValueError(
                f"DiffVAE checkpoint {checkpoint_path!r} has no {_UPSAMPLE3_WEIGHT_KEY}, which "
                "na_dsl_kernel needs to fuse the stage-4 hop into context_proj"
            )
        return cls(upsample_weight, upsample_bias, context_biases)

    def fused_results(
        self, weight_key: str, folded_weight: torch.Tensor, gates: dict[str, torch.Tensor]
    ) -> list[KeyValueOperationResult]:
        """The two fused buffers for the block owning ``weight_key``.
        ``folded_weight`` is that block's already gate-folded ``context_proj.weight``; the
        bias gets the same fold here, so the packed pair carries ``gate_ctx`` exactly once
        and matches what the unfused ``context_proj`` would have computed.
        """
        prefix = weight_key.removesuffix(_CONTEXT_PROJ_WEIGHT_SUFFIX)
        bias_key = prefix + _CONTEXT_PROJ_BIAS_SUFFIX
        raw_bias = self.context_biases.get(bias_key)
        if raw_bias is None:
            raise ValueError(f"No pre-read {bias_key!r} to fuse alongside {weight_key!r}")
        folded_bias = _fold_gate_into_linear(bias_key, raw_bias, gates)[0].new_value
        weight, bias = pack_fused_upsample_ctx_weights(
            self.upsample_weight, self.upsample_bias, folded_weight, folded_bias
        )
        return [
            KeyValueOperationResult(f"{prefix}.{FUSED_CTX_WEIGHT_BUFFER}", weight),
            KeyValueOperationResult(f"{prefix}.{FUSED_CTX_BIAS_BUFFER}", bias),
        ]


def video_decoder_sd_ops_for_checkpoint(
    checkpoint_path: str,
    *,
    na_dsl_kernel: bool = False,
) -> SDOps:
    """Build diffusion-decoder SDOps for ``checkpoint_path``.
    Pre-reads ``gate_*`` and builds fold/drop ops so standalone distilled
    checkpoints load into the ungated ``DiffusionNABlock``.
    ``na_dsl_kernel`` additionally derives the CuTe DSL kernel's softmax bound from each
    ``k_norm`` weight and fuses ``upsamples[3]`` into every block's ``context_proj``, so
    the fused block reads stage-4 directly instead of a full-resolution context. Must be
    paired with ``DiffVAEMode.BLACKWELL_DSL`` (registers the receiving buffers); apply
    alone leaves NaN / meta / empty buffers that fail at first forward. ``VideoDecoder``
    ties both to one flag -- custom builders must do the same.
    """
    gates = _read_diff_vae_gates(checkpoint_path)
    if na_dsl_kernel:
        ops = _build_diffusion_vae_decoder_sd_ops(gates, stage4_hop=_Stage4Hop.read(checkpoint_path)).with_kv_operation(
            operation=_emit_na_softmax_bound, key_suffix=_K_NORM_SUFFIX
        )
    else:
        ops = _build_diffusion_vae_decoder_sd_ops(gates)
    if not _checkpoint_has_type_emb(checkpoint_path):
        ops = ops.with_kv_operation(operation=_emit_zero_type_emb, key_suffix=_CONV_IN_WEIGHT_KEY)
    return ops


def build_diffvae_mode_op(mode: DiffVAEMode, compilation_config: CompilationConfig | None = None) -> ModuleOps:
    """Install ``mode`` on the meta decoder before weights are loaded."""

    def mutator(model: torch.nn.Module) -> torch.nn.Module:
        return apply_diffvae_mode(model, mode, compilation_config=compilation_config)  # type: ignore[arg-type]

    return ModuleOps(
        name=f"diffvae_mode_{mode.value}",
        matcher=lambda model: isinstance(model, DiffusionVideoDecoder),
        mutator=mutator,
    )
