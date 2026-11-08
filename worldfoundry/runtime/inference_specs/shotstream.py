# SPDX-FileCopyrightText: 2026 OpenEnvision
# SPDX-License-Identifier: Apache-2.0
"""Curated inference spec for the ``shotstream`` model family."""

from __future__ import annotations

from worldfoundry.core.execution.inference import (
    InferenceCheckpointRef,
    InferenceTaskProfile,
    InferenceVariantSpec,
    ModelInferenceSpec,
)

from ._common import _artifact, _field

SHOTSTREAM_INFERENCE_SPEC = ModelInferenceSpec(
    model_family_id="shotstream",
    display_name="ShotStream",
    default_variant_id="wan2.1-1.3b-causal",
    default_task_id="causal-video",
    aliases=("shot-stream", "interactive-video-generation"),
    variants=(
        InferenceVariantSpec(
            variant_id="wan2.1-1.3b-causal",
            label="Wan2.1 1.3B Causal",
            status="requires_local_checkpoints",
            checkpoints=(
                InferenceCheckpointRef(
                    role="shotstream_checkpoint",
                    uri="${WORLDFOUNDRY_CKPT_DIR}/KlingTeam--ShotStream",
                    status="required",
                ),
                InferenceCheckpointRef(
                    role="wan_base_model",
                    uri="${WORLDFOUNDRY_CKPT_DIR}/Wan-AI--Wan2.1-T2V-1.3B",
                    status="required",
                ),
            ),
            load_kwargs={
                "model_id": "shotstream",
                "checkpoint_dir": "${WORLDFOUNDRY_CKPT_DIR}/KlingTeam--ShotStream",
                "wan_model_dir": "${WORLDFOUNDRY_CKPT_DIR}/Wan-AI--Wan2.1-T2V-1.3B",
            },
            aliases=("default", "official", "causal"),
            notes=(
                "Runs the official four-step causal generator from a checkout staged under ${WORLDFOUNDRY_MODEL_SOURCE_DIR}/ShotStream.",
            ),
        ),
    ),
    tasks=(
        InferenceTaskProfile(
            task_id="causal-video",
            label="Causal Video",
            description="Generate one 81-frame shot from a prompt, or run an official multi-shot CSV fixture.",
            aliases=("default", "official-demo", "multi-shot"),
            inputs=(
                _field(
                    "prompt",
                    "Prompt",
                    target="prompt",
                    required=True,
                    default=(
                        "A cinematic science-fiction scene in a bioluminescent forest at night. An astronaut in "
                        "a white exploration suit walks between glowing blue plants."
                    ),
                ),
                _field(
                    "checkpoint_dir",
                    "ShotStream Checkpoint Dir",
                    kind="path",
                    target="load_kwargs",
                    required=True,
                    default="${WORLDFOUNDRY_CKPT_DIR}/KlingTeam--ShotStream",
                ),
                _field(
                    "wan_model_dir",
                    "Wan Base Model Dir",
                    kind="path",
                    target="load_kwargs",
                    required=True,
                    default="${WORLDFOUNDRY_CKPT_DIR}/Wan-AI--Wan2.1-T2V-1.3B",
                ),
                _field(
                    "data_path",
                    "Official Input CSV",
                    kind="path",
                    target="call_kwargs",
                    description="Optional official ShotStream CSV. Omit it to synthesize a one-shot fixture from the prompt.",
                ),
                _field("seed", "Seed", kind="integer", target="call_kwargs", default=42),
                _field("fps", "FPS", kind="integer", target="call_kwargs", default=16),
                _field("plan_only", "Plan Only", kind="boolean", target="call_kwargs", default=False),
                _field("timeout_seconds", "Timeout Seconds", kind="integer", target="call_kwargs", default=21600),
            ),
            outputs=(
                _artifact("runtime_plan", "manifest", required=True, preview=False),
                _artifact("video", "video", required=True, preview=True),
            ),
            default_call_kwargs={"seed": 42, "fps": 16, "plan_only": False, "timeout_seconds": 21600},
        ),
    ),
    notes=(
        "The compatibility launcher uses per-run checkpoint symlinks and handles FlashAttention/PyAV API drift without editing upstream source.",
    ),
)
