# SPDX-FileCopyrightText: 2026 OpenEnvision
# SPDX-License-Identifier: Apache-2.0
"""Curated inference spec for the ``egowm`` model family."""

from __future__ import annotations

from worldfoundry.core.execution.inference import (
    InferenceCheckpointRef,
    InferenceTaskProfile,
    InferenceVariantSpec,
    ModelInferenceSpec,
)

from ._common import _artifact, _field

EGOWM_INFERENCE_SPEC = ModelInferenceSpec(
    model_family_id="egowm",
    display_name="EgoWM",
    default_variant_id="svd-25dof-nav",
    default_task_id="navigation-video",
    aliases=("ego-wm", "egowm-svd-25dof"),
    variants=(
        InferenceVariantSpec(
            variant_id="svd-25dof-nav",
            label="SVD 25-DoF Navigation",
            status="requires_local_checkpoints",
            checkpoints=(
                InferenceCheckpointRef(
                    role="egowm_checkpoint",
                    uri="${WORLDFOUNDRY_CKPT_DIR}/anuragba--egowm/svd_25dof_nav.pth",
                    status="required",
                ),
                InferenceCheckpointRef(
                    role="svd_base_model",
                    uri="${WORLDFOUNDRY_CKPT_DIR}/stabilityai--stable-video-diffusion-img2vid",
                    status="required",
                ),
            ),
            load_kwargs={
                "model_id": "egowm",
                "checkpoint_path": "${WORLDFOUNDRY_CKPT_DIR}/anuragba--egowm/svd_25dof_nav.pth",
                "base_model_dir": "${WORLDFOUNDRY_CKPT_DIR}/stabilityai--stable-video-diffusion-img2vid",
            },
            aliases=("default", "official", "25dof"),
            notes=(
                "Runs the official action/state-conditioned SVD model from a checkout staged under "
                "${WORLDFOUNDRY_MODEL_SOURCE_DIR}/egowm.",
            ),
        ),
        InferenceVariantSpec(
            variant_id="svd-3dof-nav",
            label="SVD 3-DoF Navigation",
            status="requires_local_checkpoints",
            checkpoints=(
                InferenceCheckpointRef(
                    role="egowm_checkpoint",
                    uri="${WORLDFOUNDRY_CKPT_DIR}/anuragba--egowm/svd_3dof_nav.pth",
                    status="required",
                ),
                InferenceCheckpointRef(
                    role="svd_base_model",
                    uri="${WORLDFOUNDRY_CKPT_DIR}/stabilityai--stable-video-diffusion-img2vid",
                    status="required",
                ),
            ),
            load_kwargs={
                "model_id": "egowm",
                "checkpoint_path": "${WORLDFOUNDRY_CKPT_DIR}/anuragba--egowm/svd_3dof_nav.pth",
                "model_path": "${WORLDFOUNDRY_CKPT_DIR}/anuragba--egowm/svd_3dof_nav.pth",
                "base_model_dir": "${WORLDFOUNDRY_CKPT_DIR}/stabilityai--stable-video-diffusion-img2vid",
            },
            call_kwargs={"variant": "3dof"},
            aliases=("3dof",),
            notes=(
                "Pair an initial image with actions_path containing one normalized [x, y, yaw] action per generated frame.",
            ),
        ),
    ),
    tasks=(
        InferenceTaskProfile(
            task_id="navigation-video",
            label="Navigation Video",
            description="Generate an egocentric navigation rollout from a paired initial image, 25-DoF state, and per-frame actions.",
            aliases=("default", "official-demo", "25dof-nav"),
            inputs=(
                _field(
                    "input_path",
                    "Initial Scene Image",
                    kind="path",
                    target="input_path",
                    required=True,
                    description="Supply the first frame paired with the 25-DoF conditions JSON.",
                ),
                _field(
                    "conditions_path",
                    "Paired 25-DoF Conditions JSON",
                    kind="path",
                    target="call_kwargs",
                    description="Required for 25dof; omit for 3dof. Contains physical_initial_state and physical_actions.",
                ),
                _field(
                    "actions_path",
                    "Paired 3-DoF Actions JSON",
                    kind="path",
                    target="call_kwargs",
                    description="For 3dof, supply normalized_actions with one [x, y, yaw] row per generated frame, paired with the initial image.",
                ),
                _field("variant", "Checkpoint Variant", kind="string", target="call_kwargs", choices=("25dof", "3dof")),
                _field(
                    "checkpoint_path",
                    "EgoWM Checkpoint",
                    kind="path",
                    target="load_kwargs",
                    required=True,
                ),
                _field(
                    "base_model_dir",
                    "SVD Base Model",
                    kind="path",
                    target="load_kwargs",
                    required=True,
                    default="${WORLDFOUNDRY_CKPT_DIR}/stabilityai--stable-video-diffusion-img2vid",
                ),
                _field("num_frames", "Generated Frames", kind="integer", target="call_kwargs", default=8),
                _field("num_inference_steps", "Inference Steps", kind="integer", target="call_kwargs", default=25),
                _field("fps", "FPS", kind="integer", target="call_kwargs", default=7),
                _field("seed", "Seed", kind="integer", target="call_kwargs", default=42),
                _field("height", "Height", kind="integer", target="call_kwargs", default=512),
                _field("width", "Width", kind="integer", target="call_kwargs", default=512),
                _field("motion_bucket_id", "Motion Bucket", kind="integer", target="call_kwargs", default=180),
                _field(
                    "smoke_synthetic", "Synthetic Smoke Test Only", kind="boolean", target="call_kwargs", default=False
                ),
                _field(
                    "action_scale",
                    "3-DoF Action Offset",
                    kind="number",
                    target="call_kwargs",
                    description="Constant normalized x offset for a 3dof smoke test; paired inference uses actions_path.",
                ),
                _field("plan_only", "Plan Only", kind="boolean", target="call_kwargs", default=False),
                _field("timeout_seconds", "Timeout Seconds", kind="integer", target="call_kwargs", default=21600),
            ),
            outputs=(
                _artifact("runtime_plan", "manifest", required=True, preview=False),
                _artifact("video", "video", required=True, preview=True),
            ),
            default_call_kwargs={
                "variant": "25dof",
                "num_frames": 8,
                "num_inference_steps": 25,
                "fps": 7,
                "seed": 42,
                "height": 512,
                "width": 512,
                "motion_bucket_id": 180,
                "plan_only": False,
                "timeout_seconds": 21600,
            },
        ),
    ),
    notes=(
        "The compatibility wrapper passes the local SVD base model explicitly and leaves the official checkout unchanged. "
        "Paired 3-DoF actions_path and paired 25-DoF conditions_path use distinct input contracts.",
    ),
)
