# SPDX-FileCopyrightText: 2026 OpenEnvision
# SPDX-License-Identifier: Apache-2.0
"""Curated inference spec for the ``hunyuan_world_voyager`` model family."""

from __future__ import annotations

from worldfoundry.core.execution.inference import InferenceTaskProfile, InferenceVariantSpec, ModelInferenceSpec

from ._common import _TEST_CASES_ROOT, _artifact, _field

HUNYUAN_WORLD_VOYAGER_OFFICIAL_PROMPT = "An old-fashioned European village with thatched roofs on the houses."


HUNYUAN_WORLD_VOYAGER_OFFICIAL_CONDITION_DIR = str(_TEST_CASES_ROOT / "hunyuan_world_voyager" / "case1")


HUNYUAN_WORLD_VOYAGER_OFFICIAL_FIXTURE = str(_TEST_CASES_ROOT / "hunyuan_world_voyager" / "case1" / "ref_image.png")


HUNYUAN_WORLD_VOYAGER_OFFICIAL_CALL_KWARGS = {
    "num_frames": 49,
    "condition_dir": HUNYUAN_WORLD_VOYAGER_OFFICIAL_CONDITION_DIR,
    "interactions": "forward",
    "seed": 0,
    "infer_steps": 50,
    "flow_shift": 7.0,
    "embedded_cfg_scale": 6.0,
    "i2v_stability": True,
    "ulysses_degree": 1,
    "ring_degree": 1,
}


HUNYUAN_WORLD_VOYAGER_INFERENCE_SPEC = ModelInferenceSpec(
    model_family_id="hunyuan-world-voyager",
    display_name="HunyuanWorld-Voyager",
    default_variant_id="default",
    default_task_id="official-case1-video",
    aliases=("hunyuan-voyager", "world-voyager", "tencent/HunyuanWorld-Voyager"),
    variants=(
        InferenceVariantSpec(
            variant_id="default",
            label="Default",
            status="configured",
            call_kwargs=HUNYUAN_WORLD_VOYAGER_OFFICIAL_CALL_KWARGS,
            aliases=("official", "demo", "case1"),
            notes=("Matches the upstream HunyuanWorld-Voyager examples/case1 demo defaults.",),
        ),
    ),
    tasks=(
        InferenceTaskProfile(
            task_id="official-case1-video",
            label="Official Case1 Video",
            description="Run the HunyuanWorld-Voyager official case1 conditioned world-video demo.",
            aliases=("default", "official-demo", "conditioned-video"),
            inputs=(
                _field(
                    "image",
                    "Reference Image",
                    kind="path",
                    target="input_path",
                    required=True,
                    default=HUNYUAN_WORLD_VOYAGER_OFFICIAL_FIXTURE,
                ),
                _field("prompt", "Prompt", target="prompt", default=HUNYUAN_WORLD_VOYAGER_OFFICIAL_PROMPT),
                _field(
                    "condition_dir",
                    "Condition Dir",
                    kind="path",
                    target="call_kwargs",
                    required=True,
                    default=HUNYUAN_WORLD_VOYAGER_OFFICIAL_CONDITION_DIR,
                ),
                _field(
                    "interactions", "Interactions", kind="interaction_tokens", target="params", default=("forward",)
                ),
                _field("frames", "Requested Frames", kind="integer", target="params", default=49),
                _field("fps", "FPS", kind="integer", target="params", default=24),
                _field("steps", "Steps", kind="integer", target="params", default=50),
                _field("seed", "Seed", kind="integer", target="call_kwargs", default=0),
                _field("flow_shift", "Flow Shift", kind="number", target="call_kwargs", default=7.0),
                _field("embedded_cfg_scale", "Embedded CFG", kind="number", target="call_kwargs", default=6.0),
                _field("i2v_stability", "I2V Stability", kind="boolean", target="call_kwargs", default=True),
                _field("ulysses_degree", "Ulysses Degree", kind="integer", target="call_kwargs", default=1),
                _field("ring_degree", "Ring Degree", kind="integer", target="call_kwargs", default=1),
            ),
            outputs=(
                _artifact("video", "video", required=True, preview=True),
                _artifact("manifest", "manifest", required=True),
            ),
            default_call_kwargs=HUNYUAN_WORLD_VOYAGER_OFFICIAL_CALL_KWARGS,
        ),
    ),
    notes=("The official case1 requests 49 frames and emits 45 after dropping four I2V conditioning frames.",),
)
