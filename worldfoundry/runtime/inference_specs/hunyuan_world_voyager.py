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
    "condition_dir": HUNYUAN_WORLD_VOYAGER_OFFICIAL_CONDITION_DIR,
    "seed": 0,
    "infer_steps": 50,
}


HUNYUAN_WORLD_VOYAGER_INFERENCE_SPEC = ModelInferenceSpec(
    model_family_id="hunyuanworld-voyager",
    display_name="HunyuanWorld-Voyager",
    default_variant_id="default",
    default_task_id="official-case1-video",
    aliases=("hunyuan-world-voyager", "hunyuan-voyager", "world-voyager", "tencent/HunyuanWorld-Voyager"),
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
                _field("infer_steps", "Steps", kind="integer", target="call_kwargs", default=50),
                _field("seed", "Seed", kind="integer", target="call_kwargs", default=0),
            ),
            outputs=(
                _artifact("video", "video", required=True, preview=True),
                _artifact("manifest", "manifest", required=True),
            ),
            default_call_kwargs=HUNYUAN_WORLD_VOYAGER_OFFICIAL_CALL_KWARGS,
        ),
    ),
    notes=(
        "The camera path comes from the condition directory; the official case1 conditions 49 frames and emits 45.",
    ),
)
