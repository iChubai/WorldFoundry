from __future__ import annotations

from pathlib import Path

import pytest

from worldfoundry.pipelines.component_pipelines import DVLTPipeline

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_dvlt_pipeline_rejects_preflight_artifact_smoke(tmp_path: Path) -> None:
    pipe = DVLTPipeline.from_pretrained(device="cpu")
    assert isinstance(pipe, DVLTPipeline)

    report_path = tmp_path / "dvlt_preflight.json"
    with pytest.raises(RuntimeError, match="DVLT cannot run"):
        pipe(
            images="sample_image.png",
            interactions=["view_sequence"],
            output_path=report_path,
            return_dict=True,
        )

    assert not report_path.exists()

    with pytest.raises(ValueError, match="no longer supports plan_only"):
        DVLTPipeline.from_pretrained({"plan_only": True}, device="cpu")
