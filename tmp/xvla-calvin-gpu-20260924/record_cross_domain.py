"""Record one completed X-VLA cross-domain checkpoint inference probe."""

from __future__ import annotations

import json
import sys
from pathlib import Path


variant = sys.argv[1]
names = {
    "robotwin2": "RoboTwin2",
    "vlabench": "VLABench",
    "agiworld": "AgiWorld Challenge",
    "softfold": "SoftFold",
}
assert variant in names, variant
root = Path(f"tmp/xvla-{variant}-gpu-20260924")
case = json.loads((root / "case.json").read_text())
status = json.loads((root / "status.json").read_text())
validation = json.loads((root / "validation.json").read_text())
assert status["status"] == "generated" and validation["finite"]
assert validation["shape"] == [30, 20]
domain = case["load"]["domain_id"]
checkpoint = case["load"]["model_path"]
peak = status["peak_memory_allocated"]
review = {
    "status": "verified_inference_only",
    "checkpoint": checkpoint,
    "fixture": "tmp/calvin-fixture-20260922/fixture.json",
    "input": f"Real CALVIN debug-mirror episode0 frame0 cameras and exploratory 20D state; deliberately out of {names[variant]} domain; domain_id={domain}, ten denoising steps, seed42.",
    "action_shape": validation["shape"],
    "finite": validation["finite"],
    "value_min": validation["min"],
    "value_max": validation["max"],
    "value_std": validation["std"],
    "first_arm_rotation_min_norm": validation["first_arm_rotation_min_norm"],
    "first_arm_rotation_min_cross_norm": validation["first_arm_rotation_min_cross_norm"],
    "gpu_peak_allocated_bytes": peak,
    "scope_limit": f"Cross-domain checkpoint and loader/action-structure check only; no {names[variant]} observation, calibrated action, benchmark metric, closed-loop rollout, or task success.",
}
(root / "quality-review.json").write_text(json.dumps(review, indent=2) + "\n")
report = f"""# X-VLA {names[variant]} checkpoint GPU validation

WorldFoundry's in-tree X-VLA pipeline loaded the local `{Path(checkpoint).name}` checkpoint on H100 GPU 1. Domain ID {domain}, seed 42, and ten denoising steps generated a finite 30×20 action chunk. [validation.json](validation.json) records the output range {validation['min']:.3f} to {validation['max']:.3f}, standard deviation {validation['std']:.3f}, and nondegenerate first-arm rotation features. [status.json](status.json), [actions.json](actions.json), [result.json](result.json), [run.log](run.log), and [case.json](case.json) preserve the run. Peak PyTorch GPU allocation was {peak/1e9:.2f} GB; no project code change was needed.

The input was a real two-camera CALVIN debug-mirror observation with an exploratory 20D state adapter. It is **outside the {names[variant]} domain**. This is a checkpoint-loading and action-structure verification only; it does not establish physical action calibration, a benchmark score, or closed-loop task success. [quality-review.json](quality-review.json) records the scope.
"""
(root / "REPORT.md").write_text(report)

scope = (
    f"Public X-VLA {names[variant]} checkpoint loaded on H100 GPU1 with domain{domain}, seed42, ten denoising steps. "
    f"One real two-camera CALVIN debug-mirror observation with exploratory 20D state adapter produced finite 30x20 actions "
    f"(range {validation['min']:.3f}..{validation['max']:.3f}, std{validation['std']:.3f}), {peak/1e9:.2f}GB peak. "
    f"Deliberately cross-domain loader/action-structure check only; no {names[variant]} input, calibrated action, closed-loop rollout, or task score."
)
case_id = f"xvla-{variant}"
base = str(root)
obs_path = Path("tmp/all-model-validation-20260921/observations.json")
observations = json.loads(obs_path.read_text())
entry = {
    "model_id": "xvla",
    "case": case_id,
    "status": "verified_inference_only",
    "status_evidence": base + "/quality-review.json",
    "validation": base + "/validation.json",
    "report": base + "/REPORT.md",
    "scope": scope,
    "reviewed_in_current_session": True,
}
matches = [i for i, row in enumerate(observations) if row.get("case") == case_id]
if matches:
    observations[matches[0]] = entry
    for i in reversed(matches[1:]):
        del observations[i]
else:
    observations.append(entry)
obs_path.write_text(json.dumps(observations, ensure_ascii=False, indent=2) + "\n")

obligations_path = Path("tmp/all-model-validation-20260921/runtime-variant-obligations.json")
obligations = json.loads(obligations_path.read_text())
matches = [i for i, row in enumerate(obligations) if row.get("case") == case_id]
assert len(matches) == 1, matches
obligations[matches[0]].update(status="verified_inference_only", scope=scope, evidence=base + "/quality-review.json")
obligations_path.write_text(json.dumps(obligations, ensure_ascii=False, indent=2) + "\n")
print(case_id, review["status"], validation["shape"])
