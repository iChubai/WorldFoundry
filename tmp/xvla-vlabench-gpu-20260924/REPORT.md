# X-VLA VLABench checkpoint GPU validation

WorldFoundry's in-tree X-VLA pipeline loaded the local `2toINF--X-VLA-VLABench` checkpoint on H100 GPU 1. Domain ID 8, seed 42, and ten denoising steps generated a finite 30×20 action chunk. [validation.json](validation.json) records the output range -0.447 to 0.999, standard deviation 0.329, and nondegenerate first-arm rotation features. [status.json](status.json), [actions.json](actions.json), [result.json](result.json), [run.log](run.log), and [case.json](case.json) preserve the run. Peak PyTorch GPU allocation was 3.68 GB; no project code change was needed.

The input was a real two-camera CALVIN debug-mirror observation with an exploratory 20D state adapter. It is **outside the VLABench domain**. This is a checkpoint-loading and action-structure verification only; it does not establish physical action calibration, a benchmark score, or closed-loop task success. [quality-review.json](quality-review.json) records the scope.
