# `worldfoundry.training`

Training and post-training recipes (SFT, distillation, RL) that produce checkpoints for
WorldFoundry models. This package sits beside the inference and evaluation path: it produces
checkpoints, it does not score them. It is excluded from the published wheel.

## Layout

| Path | Purpose |
| --- | --- |
| `recipes/` | Recipe specifications, including post-training recipes |
| `engine/` | Per-family training engines (`wan`, `cosmos`, `ltx`, `sana`, `dmd`, …) and sessions |
| `post_training/` | Distillation, RL, rewards, causal-LM and agentic post-training |
| `data/` | Training data pipelines |
| `objectives/`, `optimizers/` | Losses and optimizers |
| `models/`, `checkpoint/`, `distributed/`, `tuning/`, `safety/`, `api/` | Model wrappers, checkpoint IO, distributed training, tuning, safety, public API |

Checkpoints produced here are consumed through the normal inference path, e.g.
`wf run <model-id> --pipeline.load.checkpoint-path <ckpt>`.
