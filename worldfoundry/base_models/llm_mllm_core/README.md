# `worldfoundry.base_models.llm_mllm_core`

Inference-only (multimodal) language models. Their main use is as **judges and feature
extractors in benchmark evaluation** — physical-commonsense raters, video-QA scorers,
anomaly detectors. They also serve as language backbones for a few VLA policies and for
prompt expansion in diffusion pipelines.

## Layout

All models live under `mllm/`:

| Path | Model / role |
| --- | --- |
| `mllm/qwen/` | Qwen family: Qwen2.5-VL prompt embedder, Qwen3 backbone, Qwen3Guard, Qwen-Image, Cosmos-Reason1, and Being-H / QwenPi policy backbones |
| `mllm/llava_next/`, `mllm/vila/`, `mllm/keye_vl/`, `mllm/perceptionlm/` | General video/image MLLMs used as evaluators |
| `mllm/internvideo2/`, `mllm/umt/` | Video–language encoders |
| `mllm/videophy2_autoeval/`, `mllm/videocon_physics/` | Physical-commonsense judges (VideoPhy-2 AutoEval, VideoCon-Physics) |
| `mllm/instance_anomaly_detector/` | Instance-level anomaly detection for generated videos |
| `mllm/open_flamingo/` | OpenFlamingo backbone (used by RoboFlamingo) |
| `mllm/modelscope_swift/` | ModelScope Swift inference framework (used by the anomaly detector) |

## Usage

These models are loaded on demand by benchmark runners under
`worldfoundry/evals/tasks/execution/runners/` (VideoScore, WorldModelBench, 4DWorldBench,
…), by VQAScore in `perception_core`, and by a few action policies under
`worldfoundry/synthesis/action_generation/`. Check what a benchmark needs before running it:

```bash
wf zoo benchmark-show --benchmark-id <benchmark-id> --include-spec --json
python scripts/setup/prepare_benchmark_assets.py --benchmark-id <benchmark-id> --json
```

## Conventions

- Inference only; checkpoints resolve through the Hugging Face cache or `WORLDFOUNDRY_CKPT_DIR`.
- Keep imports lazy — these stacks pin large `transformers` / vision dependencies.
- Hosted-API judges (OpenAI, Gemini, DashScope) are not here; they live with the runners that call them and read credentials from environment variables.
- Retain upstream attribution and license notices in vendored code.
