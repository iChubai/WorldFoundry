---
name: worldfoundry-model-inference
description: >
  Use this skill when running a WorldFoundry model to generate video, images,
  3D assets, or actions: choosing a model ID, variant, and task profile;
  preparing the conda environment and checkpoints; inspecting the resolved
  configuration without loading weights; running `wf run` or a Workspace job;
  reading output artifacts; or diagnosing a failed run (not runnable, missing
  checkpoint, import error, OOM, wrong output). Scoring outputs is
  worldfoundry-evaluation; adding or fixing a model integration is
  worldfoundry-model-integration; speedups are
  worldfoundry-inference-optimization.
license: Apache-2.0
compatibility: >
  Requires a WorldFoundry checkout with the unified environment from
  scripts/setup/bootstrap_worldfoundry.sh. Generation needs an NVIDIA GPU,
  local or downloadable checkpoints, and provider credentials for hosted APIs.
---

# Run WorldFoundry Model Inference

`wf run <model-id>` is the canonical, reproducible inference command; the same
shape appears on every model page. Everything before generation can be
checked without loading weights, so do that before allocating a GPU. Run
commands from the repository root with the unified environment active.
[Run inference](../../docs/fumadocs/content/docs/guides/inference.mdx),
[local assets](../../docs/fumadocs/content/docs/guides/local-assets.mdx), and the
[CLI reference](../../docs/fumadocs/content/docs/reference/cli.mdx) own the
details.

## Prepare the machine once

```bash
bash scripts/setup/bootstrap_worldfoundry.sh
source tmp/worldfoundry_unified_env.sh
conda activate "${WORLDFOUNDRY_UNIFIED_ENV_PREFIX}"
wf --help
```

`bash scripts/setup/bootstrap_worldfoundry.sh --verify-only` checks an existing
setup. Without the console script on `PATH`, `python -m worldfoundry` uses the
same parser.

## Workflow

### 1. Pick the model

```bash
wf zoo models --integration-status integrated
wf zoo model-show --model-id <model-id-or-alias> --include-manifest --json
```

The model page `docs/fumadocs/content/docs/guides/supported-models/<model-id>.mdx`
lists task profiles, hardware, and the published command. Use canonical IDs in
scripts; aliases are for interactive use.

### 2. Read the contract without weights

```bash
wf run <model-id> --model-status
wf run <model-id> --help
```

`--model-status` (an alias of `--print-config`) reports `runnable`,
`runner_entry_kind`, `blocked_reason`, `task_choices`, `output_artifacts`, and
every typed field with its scope, default, and `required` flag. `--help` groups
them as `--pipeline.*` (inputs and generation), `--pipeline.load.*`
(checkpoint and loading), and `--runtime.*` (device and runner). Add your
overrides to `wf run <model-id> ... --print-config --json` to see the resolved
values before running.

### 3. Prepare the environment and checkpoints

```bash
bash scripts/setup/model_env_install.sh --list
bash scripts/inference/prepare_model_infer.sh <model-id> --verify-env-only
wf zoo model-download --model-id <model-id> --check-local --json
```

- `model_env_install.sh --list` is the source of truth for environment
  routing. Most models use the unified environment; install a dedicated one
  only when listed: `bash scripts/setup/model_env_install.sh --model <model-id>`.
- `--check-local` only inspects the cache. Download with
  `bash scripts/inference/prepare_model_infer.sh <model-id> --download`, or
  `wf zoo model-download --model-id <model-id> --execute --report-path tmp/model_zoo/download/<model-id>.json`.
- Hugging Face weights resolve through `HF_HOME`/`HF_HUB_CACHE`; gated repos
  need `HF_TOKEN`. Non-HF weights resolve under `WORLDFOUNDRY_CKPT_DIR`.
  Link existing checkpoint directories with
  `bash scripts/setup/link_hf_checkpoints.sh` instead of copying them.

### 4. Run

```bash
wf run <model-id> --pipeline.task-profile <task-profile> \
  --pipeline.prompt "<prompt>" --pipeline.seed 42 \
  --output-dir tmp/worldfoundry_infer/<model-id>
```

Input field names differ by model: Wan I2V takes `--pipeline.input-path`,
Matrix-Game 2 takes `--pipeline.image`. Use the `input` fields and `required`
flags that `--model-status` lists. Hosted API models read provider credentials
from the environment variables named on their model page.

For a Workspace visual smoke run (the job path the Studio UI submits):

```bash
python -m worldfoundry.studio.workspace_job infer --model-id <model-id> \
  --prompt "<prompt>" --input-path /path/to/input.png \
  --output-dir tmp/worldfoundry_infer/<model-id> --device cuda
```

Pass model-specific call or load keys with `--call-json` and `--load-json`.

### 5. Inspect the result

A direct `wf run` executes through the evaluation core with the
`artifact_count` metric. Generated media lands in `<output-dir>/generated/`;
`results.jsonl` lists each sample's status, artifacts, and errors;
`sample_ledger.jsonl` records stage failures; `run_manifest.json` records the
resolved identity and paths; lifecycle logs go under
`<output-dir>/logs/<run-id>/`. Batch several inputs with `--requests-path`
(one `GenerationRequest` JSON object per line). A file existing is not
success: check frame count, fps, resolution, controls, and content against
the model page or official demo.

## Troubleshooting

| Symptom | Meaning | Next step |
| --- | --- | --- |
| `is not runnable: ... has no registered runner target` | Entry is `listed_only` | Pick another model, or integrate it with worldfoundry-model-integration |
| `... integration status is 'planned' rather than 'integrated'` | Entry is a `runner_candidate` without accepted evidence | Choose an integrated variant, or complete the integration |
| `unknown model-zoo entry ...; did you mean` | ID or alias typo | `wf zoo models`, then `zoo model-show` |
| Missing checkpoint or file not found | Weights not staged | `--check-local`, then download; override with the model's `--pipeline.load.*` path field |
| `ImportError`, missing `cv2`, or model dependency | Wrong environment | Activate the unified env or install the listed dedicated env |
| Workers report missing `python3.x`, `libssl`, or `libcrypto` | A conda env was copied between machines | Recreate it; `bootstrap_worldfoundry.sh --verify-only` |
| CUDA out of memory | Resident weights or activations exceed VRAM | Use the model's offload field, such as `--pipeline.load.offload-mode async-block` when listed |
| Exit code 2 | Invalid CLI input, missing files or environment, or failed preflight | Read the message; `wf run <model-id> --help` |

## Gotchas

- `wf run` writes to `./benchmark_results` unless `--output-dir` is given;
  always pass a directory under `tmp/`.
- Typed options are model-specific. Never copy `--pipeline.load.*` flags from
  another family; Wan acceleration keys are not wired on other runtimes.
- Nested options such as `feature_cache`, `approximate_attention`, and
  `quantization` belong in `--load-json` or Python kwargs, not guessed flags.
- Multi-GPU support is per model; use the device or process-count fields the
  model's `--help` lists rather than a generic launcher.
- Compiled or graph-captured first requests are slow by design; do not judge
  speed from them.
- Public commands use Hugging Face repo IDs or `${WORLDFOUNDRY_CKPT_DIR}`
  paths, never private cache paths.
- Inference output is not benchmark or leaderboard evidence; score it with
  [evaluation](../worldfoundry-evaluation/SKILL.md).

## Report

Give the exact command, model, variant, task profile, environment name, GPU,
checkpoint repo and revision (or local path), output artifact paths, wall
time, and any warnings or fallbacks.
