# Route Manifests

Read this when creating or editing the binding, runtime profile, environment,
or catalog entry for a model. Copy the closest existing entry in the same
catalog category and change only what differs. The parsers own the accepted
fields: `worldfoundry/evals/models/catalog/schema.py` for catalog entries
and `worldfoundry/evals/models/pipelines/bindings.py` for bindings.

## Files and their roles

| File | Role | Example |
| --- | --- | --- |
| `worldfoundry/data/models/bindings/pipelines/<binding-id>.yaml` | Import target and invocation form for the generic pipeline runner | `helios.yaml`, `wan2.1-t2v-1.3b.yaml` |
| `worldfoundry/data/models/runtime/profiles/<model-id>.yaml` | Checkpoint roles and local paths, input schema, artifact kind, execution metadata, hardware notes | `helios.yaml` |
| `worldfoundry/data/models/runtime/environments/<group>/<model-id>.yaml` | Dedicated conda environment; only for a documented dependency conflict | `video/helios.yaml` |
| `worldfoundry/data/models/runtime/configs/<model-id>/` | Static presets and example inputs, resolved with `package_data_path()` | `vla_va_wam/openvla.yaml` |
| `worldfoundry/data/models/catalog/<category>/<model-id>.yaml` | Public identity, provenance, variants, status claims, evidence, docs seed | `video/helios.yaml`, `video/wan2.1.yaml` |

Categories are `video`, `world_models`, `three_d_four_d`, `vla_va_wam`,
`multimodal`, and `hosted_api`.

## Binding

```yaml
schema_version: 2
binding_id: <binding-id>
model_id: <model-id>
aliases:
- <alias>
runner: worldfoundry.pipeline
pipeline:
  target: worldfoundry.pipelines.<family>.pipeline_<family>:<PipelineClass>
  loading:
    method: from_pretrained
    signature: unified
  invocation:
    mode: native_or_standard_invocation
```

Native diffusion pipelines built on `NativeVisualDiffusionPipeline` use
`mode: native_diffusion_invocation`. Write the binding after the class
imports; the binding resolver is the only route parser.

## Runtime profile

```yaml
schema_version: 2
model_id: <model-id>
display_name: <Display Name>
task_family: video_generation
source_repos:
- url: https://github.com/<org>/<repo>
  revision: <commit-sha>
checkpoints:
- repo_id: <org>/<checkpoint-repo>
  revision: <commit-sha>
  local_dir: ${WORLDFOUNDRY_CKPT_DIR}/<checkpoint-dir>
  role: <role>
input_schema:
  prompt: true
  image: optional
artifact_kind: generated_video
execution:
  backend_stage: <implementation-depth>
  pipeline_binding: <binding-id>
integration:
  status: planned
gpu_targets:
- <GPU class actually validated, or the stated requirement>
notes:
- <scope of executed evidence>
```

Public checkpoint paths are Hugging Face repo IDs, native HF cache locations,
or `${WORLDFOUNDRY_CKPT_DIR}` paths; never private machine paths. Code executed
in a subprocess declares `source_repos[].in_tree_path` so affected-case
selection can see it.

## Environment

Most models run in the unified environment; confirm with
`bash scripts/setup/model_env_install.sh --list`. Add an environment file only
for a real conflict such as pinned Torch/CUDA ABI, JAX/TensorFlow isolation, or
an incompatible Diffusers stack, and record that reason in the profile notes.
Verify it with `bash scripts/setup/model_env_install.sh --model <model-id> --verify-only`.

## Catalog entry

```yaml
id: <model-id>
name: <Display Name>
aliases:
- <alias>
tasks:
- text-to-video
output_artifacts:
- generated_video
source:
  status: open_source
  official_repo_url: https://github.com/<org>/<repo>
  revision: <commit-sha>
  license: <spdx-or-upstream-license>
checkpoints:
- repo_id: <org>/<checkpoint-repo>
  revision: <commit-sha>
  license: <license>
  role: <role>
variants:
- id: <variant-id>
  task: text-to-video
  checkpoint_refs:
  - repo_id: <org>/<checkpoint-repo>
  runtime_profile: <model-id>
  pipeline_binding: <binding-id>
  runner_target: worldfoundry.evals.models.runners.pipeline:WorldFoundryPipelineRunner
  integration:
    status: planned
integration:
  status: planned
  notes:
  - <what was executed, on which hardware>
runner_target: worldfoundry.evals.models.runners.pipeline:WorldFoundryPipelineRunner
runtime_profile: runtime-profile:<model-id>
pipeline_binding: <binding-id>
docs:
  overview: <English overview>
  overview_zh: <Chinese overview>
```

The `docs:` block seeds generated model pages; keep each English field paired
with its `_zh` field.

## Status vocabulary

| Field | Accepted values | Meaning |
| --- | --- | --- |
| `source.status` | `open_source`, `api`, `closed`, `unknown` | How code and weights are obtained, not WorldFoundry readiness |
| `integration.status` (entry and variant) | `planned`, `blocked`, `integrated` | WorldFoundry claim; `integrated` requires executed real-request evidence |
| `demo_parity.status`, `runner_parity.status` | `verified`, `pending`, `not_applicable`, `failed` | Official demo and WorldFoundry runner comparison state |

Prefer the canonical value and put the nuance in `notes`. A new freeform value
must be registered in the matching alias set in `schema.py`;
`tests/evaluation/catalog/test_model_catalog_statuses.py` fails otherwise,
because unregistered values silently fall back to `pending` or `unknown`.
Use `blocked` with the concrete reason (license, checkpoint, dependency, API,
or hardware) in `notes` rather than an invented status.

## Evidence fields

Record what ran, not what is expected to work:

- `demo_parity` and `runner_parity` list `expected_artifacts` with a
  `video_probe` (width, height, frames, fps, duration) for the official and
  WorldFoundry runs of the same prompt, seed, checkpoint, and settings.
- Variant `integration.evidence` lists report paths under `tmp/`; artifacts,
  hashes, and logs stay outside Git and are referenced from the PR.
- Native diffusion variants may carry a `native_diffusion` block with
  `recipe_id`, `strategy`, components, `checkpoint_key_validation` counts, and
  `artifact_parity` (`pending` until the CUDA artifact comparison passes).
- Keep claims variant-scoped. A shared pipeline class, alias, or family does
  not extend evidence to another checkpoint, task, or GPU class.
