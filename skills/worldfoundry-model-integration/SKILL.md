---
name: worldfoundry-model-integration
description: >
  Use this skill when adding or migrating a model into WorldFoundry: a new
  model or variant, a checkpoint port or key conversion, an upstream inference
  implementation, or wiring a model into the catalog, pipeline bindings,
  runtime profiles, CLI, and evaluation runner. Covers choosing the
  integration lane, canonical native-diffusion roles, checkpoint mapping and
  component parity, operators and pipelines, catalog status claims, and the
  evidence a first PR needs. Running an existing model is
  worldfoundry-model-inference; Studio streaming and serving are
  worldfoundry-studio-serving; replay gates are
  worldfoundry-inference-validation.
license: Apache-2.0
compatibility: >
  Requires a WorldFoundry checkout and its unified Python environment.
  Parity runs and real generation need local checkpoints and a CUDA GPU.
---

# Integrate a WorldFoundry Model

Port the smallest upstream *inference* slice into canonical owners, prove it
numerically, then expose one public route. Run commands from the repository
root in the unified environment. [CONTRIBUTING.md](../../CONTRIBUTING.md) owns
contribution rules; [Add a model](../../docs/fumadocs/content/docs/guides/add-model.mdx),
[model runtime](../../docs/fumadocs/content/docs/maintainers/architecture/model-runtime.mdx),
and [native diffusion](../../docs/fumadocs/content/docs/maintainers/architecture/native-diffusion.mdx)
own the contracts applied here.

## Gather the contract

Record these before editing; they become the PR's Scope and Runtime sections.

- Upstream repository, pinned revision, and license. Checkpoint repo IDs,
  revisions, layout (single file, sharded safetensors index, or Diffusers
  folders), and roles such as `dit`, `vae`, `text-encoder`, and `tokenizer`.
- Tasks; inputs (prompt, image, video, camera, actions); output artifact kind;
  native resolution, frames, and fps; sampler, steps, guidance; dtype; VRAM.
- The official inference command, plus one input, seed, and raw output that
  can be rerun as the parity reference.
- The closest in-tree integration by runtime contract, not by model name.

Integrate one task and one checkpoint variant first. Extend to other tasks and
variants after that route passes.

## Choose the integration lane

Native diffusion roles live under `worldfoundry/base_models/diffusion_model/`.

| Lane | Choose when | Owning code | Reference |
| --- | --- | --- | --- |
| Native diffusion, existing graph | Network, codec, and sampler already exist for the family | Recipe in `recipes/<family>.py`; thin public pipeline subclass | `wan2.1-t2v-1.3b` in `recipes/wan.py` and `worldfoundry/pipelines/wan/pipeline_wan_2p1_t2v.py` |
| Native diffusion, new components | Checkpoint-visible network, encoder, codec, or latent-initializer differences | `models/networks/<family>/`, `models/denoisers/<family>.py`, `models/encoders/`, `models/autoencoders/`, `models/initializers/` | `recipes/hunyuan_video.py`, `recipes/ltx.py` |
| Native diffusion, new sampling loop | No registered execution strategy implements the algorithm | Framework strategy in `runners/`, selected by the recipe's `ExecutionSpec` | `runners/joint_chunked.py`, `runners/compressed_memory.py` |
| Local non-diffusion runtime | 3D/geometry, perception, or LLM/MLLM inference | Family under `worldfoundry/base_models/three_dimensions/`, `perception_core/`, or `llm_mllm_core/`, plus pipeline and operator | `worldfoundry/base_models/three_dimensions/depth/depth_anything/runtime_v3.py`, `worldfoundry/pipelines/pi3/` |
| Embodied action policy | VLA/VA policy driven by simulator rollouts | Callable config in `worldfoundry/data/models/runtime/configs/vla_va_wam/`; `predict_action()` in `synthesis/action_generation/<model>/runtime.py` | `worldfoundry/synthesis/action_generation/openvla/runtime.py` |
| Hosted API | A provider is the only execution surface | Wrapper under `worldfoundry/pipelines/<provider>/` plus operator; credentials from environment variables | `worldfoundry/pipelines/kling/pipeline_kling_api.py` |

State the lane in the PR. A new network or sampling strategy is a much larger
change than a recipe over an existing graph.

## Workflow

```text
- [ ] 1. Reference frozen
- [ ] 2. Checkpoint mapping proven
- [ ] 3. Component parity proven
- [ ] 4. Implementation placed in owning layers
- [ ] 5. Public route wired
- [ ] 6. Route verified without weights
- [ ] 7. Real requests compared with the reference, then status promoted
- [ ] 8. Regressions and replay case registered
- [ ] 9. Docs updated
```

### 1. Freeze the reference

Run the official command at the pinned revision in its own environment,
outside `worldfoundry/`. Keep its input, seed, raw output, and log under
`tmp/`. Upstream code is a migration source: production modules never import
it from a checkout or cache.

### 2. Prove checkpoint mapping before activations

Declare checkpoint roles and let the canonical loaders resolve them
(`worldfoundry/base_models/diffusion_model/loaders/checkpoints.py` for native
diffusion, `worldfoundry/core/model_loading/` otherwise). Express renames with
`remap_checkpoint_keys()` from `worldfoundry/core/model_loading/checkpoints/remap.py`,
build the module on `meta`, and run `validate_state_dict_compatibility()` from
`worldfoundry/core/model_loading/checkpoints/assignment.py`: it rejects
missing, unexpected, and shape-mismatched keys without allocating weights.
List intentionally ignored source tensors explicitly. Prefer the native
release over a Diffusers port when both exist; native keys usually need fewer
renames. Then compare mapped tensor values. Read
[numerical parity](../worldfoundry-inference-validation/references/numerical-parity.md)
for fused QKV, modulation order, latent scaling, and sidecar checks.

### 3. Prove component parity

Compare the denoiser, condition encoders, latent codec, and scheduler with the
same weights and inputs: FP32 small shapes first, then deployment dtype and
shape. Execute real forwards and compare values; loading success and matching
shapes are not parity. On a mismatch, locate the first divergent boundary
before investigating later stages.

### 4. Place the implementation

- Native diffusion: one immutable `NativeDiffusionRecipe` per public model ID,
  registered in the lazy table of `recipes/registry.py` and exported from
  `recipes/__init__.py`. Components declare named checkpoint dependencies and
  never search directories. A new loop is a framework strategy in `runners/`;
  model code cannot supply its own runner.
- Attention, normalization, RoPE, placement, offload, distributed, and
  checkpoint primitives come from `worldfoundry/core/`. Never add a family copy.
- Orchestration not owned by the native framework goes in
  `worldfoundry/synthesis/`, which is infer-only: no training code, launch
  scripts, demo media, or copies of `base_models` components.
- Input validation, media loading, and camera/action parsing belong to an
  operator in `worldfoundry/operators/`. `NativeVisualDiffusionPipeline` in
  `worldfoundry/pipelines/native_diffusion.py` already wires the shared
  `RuntimeVideoOperator`; add a family operator only for model-specific
  controls.
- Keep upstream copyright headers in ported files. New first-party Python
  files carry SPDX `Apache-2.0` headers. Record provenance in
  `THIRD-PARTY-NOTICES` and the catalog `source` block.

### 5. Wire the public route

Create files in dependency order, so no manifest names a target that does not
import:

1. Pipeline under `worldfoundry/pipelines/<family>/`. `from_pretrained()` loads
   and fails loudly on a missing checkpoint; `__call__`/`process()` returns
   durable artifact paths; `stream()` exists only for incremental output.
   Native video models subclass `NativeTextToVideoPipeline` (the Wan I2V
   pipelines do too) and set `MODEL_ID` and defaults, as
   `pipeline_wan_2p1_t2v.py` does.
2. Binding in `worldfoundry/data/models/bindings/pipelines/`.
3. Runtime profile in `worldfoundry/data/models/runtime/profiles/`; a dedicated
   environment in `worldfoundry/data/models/runtime/environments/` only for a
   real dependency conflict; presets in `worldfoundry/data/models/runtime/configs/<model>/`.
4. Catalog entry in `worldfoundry/data/models/catalog/<category>/` last.

Read [route manifests](references/route-manifests.md) for minimal templates,
status vocabularies, and evidence fields.

### 6. Verify the route without weights

```bash
wf zoo model-show --model-id <model-id> --include-manifest --json
wf zoo model-specs --model-id <model-id> --json
wf zoo model-download --model-id <model-id> --check-local --json
wf run <model-id> --model-status
```

Check aliases, checkpoint roles, runtime profile, and the resolved binding.
`--model-status` (an alias of `--print-config`) never loads weights. While
the catalog says `planned`, it reports `"runner_entry_kind": "runner_candidate"`
and a `blocked_reason`; that is expected until step 7 evidence exists.

### 7. Run real requests, then promote

Before promotion, `wf run <model-id>` refuses the entry. Exercise the same
runner through the evaluation core, which accepts a runner candidate.
`inputs` carries the prompt and media; `generation_kwargs` reaches the
pipeline call:

```bash
mkdir -p tmp/model_integration/<model-id>
printf '%s\n' '{"sample_id": "ref-0001", "task_name": "text-to-video", "inputs": {"prompt": "<reference prompt>"}, "generation_kwargs": {"seed": 42}}' \
  > tmp/model_integration/<model-id>/requests.jsonl
wf evaluate --mode model --model-id <model-id> \
  --model-manifest-dir worldfoundry/data/models/catalog \
  --requests-path tmp/model_integration/<model-id>/requests.jsonl \
  --metric artifact_count --output-dir tmp/model_integration/<model-id>/eval --json
```

Use the step 1 input, seed, shape, steps, and precision, and compare the
artifact with the reference: frame count, fps, resolution, and content.
`artifact_count` only proves a file appeared. Once this comparison passes and
the step 8 regressions exist, set `integration.status: integrated` on the
validated variant; then `wf run <model-id> --model-status` must report
`"runnable": true` and `"runner_entry_kind": "runnable_runner"`, and the
public command from [model inference](../worldfoundry-model-inference/SKILL.md)
runs it.

### 8. Register regressions

- Put focused tests in the module-mirroring directory under `tests/`, marked
  `unit` or `gpu`, and add each to its Make selection as described in
  [test selection](../worldfoundry-inference-validation/references/test-selection.md).
  Test resolution, loading options, input validation, request forwarding, and
  output artifacts through the public loader.
- A native video or world-model variant needs a short replay case in
  `tests/manual/geometry_regression_cases.json` and component declarations in
  `tests/manual/geometry_regression_dependencies.json`. Copy the
  `wan21-t2v-1p3b-short` case shape: A, B, invalid request, repeat of A.
  CI rejects new uncovered native variants; check with
  `make coverage-inference PYTHON=python3`.
- Accept real-weight numerics with [inference validation](../worldfoundry-inference-validation/SKILL.md).

### 9. Document

Maintain paired pages `docs/fumadocs/content/docs/guides/supported-models/<model-id>.mdx`
and `.zh.mdx`. `npm --prefix docs/fumadocs run models:generate` seeds a page
from the catalog `docs:` block; set `pageSource: authored` before editing it by
hand. Verify with `npm --prefix docs/fumadocs run models:check` and
`npm --prefix docs/fumadocs run models:homes:check`.

## Gotchas

- A runtime profile is not a route. `wf run` treats an entry as
  `runnable_runner` only with a `runner_target` and `integration.status:
  integrated`; a runner target with another status is `runner_candidate`, and
  no runner target is `listed_only`. The generic pipeline runner also needs a
  `pipeline_binding` that imports.
- Route strings are parsed only by
  `worldfoundry/evals/models/pipelines/bindings.py`. Do not add a second
  parser in catalog loading, resolvers, or check helpers.
- Catalog statuses must be canonical values or aliases registered in
  `worldfoundry/evals/models/catalog/schema.py`;
  `tests/evaluation/catalog/test_model_catalog_statuses.py` fails on anything
  else. Model `integration.status` is `planned`, `blocked`, or `integrated`.
- `tests/base_models/diffusion_model/test_module_boundaries.py` rejects any
  canonical diffusion reference to `worldfoundry.synthesis`, `pipelines`,
  `studio`, `operators`, or `evaluation`, including lazy imports and literal
  `"module:Class"` strings.
- `models/networks/` holds checkpoint-shaped tensor math only. It cannot load
  checkpoints, choose backends, or contain pipeline, config, scheduler, or
  utils packages. Importing a network must not register or load a model.
- A lazy registry entry whose module is missing fails only when that model ID
  is resolved. Resolve every new ID once:
  `python -c "from worldfoundry.base_models.diffusion_model.recipes import default_native_diffusion_registry as r; r().resolve('<model-id>')"`.
- A seed alone does not reproduce initial latents when preprocessing consumes
  RNG. Compare or inject the same noise and conditioning tensors.
- Declare decoded output with `video_chunk` layout and value range instead of
  letting consumers infer pixels from observed values. `video_chunk: None`
  means the result has no RGB output.
- Control-plane imports stay lazy: catalog discovery, CLI help, and Studio's
  AST catalog scan must not import Torch, initialize CUDA, or load weights.
- Hosted API keys come only from environment variables, never manifests,
  tests, docs, or reports.
- Checkpoints, demo media, and generated outputs stay outside Git. Local demo
  inputs live under `testcase/` and are referenced by explicit paths.

## Done criteria

- Reference command, input, seed, and artifact recorded outside Git.
- Checkpoint bijection and component parity executed with tensor values.
- Code placed in the lane's owners; boundary test passes.
- Real requests match the reference within the owning acceptance contract;
  only then does `wf run <model-id> --model-status` report a runnable route.
- Regressions registered in Make selections; replay case declared for native
  video/world variants.
- Catalog status and evidence fields describe only the executed scope;
  untested variants, tasks, and hardware remain marked as such.
- Paired model pages updated and generated docs checked.
