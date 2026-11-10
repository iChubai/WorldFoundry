# WorldFoundry Review Checklist

Read the sections that match the changed paths. Each item names the rule
source so a finding can cite it. Report an item only when the diff actually
violates it.

## Every change

- One focused surface; unrelated edits are split out (CONTRIBUTING.md, Pull
  Requests And Review).
- New functions, methods, and public request/result contracts are annotated;
  public APIs document inputs, outputs, exceptions, and shapes, units, camera
  conventions, or devices where relevant (CONTRIBUTING.md, Coding Conventions).
- Catalog discovery, CLI help, and package initializers stay lazy: importing a
  control-plane module does not load checkpoints, import heavy model stacks,
  or initialize CUDA.
- Inputs are validated before expensive loading; errors name the model ID and
  stage; workers, streams, and temporary resources are released on success
  and failure.
- No forwarding modules at abandoned paths; callers import the canonical
  implementation.
- New first-party Python files carry SPDX copyright and `Apache-2.0` headers;
  ported code keeps upstream notices; provenance is recorded in
  `THIRD-PARTY-NOTICES` and the owning catalog.
- No checkpoints, datasets, demo media, caches, credentials, or run outputs
  are committed. A change that adds, moves, or removes a module updates its
  package-map README.
- Comments explain ownership, constraints, or why a choice matters, not the
  history of a bug.

## Core infrastructure (`worldfoundry/core/`)

- Stays model-neutral: no family names, family branches, or checkpoint
  layouts.
- Code sits in its domain: geometry in `core.geometry`, execution and process
  caches in `core.execution`, logging and timing in `core.observability`,
  media codecs and streaming processing in `core.media`, plus the existing
  `nn`, `attention`, `io`, and `model_loading` packages.

## Native diffusion (`worldfoundry/base_models/diffusion_model/`)

- No reference to `worldfoundry.synthesis`, `pipelines`, `studio`,
  `operators`, or `evaluation`, including lazy imports and literal targets
  (`tests/base_models/diffusion_model/test_module_boundaries.py`).
- `models/networks/` holds tensor math only: no checkpoint loading, backend
  selection, schedulers, configs, or utilities.
- Recipes are declarative; components declare named checkpoint dependencies;
  new sampling loops are framework strategies in `runners/` without
  model-name branches; importing a module registers nothing.
- No family copy of attention dispatch, offload, distributed bootstrap,
  kernels, or checkpoint frameworks.

## Pipelines, operators, synthesis

- Pipelines load, execute, and return durable artifacts; operators own input
  validation, media loading, and control parsing; synthesis is infer-only and
  reuses `base_models` components instead of copying them.
- Decoded RGB results declare `video_chunk` layout and value range;
  `video_chunk: None` means no RGB.
- Hosted API keys come from environment variables; API clients are
  request-scoped.
- Reusable pipelines keep per-request RNG, latents, scheduler history, and
  caches isolated (A/B/A).

## Catalog and runtime data (`worldfoundry/data/`)

- Status values are registered in the owning `schema.py`.
- `integrated` and leaderboard flags are backed by executed evidence for the
  exact variant, task, cohort, and hardware; unexecuted combinations stay
  marked as such.
- Route strings resolve only through
  `worldfoundry/evals/models/pipelines/bindings.py`; benchmark routing
  comes from the catalog `runner` block, not handwritten tables.
- Public paths are Hugging Face repo IDs or `${WORLDFOUNDRY_CKPT_DIR}`, never
  private machine paths; `docs:` fields keep English and `_zh` pairs.

## Evaluation (`worldfoundry/evals/`)

- `api/*` `to_dict()`/`from_dict()` payloads, run file names, JSONL row shapes,
  `sample_id` alignment, and scorecard fields stay compatible, or the break is
  coordinated with validation, docs, and examples.
- Metric direction, aggregation, and definitions are explicit; failed and
  skipped samples stay visible in ledgers and denominators.
- Metrics consume `GenerationResult` artifacts and never call model pipelines.

## Studio (`worldfoundry/studio/`)

- Inference modules do not import Workspace, serving, or visualization
  packages (`tests/studio/test_module_boundaries.py`).
- Studio forwards inputs and does not seed or overwrite pipeline-owned memory;
  model cadence comes from the pipeline's `RealtimeSpec`.
- Leases, deadlines, cancellation, and process-group cleanup follow the
  dispatch contract on success, failure, and timeout.

## Tests

- Placed in the module-mirroring directory, named for behavior, marked
  `unit` or `gpu`, and registered in a Make selection.
- Required selections contain no skips; optional-resource tests live in
  optional suites or `tests/manual/`.
- Assertions check observable outputs or state; a test whose only evidence is
  that a mock was called does not support behavior claims.
- Tests for deleted contracts are removed; tests exposing supported defects
  are kept.

## Docs and skills

- English and Chinese Fumadocs pages change together and are listed in both
  `meta.json` files.
- Generated data is regenerated, not hand-edited.
- CLI changes update `docs/fumadocs/content/docs/reference/cli.mdx` and keep
  JSON output compatible.
- Skills that name a changed command, path, or contract are updated in the
  same change.

## Evidence

- Inference behavior changes carry affected-case replays against accepted
  references; references are not regenerated and tolerances are not widened.
- Performance numbers state hardware, workload, repeats, and the executed
  path, and keep startup separate from steady state.
- Skipped heavy stages are listed as blockers, not described as passing.
