---
name: worldfoundry-inference-validation
description: >
  Use this skill when choosing or running validation for a WorldFoundry model,
  loader, kernel, or runtime change; when outputs drift or a numerical
  comparison fails; when adding a regression to a Make gate; or when judging
  whether inference evidence supports a claim. Covers evidence levels, the
  public CPU and tensor gates, CUDA contracts, A/B/A request isolation,
  component parity and first-divergence diagnosis, affected-case checkpoint
  replays, and the evidence report. Speed work is
  worldfoundry-inference-optimization; benchmark scoring is
  worldfoundry-evaluation.
license: Apache-2.0
compatibility: >
  Requires a WorldFoundry checkout and its Python environment. CUDA gates and
  checkpoint replays need a GPU host with accepted local weights and inputs.
---

# Validate WorldFoundry Inference

Select evidence proportional to the changed surface, run it, and label each
result by what it actually executed. Run commands from the repository root.
The [validation guide](../../docs/fumadocs/content/docs/reference/validation.mdx)
owns resources, accepted references, tolerances, and CI attestation; this
skill routes to the right gate. Before any check, name the failure it would
detect and what you would do if it failed. If neither has an answer, skip it.

## Evidence levels

| Evidence | Establishes | Does not establish |
| --- | --- | --- |
| Catalog and binding tests | The model ID resolves to the current pipeline and profile | That generation works |
| Mocked pipeline and runner tests | Loading options, input normalization, request forwarding, artifact contracts | Numerics |
| CPU tensor tests on small real components | Executed arithmetic and request-state handling | CUDA kernels or released weights |
| CUDA contracts | Selected kernels, streams, caches, and precision paths on the tested GPU | Accepted model outputs |
| Checkpoint replay against an accepted reference | Declared weights, inputs, and environment still produce accepted outputs and artifacts | Other variants, inputs, or full-demo quality |
| Benchmark execution | The configured evaluator's scores on the executed cohort | Other cohorts or settings |

Imports, mocks, `--print-config`, `--model-status`, and `plan_only` never count
as checkpoint-backed generation evidence.

## Select the gates

| Changed surface | Gates |
| --- | --- |
| Catalog, bindings, CLI, packaging, control plane | `make test`, `make lint`, `make cli-entrypoint-check` |
| CPU-testable tensor math, loaders, samplers, codecs, request state | Above plus `make test-infer-tensors` (`make test-infer-video-tensors` for the video subset) |
| Evaluation runners and metrics | `make test-evaluation`; `make test-eval-core` after installing `.[eval_core]` |
| PEFT loading, serializers, geometry exports | `make test-infer-contracts PYTHON=/path/to/model-env/bin/python` |
| CUDA kernels, graphs, attention, FP8, transport | `make test-infer-cuda-contracts PYTHON=/path/to/model-env/bin/python` on SM90+ |
| Model loading, preprocessing, inference, camera conversion, export | Affected-case checkpoint replays (below) |
| Matrix-Game 2 numerical contract | `make test-mg2-checkpoint-contracts`, `make test-mg2-trajectory-contracts`, `make test-mg2-conditioned-trajectory-contracts` with `MG2_CHECKPOINT_ROOT` on SM90+ |
| Wan2.1 LightVAE student | `make test-lightvae-checkpoint-contracts LIGHTVAE21_CHECKPOINT=/path/to/checkpoint` |

Weight-free gates never replace CUDA or checkpoint evidence for an inference
change. To register a new test in the right selection, read
[test selection](references/test-selection.md).

## Run the fast gates

Iterate on one file with `python -m pytest -q tests/<area>/test_<behavior>.py`,
then run the Make target. Make targets use a strict JUnit runner: they collect
an independent node manifest first and reject skips, collection errors,
missing, extra, or duplicate nodes, and stale reports. `TEST_ARGS` applies only
to execution, so filtering with `-k` or `-m` fails the gate.

```bash
make test
make test-infer-tensors
```

`make test-infer-tensors` hides CUDA and pins two CPU threads. CUDA targets
fail when no GPU is available instead of reporting a skipped suite.

## Cover state on reusable models

For pipelines, workers, caches, or decoders reused across requests, run
A/B/A: requests one and three identical, request two changing only the
intended conditioning. Also cover reset, a failed request followed by reuse,
and cancellation where the route exposes it. Compare eager, compiled, graph,
and offloaded paths on the same state and shapes. The first and third outputs
must match exactly unless the owning contract states a tolerance.

## Compare components and locate drift

For model ports and changed tensor math, compare deterministic component
outputs before the full sampling loop: aligned weights, inputs, timesteps,
dtype, and executed attention backend; FP32 small shapes first, deployment
precision second. When parity fails, read
[numerical parity](references/numerical-parity.md) and find the first divergent
boundary instead of explaining the final image. Keep captures under `tmp/`
and remove temporary hooks.

## Replay affected checkpoint cases

Commit the candidate first; replays and verification run against committed
source, and dirty files are never replayed.

```bash
make plan-inference PYTHON=python3 INFERENCE_BASE=<comparison-commit>
make coverage-inference PYTHON=python3
make replay-inference PYTHON=/path/to/model-env/bin/python \
  INFERENCE_PROFILE=/private/path/inference-regression.json
make verify-inference PYTHON=python3 \
  INFERENCE_PROFILE=/private/path/inference-regression.json \
  INFERENCE_REPORT=/private/path/completed-run/report.json
```

- The plan (`tmp/inference-regression-plan.json`) needs no GPU or weights. An
  `uncovered` result means unknown paths or unresolved imports; add component
  declarations to `tests/manual/geometry_regression_dependencies.json` instead
  of ignoring it. `no_inference_changes` still requires the CPU gates.
- The private host profile lives outside Git and pins the Python, Torch, CUDA,
  and assets used for the accepted references; a changed environment fails
  preflight rather than weakening the comparison.
- For one geometry case, run `tests/manual/geometry_regression.py run` for the
  reference worktree and the candidate, then `compare`; audit all affected
  cases with `make test-geometry GEOMETRY_REFERENCE=<dir> GEOMETRY_CANDIDATE=<dir>`.
  The validation guide lists the case environment variables.

## Report the evidence

Use this structure in the PR or review reply:

```markdown
### Validation
| Command | Result | Scope | Artifacts |
| --- | --- | --- | --- |
| make test-infer-tensors | pass (N nodes, 0 skipped) | CPU tensors | tmp/inference-cpu-contracts.xml |
| make replay-inference ... | pass, cases: <ids> | released weights, GPU <model> | /private/.../report.json |

Not run: <gate> — <missing resource or reason>.
Environment: GPU, driver, Python, Torch, CUDA; checkpoint revisions.
```

Label component-only versus public-loader coverage, and keep functional,
numerical, performance, and lifecycle evidence in separate rows.

## Gotchas

- Accepted references are never regenerated to make a candidate pass, and
  tolerances are never widened to hide drift. A changed accepted output needs
  its own documented acceptance.
- Exact comparison is the default. For nondeterministic GPU paths, measure the
  reference's self-run variation before proposing any tolerance.
- A seed set before model loading can be consumed by initialization; pin the
  inference seed after loading, as the WorldFM case does.
- `OMP_NUM_THREADS` and `MKL_NUM_THREADS` change CPU reconstruction numerics;
  pin them in replay profiles.
- `tests/manual/` and `tests/stream/` are excluded from collection. Run them
  directly (`PYTHONPATH=. python tests/manual/<script>.py`); they may download
  weights, use GPUs, call paid APIs, and write artifacts.
- Keep a failing test that exposes a defect in a supported integration.
  Delete tests only with the implementation or contract they cover.
- A shared pipeline class or alias does not extend replay coverage to another
  model ID, checkpoint, or mode.

## Done criteria

- Every executed gate is listed with node counts and zero unexpected skips.
- Inference changes have affected-case replays verified against accepted
  references, or the missing resource is stated as a blocker.
- Reusable-state changes have A/B/A, reset, and failure-reuse coverage.
- Results are labeled by scope; nothing unexecuted is described as passing.
