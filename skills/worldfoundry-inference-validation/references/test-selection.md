# Test Placement and Gate Selection

Read this when adding, moving, or deleting a test, or when a Make gate
reports a missing, extra, or skipped node. The `Makefile` and
`[tool.pytest.ini_options]` in `pyproject.toml` are the sources of truth for
selections; [CONTRIBUTING.md](../../../CONTRIBUTING.md) owns markers and naming.

## Place the test

- Mirror the owning module: `tests/base_models/`, `tests/synthesis/`,
  `tests/pipelines/`, `tests/operators/`, `tests/core/`, `tests/runtime/`,
  `tests/cli/`, `tests/scripts/`, `tests/studio/`, `tests/studio_visualization/`,
  `tests/evaluation/`, or `tests/eval_core/`. Cross-layer contracts go in
  `tests/integration/`; the `tests/` root holds only shared setup.
- Name files `test_<behavior>.py`, using the model name where relevant, never
  an issue ID or review round.
- Mark weight-free tests `unit` and CUDA tests `gpu`, usually with a
  module-level `pytestmark = pytest.mark.unit`.
- Test current public bindings, supported inputs, and observable outputs.
  Remove tests for deleted implementations and replace assertions about
  superseded internal paths with the current contract.
- Scripts that run inference, download weights, or call provider APIs at
  import time go in `tests/manual/` or `tests/stream/`, which pytest does not
  collect. Read checkpoint locations from environment variables.

## Register it in a selection

| Make target | Selection source | Runs in | Put here |
| --- | --- | --- | --- |
| `make test` | `testpaths` in `pyproject.toml` plus `PUBLIC_GATE_CONTRACTS` | Torch-free `.[test]` environment (CI `cpu-tests`) | Catalog, CLI, packaging, docs guidance, Studio lifecycle with fakes |
| `make test-infer-tensors` | `INFER_TENSOR_CONTRACTS`, which includes `VIDEO_TENSOR_CONTRACTS`, `STREAMING_CPU_CONTRACTS`, `DIFFUSION_ACCELERATION_CPU_CONTRACTS`, and `EVALUATION_CPU_CONTRACTS` | CPU Torch, safetensors, einops; CUDA hidden; `not gpu` nodes (CI `inference-tensors`) | Small real components, samplers, codecs, request isolation |
| `make test-infer-video-tensors` | `VIDEO_TENSOR_CONTRACTS` | Same as above | Video component numerics |
| `make test-evaluation` | `EVALUATION_CPU_CONTRACTS` | CPU Torch | Evaluation metric and runner contracts |
| `make test-infer-contracts` | `INFER_TENSOR_CONTRACTS` plus geometry export tests | Prepared model environment | PEFT, serializer, and geometry export contracts |
| `make test-infer-cuda-contracts` | `INFER_CUDA_CONTRACTS`, as explicit node IDs | SM90+ GPU with Triton and cuDNN frontend | CUDA kernels, graphs, attention, FP8, transport |
| `make test-docs-contracts` | Docs and catalog contract tests | Public CPU environment | Documented commands and catalog claims |
| `make test-eval-core` | All of `tests/eval_core/` through plain pytest | `.[eval_core]` | Extended evaluation contracts |
| `make test-realtime-client` | Node tests in `tests/studio_visualization/` | Node.js | Browser realtime client |

A test that is in no selection never runs in CI. Torch imports in a
`make test` node break the Torch-free public gate; move such tests to a
tensor selection. CUDA nodes are listed individually because older CUDA tests
do not all carry the `gpu` marker.

## Strict runner behavior

Targets built on `tests/manual/validate_junit_contract.py` first collect an
independent manifest, then require every collected node to execute and pass.

- A skip, collection error, missing or extra node, duplicate, changed source,
  or stale report fails the gate. Optional resources must not be skipped
  inside a required selection; put such tests in an optional suite instead.
- `TEST_ARGS` affects execution only. Selecting a subset with `-k` or `-m`
  makes the executed set differ from the manifest and fails.
- Reports land in `tmp/` (for example `tmp/public-cpu-contracts.xml` and
  `tmp/inference-cpu-contracts.xml`); CI uploads them as evidence.

## CI mapping

| CI job | Runs |
| --- | --- |
| `cpu-tests` | `make test`, `make test-docs-contracts` |
| `inference-tensors` | `make test-infer-tensors`, `make test-realtime-client` |
| `public-surface` | Docs build, `make docs-check`, `make lint`, `make compile-eval`, `make cli-check`, `make cli-entrypoint-check` |
| `packaging-license-gate` | Clean wheel and sdist build, `make packaging-check` |
| `geometry-impact`, `inference-replay` | Affected-case plan and verification of trusted GPU evidence |
| `Python quality`, `DCO` | Ruff and ty on changed Python files; sign-off check |

CUDA contracts and checkpoint replays have no hosted runner; run them on a
prepared GPU host and attach their reports.
