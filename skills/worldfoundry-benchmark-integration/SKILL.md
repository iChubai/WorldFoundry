---
name: worldfoundry-benchmark-integration
description: >
  Use this skill when adding a benchmark to WorldFoundry, porting an official
  evaluator or judge, importing official-shaped results, or fixing benchmark
  readiness, task materialization, metric correctness, or scorecard
  normalization. Covers protocol capture, catalog and task manifests, runtime
  profiles, in-tree official runners and normalizers, dataset contracts,
  Benchmark Hub pages, and honest integration and leaderboard claims. Scoring
  a model with an existing benchmark is worldfoundry-evaluation; adding a
  model is worldfoundry-model-integration.
license: Apache-2.0
compatibility: >
  Requires a WorldFoundry checkout and its unified environment. Official runs
  need the declared datasets, judge weights, credentials, or simulators.
---

# Integrate a WorldFoundry Benchmark

Wire one benchmark's official protocol into the shared evaluation core so that
generated artifacts become a reviewable `scorecard.json`. Run commands from the
repository root in the unified environment. [CONTRIBUTING.md](../../CONTRIBUTING.md)
owns contribution rules; [Add a benchmark](../../docs/fumadocs/content/docs/guides/add-benchmark.mdx)
and [benchmarks and data](../../docs/fumadocs/content/docs/maintainers/architecture/benchmarks-data.mdx)
own the contracts. Integration evidence and leaderboard eligibility are
separate claims throughout.

## Capture the protocol

Record before editing:

- Upstream evaluator repository, pinned revision, license, and paper.
- Data: prompt suite or dataset, split, sample IDs, access, gating, and unsafe
  content flags; whether generated artifacts are the only input.
- Every metric: ID, direction, range and unit, aggregation, primary metric,
  leaderboard key, and any official normalization.
- Execution needs: judge or VLM weights, hosted API, simulator or Docker image,
  GPU, and the official result file format.
- The official cohort and denominator: how missing, failed, or skipped
  samples count in published numbers.

## Choose the closest existing contract

| Contract | Reference | Where to look |
| --- | --- | --- |
| Vendored official runtime scoring generated videos | `vbench` | `worldfoundry/data/benchmarks/catalog/video/vbench.yaml`, `worldfoundry/evals/tasks/execution/runners/vbench/` |
| Official runner over a local benchmark data root | `phyground` | `worldfoundry/evals/tasks/execution/runners/phyground/`, `tests/eval_core/test_phyground_official_runner.py` |
| Judge or hosted-API evaluator | `video-bench` | `worldfoundry/data/benchmarks/catalog/video/video-bench.yaml`, `worldfoundry/evals/tasks/execution/runners/videobench/` |
| Embodied simulator rollouts, official results imported | `libero` | `worldfoundry/data/benchmarks/catalog/embodied/libero.yaml`, [embodied runtime](../../docs/fumadocs/content/docs/evaluation/embodied-official-runtime.mdx) |

Match the execution contract, not the benchmark's topic. Reuse its runner,
normalizer, and test layout instead of creating a parallel evaluator.

## Workflow

```text
- [ ] 1. Manifests written and parsed
- [ ] 2. Task materialization proven
- [ ] 3. Runner or normalizer connected
- [ ] 4. Deterministic regressions added
- [ ] 5. Supported mode exercised and scorecard read
- [ ] 6. Status and Hub page match the evidence
```

### 1. Write the manifests

- Catalog entry `worldfoundry/data/benchmarks/catalog/<family>/<benchmark-id>.yaml`:
  identity and aliases, source provenance, dataset access, `runner` block,
  protocol metrics, evidence, blockers, and status fields.
- Task manifest `worldfoundry/data/benchmarks/tasks/external/<benchmark-id>.yaml`:
  splits, metric aggregation, and how samples become `GenerationRequest` rows.
- Runtime profile `worldfoundry/data/benchmarks/runtime_profiles/official/<benchmark-id>.yaml`:
  environment, required and optional env vars, assets, and preflight command.

Then confirm the zoo parsed what you meant; fix YAML before touching a GPU:

```bash
wf zoo benchmark-show --benchmark-id <benchmark-id> --include-spec --json
wf zoo benchmark-specs --benchmark-id <benchmark-id> --json
wf tasks show --task-type <benchmark-id> --json
wf task validate worldfoundry/data/benchmarks/tasks/external/<benchmark-id>.yaml --json
```

### 2. Prove task materialization

Fix sample IDs, splits, and input keys before any generation:

```bash
wf task materialize <task-name> --benchmark <benchmark-id> \
  --dataset-manifest <dataset_manifest.json> --num-samples 8 \
  --output-jsonl tmp/benchmark_integration/<benchmark-id>/requests.jsonl --json
```

Inspect stable `sample_id` values, resolved input paths, and expected artifact
keys. Every request needs a unique `sample_id`; multiple seeds use distinct IDs.

### 3. Connect the runner or normalizer

- Runtime execution lives in
  `worldfoundry/evals/tasks/execution/runners/<benchmark>/`. Vendored
  official source stays under that runner's `runtime/` directory with its
  license and is recorded in `THIRD-PARTY-NOTICES`.
- Normalize vendor output with the shared framework in
  `worldfoundry/evals/tasks/execution/framework/` (`result_normalizer.py`,
  `official_result_scoring.py`) so metrics land in `metrics/summary.json`.
- Routing is derived from the catalog `runner` block and contracts in
  `worldfoundry/evals/tasks/contracts/external.py`. Never add a handwritten
  runner or dispatch table.
- Benchmark scoring consumes `GenerationResult` artifacts. Never call a model
  pipeline from inside a metric, and never teach a model the benchmark's
  dimensions.

### 4. Add deterministic regressions

Use small fixtures with known values to test parsing, materialization, metric
formulas, aggregation, normalization, and visible failed or skipped samples.
Assert numbers, not the presence of metric keys. Place them with the closest
existing benchmark tests in `tests/eval_core/` or `tests/evaluation/`, and add
CPU contracts to the Make selection described in
[test selection](../worldfoundry-inference-validation/references/test-selection.md).

### 5. Exercise the supported mode

```bash
wf preflight runtime --profile <benchmark-id> \
  --manifest worldfoundry/data/benchmarks/runtime_profiles/official/<benchmark-id>.yaml \
  --output-dir tmp/preflight/runtime/<benchmark-id> --json
wf zoo benchmark-run --benchmark-id <benchmark-id> --mode official-validation \
  --official-results-path <official_results.json> \
  --generated-artifact-dir <generated_artifacts> \
  --output-dir tmp/benchmark_zoo/official_validation/<benchmark-id> --json
```

| Mode | Proves |
| --- | --- |
| `contract` | Wiring and output contracts; never scoring evidence |
| `normalizer` | Supplied results convert to WorldFoundry artifacts; the official scorer did not run |
| `official-validation` | Official-shaped results import through the runner; integration evidence |
| `official-run` | The declared evaluator executed on prepared data, weights, and services |

Add `--limit N` for a bounded smoke and `--benchmark-data-root` for local data.
Read the validity fields in `scorecard.json`: `score_valid`,
`leaderboard_valid`, and blocking reasons (under `eligibility` in shared
scorecards), plus `normalizer_only` and `official_benchmark_verified` where
the official runner writes them. Check that failed and skipped samples remain
visible. Exit code 0 only means the command completed.

### 6. Align status and docs with the evidence

- `integration.status` is `planned`, `blocked`, or `integrated`; `maturity` is
  `verified_runner`, `contract_ready`, `planned`, or `blocked`. Values outside
  `worldfoundry/evals/tasks/catalog/schema.py` degrade with a warning.
- A normalizer-only integration keeps `verification_status: normalizer_only`,
  `official_benchmark_verified: false`, and `leaderboard_valid: false` until
  the full official scorer or judge has run on the declared cohort.
- Write paired Benchmark Hub pages
  `docs/fumadocs/content/docs/evaluation/benchmark-hub/<benchmark-id>.mdx` and
  `.zh.mdx`, list them in both `meta.json` files, embed
  `<BenchmarkPageSections id="<benchmark-id>" locale="en" />`, and refresh
  generated data with `npm --prefix docs/fumadocs run benchmarks:generate` and
  `npm --prefix docs/fumadocs run coverage:generate`. Curated copy belongs in
  `docs/fumadocs/lib/benchmark-page-overrides.json`, not the generated JSON.

## Gotchas

- Hub page IDs, catalog IDs, and task manifest names must agree;
  `make test-docs-contracts` checks them together with documented commands.
- `make lint` runs `workspace-registry-check`, which rejects runnable video
  catalog entries missing from the derived runner registry, handwritten runner
  tables, markdown other than READMEs under `worldfoundry/evals/`, and
  runner `requirements.txt` files; pip requirements belong in
  `worldfoundry/data/benchmarks/requirements/`. Benchmark documentation lives
  in Fumadocs.
- A bounded run on one dimension or `--limit` subset is integration evidence,
  not full-suite validity. Keep `leaderboard_valid: false` and record the
  scope, as `vbench` does for its single validated dimension.
- Successful samples alone do not make a valid score. Missing outputs follow
  the benchmark's published denominator policy and stay visible in the ledger.
- Do not substitute an approximation (for example CLIPScore for a VLM judge)
  under the official metric ID. Use a distinct ID and say so on the Hub page.
- Credentials come from environment variables declared in the runtime
  profile. Datasets, judge weights, simulator images, and generated videos stay
  outside Git; reference them by path and hash.

## Done criteria

- `benchmark-show`, `benchmark-specs`, `tasks show`, and `task validate` agree
  on the ID, metrics, and blockers.
- Materialized requests have stable sample IDs before generation.
- Deterministic metric and normalizer tests assert known values and are in a
  Make selection.
- The supported mode wrote `scorecard.json`; its validity fields match the
  catalog claims, and failed or skipped samples are visible.
- Paired Hub pages and generated docs data are current
  (`make docs-check`, `make test-docs-contracts`).
- The PR lists exact commands, data access conditions, artifacts, and any
  unexecuted official stages with their blockers.
