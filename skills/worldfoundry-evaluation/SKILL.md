---
name: worldfoundry-evaluation
description: >
  Use this skill when evaluating models or generated outputs with
  WorldFoundry: running a model on a benchmark, planning or resuming a
  model-by-benchmark matrix, scoring a directory of generated videos with a
  benchmark protocol, scoring result rows with metric IDs, reproducing a
  paper configuration from a recipe, importing official results, running
  embodied closed-loop evaluations, or reading and comparing scorecards.
  Covers command choice, plan-first execution, benchmark modes, resume and
  generation cache, run artifacts, and score versus leaderboard validity.
  Adding or fixing a benchmark integration is
  worldfoundry-benchmark-integration; running a model without scoring is
  worldfoundry-model-inference.
license: Apache-2.0
compatibility: >
  Requires a WorldFoundry checkout with the unified environment. Benchmarks
  may need datasets, judge weights, API credentials, simulators, or a GPU, as
  declared on their Benchmark Hub pages.
---

# Evaluate with WorldFoundry

The CLI, TUI, Workspace, and MCP compile every evaluation intent onto the same
core and its shared run artifacts. The scorecard's validity fields, not the
exit code, decide what a number can claim. Run commands from the repository
root with the unified environment active. The
[evaluation guide](../../docs/fumadocs/content/docs/evaluation/index.mdx), the
[CLI reference](../../docs/fumadocs/content/docs/reference/cli.mdx), and each
benchmark's Hub page under `docs/fumadocs/content/docs/evaluation/benchmark-hub/`
own the details.

## Choose the command

| Intent | Command |
| --- | --- |
| Run an in-tree model under a benchmark's protocol | `wf run --model <model-id> --benchmark <benchmark-id>` |
| Score your own generated artifacts with a benchmark's full protocol | `wf score --benchmark <benchmark-id> --artifacts <dir>` |
| Score WorldFoundry result rows with metric IDs | `wf score --results <results.jsonl> --metric <metric-id>` |
| Generate from a dataset manifest, then score metric IDs | `wf generate-score --model <model-id> --dataset-manifest <file> --metric <metric-id>` |
| Reproduce a pinned model-by-benchmark configuration | `wf reproduce --recipe <recipe.yaml>` |
| Score materialized requests or results through the low-level core | `wf evaluate --mode existing-results --results-path <results.jsonl>` |
| Import official-shaped results as framework evidence | `wf zoo benchmark-run --benchmark-id <benchmark-id> --mode official-validation` |
| Simulator closed-loop rollouts | `wf embodied plan`, then `wf embodied run` |
| Index, compare, or validate finished runs | `wf index-runs`, `wf compare-runs`, `wf validate-artifact` |

Embodied evaluations take a config, assets, and often a Docker image; follow
[embodied official runtime](../../docs/fumadocs/content/docs/evaluation/embodied-official-runtime.mdx).

## Workflow

### 1. Discover the protocol and its needs

```bash
wf zoo benchmarks --ready-now
wf zoo benchmark-show --benchmark-id <benchmark-id> --include-spec --json
wf metric list
wf suites list
```

Read the benchmark's Hub page for data roots, environment variables, metric
subsets, artifact naming, and judge credentials before preparing assets.

### 2. Plan before executing

`run`, `score`, `generate-score`, and `reproduce` accept `--plan-only`. The
plan writes `prepared_evaluation.json` with the intent classification,
resolved request, configuration sources, and preflight diagnostics. Remove
`--plan-only` only after it reports `ready: true`.

```bash
wf run --all-benchmarks --model <model-id> --plan-only \
  --output-dir tmp/evaluation/plan --json
```

### 3. Execute one cell, then the matrix

```bash
wf run --model <model-id> --benchmark <benchmark-id> --mode official-run \
  --output-dir tmp/evaluation/<model-id>/<benchmark-id> --json
```

- Repeat `--model` or `--benchmark`, or use `--suite <suite-id>`, for a matrix.
- `--resume` reuses completed cells, generation, and metric batches whose
  inputs still match; keep the same output directory.
- For deterministic generation, add `--generation-cache-dir tmp/generation-cache
  --generation-cache-mode read-write`; hits are recorded in the run manifest.
- `--model-worker-device` pins model workers to disjoint, equal-size GPU groups.
- `--fail-on-sample-error` makes any failed sample fail the command, as CI
  jobs usually require.

| `--mode` | Meaning |
| --- | --- |
| `official-run` | Executes the declared official evaluator; requires its data, weights, and services |
| `official-validation` | Imports official-shaped results; integration evidence, not a leaderboard run |
| `normalizer` | Converts supplied results; the official scorer did not run |
| `contract` | Wiring check only; never scoring evidence |

### 4. Read the artifacts

| File | Check |
| --- | --- |
| `scorecard.json` | `score_valid`, `leaderboard_valid`, and blocking reasons (under `eligibility` in shared scorecards); `normalizer_only` where official runners write it; `provenance.claim.level` when the run records provenance |
| `metrics/summary.json` | Aggregates, failed and skipped IDs, metric definitions |
| `sample_ledger.jsonl` | Per-sample status with stage errors; nothing hidden |
| `results.jsonl` | Each sample's artifacts, status, and errors |
| `run_manifest.json` | Command identity, status, counts, paths, cache evidence |
| `suite_manifest.json`, `suite_report.md` | One row per matrix cell |

Validate them before reporting:

```bash
wf validate-artifact tmp/evaluation/<model-id>/<benchmark-id>/scorecard.json --strict --json
```

### 5. Compare runs

```bash
wf index-runs tmp/evaluation --output-dir tmp/evaluation/index
wf compare-runs --index tmp/evaluation/index/index.json --require-score-valid \
  --metric <metric-id> --output-md tmp/evaluation/compare.md
```

Comparisons reject different metric definitions; winners require valid scores,
complete coverage, and a known metric direction.

## What a number can claim

- `score_valid` requires zero failed samples, at least one valid metric, and
  no invalid metrics. `leaderboard_valid` additionally requires complete
  metric coverage, an official evidence gate, comparable provenance, and a
  non-mock backend.
- `provenance.claim.level` is `exact_reproduction` (pinned generation,
  official data and evaluation, pinned runtime and reference),
  `benchmark_comparable` (official data and evaluation), or `diagnostic`.
  Model-by-benchmark plans are capped by the benchmark's checked-in evidence;
  caller flags can only narrow a claim.
- A run limited with `--num-samples`, `--limit`, or a dimension subset is a
  bounded run, whatever its scores.

## Gotchas

- Exit code 0 means the operation completed under its contract; 1 means a
  validation, metric, sample, or run failure; 2 means invalid input, missing
  files or environment, or failed preflight.
- `score --benchmark --artifacts` runs the official protocol and rejects
  `--resume` and `--metric-batch-size`; those apply to in-process scoring.
- Every request needs a unique `sample_id`; multiple seeds use distinct IDs.
  Imported results need an explicit `status`; a missing one is `unknown`.
- VBench's standard cohort has five videos per prompt, so a sample limit
  counts videos, not prompts.
- Changing official steps, CFG, resolution, or enabling lossy caches creates a
  different workload; label such runs as custom, not official.
- `--task-type`, `--benchmark-name`, and `--data-path` are retired flags that
  always fail; use `--benchmark`/`--model` or `wf task materialize`.
- Put `--json` on the leaf command and repeat list flags instead of
  comma-joining, except where `--help` documents commas.
- Credentials come from environment variables. Datasets, generated media, and
  run directories stay outside Git.

## Report

```markdown
| Model | Benchmark | Mode | Samples (ok/failed/skipped) | Primary metric | score_valid | leaderboard_valid | Claim level |
| --- | --- | --- | --- | --- | --- | --- | --- |

Command: <exact command>
Output: <output-dir> (scorecard.json, sample_ledger.jsonl)
Blocking reasons: <eligibility.blocking_reasons>
```
