# Contributing

Thanks for helping improve WorldFoundry. Keep changes scoped, document public
behavior, and prefer manifests, task YAML, runtime profiles, and public CLI
commands over private scripts.

The detailed maintainer guide lives in
[`docs/fumadocs/content/docs/maintainers/contributing.mdx`](docs/fumadocs/content/docs/maintainers/contributing.mdx).

## Before Opening A Pull Request

- Keep the change scoped to one pipeline, benchmark path, CLI surface, or docs area.
- Do not commit checkpoints, generated videos, local caches, API keys, tokens, or credentials.
- State any GPU, API quota, checkpoint, simulator, official repo, or gated-data requirement.
- Run the public validation commands relevant to the changed surface.
- Run `bash scripts/docs/build.sh --skip-bootstrap` when docs or README links change.
- Include scorecard, preflight, or validation evidence for any readiness claim.

## Change Checklists

### Add Or Update A Model

- Update `data/models/catalog/` and, when needed, `data/models/runtime_profiles/`.
- Declare aliases, source status, integration status, auth, checkpoint refs, license, and blockers.
- Add or reuse a public adapter target and a minimal validation command before claiming `integrated`.
- Keep API/GPU/checkpoint requirements out of quickstart defaults.
- Verify with `worldfoundry-eval zoo models --json` and the smallest documented model smoke run.

### Add Or Update A Benchmark

- Update `data/benchmarks/catalog/` and task YAML under `data/benchmarks/tasks/external/` when applicable.
- Declare dataset refs, auth, unsafe/gated data, official repo, simulator, metrics, and blockers.
- Add a contract adapter or normalizer before claiming `contract_ready`.
- Add official runtime evidence before claiming leaderboard validity.
- Verify with `worldfoundry-eval zoo benchmarks --json` and the documented readiness checks.

### Add Or Update A Metric

- Add deterministic validation examples for metric correctness.
- Document whether higher is better, output units, primary metric status, and leaderboard mapping.
- Ensure scorecards expose the metric under stable keys.

### Add Or Update Docs

- Keep first-run docs CPU-only, no-download, no-token, and no-GPU.
- Link complex environment, official runtime, and API paths to reference docs.
- Run the docs build and docs checks.

### Add Or Update A CLI Argument

- Update `docs/fumadocs/content/docs/reference/cli.mdx` when public behavior changes.
- Preserve JSON output compatibility or document the breaking change.
- Run `make docs-check` and document help, error handling, and no-heavy-import behavior.

## Synthesis Is Infer-Only

`worldfoundry/synthesis/**` packages upstream **inference** runtimes only. Do not add:

- training launchers, dataset builders, finetune scripts, or benchmark-only tooling
- demo media, generated outputs, checkpoints, or downloaded datasets
- notebooks, README files, or other documentation inside the synthesis tree

Put reusable demo assets under `worldfoundry/data/test_cases/`.

Before release or large runtime imports, run the public quality gates:

```bash
make lint
make docs-check
```

On a fresh CPU-only environment, install the release-gate Torch wheels from
the official CPU index before installing the normal `[eval_core]` extra:

```bash
python -m pip install --index-url https://download.pytorch.org/whl/cpu \
  "torch>=2.7,<2.12.0" "torchvision>=0.22,<0.27.0"
python -m pip install -e ".[eval_core]"
make test-eval-core
```

When model manifests or homepage recipes change, refresh and verify the
checked-in recipe data:

```bash
npm --prefix docs/fumadocs run models:generate
npm --prefix docs/fumadocs run models:check
```
