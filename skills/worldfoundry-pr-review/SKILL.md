---
name: worldfoundry-pr-review
description: >
  Use this skill when reviewing a WorldFoundry pull request or local branch,
  self-reviewing a change before opening a PR, writing the PR description, or
  checking merge readiness. Covers freezing the reviewed diff, routing checks
  by changed area, WorldFoundry-specific rules (ownership boundaries,
  infer-only synthesis, lazy imports, catalog claims versus evidence, test
  selection, paired docs, licensing, artifacts kept out of Git), commit and PR
  title format, DCO, and a severity-ranked findings report. Running the
  validation gates is worldfoundry-inference-validation; repository cleanup is
  worldfoundry-repository-maintenance.
license: Apache-2.0
compatibility: >
  Requires a WorldFoundry checkout with history for the base branch. The
  GitHub CLI is optional for reading remote pull requests.
---

# Review WorldFoundry Changes

Review against the repository's own contracts and report only what is
actually wrong. Run commands from the repository root.
[CONTRIBUTING.md](../../CONTRIBUTING.md) owns the rules; the
[PR template](../../.github/PULL_REQUEST_TEMPLATE.md) owns the description.

- **Review** (another author's PR or branch): report findings; edit nothing
  unless fixes are requested, and never post to GitHub unless asked.
- **Self-review** (before opening a PR): apply the same rubric, fix the
  blockers, then write the PR description.

Every finding must be reachable (a concrete failing path, not a suspicion),
cite `path:line` and the rule or call path that establishes it, state its
impact, and propose the smallest safe fix. Zero findings is a valid result.
Do not report formatting that Ruff and pre-commit enforce, or backlog the
change did not touch.

## 1. Freeze the diff

```bash
BASE=$(git merge-base origin/main HEAD)
git log --format='%h %s' "$BASE"..HEAD
git diff --stat "$BASE"...HEAD
git diff "$BASE"...HEAD
git status --short
```

Include staged, unstaged, and in-scope untracked files for a local review.
For a GitHub PR, `gh pr diff <number>` and `gh pr view <number> --comments`
read the diff and conversation; inline review comments come from
`gh api --paginate repos/OpenEnvision/WorldFoundry/pulls/<number>/comments`.
Record the head commit you reviewed.

## 2. Map the change

Group changed files into production code, tests, catalog and data YAML,
docs, CI and build, and generated output. Compare the title and description
with the diff and flag unrelated scope: a PR covers one pipeline, benchmark
path, public surface, or docs area.

## 3. Route the checks

| Changed paths | Read first | Domain skill |
| --- | --- | --- |
| `worldfoundry/base_models/diffusion_model/` | Native diffusion boundaries | [model integration](../worldfoundry-model-integration/SKILL.md) |
| `worldfoundry/pipelines/`, `worldfoundry/operators/`, `worldfoundry/synthesis/`, `worldfoundry/data/models/` | Route ownership and status claims | [model integration](../worldfoundry-model-integration/SKILL.md) |
| `worldfoundry/data/benchmarks/`, `worldfoundry/evals/` | Protocol, artifact contracts, eligibility | [benchmark integration](../worldfoundry-benchmark-integration/SKILL.md), [evaluation](../worldfoundry-evaluation/SKILL.md) |
| `worldfoundry/core/` | Model-neutral placement, lazy imports | [optimization](../worldfoundry-inference-optimization/SKILL.md) |
| `worldfoundry/studio/` | Lifecycle and transport ownership | [studio serving](../worldfoundry-studio-serving/SKILL.md) |
| `tests/`, `Makefile` | Placement and selection | [inference validation](../worldfoundry-inference-validation/SKILL.md) |
| `docs/fumadocs/`, `skills/`, `MANIFEST.in`, `THIRD-PARTY-NOTICES` | Pairing, generation, packaging | [repository maintenance](../worldfoundry-repository-maintenance/SKILL.md) |

Apply the matching sections of [the review checklist](references/review-checklist.md).
Trace callers before calling code dead: registries, literal `"module:Class"`
targets, lazy imports, subprocess entrypoints, and Studio's static scan.

## 4. Check evidence against claims

- Inference behavior changes need affected-case checkpoint replays; CUDA
  kernel changes need CUDA contracts. CI's CPU jobs cannot supply either, so a
  missing replay is a blocker, not a pass.
- Performance claims need a matched stack, the effective executed path,
  repeated measurements with spread, and numerical evidence.
- `integration.status: integrated`, `leaderboard_valid: true`, and similar
  promotions need executed evidence for the exact variant or cohort.
- A bug fix includes a regression that fails without the fix; numerical claims
  are not supported by mocks alone.

## 5. Run the cheap checks

```bash
make pr-check BASE_REF="$BASE"
make lint
make test
```

`make pr-check` runs Ruff and ty on committed Python changes and verifies DCO
sign-offs. Add the Make targets or focused tests for the touched area; GPU
gates belong to whoever has the prepared host, and their absence is reported.

## 6. Report

```markdown
## Review of <branch or PR #> at <head-sha> (base <base-sha>)

### Blocking
1. **<Title>** — `path:line`. <Observed impact>. <Rule or call path>. Fix: <smallest safe change>.

### Non-blocking
1. **<Title>** — `path:line`. <Why it matters>. Suggestion: <change>.

### Verified
- `<command>` — <result>

### Not verified
- <gate> — <missing resource or reason>

Verdict: READY or NEEDS CHANGES
```

## Prepare the PR

- Commit subjects and the PR title use `[type]Description` with no space after
  the bracket and an English imperative, for example
  `[fix]Correct model session cleanup`. Types include `fix`, `feat`,
  `refactor`, `test`, `docs`, and `chore`.
- Sign off every commit with `git commit --signoff`; squash merges keep the
  trailers.
- Fill every template section: Summary, Scope, Runtime And Assets, the
  Validation table with exact commands and results, Known Limitations, and
  the Contribution Requirements checklist. Record failed or unexecuted stages
  with their reasons.
- Link the replay report and a representative output for model execution
  changes, the scorecard and any `leaderboard_valid` change for benchmark
  changes, and the docs build plus both language pages for docs changes.

## Gotchas

- The `Python quality` check covers only changed Python files, and ty excludes
  `worldfoundry/base_models/` and `worldfoundry/synthesis/`; run focused tests
  for those trees.
- A test that is in no Make selection never runs in CI.
- Generated docs data must come from its generator; hand edits are
  overwritten and fail `make docs-check`.
- Review the committed diff the PR will merge, not only the working tree.
