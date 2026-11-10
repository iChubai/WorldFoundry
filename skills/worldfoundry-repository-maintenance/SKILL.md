---
name: worldfoundry-repository-maintenance
description: >
  Use this skill when maintaining the WorldFoundry repository itself: removing
  obsolete code or files, reorganizing tests, updating paired English and
  Chinese Fumadocs pages or generated docs data, changing CLI help or
  arguments, checking source packaging and third-party licenses, editing
  repository policy files, maintaining the Agent Skills in skills/, or
  preparing a release. Covers ownership tracing before deletion, published
  surfaces and their checks, generated documentation, and packaging and
  license gates. Reviewing or preparing a pull request is worldfoundry-pr-review.
license: Apache-2.0
compatibility: >
  Requires a WorldFoundry checkout, Git, and the unified environment.
  Fumadocs work also requires Node.js and npm.
---

# Maintain the WorldFoundry Repository

Keep published surfaces consistent with their sources and remove only what is
proven unused. Run commands from the repository root.
[AGENTS.md](../../AGENTS.md) and [CONTRIBUTING.md](../../CONTRIBUTING.md) own
repository rules; the [maintainer guide](../../docs/fumadocs/content/docs/maintainers/contributing.mdx)
owns documentation generation, Agent Skills authoring, and release evidence.

## Before editing

Run `git status --short` and note changes you did not make; leave them
untouched and out of your commits. Identify the document or test that owns
the surface you are about to change.

## Remove obsolete code safely

1. Trace every caller before deleting: Python imports, lazy imports, literal
   `"module:Class"` targets in catalogs and bindings, runtime-profile
   `source_repos[].in_tree_path`, subprocess entrypoints, serialized
   checkpoint classes, Studio's static catalog scan, docs, and tests. Search
   both forms, for example `git grep -n "worldfoundry.pipelines.<family>"` and
   `git grep -n "worldfoundry/pipelines/<family>"`.
2. An unused-import or coverage report alone is not evidence of dead runtime
   code.
3. Remove an abandoned implementation together with its private helpers,
   obsolete tests, and entrypoints. Do not leave forwarding modules at the
   abandoned path.
4. Keep tests that expose defects in supported integrations. Delete tests
   only with the implementation or contract they cover.
5. Update the docs file maps and catalog provenance that named the old path.

## Published surfaces

| Surface | Rules | Checks |
| --- | --- | --- |
| README files | Any directory may keep a README for its own code or tooling. `worldfoundry/`, each top-level package, and each `base_models` subpackage keep a package-map README, updated when a module is added, moved, or removed; guides belong in Fumadocs | Review |
| Fumadocs pages | Paired `.mdx` and `.zh.mdx`; listed in both `meta.json` and `meta.zh.json` | `bash scripts/docs/build.sh`, `make docs-check` |
| Generated docs data | Regenerate with `npm --prefix docs/fumadocs run models:generate`, `benchmarks:generate`, or `coverage:generate`; the docs build regenerates the Python API metadata itself. Never hand-edit generated JSON | `make docs-check` |
| CLI | Update `docs/fumadocs/content/docs/reference/cli.mdx`; keep JSON output compatible; help stays import-light | `make cli-entrypoint-check`, `make cli-check`, `make test-docs-contracts` |
| Tests | Module-mirroring placement, markers, and Make selection per [test selection](../worldfoundry-inference-validation/references/test-selection.md) | The affected Make target |
| Source distribution | `MANIFEST.in` and package data include sources and skills; no weights, media, caches, or run outputs | `make packaging-check` |
| Licenses | Upstream headers kept; new first-party Python files carry SPDX `Apache-2.0`; provenance in `THIRD-PARTY-NOTICES` and the owning catalog | `make packaging-check` |
| Agent Skills | Rules below | `python -m pytest -q tests/core/test_agent_skills.py` |
| Policy files | `SECURITY.md` and `CODE_OF_CONDUCT.md` stay linked from `README.md`, `CONTRIBUTING.md`, and both maintainer guides | `python -m pytest -q tests/core/test_repository_policies.py` |

Model pages with `pageSource: authored` keep hand edits; generated pages are
rewritten on the next `models:generate`. Curated benchmark copy lives in
`docs/fumadocs/lib/benchmark-page-overrides.json`.

## Maintain the Agent Skills

- Each skill is `skills/<name>/SKILL.md` with `name` (matching the directory,
  `worldfoundry-` prefix), a `description` starting "Use this skill when…"
  that also names the sibling skill for nearby tasks, `license: Apache-2.0`,
  and `compatibility` for checkout and hardware needs.
- Bodies stay under 500 lines: scope and owning docs, workflow with commands,
  gotchas, and done criteria or a report template. Detailed material goes in
  `references/` with an explicit "read when" link one level deep.
- Commands run from the repository root and use `wf`. When a command, path, or
  contract changes, update every skill that names it in the same change.
- `tests/core/test_agent_skills.py` checks frontmatter, links, index coverage,
  packaging, and that repository paths, `make` targets, and `python -m`
  modules named in skills exist. Optional format validation:
  `skills-ref validate skills/<name>`.
- Adding or removing a skill updates [the skill index](../index.md) and the
  skill list in both maintainer guide languages.

## Checks for a maintenance change

Pick the rows that match the changed surfaces:

```bash
pre-commit run
make lint
make test
make cli-entrypoint-check
make packaging-check
bash scripts/docs/build.sh
make docs-check
make test-docs-contracts
```

Before a release, also build the wheel and sdist and audit them with
`make packaging-check WHEEL=<wheel> SDIST=<sdist>`, run `make compile-eval` and
`make cli-check`, and replay inference cases for changed shared components with
[inference validation](../worldfoundry-inference-validation/SKILL.md).

## Gotchas

- `pre-commit run` checks staged files; `make precommit` runs every hook on
  all files and can rewrite files you did not intend to touch.
- `make lint` also loads the model and benchmark catalogs and validates the
  runtime and workspace registries, so a YAML typo fails it.
- Markdown other than READMEs under `worldfoundry/evals/` and runner
  `requirements.txt` files are rejected; pip requirements go in
  `worldfoundry/data/benchmarks/requirements/`.
- `synthesis` is infer-only: no training code, launch scripts, notebooks, or
  demo media.
- Local demo inputs live in `testcase/`, which is not part of the source
  distribution; reference them by explicit path.
- Generated docs data in `docs/fumadocs/lib/` and the Python API metadata are
  overwritten by their generators; edit the source or the overrides file.

## Done criteria

- Every deleted file's callers were traced and removed or redirected.
- Each changed surface's check ran and passed, or its blocker is recorded.
- English and Chinese pages changed together; generated data is current.
- No unrelated working-tree changes are included.
