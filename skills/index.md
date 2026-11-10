# WorldFoundry Agent Skills

Task-specific workflows use the [Agent Skills format](https://agentskills.io/specification).
Each directory contains a `SKILL.md` with `name`, `description`, `license`, and
`compatibility` frontmatter followed by Markdown instructions; some skills add
`references/` files that load only when a task needs them. Skills are
first-party source documentation under [Apache-2.0](../LICENSE), included in
source distributions.

## Available Skills

| Skill | Use when |
| --- | --- |
| [Model inference](worldfoundry-model-inference/SKILL.md) | Running a model with `wf run` or a Workspace job, preparing environments and checkpoints, or diagnosing a failed run |
| [Evaluation](worldfoundry-evaluation/SKILL.md) | Running models on benchmarks, scoring artifacts or results, reproducing recipes, or reading and comparing scorecards |
| [Model integration](worldfoundry-model-integration/SKILL.md) | Adding or migrating a model or variant, converting checkpoints, proving component parity, or wiring the catalog route |
| [Benchmark integration](worldfoundry-benchmark-integration/SKILL.md) | Adding a benchmark, official runner, or normalizer, materializing tasks, or fixing scorecard normalization |
| [Studio serving](worldfoundry-studio-serving/SKILL.md) | Supporting a model in Studio or the Workspace, streaming and realtime sessions, worker lifecycle, or previews |
| [Inference optimization](worldfoundry-inference-optimization/SKILL.md) | Profiling and changing attention, compile, graphs, warmup, caches, offload, precision, parallelism, or decode |
| [Inference validation](worldfoundry-inference-validation/SKILL.md) | Selecting gates, comparing components, locating numerical drift, replaying checkpoint cases, or registering tests |
| [PR review](worldfoundry-pr-review/SKILL.md) | Reviewing or self-reviewing a change, writing the PR description, or checking merge readiness |
| [Repository maintenance](worldfoundry-repository-maintenance/SKILL.md) | Removing obsolete code, updating docs and generated data, packaging and licenses, skills, or release readiness |

Common sequences:

- New model: model integration, inference validation, studio serving when the
  model is served interactively, then PR review.
- New benchmark: benchmark integration, evaluation, then PR review.
- Speedup: inference optimization, inference validation, then PR review.
- Using WorldFoundry: model inference, then evaluation.

## Using Skills

Load the relevant `SKILL.md` explicitly, or configure your client to discover
this directory. Client setup and selective linking are documented in the
[maintainer guide](https://openenvision.github.io/WorldFoundry/docs/maintainers/contributing#agent-skills).
Local client configuration is opt-in and remains outside Git.

[AGENTS.md](../AGENTS.md) and [CONTRIBUTING.md](../CONTRIBUTING.md) own repository
rules. Skills describe task workflows; the linked Fumadocs pages own detailed
architecture, environment setup, and validation contracts.

## Workflow References

The skills adapt practices from the agent skills of
[vLLM](https://github.com/vllm-project/vllm) (PR checklist, microbenchmark
caveats), [SGLang](https://github.com/sgl-project/sglang) (diffusion model
porting, diagnosis-first benchmarking, executed-backend gates),
[FlashDreams](https://github.com/NVIDIA/flashdreams) (model integration
phases, profiling before optimization, performance and quality validation),
[vLLM-Omni](https://github.com/vllm-project/vllm-omni) (accuracy and
performance test contracts, evidence-based review),
[Diffusers](https://github.com/huggingface/diffusers) (component parity,
report-first self-review), [FastVideo](https://github.com/hao-ai-lab/FastVideo)
(reference-backed parity), and [LightX2V](https://github.com/ModelTC/LightX2V)
(native porting, compile and warmup lifecycles).

WorldFoundry owns the implementation boundaries, commands, and acceptance
contracts. Basic model integration, numerical acceptance, performance, and
serving reliability remain separate evidence scopes.

## Adding a Skill

Follow the official [specification](https://agentskills.io/specification) and
[authoring best practices](https://agentskills.io/skill-creation/best-practices).

- Use a lowercase, hyphenated `worldfoundry-` name matching its directory
  (1–64 characters). Declare `license: Apache-2.0` and use `compatibility`
  for checkout, hardware, and asset requirements (at most 500 characters).
- Start the `description` with "Use this skill when…", list the user intents
  that should activate it, and name the sibling skill for nearby tasks it
  does not cover (at most 1024 characters).
- Keep `SKILL.md` below 500 lines: scope and owning docs, a workflow with
  commands, gotchas, and done criteria or a report template. Move detailed
  material to `references/`, linked one level deep with when to read it.
- Commands run from the repository root and use `wf`. Repository paths,
  `make` targets, and `python -m` modules named in a skill must exist.
- Update this index and the skill list in both maintainer guide languages
  when adding or removing a skill.

Validate formatting with the official `skills-ref validate <skill-directory>`
reference tool and run `python -m pytest -q tests/core/test_agent_skills.py`
from the repository root. Evaluate activation and task outcomes separately,
following the [maintainer guide](https://openenvision.github.io/WorldFoundry/docs/maintainers/contributing#agent-skills).
