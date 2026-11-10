---
name: worldfoundry-inference-optimization
description: >
  Use this skill when WorldFoundry inference is too slow or uses too much
  memory, or when changing attention backends, torch.compile, CUDA graphs,
  warmup, request batching, KV or feature caches, acceleration plugins,
  offload, precision or quantization, VAE decode, sequence or CFG
  parallelism, or Studio streaming latency. Covers matched-stack baselines,
  stage attribution, confirming the executed fast path, rejected knob
  combinations, lifecycle-safe changes, and accepting a speedup only with
  numerical evidence. Correctness gates are worldfoundry-inference-validation;
  Studio workers and transport are worldfoundry-studio-serving.
license: Apache-2.0
compatibility: >
  Requires a WorldFoundry checkout and a prepared CUDA environment.
  Measurements need local checkpoints and representative inputs.
---

# Optimize WorldFoundry Inference

Diagnose before changing code, change one variable at a time, prove the fast
path actually executed, and accept a speedup only with numerical evidence.
Run commands from the repository root. The
[inference optimization guide](../../docs/fumadocs/content/docs/guides/inference-optimization.mdx)
owns available knobs and rejected combinations;
[diffusion acceleration](../../docs/fumadocs/content/docs/guides/diffusion-acceleration.mdx)
owns plugin contracts; the
[validation guide](../../docs/fumadocs/content/docs/reference/validation.mdx)
owns acceptance.

## Start from the supported defaults

- Official-looking output: keep catalog defaults, BF16, `offload_mode=resident`
  when VRAM fits (otherwise `async-block`), and leave feature caches,
  approximate attention, quantization, CUDA Graph, and preview decoders off.
- Throughput on the same native Wan checkpoint: a real Flash or Sage
  `attention_backend`, then `fuse_qkv`, `fused_rope`, and `static_cross_kv`;
  `compile` at load followed by representative warmup; Wan VAE tiling or
  temporal chunks when decode OOMs or dominates.
- Most named knobs exist only on native Wan 2.1/2.2. Other families accept a
  subset; list a model's real load keys with `wf run <model-id> --help`
  (`--pipeline.load.*`) before setting one.

## Workflow

```text
- [ ] 1. Workload and stack frozen
- [ ] 2. Baseline measured with stage attribution
- [ ] 3. Bottleneck classified; existing fast paths checked
- [ ] 4. One change made in its owning layer
- [ ] 5. Executed path confirmed from receipts
- [ ] 6. Numerical and lifecycle evidence collected
- [ ] 7. Candidate measured against the matched baseline
```

### 1. Freeze the workload and stack

Fix weights, input, prompt or control schedule, seed, resolution, frame count,
sampler, steps, guidance, and precision. Record the command, commit, GPU model
and count, driver, CUDA, PyTorch, and resolved runtime policy. For Studio,
record worker startup, one request, and the repeated benchmark command.
Official benchmark settings (steps, CFG, size) are part of the workload;
changing them creates a different workload.

### 2. Measure and attribute

Report loading, cold compilation or capture, warmup, cache fill, steady-state
denoise, decode, transfer, encoding, and queue wait separately. Use CUDA
events or explicit synchronization for GPU time and at least three measured
repeats with per-run values. Profiler timings attribute cost; they are never
headline latency. For a timeline without added synchronization:

```bash
WORLDFOUNDRY_NVTX=1 nsys profile --trace=cuda,nvtx,osrt -o tmp/trace/<name> <command>
```

The standard runner emits `worldfoundry.run`, `encode`, `denoise_step`,
`denoiser.positive`/`denoiser.negative`, and `decode` ranges; postprocess,
H.264 output, and offload H2D copies have their own ranges.

### 3. Classify and look for an existing fast path

Name the dominant stage: attention or GEMM, normalization and elementwise
glue, launch overhead, decode, offload copies, host transfer and encoding,
queueing, or collectives. Check existing implementations before writing
kernels: backend dispatch in `worldfoundry/core/attention/`, compile and
graphs in `worldfoundry/core/execution/`, plugins in
`worldfoundry/core/acceleration/`, and diffusion transforms in
`worldfoundry/base_models/diffusion_model/optimizations/`.

### 4. Change one variable in its owning layer

- Shared backend selection, kernels, compile, graph, cache, and offload code
  belongs in `worldfoundry/core/`; family packages never get private copies.
- Diffusion-specific composition goes through `RuntimePolicy`, components,
  plugins, and runner strategies; no model-name branches in shared runners.
- Compiled or captured regions hold stable tensor compute only. Loading,
  offload transfers, Python request setup, and cleanup stay outside.
- Prefer an existing configuration surface. Add a public option only for a
  distinct supported behavior; keep exact and approximate paths separate.

For compile, CUDA Graph, warmup, cache, batching, or offload changes, read
[execution lifecycle](references/execution-lifecycle.md) before editing.

### 5. Confirm what executed

An accepted option is not a speedup. After one forward on a native pipeline,
read `pipe.native_pipeline.components.denoiser.runtime_optimization_report()`
(and the same on `components.decoder`) and compare `requested`, `effective`,
and `fallbacks`:

- Attention: attempts equal successes and no target fallback; compile shows
  `compiled_graph_traces > 0`.
- CUDA Graph: `replay > 0` with zero eager and capture failures.
- Static cross-KV: hits in this request and zero bypasses.
- Quantization: `low_precision_kernel_calls > 0`; `dense_policy_calls` ran BF16.
- Plugins: installation reports `execution_verified=False` by design; use the
  per-request receipts to prove execution.

### 6. Collect numerical and lifecycle evidence

Exact or numerically equivalent changes must reproduce the dense reference
under the owning contract; approximations need explicit opt-in and measured
quality limits. Existing harnesses:

```bash
python -m benchmarks.inference.wan_execution_preparation \
  --checkpoint /path/to/Wan2.1-T2V-1.3B --output-dir tmp/wan-execution --request-batching
python -m benchmarks.operators.attention_adapter_parity --backend flash2 --strict --out tmp/attention-parity
```

`wan_execution_preparation` replays dense, static-graph, and batched A/B/A
requests through the public path; its `report.json` must show both
`complete` and `passed`, and its timings are diagnostic. For sequence
parallelism, `benchmarks.inference.wan22_multigpu_e2e` saves single-rank
reference latents and compares the multi-rank candidate with
`--reference-latents`, `--reference-video`, and `--fail-on-fallback`. Then run
the CUDA and replay gates from
[inference validation](../worldfoundry-inference-validation/SKILL.md).

### 7. Measure the candidate

Use the same stack, settings, and GPU set as the baseline. Use fresh processes
for compile, autotune, backend probing, and persistent caches. Report startup
and steady state separately, including warmup plus the first completed
request for one-shot workloads. Framework A/B claims use
`benchmarks.inference.framework_ab_wan22` on one GPU in `Exclusive_Process`
mode; timings from shared `Default`-mode GPUs are diagnostic only.

## Gotchas

- Rejected at build: sequence parallelism with feature caches, sparse or
  approximate attention, or CUDA Graph (fused RoPE is SP-aware); CUDA Graph
  with feature caches or approximate attention; quantization with block or
  disk offload; `device_map=balanced` with any offload; two feature caches at
  once; `cfg_parallel` degrees other than 1 or 2.
- The first compiled or captured request is slow; never time it. Moving
  compilation into warmup moves the cost, it does not remove it.
- After warmup freezes CUDA Graph capture, unseen request signatures bypass
  the graph and may recompile. Warm the deployed shapes, batch sizes, and
  condition lengths.
- `vae_preview_decoder_path` (TAEW2.2) is a preview decoder; release and
  benchmark videos use the official VAE.
- Packed reusable offload buffers are opt-in and were slower than per-parameter
  copies for small layers; measure the real layer sizes.
- Peak allocated and reserved memory differ; report both when diagnosing OOM.
- Plugins require an idle model; compiled models and CUDA Graph owners must be
  rebuilt after plugin or weight changes.

## Report

| Config | Command | Effective path | Latency (ms, N runs, spread) | Peak alloc/reserved | Startup | Numerics | Decision |
| --- | --- | --- | --- | --- | --- | --- | --- |

State the hardware and workload scope of every number, the measured memory,
startup, precision, or quality tradeoffs, and whether the change is default,
opt-in, or rejected and why. Update the paired guide or model pages that own
the changed behavior. Traces, media, and replay outputs stay outside Git.
