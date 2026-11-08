# `worldfoundry.core`

Shared inference infrastructure used by every model family. Anything reusable across
families — loading, offload, attention backends, distributed execution, kernels,
quantization, media IO — lives here exactly once; model packages must not reimplement it.

## Layout

| Path | Purpose |
| --- | --- |
| `model_loading/` | Checkpoint materialization, state-dict loading, LoRA/PEFT, model factories |
| `vram/` | Device maps, layer-wise and disk offload, encoder lifecycle |
| `attention/` | Attention backends (SDPA, flash, sparse, …), KV cache, RoPE, sequence/context parallel attention |
| `distributed/` | Collectives, model/sequence parallel, sharding |
| `execution/` | Device selection, `torch.compile` and CUDA graph helpers, queued batching, realtime execution |
| `acceleration/` | Caching (EasyCache, …), quantization, token pruning, convolution layout |
| `kernels/` | Triton and native kernels with a capability registry and autotune cache |
| `nn/` | Reusable blocks: transformer layers, diffusion blocks, latent utilities, checkpointing |
| `memory/` | Memory stores and retrieval for streaming and multi-turn models (`BaseMemory`) |
| `media/` | Video/image codecs, artifact writing, resolution presets |
| `io/` | Packaged-data paths (`package_data_path`), input loading, manifests, filesystem helpers |
| `geometry/` | Camera poses, trajectories, warping, surfels |
| `configuration/` | Config loading (Hydra, lazy config), overrides, validators |
| `observability/` | Logging setup, NVTX ranges, realtime timing, profiling |
| `safety/` | Guardrails |
| `utils/` | Python and tensor utilities |
| `contracts.py`, `registry.py`, `env_registry.py`, `prompting.py`, `action_normalization.py` | Shared contracts, registries, prompt and action helpers |

Inference tuning flags exposed as `--pipeline.load.*` (offload, attention backend, compile,
quantization, CUDA graphs) are implemented here. See
[Core infrastructure](https://openenvision.github.io/WorldFoundry/docs/maintainers/architecture/core-infrastructure)
and [Inference optimization](https://openenvision.github.io/WorldFoundry/docs/guides/inference-optimization).
