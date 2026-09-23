# GigaBrain-0 / 0.1 checkpoint-backed CUDA smoke validation

Both local 3.5B Base checkpoints completed WorldFoundry public `GigaBrain0Synthesis.predict` on one CUDA-visible GPU (`CUDA_VISIBLE_DEVICES=3`) and emitted distinct, finite `50 × 14` action traces. This is **inference readiness only**. The three 224×224 RGB views were deterministic synthetic patterns; the 14D robot state was the public RoboTwin dataset mean. There was no robot rollout, task outcome, or official score.

| Variant | Trace | Finite | Distinct rows | Mean absolute step change | Outside public action min/max |
| --- | --- | --- | ---: | ---: | ---: |
| GigaBrain-0-3.5B-Base | [action_trace.json](giga-brain-0/action_trace.json) | yes, 50×14 | 50/50 | 0.01565 | 0/700 |
| GigaBrain-0.1-3.5B-Base | [action_trace.json](giga-brain-0.1/action_trace.json) | yes, 50×14 | 48/50 | 0.00474 | 0/700 |

The traces differ by mean absolute action value 0.29794, so this run did not silently reuse one output for both checkpoints. Forward execution took 1.398 s and 1.467 s respectively after checkpoint loading. Each checkpoint has four readable safetensors shards with exactly 1,076 indexed tensors; see [checkpoint-integrity.json](checkpoint-integrity.json). The public `lerobot/robotwin_unified` `meta/stats.json` was downloaded from revision `1287871839fae2296bc27b88a5457c3e1eba8e1f`, SHA256 `eff383af77c6b8770f9523f91a1f3c90c263c3e5c15cca7c29c5d345a5181b6b`; see [stats-source.json](stats-source.json). No task-specific norm statistics or verified `delta_mask` were available, so the traces do not establish RoboTwin policy accuracy.

The GPU 3 launch used:

```bash
CUDA_VISIBLE_DEVICES=3 OMP_NUM_THREADS=4 NUMEXPR_MAX_THREADS=64 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 ../envs/worldfoundry-unified-cu121/bin/python -u tmp/gigabrain-gpu-20260923/run_case.py 0
CUDA_VISIBLE_DEVICES=3 OMP_NUM_THREADS=4 NUMEXPR_MAX_THREADS=64 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 ../envs/worldfoundry-unified-cu121/bin/python -u tmp/gigabrain-gpu-20260923/run_case.py 0.1
```

The successful logs are [GigaBrain-0](giga-brain-0-pre-telemetry.log) and [GigaBrain-0.1](giga-brain-0.1.log). Their runtime plans record `device: cuda:0`, where visible device 0 maps to physical GPU 3. Per-case validation and numeric checks are in `giga-brain-0{,.1}/validation.json` and `action-quality.json`.

Three integration defects surfaced and were fixed: the OpenPI package imported Flax eagerly when GigaBrain needed only the FAST tokenizer; the default stats path pointed to a removed `../data` tree; and BF16 SigLIP features reached a BF16 multimodal projector as FP32. The import, canonical stats-token resolution, and both checkpoint-backed forwards passed after those fixes. An optional later telemetry rerun failed with CUDA OOM because another project's supervisor restarted its four-card training and occupied ~74 GB per GPU. That resource failure is preserved in [giga-brain-0.log](giga-brain-0.log); it does not replace the earlier successful trace. Peak GPU allocation was not measured.
