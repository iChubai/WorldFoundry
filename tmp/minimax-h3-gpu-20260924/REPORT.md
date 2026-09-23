# minimax-h3 validation audit

Status: `failed_integration`.

All 14 DiT, 14 Qwen text-encoder, three video-VAE, and one audio-VAE checkpoint shards are present; safetensors headers open and a small sampled tensor from each shard is finite/nonzero. Their combined 134.13 GiB exceeds one H100 79.65 GiB; current public loader moves every component to the same device. The 61.73-GiB DiT alone leaves 17.92 GiB before activations. Sequential CPU offload at reduced geometry is plausible but not implemented or verified. No cross-card job was displaced and no GPU inference claim is made.

Evidence: asset-audit.json; memory-assessment.json; worldfoundry/pipelines/minimax/pipeline_minimax_h3.py.
