# emu3.5 validation audit

Status: `failed_integration`.

15/15 Emu3.5 checkpoint shards (68.21 GB) and index are present and safetensors headers open. The official checkpoint card describes interleaved text/image and image generation, not video output. Current Transformers 4.57.6 fails AutoConfig on config.model_type="Emu3"; WorldFoundry uses AutoModelForCausalLM with this config and emits JSON model_response. No video GPU inference was attempted.

Evidence: asset-audit.json; runtime-preflight.json; ../ckpts/BAAI--Emu3.5/README.md.
