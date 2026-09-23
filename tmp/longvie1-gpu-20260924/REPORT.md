# longvie-1 validation audit

Status: `blocked_external`.

LongVie-1 has no published default first-generation control checkpoint. The runtime raises FileNotFoundError without an explicit first-generation control_weight_path or weight_dir and rejects LongVie2 DiT/control weights. The Wan2.1-I2V-14B-480P base exists under an unprefixed local alias, so the missing LongVie-1 control weight is the blocker.

Evidence: asset-preflight.json; worldfoundry/synthesis/visual_generation/longvie/worldfoundry_runtime.py.
