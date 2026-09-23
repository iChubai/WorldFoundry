# ltx-2.5 validation audit

Status: `blocked_external`.

The official LTX-2.5 preflight is blocked: required transformer and Gemma4 text-encoder safetensors have .aria2 sidecars and are 2,335,547,254 and 940,793,130 bytes short of official expected sizes. Source/runtime files are present. No GPU inference is valid before these checkpoint downloads complete.

Evidence: preflight.json; size-comparison.json.
