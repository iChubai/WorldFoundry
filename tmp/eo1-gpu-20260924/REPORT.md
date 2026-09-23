# eo1 preflight, 2026-09-24

Default runtime config rejected its external source checkout before weight loading; the checkpoint declares AutoConfig/AutoModel code requiring trust_remote_code, while the in-tree runtime forbids it.

No action chunk was produced. Evidence: `tmp/vla-blockers-gpu-20260924/loader-preflight.json`.
