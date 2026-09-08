# Third-party dependencies

This directory holds vendored or submodule-backed dependencies that are not part
of the main `worldfoundry` Python package layout.

## Layout

| Path | Integration |
|------|-------------|
| FlashAttention | External Python dependency — install with `scripts/setup/install_flash_attn.sh` when it is not already available in the runtime |
| `diff-gaussian-rasterization/` | Canonical depth-capable Gaussian rasterizer; auxiliary median-depth and opacity outputs are opt-in |
| `simple-knn/` | CUDA KNN extension vendored for 3D pipelines |

## Guidelines

- Prefer **git submodules** for large, independently versioned upstream projects.
- Keep **in-tree copies** only when heavy patching or tight coupling is required.
- Document upstream provenance in `UPSTREAM.md` / `LICENSE` next to the vendored tree when adding new entries.
- Do not duplicate code that already exists under `worldfoundry/base_models/` or `worldfoundry/synthesis/` — import the canonical integration path instead.
