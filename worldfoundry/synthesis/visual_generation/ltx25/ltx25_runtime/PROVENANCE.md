# LTX-2.5 inference runtime

`ltx_core/` and `ltx_pipelines/` are adapted from Lightricks/LTX-2, revision
`a95ab856bf29407b6b066ede0abe1846050db56c` (LTX-2.x Community License Agreement, August 11,
2026). See root `THIRD-PARTY-NOTICES` for the license.

The directory holds the import closure of `ltx_pipelines.distilled`. Every file was modified by
WorldFoundry: imports are package-relative and the code is formatted to the repository style
without file-wide lint suppressions. `ltx_pipelines/__init__.py` keeps only a package docstring,
so the other upstream pipelines (DFR, IC-LoRA, retake, Dub-It, keyframe interpolation, audio-only
and guided one- and two-stage pipelines) and the modules only they import are omitted. The
`ltx_kernels`, `natten`, FlashAttention and `mps_sdpa` accelerators stay optional imports, as
upstream.
