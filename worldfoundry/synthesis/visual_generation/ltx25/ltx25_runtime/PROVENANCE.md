# LTX-2.5 inference runtime

`ltx_core/` and `ltx_pipelines/` are adapted from Lightricks/LTX-2, revision
`a95ab856bf29407b6b066ede0abe1846050db56c` (LTX-2.x Community License Agreement, August 11,
2026). See root `THIRD-PARTY-NOTICES` for the license.

The directory holds the inference orchestration of `ltx_pipelines.distilled`: the two-stage
distilled pipeline, staged checkpoint builders and their key mappings, CPU/disk block streaming,
FP8 and NVFP4 quantization policies, conditioning, guidance and Euler sampling steps, and media
I/O. The networks are the canonical WorldFoundry LTX implementations under
`worldfoundry/base_models/diffusion_model`: the joint audio-video transformer
(`models/networks/ltx`, including the LTX-2.5 bias-free feed-forward, static prompt AdaLN and
keyframe position embedding), the duration head, the Gemma text encoder and embeddings connectors
(`models/encoders/ltx`), the video VAE with its neighborhood-attention diffusion decoder and the
audio VAE and vocoder (`models/autoencoders/ltx`), the latent upsampler, the tiling geometry
(`models/representations/ltx`), and the attention and compilation policies
(`optimizations/ltx`).

Every file was modified by WorldFoundry: imports are package-relative or canonical, unused
definitions and the other upstream pipelines (DFR, IC-LoRA, retake, Dub-It, keyframe
interpolation, audio-only, guided one- and two-stage) are omitted, and the code is
formatted to the repository style without file-wide lint suppressions. The `ltx_kernels`,
`natten`, FlashAttention and `mps_sdpa` accelerators stay optional imports, as upstream.
