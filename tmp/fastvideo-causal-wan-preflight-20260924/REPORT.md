# FastVideo CausalWan2.2 I2V validation

The pinned Hugging Face release is `FastVideo/CausalWan2.2-I2V-A14B-Preview-Diffusers` at revision `0977572ccf137da5e577d62e3231ca840151af38`. Its README specifies image-to-video with an input image, 81 output frames, and eight inference steps. The former WorldFoundry route identified it as text-to-video, exposed 717 default frames, and would ignore the source image. The public recipe, pipeline, binding, profile, catalog, and English/Chinese guides now use the I2V ID and require an image. The image is VAE-encoded as one clean first-frame latent, committed to both expert caches at timestep zero, and preserved in the output while subsequent latent frames are generated.

`../envs/worldfoundry-unified-cu121/bin/python -m pytest tests/base_models/test_flashdreams_fastvideo_causal_wan.py -q` passed all 10 CPU contract tests. These cover the fixed schedule, self-forcing noise, expert boundary and separate caches, weight conversion, dtype, registered recipe, VAE encoder binding, and first-frame cache/output behavior. This validates routing and fake-component execution, not checkpoint-backed video quality or runner parity.

The actual-read preflight in `actual-read.json` opened every safetensors component and read sampled tensor values, counting 242 text-encoder, 1095 tensors in each transformer, and 194 VAE tensors. Both local text-encoder shards 1 and 2 retain `.aria2` sidecars. Although each has the published byte count, full-file SHA-256 differs from the pinned release's LFS hash:

| File | Local SHA-256 | Official SHA-256 |
| --- | --- | --- |
| `text_encoder/model-00001-of-00003.safetensors` | `7100b5eca8a7ac8d35976ae11a74bc090f45380c71917b2b67b813e2816ede63` | `a8e861969c7433e707cc5a74065d795d36cca07ec96eb6763eb4083df7248f58` |
| `text_encoder/model-00002-of-00003.safetensors` | `a4ff3c9d96a7410fcd2626c77bf4cd9bc2dfd9dcd57b243efcf9631a88d721cb` | `d57d948ece4837d850b7a859a4415121d57cacf8b9ee1d4db200c67f592902d7` |

The transformer index refers to 12 shard names, but the release stores the same complete tensor key set in one safetensors file per expert; the recipe intentionally selects these files. No weight or `.aria2` file was removed or altered. Real GPU inference is **blocked_external** until the two text-encoder shards are restored and verified.
