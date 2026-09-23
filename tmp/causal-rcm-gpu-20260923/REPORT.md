# Causal-rCM local GPU validation

The in-tree official entrypoint generated `causal-rcm-teapot-41f/generated.mp4` on GPU 2 from a text prompt and the local 1.3B checkpoint. FFprobe and frame decoding confirm 41 frames at 832×480. Sampled frames retain the red teapot, tabletop, and window consistently, with slight motion. This supports inference and output validity for the recorded 480p, 4-step, 41-frame configuration; it does not establish quality across other settings.

Evidence: `causal-rcm-teapot-41f/{plan.json,result.json,status.json,generated.mp4,generated.mp4.log}` and `../all-model-validation-20260921/causal-rcm-contact.jpg`. The real run exposed and drove fixes for checkpoint path resolution, `randn_like(generator=...)` compatibility, FlashAttention 3 tuple output, and VAE cache accounting. The final run completed MP4 decoding after these fixes.
