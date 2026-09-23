# TesserAct local GPU validation

The public pipeline loaded the local RGB-depth-normal SFT transformer and CogVideoX base assets on GPU 3, then generated a decodable 9-frame, 960×256 MP4. Each frame contains RGB, depth, and normal panels. The sampled `contact.jpg` retains the blue plush toy in the RGB panel; the RGB motion is minimal and the geometry panels came from the runtime's deterministic synthetic-gradient smoke conditioning. This establishes inference and output layout only, not estimated physical geometry or action semantics.

Evidence: `status.json`, `generated.mp4`, `generated.json`, `result-summary.json`, and `contact.jpg`. The initial blocked plan used the wrong checkpoint-root environment variable in the validation command; the successful retest used `WORLDFOUNDRY_CKPT_DIR` and the staged local weights.
