# Genie Envisioner local GPU validation

The public pipeline loaded the local GE-base fast checkpoint and LTX-Video base components on GPU 1. It accepted a real RGB photo in explicitly labeled synthetic-three-view mode and generated a decodable 13-frame 768×192 MP4 containing three 256-pixel views. The sampled `contact.jpg` keeps the plush toy and table coherent across panels but shows little motion. This establishes checkpoint loading, synthetic input acceptance, and output layout, not real multi-view geometry or robot-action fidelity.

Evidence: `status.json`, `generated.mp4`, `generated.json`, `result-summary.json`, and `contact.jpg`. The recorded run used 192×256 per view, five denoising steps, and seed 42.
