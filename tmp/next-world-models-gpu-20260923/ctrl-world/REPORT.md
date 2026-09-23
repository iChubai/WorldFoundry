# Ctrl-World local GPU validation

The public Ctrl-World pipeline loaded the local checkpoint, SVD base, and CLIP text model on GPU 0. It accepted a real RGB photo in explicitly labeled synthetic-three-view mode plus a rightward action, then generated a decodable five-frame 960×192 MP4 containing three 320-pixel views. The sampled `contact.jpg` shows the first two panels retain the plush toy, while the third view blurs and largely collapses. This is a quality concern; synthetic views cannot establish real multi-camera or robot-action semantics.

Evidence: `status.json`, `generated.mp4`, `generated.json`, `result-summary.json`, and `contact.jpg`. The recorded run used 192×320 per view, four denoising steps, and seed 42.
