# Gamma-World local GPU validation

The causal few-step, causal, and bidirectional checkpoints each loaded and generated a decodable 9-frame video on GPU 3 from the same real RGB photo. The bidirectional case also accepted two synthetic keyboard/camera action streams. Evidence is in each case directory's `status.json`, `generated.mp4`, and `result-summary.json`, with sampled frames in `../all-model-validation-20260921/gamma-*-contact.jpg`.

The few-step output is almost static. The causal and bidirectional outputs retain the plush toy but have visible scene distortions. The two input images produce a side-by-side output; the synthetic actions have no demonstrated real control effect. These runs establish loading, accepted input shape, and decodable output, not camera-motion adherence or action semantics. The no-action path previously failed in `input_encoder(None)` and now runs through the public pipeline.
