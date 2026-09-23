# HMA discrete and continuous GPU validation

The HMA catalog now maps MAGVIT 362M to `liruiw/hma-base-disc` and MAR 1B to `liruiw/hma-base-cont`, matching the upstream README. The launcher chooses STMaskGIT plus the official MAGVIT2 tokenizer for the discrete checkpoint and retains STMAR plus the SVD temporal VAE for the continuous checkpoint. The official tokenizer was downloaded from `1x-technologies/world_model_tokenized_data/magvit2.ckpt` and its SHA256 recorded.

Both routes completed real single-H100 inference from the official LangTable frame with four small rightward actions, each producing five decodable 256×256 frames. The discrete output retains the table, colored blocks and gripper while the gripper/blue block position changes; the continuous regression retains the scene with weaker movement. This validates loading, shape and short video generation, not action calibration or robot task success.

Evidence: `hma-discrete-fixed/status.json`, `hma-discrete-fixed/generated.mp4.log`, `hma-discrete-fixed/quality-review.json`, `hma-cont-regression/status.json`, and the generated videos in those directories.
