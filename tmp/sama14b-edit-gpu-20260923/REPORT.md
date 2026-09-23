# SAMA-14B local GPU video-edit validation

The public `SAMA14BPipeline` loaded the local SAMA and Wan 2.1 T2V-14B weights on GPU 2 and completed a real 17-frame video edit. The input was a previously generated 15-frame snowy landscape with a central thatched structure and moving camera. The request was to change snowy trees to autumn orange foliage while preserving the structure and camera motion.

- Case: `sama14b-autumn-edit`; status and exact invocation are in `sama14b-autumn-edit/status.json`, `result.json`, and `edited.mp4.log`.
- Output: `sama14b-autumn-edit/edited.mp4`, SHA-256 `e8f36564eaaf17c7adbef08e90cbe19f5cfdf2c9f4e7e4d4430bfcda9b544afa`; decoded 17 frames, 640×352, 12 fps. The runner recomputed the artifact hash and decoded every frame.
- Checkpoint: `syxbb/SAMA-14B` revision `a91358332decca9adc7d0b77dc9345629cbf0852`, `sama_14b_ema.safetensors`, 28,601,809,776 bytes. Local SHA-256 `a3bf2bef8366573e04798b9454fabef7aeb2cfd514952a477a09572903e22e98` matches the official LFS metadata in `official-assets.json`.
- Visual review: source and output frame 8 are in `review/`; output frames 0 and 16 were also inspected. The model turns the background trees bright orange and retains the central structure and camera movement. However, orange saturates almost the entire snowy ground and background, far beyond the requested tree edit. This is a **quality concern**, not a load or execution failure.

Scope: one short video-edit configuration and a visual check. It does not validate mask-conditioned editing, long sequences, other prompts, or every model parameter.
