# Dependency-fix validation

Installed jmespath1.1.0 without dependency upgrades into MatrixGame1, MatrixGame2 and StableVirtualCamera environments. MatrixGame2 Universal checkpoint now passes real CUDA inference. Its num_frames=15 argument counts latent frames; pipeline_matrix_game_2.py computes(15-1)*4+1=57 RGB frames,confirmed by full decode. Output640x352/12fps,forward camera motion around snowy hut visually coherent. Other checkpoints/actions untested.

MatrixGame1 also passes after dependency repair:30steps,33frames,1280x720/16fps. Official Minecraft fixture,forward then camera_r; contact sheet shows matching translation then right turn. Full decode passes. StableVirtualCamera generated; review pending.

StableVirtualCamera v1.1 now generates33frames576x576/16fps after jmespath repair. All frames decode,but contact sheet exhibits severe exposure flicker and geometry instability. Recorded quality_concern; no visual pass. This repeats a historical concern but current inference is independently recorded.

StableCamera investigation: all15 inference source files match official revisionfe19948e9b7bea261ab2db780a59656131404a83. Main checkpoint size and SHA256 match cached HF revisione538e251c1009e9a41cf8b7fee5f21332a1960de;anonymous HF metadata redacts the hash,which is not a mismatch. Export is byte-identical to native samples-rgb.mp4. FP32/efficient-SDPA diagnostic reproduced strong flicker and warping;half precision is not a sufficient explanation. Evidence:../stable-quality-20260921 and ../stable-fp32-efficient-gpu-20260921. Quality concern remains.
