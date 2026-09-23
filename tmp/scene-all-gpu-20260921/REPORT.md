# Scene validation

MonST3R passed three-view reconstruction at512 with100 global alignment iterations. Scene contains341522 vertices,positive finite depth and unit camera quaternions. Depth and point-cloud views preserve room/furniture structure. Checkpoint audit proves omitted DPT keys are exact shared aliases. Flow loss is disabled and pair dynamic masks skipped; this result does not cover RAFT/SAM2 dynamic processing. Restored local ignored dependencies are not a packaging claim.

FlashWorld passed16-view generation and226-frame interpolation render at704x480/15fps. Scene tensors finite,full FFmpeg decode passes,room and forward trajectory remain coherent in contact sheet; some edge geometry stretching is visible. No ground-truth reconstruction accuracy claim.

StableVirtualCamera failed before inference due to missing jmespath; dependency installed and new run queued in ../scene-matrix-fix-gpu-20260921. WorldFM remains queued.
