# Segmentation and geometry validation

MobileSAM and RepViT-SAM image point+box car masks passed exact checkpoint loads and visual review. SAM2 tracking is reported separately.

Original Pi3(mode=pi3) passed3views at378x658:587690 finite points,positive depth,proper rotation matrices. Global coordinates agree with local points transformed by predicted poses to4.77e-7. World frame is arbitrary; depth and point-cloud views reviewed. Exact model loading has no missing/unexpected keys.

DA3-LARGE-1.1 passed3views at294x504 with positive finite depths/focals and proper rotations. Six missing auxiliary LayerNorm names are shared aliases with exact checkpoint tensors; see shared-norm-check.json. Depth contact reviewed.

Both new downloads match pinned HF LFS SHA256 and size:../all-model-validation-20260921/pi3-da3-download-verified.json. No ground-truth metric accuracy claim.
