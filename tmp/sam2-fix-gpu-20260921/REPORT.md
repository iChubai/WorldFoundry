# DUSt3R / SAM2 validation

DUSt3R public directory-loading fix has passed real CUDA inference at512 with two views: finite point predictions,20000 exported points,proper camera rotations and positive focal lengths. Point cloud visualization is consistent with the room floor and furniture. All eight omitted DPT keys are proven module aliases with exact checkpoint tensors; see checkpoint-alias-review.json. No metric ground-truth accuracy claim. The separate DUSt3RBaseModel public entrypoint also passed an independent GPU run with the same numeric/visual checks.

SAM2 fixture errors and failed runs retained here; corrected image+tracking evidence and report are in ../sam2-tracking-gpu-20260921.
