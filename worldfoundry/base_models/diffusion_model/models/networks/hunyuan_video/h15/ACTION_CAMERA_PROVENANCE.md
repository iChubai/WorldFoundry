# HY15 Action2V inference

`action_camera.py` and `camera_attention.py` adapt the camera-conditioned HY15
inference graph from shengshu-ai/minWM, revision
`df522a26cd4409d3e3e8f269cc98eac069b5df47`.
The upstream files retain Tencent Hunyuan license headers (LICENSE-HUNYUAN);
minWM's repository license is included as LICENSE-MINWM.
Teacher-forcing and distributed training branches are omitted. Common layers,
PRoPE, VAE, text encoders and attention kernels reuse WorldFoundry implementations.
`camera_modulation.py` preserves frame-wise conditioning of cached inference.
