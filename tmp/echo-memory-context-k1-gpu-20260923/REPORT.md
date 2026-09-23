# Echo-Memory context K=1 GPU validation

The public `EchoMemoryContextK1Pipeline` loaded the local `context_k1/epoch-0.safetensors` checkpoint and generated a two-chunk I2V clip on GPU 1 with a real RGB photo, 81 requested frames per chunk, 50 steps, seed 42, and 15 fps. The pipeline returned `generated_video_path`; the first validation script incorrectly expected `artifact_path`, so its nonzero exit was a harness error after generation. The harness was corrected and the existing video was checked directly.

Evidence: `echo-memory-k1-two-chunks/status.json`, `result-summary.json`, `generated.mp4`, and `contact.png`. FFprobe reports 161 decoded frames at 640×352, 15 fps, 10.73 seconds. Five sampled frames show a coherent plush toy and red table across the chunk boundary. The requested camera move is weak; this is a quality concern, not a demonstrated semantic pass. A separate frame or camera-motion comparison is needed before claiming prompt adherence. The input is a single photo, so this case does not cover action-conditioned variants.

The actual model bug fixed during this run was `EchoWanAttentionBlock.forward` rejecting Wan's internal RoPE keyword arguments. The complete pipeline generated after that fix.
