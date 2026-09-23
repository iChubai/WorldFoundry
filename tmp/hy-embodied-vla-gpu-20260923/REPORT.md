# Hy-Embodied-0.5-VLA local GPU validation

The public `HyEmbodiedVLAPipeline` loaded both released variants and completed action inference on a real RGB photo plus a clearly synthetic 32-value zero robot state. After fixing the tokenizer's Mistral regex flag, RoboTwin and UMI were rerun independently on GPU 0 and GPU 3. Both `*-regexfix/status.json` files report `generated`; neither rerun emits the incorrect-regex warning.

| Variant | Output | Absolute action range | Artifact SHA-256 |
| --- | --- | --- | --- |
| RoboTwin | finite, nonzero 20×20 action trace | -0.0974 to 1.0048 | `b1ad86cd4eb124efda001f9b7ea2c7662775d4b9539786e9bc1dc29dc07a7e72` |
| UMI | finite, nonzero 50×20 action trace | -13.4444 to 38.9102 | `b92fdbb85dd0b87488e461ffbafd6253f5ad24f0b692071c84b329e8bd0d3cf2` |

Both results name `HyVLA.forward_evaluate(batch)['pred']` as the inference backend and include checkpoint configuration hashes. The local `model.safetensors` SHA-256 values match public Hugging Face LFS metadata captured in `official-assets.json`: RoboTwin `3bd6c16225f905a298340489d519498d4e5ecf5bcdd28a5c1df63e29894fef60`; UMI `bc1f3f0de1bd1ca5ce6830c5933fbb09c358d12ea71d212c8e7cb66f22c4b246`. Each file is 9,053,587,008 bytes.

Scope is **inference and output-structure validation only**. A photo of a tabletop toy and zero state are not a calibrated RoboTwin or UMI observation. There is no robot ground truth or safety validation, and UMI's large absolute values especially must not be interpreted as usable commands. The pre-fix artifacts remain available for comparison, but only the reruns support the current code.
