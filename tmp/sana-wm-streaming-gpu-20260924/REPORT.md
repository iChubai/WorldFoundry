# SANA-WM Streaming GPU2 验证进展

完整本地 streaming 权重已由公开 pipeline 加载；25 帧、4 步请求进入 LTX-2 refiner，但本机 Diffusers 的 `LTX2VideoTransformer3DModel.forward` 不接受 `video_self_attention_mask`，见 `status.json` 与 `run.log`。首轮 33 帧请求因 active latent 4 无法被 chunk size 3 整除，被预期参数约束拒绝；25 帧是有效请求。

已在 `sana_refiner.py` 实施按 block 注入布尔前缀遮罩的兼容修复，GPU 复测尚待完成；当前不得声称推理通过。
