# UniK3D 验证与修复

ViT-L checkpoint 在 automatic/panorama 两个公共入口完成 GPU 推理；704×1280 室内图、1024×2048 全景图均生成有限正深度，图中家具、门洞和地面层次符合输入。权重加载无 missing/unexpected keys。

本地恢复的 UniK3D 编码器仍引用已被共享 DINOv2 重构移除的工厂。现使用共享 DINOv2 并保留原始 register checkpoint 参数、LayerNorm epsilon、逐层 BHWC 特征和 class token 契约。真实权重对旧参考实现/新编码器均严格加载成功，固定输入 56×70 的全部 24 层特征与 class token 在 CPU 上逐值一致（encoder-parity.json），另有完整 CUDA 推理结果。仅恢复的 ignored 本地推理依赖已验证，安装包覆盖尚未验证。

DepthPro 的证据在 ../base-extra-gpu-20260921/depth-pro/，完整 checkpoint 无缺失，704×1280 深度 0.440–14.101，预测 focal=638.403，深度图人工审核通过，缺少标定真值，不声明米制精度。

CLIP 在统一环境被 PyTorch 版本安全要求阻止加载，转用 PyTorch2.7.1 环境后需要补齐 OpenCV。失败均保留，后续复测单列，不计通过。
