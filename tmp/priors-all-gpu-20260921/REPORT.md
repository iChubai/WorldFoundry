# 几何先验 GPU 验证

DAP（真实 2:1 全景图）、DA3 Metric、Metric3D v2 large 完成实际推理、数值和深度图检查。DAP 的模型距离保留 checkpoint 原始约 1/100 尺度，不宣称真实米制精度。DA3 与 Metric3D 使用显式 focal_length=1000，缺少真实相机标定，不能据此声明真实距离准确。

Metric3D 唯一缺失 key 为 depth_model.encoder.mask_token，仅在 masks 非空时使用；实际图像推理不传 masks，encoder.forward_features 的默认 masks=None，不影响本次推理。其他参数均加载，输出有限且结构可辨。

Video Depth Anything 原始输出虽数值有限，但人工检查发现纹理破碎。根因是公共 wrapper 传 uint8 0–255，违背 DepthEstimationInput 要求的 float 0–1。原始结果保留，不计为通过。修复复测证据在 ../fix-regression-gpu-20260921/video-depth-anything-prior/。

公开 checkpoint 下载均锁定版本，SHA256 校验记录在 ../all-model-validation-20260921/depth-checkpoint-metadata/integrity.json。
