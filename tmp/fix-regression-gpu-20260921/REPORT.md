# Video Depth Anything / SAM3 修复复测

Video Depth Anything：公共几何先验 wrapper 现在把 uint8 RGB 转为 float32 0–1，与已有 DepthEstimationInput 契约保持一致。真实三帧 GPU 复测检查了入模 dtype/range（input-contract.json），权重严格加载、输出有限，深度图恢复可辨认的地板与桌椅结构。原始破碎结果保存在 ../priors-all-gpu-20260921/，没有计为通过。

SAM3：默认词表在本包缺失时使用已安装 OpenCLIP 的相同 BPE 文件（与上游 SAM3 解压后 SHA256 完全一致），找不到时给出可操作错误。融合 Linear+activation 原先强制输出 BF16，下一层 FP32 Linear 在非 autocast 路径报错；现在遵循调用者 autocast 设置，未开启时保持输入精度。CPU FP32/BF16、ReLU/GELU 四组结果与标准 Linear+activation 逐值一致；完整 FP32 GPU 模型复测成功。

SAM3 的 toy car 文本提示找到一个掩码，置信度 0.934，人工检查覆盖输入中的红色玩具车。当前 image builder 不启用 SAM2 交互分支，checkpoint 的 22 个 sam2_convs 参数因而是多余项；image 模型没有缺失参数。未声称视频跟踪或 SAM3.1 multiplex 已通过。
