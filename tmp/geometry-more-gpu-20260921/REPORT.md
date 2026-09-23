# GeoCalib / InfiniteVGGT GPU 验证

GeoCalib pinhole 与 distorted 权重通过公共 wrapper 完整 CUDA 推理，无缺失/多余权重。记录实际入模 float32 RGB0–1。两者焦距分别约 948.7、839.6 像素，roll 约 0.059°、0.192°，pitch 约 -2.47°、-3.29°，重力向量归一化，协方差有限且半正定。up-field 叠加图方向符合桌椅和墙面，缺少真实相机标定，不能用两个预测相近当成精度证明。

InfiniteVGGT 使用 StreamVGGT checkpoints.pth，严格加载全部参数。三视图深度、点图、置信度、相机全部有限；旋转有效，456876 点 PLY 可读。深度图和点云两个方向投影已人工检查，室内结构可辨，允许稀疏和离群区域。未覆盖长序列稳定性/缓存极限。

CLIP 本批失败因新版独立环境缺少 torchvision，下一批补齐 PyTorch 匹配的 torchvision0.22.1 后复测成功，证据在 ../omega-all-gpu-20260921/。
