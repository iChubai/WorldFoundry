# VGGT-Omega / LingBot-Map / CLIP GPU 验证

VGGT-Omega 普通 checkpoint 生成 3×384×672 深度，文本对齐 checkpoint 生成 3×192×336 深度及有限非恒定的 text_alignment_embedding。权重与数值、相机旋转、深度图通过检查。未跑文本检索语义评测。原始脚本错误要求所有结果必须有 PLY，实际 VGGTResult.save 的约定是 numpy、相机 JSON、深度可视化；保留 harness-failed-status.json，对已有真实导出重校验成功，无模型更改或重复推理。

LingBot-Map 的 lingbot-map.pt/windowed 模式完成三视图推理，权重无缺失/多余项，相机有效，有限非空点云、深度图已检查。long/streaming 的先前结果独立保留；stage1 尚待验证。

CLIP ViT-B/32 经项目 EvalCrafter 实际 CUDA 评分。两份相同森林视频分别配森林和寿司提示，文本匹配得分 0.21049 > 0.16567；33 帧完整参与，两份时间一致性均为 0.9999904。验证覆盖已实现的两个评分入口，不声称整个 EvalCrafter 或指标准确率通过。环境为 PyTorch2.7.1+cu126、Transformers5.14.1、torchvision0.22.1+cu126，补装 OpenCV4.11，未绕过旧 Torch 的加载安全检查。
