# Cosmos GPU 验证

Predict2 2B、Predict2.5 2B、Transfer2.5 2B edge 均完成 1280×704、33 帧、16 FPS、35 步生成，全部帧解码无错误。画面审核可见木地板、桌椅、玩具车等输入特征，前向运动连续。Transfer 的后续帧比起始输入更明亮、低对比，未见整片噪声或黑屏；这是带 edge 控制的外观变化，不声称首帧之后精确还原。

Transfer 控制视频复用旧的 Predict2.5 生成素材作为固定输入，模型本轮重新推理，没有将旧输出充当新结果。仅覆盖各 2B 权重的指定配置；14B、其他 Transfer 控制类型另需验证。

Wan2.2 本批因 dreamx-world 环境缺少 jmespath 失败；已补依赖并单独补跑，见 ../geometry-wan-followup-gpu-20260921/。本报告不将失败项计为通过。
