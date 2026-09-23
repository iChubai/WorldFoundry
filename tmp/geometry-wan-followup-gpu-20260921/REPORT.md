# Wan2.2 / LingBot-Map 补充验证

Wan2.2 TI2V5B 在 dreamx-world 环境补装 jmespath 后完成实际 CUDA 推理。1280×704、33 帧、16 FPS、30 步视频全部解码，输入场景、玩具车、桌椅和前向运动连续可辨；没有整片噪声或黑屏。仅覆盖 5B I2V 配置。

LingBot-Map stage1/streaming checkpoint 完成三视图推理，输出几何有限、相机和 PLY 非空，深度图检查符合场景结构。该 checkpoint 的精度与最终/long 权重不作等价保证。

DUSt3R 的实际执行暴露公共 wrapper 只接受文件而忽略有效 HF 权重目录，最终给 loader 传空字符串。已修复目录接受逻辑，补跑在 ../sam2-fix-gpu-20260921/；初始失败保留，不计通过。17 个缺失本地推理依赖从原提交只恢复 .py 至 ignored 目录，安装包包含性仍未验证。

SAM2 原始版本也被旧脚本的 float 二值 mask 索引错误打断，修正检查后在 ../sam2-fix-gpu-20260921/ 单列补跑完整图像+视频验证。
