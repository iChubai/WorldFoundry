# ATI-Wan2.1-14B GPU 验证（2026-09-24）

通过公共 `ATIPipeline.from_pretrained` 加载本地 `bytedance-research/ATI` 权重，以及 `Wan2.1-I2V-14B-480P` 底座（`../ckpts/` 中的链接指向 WorldArena 的真实文件）。输入为底座示例中的戴墨镜白猫照片；使用 25 条合成可见点轨迹，121 个轨迹采样点，各点水平移动 8 像素。模型要求输出 81 帧。

先完成 8 步短测，再以默认 40 步完整复测；两次均通过真实 GPU 推理并生成 81 帧、16 fps 的视频。输出保留输入纵横比，实际为 544×720（`width=832, height=480` 在此路径中作为最大面积约束）。40 步视频 81/81 帧全部解码，首/中/末帧均保留猫、墨镜、冲浪板和海面，画面运动连续；相邻帧像素 MAE 中位数 11.43/255、最大 23.03/255，首末帧 MAE 68.13/255。`status-40step.json`、`ati-cat-40step.quality.json` 和 `run-40step.log` 保存结构与数值证据。

状态为 `verified_inference_only`：合成轨迹只用于检查加载、条件输入与输出结构，没有点轨迹真值、相机控制精度或官方产物对照。原先“缺 Wan2.1 I2V 480P 底座”的判断已由实际可读的链接和成功推理纠正。

复现：`ATI_STEPS=40 NUMEXPR_NUM_THREADS=16 OMP_NUM_THREADS=16 CUDA_VISIBLE_DEVICES=2 ../envs/worldfoundry-unified-cu121/bin/python tmp/ati-gpu-20260924/validate.py`，再运行 `check_video.py` 检查视频。模型、权重与视频产物不纳入 Git。
