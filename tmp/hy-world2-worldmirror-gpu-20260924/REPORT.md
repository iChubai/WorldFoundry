# HY-World 2.0 WorldMirror GPU 验证（2026-09-24）

- 公共入口：`HYWorld2Pipeline.from_pretrained(..., task="worldrecon", device="cuda")`，使用本地 `tencent/HY-World-2.0/HY-WorldMirror-2.0` 的真实 `model.safetensors`；GPU 2。
- 输入：`tmp/uni3c-validation-20260923/reference/data/demo_uni3c/video.mp4`。运行时以 DIS flow 和清晰度筛选抽取 4 帧，尺寸 518×294。
- 权重：1545/1593 个模型状态键与 checkpoint 完全匹配；其余 48 个均为由配置确定、初始化时生成的 RoPE 周期缓存。加载器现对其他缺失键或形状不符直接报错。修改后完整重跑成功，见 `run-after-loader-guard.log`。
- 输出：4 张深度数组/图、4 张法线图、4 组内外参、286,849 顶点点云及 290,156 顶点 Gaussian PLY。深度均为正且有限，标准差 0.131–0.143；相机旋转矩阵行列式接近 1，PLY 文件大小与头部顶点数一致且 XYZ 均有限。完整数值检查见 `output-check.json`。
- 目检首帧深度图可辨输入中的人形轮廓。该检查没有相机真值、深度真值或新视角渲染对照，故状态为 `verified_inference_only`，不宣称三维重建精度。

复现：`CUDA_VISIBLE_DEVICES=2 ../envs/worldfoundry-unified-cu121/bin/python tmp/hy-world2-worldmirror-gpu-20260924/validate.py`，随后用同环境运行 `check_output.py`。重跑产物在 `output/`；修改前产物保存在 `output-before-loader-guard/`。
