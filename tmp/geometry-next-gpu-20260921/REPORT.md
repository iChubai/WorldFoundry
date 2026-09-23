# 本地 GPU 验证

|配置|深度尺寸|结果|
|---|---|---|
|vggt|[[3, 518, 518, 1]]|实际推理、数值、图像检查通过|
|cut3r|[[3, 1, 288, 512]]|实际推理、数值、图像检查通过|
|depth-anything-v3|[[3, 294, 504]]|实际推理、数值、图像检查通过|

深度图已逐项人工检查，输入场景的地板、门洞、桌椅结构可辨认，无整幅常量或非有限值。Depth Anything 1/2 为相对逆深度，近处亮、远处暗；不能与米制深度直接比较，也不能将尺度数值当成真实距离。无真值误差评估。

所有检查使用公共 pipeline；Depth Anything 1 另外调用同一 pipeline.process 获取原始深度，避免仅检查保存的彩色图。

三模型点云投影和相机矩阵已检查，geometry-checks.json 包含旋转正交误差与点数；有内参时检查正焦距。点云中仍有稀疏区域，无真值重建精度结论。

VGGT 初次被验证脚本误把带 depth 名称的 XYZ 点图判成非负深度；原始推理与导出有效，保留 harness-failed-status.json，并对导出重新校验，没有重跑或修改模型。Depth Anything 2 同批只执行 preflight，该失败未计为通过，实际补跑证据位于 ../depth-all-gpu-20260921/。

Depth Anything 3 的六个 missing 名称属于四个辅助头共用的 LayerNorm；../da3-load-review-20260921/shared-norm-check.json 证明对象共享且逐参数精确等于 checkpoint 中存储的别名。无需修改模型。
