# 几何模型本地 GPU 验证

四个配置均经过公共 pipeline 真实 CUDA 推理，权重加载无缺失/多余项，三视图数值有限，相机旋转矩阵有效，PLY 可读且非空。深度接触图和两个方向的点云投影已人工检查。

|配置|导出点数|证据|
|---|---:|---|
|pi3x|584511|pi3x/status.json|
|loger|575141|loger/status.json|
|loger-star|574344|loger-star/status.json|
|lingbot-map|49998|lingbot-map/status.json|

Pi3 使用 Pi3X 默认模式；LoGeR 普通与 star 模式分别测试；LingBot-Map 使用 long 权重和 streaming 默认模式。仅覆盖 cases.json 中的配置，未声明整个模型族完成。

输入来自相同室内视频的三个视角，无真值；检查支持推理集成与结构合理性，不代表尺度、重建精度或复杂场景鲁棒性。模型间坐标尺度与朝向不同。点云中仍有稀疏与离群区域。

逐次运行记录 imported-source-hashes.json；本批未修改模型源码。
