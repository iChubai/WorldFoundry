# 动作模型本地 GPU 验证

四个配置均调用公共 pipeline，实际在独立 H100 上加载权重并生成动作。所有验证、输入和输出均保存在本目录。

|配置|输出|验证状态|
|---|---|---|
|openvla-libero-10|[1, 7]|离线推理及动作结构检查通过|
|openvla-7b|[1, 7]|离线推理及动作结构检查通过|
|xvla-libero|[30, 20]|离线推理及动作结构检查通过|
|xvla-foundation-libero|[30, 20]|离线推理及动作结构检查通过|

OpenVLA 检查了动作维度、有限值及 checkpoint 指定归一化键对应的分位范围。X-VLA 检查了 30×20 动作块、有限值、夹爪 [0,1] 范围及第一机械臂旋转表示不退化。

输入来自 lerobot/libero 的 episode 0/frame 0，并通过 episodes 元数据确认两段视频的起点均为 0。任务是把白色杯子放到左侧盘子、黄白杯子放到右侧盘子。X-VLA 使用官方 LIBERO 客户端首步构造规则，将 xyz、轴角转换后的旋转列、零夹爪和空置第二机械臂状态组成 20D proprio；主视图旋转 180 度，腕部视图保持原样。

限制：这里只验证离线推理集成，不宣称闭环任务完成或动作准确率。数据状态语义按照 LIBERO 的 xyz/axis-angle/finger convention 解释；本次没有重建原仿真环境逐帧核对摄像机与状态。OpenVLA 基础版使用 Bridge 归一化和 LIBERO 输入，属于跨分布检查。X-VLA 基础版在 domain=3 的输出也不能据此断言已掌握任务。因此台账使用 verified_inference_only，与视觉审核通过的配置区分。

权重加载诊断：Transformers 会对单个 parameter 分别调用 load_state_dict，导致记录中出现同模块另一 parameter 为 missing 的中间结果；不能把这些中间结果直接当作完整模型缺失权重，也不以此声称所有参数严格一一对应。实际推理成功，无最终缺失权重警告。

修复：公共 OpenVLA model_path 映射到 checkpoint_dir；显式路径不存在时抛出错误，不再静默选择默认 checkpoint。openvla-path-check.json、openvla-missing-path-check.json 验证路径选择。后一项为 GPU 推理后补充的加载前检查，只改变非法显式路径处理。task-only.patch 是本任务增量，保留并发修改。
