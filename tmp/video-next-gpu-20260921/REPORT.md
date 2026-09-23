# 本地 GPU 验证：VideoCrafter 1 T2V / T2V-Turbo / ZeroScope / I2VGen-XL

4 个模型均完成实际推理并通过全帧解码、尺寸、帧率及帧数检查；3 个样本画面检查通过，I2VGen-XL 存在质量问题，不能记为完整通过。

|模型|尺寸|帧数/FPS|画面|视频|
|---|---|---|---|---|
|videocrafter1-t2v|1024×576|16/8|passed_sample_review|[MP4](videocrafter1-t2v/demo.mp4) / [抽帧](videocrafter1-t2v/demo.contact.jpg)|
|t2v-turbo|512×320|16/8|passed_sample_review|[MP4](t2v-turbo/demo.mp4) / [抽帧](t2v-turbo/demo.contact.jpg)|
|zeroscope|576×320|24/8|passed_sample_review|[MP4](zeroscope/demo.mp4) / [抽帧](zeroscope/demo.contact.jpg)|
|i2vgen-xl|1280×704|16/16|quality_concern|[MP4](i2vgen-xl/demo.mp4) / [抽帧](i2vgen-xl/demo.contact.jpg)|

ZeroScope 公共入口原先把 model_path 映射成 repo_root，现修正为 model_path。独立公共加载检查确认显式替代路径直达运行时，见 zeroscope-path-check.json。未改动其他并发修改，增量补丁见 task-only.patch。

T2V-Turbo 的 UNet、文字编码器和 VAE 均无缺失/多余键；VideoCrafter 整模型加载完整。Diffusers 可能绕过 Module.load_state_dict，未据此宣称逐参数检查通过。

I2VGen-XL 的后半段变暗、末帧异常。此前不同配置的官方实现对照已复现质量问题，但不能代替本配置验证。当前正在 tmp/i2vgen-quality-gpu-20260921 做 FP32 和同参数官方实现对照。

样本使用 seed 42，详细参数在 cases.json。本报告仅覆盖记录的配置，不代表全部模型或变体通过。全量 goal 和覆盖台账在 ../all-model-validation-20260921/。
