# 本地 GPU 视频模型复测（2026-09-21）

本批四个模型均完成真实 H100 推理、完整视频解码、尺寸/帧率/帧数检查和抽帧人工检查。三个图生视频模型修复后重跑通过。

| 模型 | 分辨率 | 帧数 / FPS | 当前耗时（含依赖导入、权重加载） | 视频 | 抽帧 |
|---|---|---|---|---|---|
| dynamicrafter-512 | 512×320 | 16 / 8 | 222.86s | [MP4](dynamicrafter-512/demo.mp4) | [预览](dynamicrafter-512/demo.contact.jpg) |
| dynamicrafter-1024 | 1024×576 | 16 / 8 | 249.52s | [MP4](dynamicrafter-1024/demo.mp4) | [预览](dynamicrafter-1024/demo.contact.jpg) |
| videocrafter1-i2v | 512×320 | 16 / 8 | 236.72s | [MP4](videocrafter1-i2v/demo.mp4) | [预览](videocrafter1-i2v/demo.contact.jpg) |
| videocrafter2-t2v | 512×320 | 16 / 28 | 266.25s | [MP4](videocrafter2-t2v/demo.mp4) | [预览](videocrafter2-t2v/demo.contact.jpg) |

[视频浏览页](index.html)

## 本批修复

- `FrozenOpenCLIPImageEmbedderV2`：恢复上游的 bicubic resize、`[-1,1] → [0,1]` 和 CLIP 均值/标准差归一化。原实现直接归一化负值域，图像条件发生偏移。
- 同一编码器：适配现代 OpenCLIP 的 NLD 输入接口，避免把 patch 序列错当 batch；保留旧版 LND 接口兼容。现代 Transformer 即使内部 `batch_first=False`，公开 forward 仍接收 NLD。
- DynamiCrafter 512/1024 默认帧数从 50 改为模型配置的 16。采样步数仍为 50。

仅修改上述两个源码/配置文件；本批增量见 [task-only.patch](task-only.patch)，修改前快照在 `before/`。未提交或推送混有其他工作内容的工作树。

## 画面对比

修复前视频和日志保存在 `pre-fix/`，原始 JSON 内的视频路径为归档前路径。VideoCrafter 1 从无关纹理恢复为输入房间；DynamiCrafter 512 曝光闪烁明显改善。
平均相邻帧像素差（缩放至 320 宽；仅辅助描述变化，不代表准确率或视频质量评分）：

| 模型 | 修复前 | 修复后 |
|---|---|---|
| dynamicrafter-512 | 40.302 | 4.692 |
| dynamicrafter-1024 | 9.301 | 3.808 |
| videocrafter1-i2v | 5.737 | 1.880 |

## 验证证据

- 四个整模型权重加载均无 missing/unexpected keys；三个图生视频的附加 CLIP 权重加载也完整。
- 三次图生视频复测逐次计数确认 50 次 DDIM 采样；VideoCrafter 2 初次运行请求 50 步，未单独插入采样计数。
- 每个视频全部 16 帧成功解码，非空白、非静止，尺寸和帧率匹配请求。
- 小型真实 OpenCLIP 模块数值回归：两种内部布局与参考路径最大误差为 0；批处理/逐张计算一致；旧 LND 适配路径通过。旧实现有可测偏差，详见 [encoder-regression.json](encoder-regression.json)。
- DynamiCrafter 默认帧数与 UNet、图像投影器配置一致；Python 语法和限定文件 `git diff --check` 通过。

## 复现与边界

配置在 `all-cases.json` 和各模型 JSON；seed=42，使用公共 `from_pretrained` / pipeline 调用。I2V 输入为 `testcase/012_abandoned-room.png`，明确指定本地 checkpoint 路径。权重大小和输入 SHA256 见 `run-environment.json`。
GPU Python：`../envs/worldfoundry-unified-cu121/bin/python`。`run_video.py` 是本地验证工具；`supervise.py` 会暂停并恢复已核验的 gg 进程（重跑前应核验 `gg-restored.json` 中 PID）。
范围仅限本批四个配置，不代表模型目录全量通过。共享编码器的其他调用者未做端到端 GPU 复测；旧 OpenCLIP 使用兼容适配测试，未安装旧发行版。未声称整模型与上游输出逐像素一致。
测试脚本、下载的对照源码、报告和输出均位于忽略目录 `tmp/`，未增加训练功能或许可证文档。

GPU 任务结束后已恢复 `gg`，进程记录见 [gg-restored.json](gg-restored.json)。

## dynamicrafter: dynamicrafter-512

verified_configuration: Recorded configuration only; no full-family or all-parameter guarantee Same concrete checkpoint/run as dynamicrafter-512-i2v;shared evidence for alias/family label,not another GPU run or all-family validation.

## videocrafter: videocrafter2-t2v

verified_configuration: Recorded configuration only; no full-family or all-parameter guarantee Same concrete checkpoint/run as videocrafter2-t2v;shared evidence for alias/family label,not another GPU run or all-family validation.
