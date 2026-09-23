# Cosmos3 / LTX2.3 / SkyReels GPU 验证

Cosmos3 Nano：1280×720、33 帧、24 FPS、35 步实际推理，全帧解码及画面检查通过。保留室内场景与玩具车，出现顺光灰尘/轻雾及轻微前向运动，无整片损坏。

LTX2.3 I2V：distilled1.1 权重+空间上采样器，768×512、33 帧、24 FPS、11 步。视频完整解码并保持场景与前向运动。当前公共调用输出仅含 H264 视频流，未验证音频输出或 T2V/V2V 模式。

SkyReelsV2：选定 T2V 配置 960×544、33 帧、16 FPS、30 步，森林/蕨类与阳光匹配提示，完整解码无误。画面对比度高，不声称其他权重或长视频扩展已验证。

SkyReelsV3 正在执行 40 步、121 帧配置，尚未计为通过。

SkyReelsV3 completed40steps/121frames at1296x704/24fps in1903.6sec. Full decode passes,room coherent,but the visual camera movement pulls back instead of the requested forward movement. Recorded quality_concern pending conditioning/parity investigation; no semantic pass.
