# Helios / LongCat 本地 GPU 验证

Helios Base、Mid、Distilled 分别加载各自发布权重，640×384、33 帧、24 FPS。Base/Mid 50 步，Distilled 使用对应金字塔 [2,2,2] 步配置。三条视频全帧解码和尺寸/帧数/帧率检查通过；森林、蕨类、阳光、前向运动可辨。Mid/Distilled 对比度更强，未见整片噪声、黑屏或场景跳切。

LongCat 普通权重完成 832×480、49 帧、15 FPS、30 步 T2V。视频完整解码，森林近处蕨叶随相机前移，画面连续。此处没有把普通模型结果扩展为蒸馏权重、图生视频或长时续写通过。

证据包括逐 case manifests、status、输出视频、接触图、完整解码统计和 imported-source-hashes。原有历史样例仅作为输入配方来源，输出为本轮重新生成。
