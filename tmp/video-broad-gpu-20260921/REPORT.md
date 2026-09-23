# 视频模型当前源码复测

LTX-Video 0.9.8、Wan2.1 T2V 1.3B 和 Wan2.1 VACE 14B 已完成真实 GPU 推理，完整解码及帧数/分辨率/帧率检查通过，接触图已审核。Mochi 已完成 64 步、84 帧、848×480、30 FPS 推理，完整解码和接触图检查通过，森林镜头运动连续。

LTX 场景保持室内结构并随镜头运动，Wan T2V 呈现森林和蕨类，VACE 呈现参考风格的日照房间。VACE 使用参考图条件，其首帧不是原图逐像素复制。均仅覆盖 cases.json 的配置；历史记录没有直接计入本次通过。

每个结果记录 imported-source-hashes.json；本批未修改模型源码。

## ltx-video-i2v: ltx-video-098

verified_configuration: Recorded checkpoint/config only; full ffmpeg decode, exact metadata and contact sheet inspected. VACE reference-image conditioning does not guarantee identical first frame. Same concrete checkpoint/run as ltx-video;shared evidence for alias/family label,not another GPU run or all-family validation.

## mochi-1-preview-t2v: mochi-1-fixed

verified_configuration: Recorded configuration and checkpoint only; no accuracy/whole-family guarantee Same concrete checkpoint/run as mochi-1;shared evidence for alias/family label,not another GPU run or all-family validation.

## wan2.1: wan21-t2v-13b

verified_configuration: Recorded checkpoint/config only; full ffmpeg decode, exact metadata and contact sheet inspected. VACE reference-image conditioning does not guarantee identical first frame. Same concrete checkpoint/run as wan2.1-t2v-1.3b;shared evidence for alias/family label,not another GPU run or all-family validation.
