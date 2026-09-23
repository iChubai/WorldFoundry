# 全模型验证台账

范围合并模型目录、公共绑定、运行配置与额外变体。基础 checkpoint 能力另列于 base-model-checks.json。历史运行仅作为线索，不自动计为当前通过。

当前状态：{"blocked_external": 33, "verified_configuration": 139, "verified_inference_only": 119, "not_run": 91, "quality_concern": 47, "failed_integration": 11}

verified_configuration 仅代表记录配置的真实推理与适用输出检查通过；verified_inference_only 仅通过推理/输出结构，未完成任务语义验证；blocked_external 为已实际核查的外部阻塞；failed_integration 为已实际复现、尚未完成修复复测的项目集成失败；failed_resource_conflict 为共享 GPU 资源冲突导致的运行失败，需换卡复测。接口检查不能替代推理。另见 runtime-variant-obligations.json 中未被目录显式列出的运行变体。

|模型/变体|分类|当前验证|接口检查|证据|
|---|---|---|---|---|
|hailuo-2p3|hosted_api|blocked_external|passed|tmp/all-model-validation-20260921/hosted-preflight.json|
|kling-api|hosted_api|blocked_external|passed|tmp/all-model-validation-20260921/hosted-preflight.json|
|luma-ray2|hosted_api|blocked_external|passed|tmp/all-model-validation-20260921/hosted-preflight.json|
|runway-gen4p5|hosted_api|blocked_external|passed|tmp/all-model-validation-20260921/hosted-preflight.json|
|sora2|hosted_api|blocked_external|passed|tmp/all-model-validation-20260921/hosted-preflight.json|
|veo3|hosted_api|blocked_external|passed|tmp/all-model-validation-20260921/hosted-preflight.json|
|wan-2p5|hosted_api|blocked_external|passed|tmp/all-model-validation-20260921/hosted-preflight.json|
|wan-2p6|hosted_api|blocked_external|passed|tmp/all-model-validation-20260921/hosted-preflight.json|
|wan-2p7|hosted_api|blocked_external|passed|tmp/all-model-validation-20260921/hosted-preflight.json|
|worldlabs|hosted_api|blocked_external|passed|tmp/all-model-validation-20260921/hosted-preflight.json|
|cut3r|three_d_four_d|verified_configuration|passed|tmp/geometry-next-gpu-20260921/cut3r/status.json|
|dap|three_d_four_d|verified_configuration|passed|tmp/priors-all-gpu-20260921/dap/status.json|
|depth-anything-v1|three_d_four_d|verified_configuration|passed|tmp/depth-all-gpu-20260921/depth-anything-v1-small/status.json; tmp/depth-all-gpu-20260921/depth-anything-v1-base/status.json; tmp/depth-all-gpu-20260921/depth-anything-v1-large/status.json|
|depth-anything-v2-prior|three_d_four_d|verified_configuration|passed|tmp/depth-all-gpu-20260921/depth-anything-v2-prior/status.json|
|depth-anything-v3|three_d_four_d|verified_configuration|passed|tmp/geometry-next-gpu-20260921/depth-anything-v3/status.json; tmp/sam-pi3-da3-gpu-20260921/da3-large-1.1/status.json|
|depth-anything-v3-prior|three_d_four_d|verified_configuration|passed|tmp/priors-all-gpu-20260921/depth-anything-v3-prior/status.json|
|dust3r|three_d_four_d|verified_configuration|passed|tmp/sam2-fix-gpu-20260921/dust3r/status.json|
|dust3r-base-model|three_d_four_d|verified_configuration|passed|tmp/sam2-fix-gpu-20260921/dust3r-base-model/status.json|
|dvlt|three_d_four_d|verified_configuration|passed|tmp/resume-gpu-20260921/dvlt/status.json|
|fantasyworld|three_d_four_d|verified_inference_only|passed|tmp/fantasyworld-wan21-gpu-20260923/fantasyworld-wan21-camera/quality-review.json|
|flashworld|three_d_four_d|verified_configuration|passed|tmp/scene-all-gpu-20260921/flashworld/status.json|
|geocalib-prior|three_d_four_d|verified_configuration|passed|tmp/geometry-more-gpu-20260921/geocalib-pinhole/status.json; tmp/geometry-more-gpu-20260921/geocalib-distorted/status.json|
|geometry-prior|three_d_four_d|not_run|passed||
|infinite-vggt|three_d_four_d|verified_configuration|passed|tmp/geometry-more-gpu-20260921/infinite-vggt/status.json|
|lagernvs|three_d_four_d|verified_inference_only|passed|tmp/lagernvs-native-gpu-20260923/lagernvs-single/quality-review.json; tmp/lagernvs-native-gpu-20260923/lagernvs-two-view/quality-review.json|
|lingbot-map|three_d_four_d|verified_configuration|passed|tmp/geometry-all-gpu-20260921/lingbot-map/status.json; tmp/omega-all-gpu-20260921/lingbot-map-windowed/status.json; tmp/geometry-wan-followup-gpu-20260921/lingbot-map-stage1/status.json|
|loger|three_d_four_d|verified_configuration|passed|tmp/geometry-all-gpu-20260921/loger/status.json; tmp/geometry-all-gpu-20260921/loger-star/status.json|
|lyra|three_d_four_d|verified_inference_only|passed|tmp/lyra2-vaefix-gpu-20260923/lyra2-forward/status.json|
|metric3d-prior|three_d_four_d|verified_configuration|passed|tmp/priors-all-gpu-20260921/metric3d-prior/status.json|
|monst3r|three_d_four_d|verified_configuration|passed|tmp/scene-all-gpu-20260921/monst3r/status.json|
|mvdiffusion|three_d_four_d|quality_concern|passed|tmp/mvdiffusion-artifactfix-gpu-20260923/mvdiffusion-outpaint-artifactfix/quality-review.json|
|neoverse|three_d_four_d|quality_concern|passed|tmp/neoverse-final-gpu-20260923/neoverse-lora-final-image/quality-review.json; tmp/neoverse-final-gpu-20260923/neoverse-lora-final-video/quality-review.json; tmp/neoverse-final-gpu-20260923/neoverse-base-final-image/quality-review.json; tmp/neoverse-final-gpu-20260923/neoverse-base-final-video/quality-review.json; tmp/neoverse-validation-20260923/family-review.json|
|pi3|three_d_four_d|verified_configuration|passed|tmp/geometry-all-gpu-20260921/pi3x/status.json; tmp/sam-pi3-da3-gpu-20260921/pi3-original/status.json|
|pixelsplat|three_d_four_d|verified_configuration|passed|tmp/pixelsplat-final-gpu-20260923/pixelsplat-re10k-final/quality-review.json; tmp/pixelsplat-final-gpu-20260923/pixelsplat-acid-final/quality-review.json|
|prior-depth-anything|three_d_four_d|verified_configuration|passed|/mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-hl02/hadoop-aipnlp/3A/multimodal/yangboxue/arena/WorldFoundry/tmp/priorda-execute-gpu-20260922/prior-depth-anything-vitb/status.json|
|recammaster|three_d_four_d|verified_inference_only|passed|tmp/recammaster-native-gpu-20260923/recammaster-static/quality-review.json; tmp/recammaster-native-gpu-20260923/recammaster-tilt/quality-review.json|
|splatt3r|three_d_four_d|verified_configuration|passed|tmp/resume-gpu-20260921/splatt3r/status.json|
|stable-virtual-camera|three_d_four_d|quality_concern|passed|tmp/scene-matrix-fix-gpu-20260921/stable-camera-zoom-fix/status.json|
|track-anything-prior|three_d_four_d|not_run|passed||
|unidepth-v2-prior|three_d_four_d|verified_configuration|passed|tmp/resume-gpu-20260921/unidepth-v2/status.json|
|unik3d-prior|three_d_four_d|verified_configuration|passed|tmp/base-extra-fix-gpu-20260921/unik3d-auto/status.json|
|vggt|three_d_four_d|verified_configuration|passed|tmp/geometry-next-gpu-20260921/vggt/status.json|
|vggt-omega|three_d_four_d|verified_configuration|passed|tmp/omega-all-gpu-20260921/vggt-omega/status.json; tmp/omega-all-gpu-20260921/vggt-omega-alignment/status.json|
|video-depth-anything-prior|three_d_four_d|verified_configuration|passed|tmp/fix-regression-gpu-20260921/video-depth-anything-prior/status.json|
|wonderjourney|three_d_four_d|not_run|passed||
|wonderworld|three_d_four_d|not_run|passed||
|worldgen|three_d_four_d|blocked_external|passed|tmp/external-blockers-20260924/asset-preflight.json|
|4dworldbench|uncataloged|not_run|pending||
|allegro-ti2v|uncataloged|verified_configuration|passed|tmp/video-more-gpu-20260921/allegro-native/status.json|
|cogvideox-2b-t2v|uncataloged|verified_configuration|passed|tmp/cog-all-gpu-20260921/cogvideox-2b-t2v/status.json|
|cogvideox-5b-i2v|uncataloged|verified_configuration|passed|tmp/cog-all-gpu-20260921/cogvideox-5b-i2v/status.json|
|cogvideox-5b-t2v|uncataloged|verified_configuration|passed|tmp/cog-all-gpu-20260921/cogvideox-5b-t2v/status.json|
|cosmos-predict2-14b-video2world|uncataloged|verified_configuration|passed|/mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-hl02/hadoop-aipnlp/3A/multimodal/yangboxue/arena/WorldFoundry/tmp/cosmos2-14-gpu-20260922/cosmos-predict2-14b/status.json|
|cosmos-predict2-2b-video2world|uncataloged|verified_configuration|passed|tmp/cosmos-wan-gpu-20260921/cosmos-predict2-2b/status.json|
|cosmos-predict2.5-14b|uncataloged|quality_concern|passed|/mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-hl02/hadoop-aipnlp/3A/multimodal/yangboxue/arena/WorldFoundry/tmp/cosmos14-gpu-20260922/cosmos-predict25-14b/status.json|
|cosmos-predict2.5-2b|uncataloged|verified_configuration|passed|tmp/cosmos-wan-gpu-20260921/cosmos-predict25-2b/status.json|
|cosmos-transfer2.5-2b-controlled-video|uncataloged|verified_configuration|passed|tmp/cosmos-wan-gpu-20260921/cosmos-transfer25-edge/status.json|
|cosmos3-nano|uncataloged|verified_configuration|passed|tmp/flagship-video-gpu-20260921/cosmos3-nano-fixed/status.json|
|cosmos3-super|uncataloged|verified_inference_only|passed|tmp/next-world-models-gpu-20260923/cosmos3-super/status.json|
|cuda-gpu|uncataloged|not_run|pending||
|diamond-csgo|uncataloged|verified_configuration|passed|tmp/diamond-csgo-preflight-20260922/result.json; tmp/diamond-csgo-integration-20260922/RECORDED.json|
|dynamicrafter-1024-i2v|uncataloged|verified_configuration|passed|tmp/crafter-gpu-20260921/dynamicrafter-1024/status.json|
|dynamicrafter-512-i2v|uncataloged|verified_configuration|passed|tmp/crafter-gpu-20260921/dynamicrafter-512/status.json|
|easyanimate-i2v|uncataloged|verified_configuration|passed|tmp/video-more-gpu-20260921/easyanimate/status.json|
|fantasyworld-wan21|uncataloged|verified_inference_only|passed|tmp/fantasyworld-wan21-gpu-20260923/fantasyworld-wan21-camera/quality-review.json|
|fantasyworld-wan22|uncataloged|verified_inference_only|passed|tmp/fantasyworld-wan22-gpu-20260923/fantasyworld-wan22-camera/quality-review.json|
|fastvideo-causal-wan2.2-i2v-14b|uncataloged|blocked_external|passed|tmp/fastvideo-causal-wan-preflight-20260924/integrity.json|
|hunyuanvideo-1.5-i2v|uncataloged|verified_configuration|passed|/mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-hl02/hadoop-aipnlp/3A/multimodal/yangboxue/arena/WorldFoundry/tmp/hunyuan15-fixed-gpu-20260922/hunyuanvideo15-i2v/status.json|
|hunyuanvideo-1.5-t2v|uncataloged|verified_configuration|passed|/mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-hl02/hadoop-aipnlp/3A/multimodal/yangboxue/arena/WorldFoundry/tmp/hunyuan15-fixed-gpu-20260922/hunyuanvideo15-t2v/status.json|
|hunyuanvideo-i2v|uncataloged|quality_concern|passed|tmp/native-video-more-gpu-20260921/hunyuanvideo-i2v/status.json; tmp/hunyuanvideo-i2v-debug-20260924/quality-review.json|
|hunyuanvideo-t2v|uncataloged|quality_concern|passed|tmp/native-video-more-gpu-20260921/hunyuanvideo-t2v/status.json|
|longsana-video-2b-480p|uncataloged|verified_configuration|passed|tmp/longsana-fixed-gpu-20260922/longsana-longlive-81f/status.json; tmp/longsana-fixed-gpu-20260922/longsana-longlive-321f/status.json|
|ltx-2-i2v|uncataloged|verified_inference_only|passed|/mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-hl02/hadoop-aipnlp/3A/multimodal/yangboxue/arena/WorldFoundry/tmp/ltx-more-gpu-20260922/ltx2-i2v/status.json|
|ltx-2-t2v|uncataloged|verified_inference_only|pending|tmp/ltx2-t2v-gpu-20260922/ltx2-t2v/status.json|
|ltx-2-v2v|uncataloged|failed_integration|pending|tmp/ltx2-v2v-gpu-20260924/preflight.json|
|ltx-2.3-i2v|uncataloged|verified_configuration|passed|tmp/flagship-video-gpu-20260921/ltx23-i2v/status.json|
|ltx-2.3-t2v|uncataloged|verified_inference_only|passed|/mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-hl02/hadoop-aipnlp/3A/multimodal/yangboxue/arena/WorldFoundry/tmp/ltx-more-gpu-20260922/ltx23-t2v/status.json|
|ltx-2.3-v2v|uncataloged|failed_integration|pending|tmp/ltx23-v2v-gpu-20260924/preflight.json|
|ltx-video-i2v|uncataloged|verified_configuration|passed|tmp/video-broad-gpu-20260921/ltx-video-098/status.json|
|lyra-1|uncataloged|not_run|passed||
|lyra-2|uncataloged|verified_inference_only|passed|tmp/lyra2-vaefix-gpu-20260923/lyra2-forward/quality-review.json|
|matrix-game-2-universal-ref-image-seed42-15f|uncataloged|verified_configuration|passed|tmp/matrix-game2-actions-seedfix-gpu-20260923/quality-review.json|
|mochi-1-preview-t2v|uncataloged|verified_configuration|passed|tmp/video-broad-gpu-20260921/mochi-1-fixed/status.json|
|sana-1600m-1024px|uncataloged|quality_concern|passed|tmp/sana-scale-fixed-gpu-20260921/sana-1600m-1024px/status.json; tmp/sana-shift-fixed-gpu-20260921/sana-1600m-1024px/status.json; tmp/sana-shift-fixed-gpu-20260921/sana-1600m-1024px-fp32/status.json|
|sana-1600m-1024px-bf16|uncataloged|quality_concern|passed|tmp/sana-shift-fixed-gpu-20260921/sana-1600m-1024px-bf16/status.json|
|sana-1600m-1024px-multiling|uncataloged|quality_concern|passed|tmp/sana-images-gpu-20260921/sana-1600m-1024px-multiling/status.json|
|sana-1600m-2k-bf16|uncataloged|quality_concern|passed|tmp/sana-shift-fixed-gpu-20260921/sana-1600m-2k-bf16/status.json|
|sana-1600m-4k-bf16|uncataloged|quality_concern|passed|tmp/sana-4k-tiled-gpu-20260921/sana-1600m-4k-bf16/status.json|
|sana-1600m-512px|uncataloged|quality_concern|passed|tmp/sana-images-gpu-20260921/sana-1600m-512px/status.json|
|sana-1600m-512px-multiling|uncataloged|quality_concern|passed|tmp/sana-shift-fixed-gpu-20260921/sana-1600m-512px-multiling/status.json|
|sana-600m-1024px|uncataloged|verified_configuration|passed|tmp/sana-images-gpu-20260921/sana-600m-1024px/status.json|
|sana-600m-512px|uncataloged|verified_configuration|passed|tmp/sana-images-gpu-20260921/sana-600m-512px/status.json|
|sana-controlnet-1600m-1024px-bf16|uncataloged|verified_configuration|passed|tmp/sana-control-sprint-gpu-20260921/sana-controlnet-1600m-1024px-bf16/status.json|
|sana-controlnet-600m-1024px|uncataloged|verified_configuration|passed|tmp/sana-control-sprint-gpu-20260921/sana-controlnet-600m-1024px/status.json|
|sana-sprint-1600m-1024px|uncataloged|quality_concern|passed|tmp/sana-images-gpu-20260921/sana-sprint-1600m-1024px/status.json|
|sana-sprint-600m-1024px|uncataloged|quality_concern|passed|tmp/sana-control-sprint-gpu-20260921/sana-sprint-600m-1024px/status.json|
|sana-streaming-2b-720p|uncataloged|quality_concern|passed|tmp/sana-video-gpu-20260921/sana-streaming-2b-720p/status.json|
|sana-streaming-bidirectional-2b-720p|uncataloged|verified_inference_only|passed|tmp/sana-video-gpu-20260921/sana-streaming-bidirectional-2b-720p/status.json|
|sana-video-2b-480p|uncataloged|verified_configuration|passed|tmp/sana-video-gpu-20260921/sana-video-2b-480p/status.json|
|sana-video-2b-720p|uncataloged|verified_configuration|passed|tmp/sana-video-gpu-20260921/sana-video-2b-720p/status.json|
|sana-wm-streaming|uncataloged|failed_integration|passed|tmp/sana-wm-streaming-gpu-20260924/status.json|
|sana1p5-1600m-1024px|uncataloged|verified_configuration|passed|tmp/sana-shift-fixed-gpu-20260921/sana1p5-1600m-1024px/status.json|
|sana1p5-4800m-1024px|uncataloged|verified_configuration|passed|tmp/sana-images-gpu-20260921/sana1p5-4800m-1024px/status.json|
|wan2.1-i2v-14b-480p|uncataloged|verified_configuration|passed|tmp/native-video-more-gpu-20260921/wan21-i2v-480p/status.json|
|wan2.1-i2v-14b-720p|uncataloged|verified_configuration|passed|tmp/native-video-more-gpu-20260921/wan21-i2v-720p/status.json|
|wan2.1-t2v-1.3b|uncataloged|verified_configuration|passed|tmp/video-broad-gpu-20260921/wan21-t2v-13b/status.json|
|wan2.1-t2v-1.3b-832x480-81f|uncataloged|verified_configuration|passed|tmp/wan-full-gpu-20260922/wan21-t2v-13b-81f/status.json|
|wan2.1-t2v-14b|uncataloged|verified_configuration|passed|tmp/native-video-more-gpu-20260921/wan21-t2v-14b/status.json|
|wan2.2-i2v-a14b|uncataloged|verified_configuration|passed|tmp/native-video-more-gpu-20260921/wan22-i2v-a14b/status.json|
|wan2.2-t2v-a14b|uncataloged|verified_configuration|passed|tmp/native-video-more-gpu-20260921/wan22-t2v-a14b/status.json|
|wan2.2-ti2v-5b|uncataloged|verified_configuration|passed|tmp/geometry-wan-followup-gpu-20260921/wan22-ti2v-5b/status.json|
|wan2.2-ti2v-5b-1280x704-121f|uncataloged|quality_concern|passed|tmp/wan-full-gpu-20260922/wan22-ti2v-5b-121f/status.json|
|yume-1p5|uncataloged|verified_inference_only|passed|tmp/yume-1p5-gpu-20260923/status.json; tmp/yume-1p5-gpu-20260923/status.json|
|allegro|video|verified_configuration|passed|tmp/video-more-gpu-20260921/allegro-native/status.json|
|animatediff|video|verified_inference_only|passed|/mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-hl02/hadoop-aipnlp/3A/multimodal/yangboxue/arena/WorldFoundry/tmp/classic-video-audited-gpu-20260922/animatediff-v1/status.json; /mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-hl02/hadoop-aipnlp/3A/multimodal/yangboxue/arena/WorldFoundry/tmp/classic-video-audited-gpu-20260922/animatediff-v2/status.json; /mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-hl02/hadoop-aipnlp/3A/multimodal/yangboxue/arena/WorldFoundry/tmp/classic-video-audited-gpu-20260922/animatediff-v3/status.json|
|bernini|video|quality_concern|passed|tmp/bernini-planner-final-load-gpu-20260922/bernini-t2v/quality-review.json; tmp/bernini-planner-tasks-gpu-20260923/bernini-t2i/quality-review.json; tmp/bernini-planner-tasks-gpu-20260923/bernini-i2i/quality-review.json; tmp/bernini-planner-tasks-gpu-20260923/bernini-r2v/quality-review.json; tmp/bernini-planner-tasks-gpu-20260923/bernini-v2v/quality-review.json; tmp/bernini-planner-tasks-gpu-20260923/bernini-rv2v/quality-review.json|
|bernini-r-1.3b|video|quality_concern|passed|tmp/bernini-native-gpu-20260922/bernini-r-1.3b-t2v/quality-review.json; tmp/bernini-renderer-tasks-gpu-20260922/bernini-r-1.3b-t2i/quality-review.json; tmp/bernini-renderer-tasks-gpu-20260922/bernini-r-1.3b-i2i/quality-review.json; tmp/bernini-renderer-tasks-gpu-20260922/bernini-r-1.3b-r2v/quality-review.json; tmp/bernini-renderer-tasks-gpu-20260922/bernini-r-1.3b-v2v/quality-review.json; tmp/bernini-renderer-tasks-gpu-20260922/bernini-r-1.3b-rv2v/quality-review.json|
|bernini-r-14b|video|quality_concern|passed|tmp/bernini-native-gpu-20260922/bernini-r-14b-t2v/quality-review.json; tmp/bernini-renderer14-tasks-gpu-20260922/bernini-r-14b-t2i/quality-review.json; tmp/bernini-renderer14-tasks-gpu-20260922/bernini-r-14b-i2i/quality-review.json; tmp/bernini-renderer14-tasks-gpu-20260922/bernini-r-14b-r2v/quality-review.json; tmp/bernini-renderer14-tasks-gpu-20260922/bernini-r-14b-v2v/quality-review.json; tmp/bernini-renderer14-tasks-gpu-20260922/bernini-r-14b-rv2v/quality-review.json|
|cogvideox|video|verified_configuration|passed|tmp/cog-all-gpu-20260921/cogvideox-5b-t2v/status.json|
|dynamicrafter|video|verified_configuration|passed|tmp/crafter-gpu-20260921/dynamicrafter-512/status.json|
|easyanimate|video|verified_configuration|passed|tmp/video-more-gpu-20260921/easyanimate/status.json|
|emu3.5|video|failed_integration|passed|tmp/emu35-gpu-20260924/runtime-preflight.json|
|fastvideo-causal-wan2.2|video|blocked_external|passed|tmp/fastvideo-causal-wan-preflight-20260924/integrity.json|
|framepack|video|verified_configuration|passed|/mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-hl02/hadoop-aipnlp/3A/multimodal/yangboxue/arena/WorldFoundry/tmp/classic-video-audited-gpu-20260922/framepack-original/status.json|
|helios|video|verified_configuration|passed|tmp/helios-longcat-gpu-20260921/helios-base/status.json|
|hunyuanvideo|video|quality_concern|passed|tmp/native-video-more-gpu-20260921/hunyuanvideo-t2v/status.json|
|hunyuanvideo-1.5|video|verified_configuration|passed|/mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-hl02/hadoop-aipnlp/3A/multimodal/yangboxue/arena/WorldFoundry/tmp/hunyuan15-fixed-gpu-20260922/hunyuanvideo15-i2v/status.json|
|hyperflow|video|verified_inference_only|passed|tmp/video-more-gpu-20260921/hyperflow-t2va-smoke/status.json; tmp/video-more-gpu-20260921/hyperflow-fl2va-smoke/status.json; tmp/video-more-gpu-20260921/hyperflow-ref2va-smoke/status.json|
|i2vgen-xl|video|quality_concern|passed|tmp/video-next-gpu-20260921/i2vgen-xl/status.json|
|joyai-echo-longvideo|video|blocked_external|passed|tmp/joyai-echo-gpu-20260923/preflight.json; tmp/joyai-echo-gpu-20260923/preflight.json; tmp/joyai-echo-gpu-20260923/preflight.json|
|krea-realtime-video|video|verified_configuration|passed|tmp/krea-realtime-video-gpu-20260923/krea-realtime-toy-2blocks/status.json|
|lingbot-video|video|quality_concern|passed|tmp/causal-more-gpu-20260921/lingbot-video-dense/status.json; tmp/video-more-gpu-20260921/lingbot-video-dense-121f/status.json; tmp/video-quality-more-gpu-20260921/lingbot-forest-121f/status.json; tmp/video-quality-more-gpu-20260921/lingbot-forest-49f-40steps/status.json|
|longcat-video|video|verified_configuration|passed|tmp/helios-longcat-gpu-20260921/longcat-video/status.json; tmp/world-variants-gpu-20260921/longcat-video-distilled/status.json|
|longvie-1|video|blocked_external|passed|tmp/longvie1-gpu-20260924/asset-preflight.json|
|longvie-2|video|verified_inference_only|passed|tmp/longvie2-gpu-20260923/validation.json|
|ltx-2.5|video|blocked_external|passed|tmp/ltx25-gpu-20260924/preflight.json|
|ltx-2.x|video|verified_configuration|passed|tmp/flagship-video-gpu-20260921/ltx23-i2v/status.json|
|ltx-video|video|verified_configuration|passed|tmp/video-broad-gpu-20260921/ltx-video-098/status.json|
|magi-1|video|blocked_external|passed|tmp/next-video-preflight-20260923/magi-1.json|
|magi2-preview|video|quality_concern|passed|tmp/magi2-audited-gpu-20260922/magi2-preview-native/quality-review.json|
|minimax-h3|video|failed_integration|passed|tmp/minimax-h3-gpu-20260924/memory-assessment.json|
|mmaudio|video|verified_inference_only|passed|tmp/mmaudio-gpu-20260922/mmaudio-small_44k/post-validation.json|
|mochi-1|video|verified_configuration|passed|tmp/video-broad-gpu-20260921/mochi-1-fixed/status.json|
|modelscope-t2v|video|quality_concern|passed|/mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-hl02/hadoop-aipnlp/3A/multimodal/yangboxue/arena/WorldFoundry/tmp/modelscope-layout-fixed-gpu-20260922/modelscope-t2v/status.json|
|omnivinci|video|verified_configuration|passed|tmp/omnivinci-image-gpu-20260923/omnivinci-image-qa-dtypefix/status.json|
|open-magvit2|video|verified_configuration|passed|tmp/openmagvit-gpu-20260922/openmagvit-ar256-b/status.json|
|open-sora|video|failed_integration|passed|tmp/open-sora-gpu-20260923/opensora-51f-default/plan.json|
|open-sora-plan|video|blocked_external|passed|tmp/next-video-preflight-20260923/open-sora-plan.json|
|pusa-vidgen|video|verified_configuration|passed|tmp/pusa-gpu-20260922/pusa-i2v-light4/status.json|
|qwen2.5-omni|video|verified_configuration|passed|tmp/qwen25omni-image-textfix-gpu-20260923/qwen25omni-image-qa-textfix/quality-review.json|
|sama-14b|video|quality_concern|passed|tmp/sama14b-edit-gpu-20260923/sama14b-autumn-edit/status.json|
|sana|video|quality_concern|passed|tmp/sana-shift-fixed-gpu-20260921/sana-1600m-1024px/status.json|
|show-o|video|verified_configuration|passed|tmp/showo-gpu-20260922/show-o/status.json|
|skyreels-v2|video|verified_configuration|passed|tmp/flagship-video-gpu-20260921/skyreels-v2-options/status.json|
|skyreels-v3|video|quality_concern|passed|tmp/flagship-video-gpu-20260921/skyreels-v3/status.json|
|spatial-ladder|video|quality_concern|passed|tmp/spatial-ladder-image-gpu-20260923/spatial-ladder-image-qa/quality-review.json|
|spatial-reasoner|video|verified_configuration|passed|tmp/spatial-reasoner-image-videofix-gpu-20260923/spatial-reasoner-image-videofix/quality-review.json|
|stable-video-infinity|video|quality_concern|passed|tmp/svi-linked-gpu-20260923/svi-two-linked-clips/quality-review.json|
|step-video-t2v|video|verified_inference_only|passed|tmp/stepvideo-gpu-20260923/result-quality.json|
|t2v_turbo_t2v|video|verified_configuration|passed|tmp/video-next-gpu-20260921/t2v-turbo/status.json|
|unianimate-dit|video|verified_inference_only|passed|tmp/unianimate-dit-gpu-20260923/validation.json|
|vchitect-2-t2v|video|quality_concern|passed|/mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-hl02/hadoop-aipnlp/3A/multimodal/yangboxue/arena/WorldFoundry/tmp/vchitect-gpu-20260922/vchitect-2-t2v/status.json|
|videocrafter|video|verified_configuration|passed|tmp/crafter-gpu-20260921/videocrafter2-t2v/status.json|
|videocrafter1-i2v|video|verified_configuration|passed|tmp/crafter-gpu-20260921/videocrafter1-i2v/status.json|
|videocrafter1-t2v|video|verified_configuration|passed|tmp/video-next-gpu-20260921/videocrafter1-t2v/status.json|
|videocrafter2-t2v|video|verified_configuration|passed|tmp/crafter-gpu-20260921/videocrafter2-t2v/status.json|
|vmem|video|verified_configuration|passed|tmp/vmem-gpu-20260923/vmem-forward-fast/status.json|
|wan2.1|video|verified_configuration|passed|tmp/video-broad-gpu-20260921/wan21-t2v-13b/status.json|
|wan2.1-vace|video|verified_configuration|passed|tmp/video-broad-gpu-20260921/wan21-vace-14b/status.json|
|wan2.2|video|verified_configuration|passed|tmp/geometry-wan-followup-gpu-20260921/wan22-ti2v-5b/status.json|
|zeroscope|video|verified_configuration|passed|tmp/video-next-gpu-20260921/zeroscope/status.json|
|a1|vla_va_wam|not_run|passed||
|abot-m0|vla_va_wam|not_run|passed||
|act|vla_va_wam|verified_inference_only|passed|tmp/act-gpu-20260922/act-thirdparty-transfercube-chunk/status.json; tmp/act-gpu-20260922/act-thirdparty-transfercube-temporal/status.json|
|ahawam|vla_va_wam|not_run|passed||
|being-h05|vla_va_wam|verified_inference_only|passed|tmp/being-gpu-20260921/being-h05-libero/status.json|
|being-h07|vla_va_wam|not_run|passed||
|cogact|vla_va_wam|verified_inference_only|passed|tmp/cogact-actions-gpu-20260921/cogact-small/status.json; tmp/cogact-actions-gpu-20260921/cogact-base/status.json; tmp/cogact-actions-gpu-20260921/cogact-large/status.json|
|db-cogact|vla_va_wam|verified_inference_only|passed|/mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-hl02/hadoop-aipnlp/3A/multimodal/yangboxue/arena/WorldFoundry/tmp/db-cogact-shape-gpu-20260921/db-cogact-libero/status.json|
|dexora-1b|vla_va_wam|not_run|passed||
|diffusion-policy|vla_va_wam|verified_inference_only|passed|tmp/diffusion-policy-gpu-20260922/diffusion-policy-pusht/status.json|
|dm0|vla_va_wam|not_run|passed||
|dreamzero|vla_va_wam|verified_inference_only|passed|tmp/dreamzero-gpu-20260923/droid/quality-review.json; tmp/dreamzero-gpu-20260923/agibot/quality-review.json|
|eo1|vla_va_wam|failed_integration|passed|tmp/eo1-gpu-20260924/status.json|
|eventvla|vla_va_wam|not_run|passed||
|fastwam|vla_va_wam|not_run|passed||
|galaxea-vla|vla_va_wam|not_run|passed||
|gaussian-actor|vla_va_wam|not_run|passed||
|giga-brain-0|vla_va_wam|verified_inference_only|passed|tmp/gigabrain-gpu-20260923/giga-brain-0/validation.json|
|giga-world-policy-0.5|vla_va_wam|not_run|passed||
|go1|vla_va_wam|not_run|passed||
|gr00t|vla_va_wam|verified_inference_only|passed|tmp/gr00t-all-gpu-20260921/libero_10/status.json|
|h-rdt|vla_va_wam|not_run|passed||
|hy-embodied|vla_va_wam|failed_integration|passed|tmp/hy-embodied-gpu-20260924/status.json|
|hy-embodied-vla|vla_va_wam|verified_inference_only|passed|tmp/hy-embodied-vla-gpu-20260923/hy-vla-robotwin-synthetic-state-regexfix/status.json|
|internvla-a1|vla_va_wam|not_run|passed||
|lapa|vla_va_wam|not_run|passed||
|last-r1|vla_va_wam|not_run|passed||
|lda-1b|vla_va_wam|not_run|passed||
|libero-para|vla_va_wam|not_run|passed||
|lingbot-va|vla_va_wam|verified_inference_only|passed|tmp/lingbot-va-gpu-20260923/libero-long-synthetic/status.json|
|lingbot-vla|vla_va_wam|verified_inference_only|passed|/mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-hl02/hadoop-aipnlp/3A/multimodal/yangboxue/arena/WorldFoundry/tmp/lingbot-vla-gpu-20260922/lingbot-vla-robotwin/status.json; tmp/lingbot-vla-audited-gpu-20260922/lingbot-vla-robotwin/status.json|
|lingbot-vla-v2|vla_va_wam|verified_inference_only|passed|tmp/lingbot-vla-v2-gpu-20260923/lingbot-vla-v2-synthetic-state-defaultfix/status.json|
|mem-0|vla_va_wam|not_run|passed||
|mme-vla|vla_va_wam|blocked_external|passed|tmp/mme-vla-gpu-20260924/status.json|
|molmoact2|vla_va_wam|verified_inference_only|passed|tmp/molmo-more-gpu-20260921/molmoact2-droid/status.json|
|molmobot|vla_va_wam|verified_inference_only|passed|/mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-hl02/hadoop-aipnlp/3A/multimodal/yangboxue/arena/WorldFoundry/tmp/molmobot-gpu-20260922/molmobot-droid/status.json|
|multi-task-dit|vla_va_wam|not_run|passed||
|octo|vla_va_wam|verified_inference_only|passed|tmp/octo-gpu-20260922/octo-small/status.json|
|openpi|vla_va_wam|not_run|passed||
|openpie-0.6|vla_va_wam|failed_integration|passed|tmp/openpie-0.6-gpu-20260924/status.json|
|openvla|vla_va_wam|verified_inference_only|passed|tmp/openvla-fixed-gpu-20260922/openvla-7b-finetuned-libero-10/status.json|
|openvla-oft|vla_va_wam|verified_inference_only|passed|/mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-hl02/hadoop-aipnlp/3A/multimodal/yangboxue/arena/WorldFoundry/tmp/action-pathfix-gpu-20260921/openvla-oft-libero-10/status.json|
|pi0|vla_va_wam|not_run|passed||
|pi0-fast|vla_va_wam|not_run|passed||
|pi0-worldfoundry|vla_va_wam|not_run|passed||
|pi05|vla_va_wam|not_run|passed||
|rdt-1b|vla_va_wam|not_run|passed||
|real-time-chunking|vla_va_wam|not_run|passed||
|rise|vla_va_wam|not_run|passed||
|roboflamingo|vla_va_wam|blocked_external|passed|tmp/roboflamingo-gpu-20260924/status.json|
|rt-1|vla_va_wam|not_run|passed||
|smolvla|vla_va_wam|verified_inference_only|passed|tmp/smolvla-camera-fixed-gpu-20260922/smolvla-libero/status.json|
|spatial-forcing|vla_va_wam|verified_inference_only|passed|/mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-hl02/hadoop-aipnlp/3A/multimodal/yangboxue/arena/WorldFoundry/tmp/spatial-openvla-gpu-20260921/spatial-forcing-libero-10/status.json|
|spirit-v1.5|vla_va_wam|not_run|passed||
|starvla|vla_va_wam|verified_inference_only|passed|tmp/star-xiaomi-gpu-20260922/starvla-qwen3-vl-oft-libero-4in1/status.json; tmp/star-xiaomi-gpu-20260922/starvla-wm4a-wan2d2-oft-libero-4in1/status.json|
|tdmpc|vla_va_wam|not_run|passed||
|tinyvla|vla_va_wam|not_run|passed||
|vlanext|vla_va_wam|verified_inference_only|passed|tmp/vlanext-exact-gpu-20260922/vlanext-libero-10-exact256/status.json|
|vqbet|vla_va_wam|not_run|passed||
|wall-oss|vla_va_wam|failed_integration|passed|tmp/wall-oss-gpu-20260924/status.json|
|x-wam|vla_va_wam|quality_concern|passed|tmp/next-world-models-gpu-20260923/x-wam-robotwin/result.json; tmp/next-world-models-gpu-20260923/x-wam-robocasa/result.json; tmp/xwam-world-gpu-20260923/quality-review.json|
|xiaomi-robotics-0|vla_va_wam|verified_inference_only|passed|tmp/star-xiaomi-gpu-20260922/xiaomi-robotics-0-libero/status.json|
|xiaomi-robotics-1|vla_va_wam|not_run|passed||
|xvla|vla_va_wam|verified_inference_only|passed|tmp/action-all-gpu-20260921/xvla-libero/status.json; tmp/action-all-gpu-20260921/xvla-foundation-libero/status.json; tmp/xvla-calvin-gpu-20260924/quality-review.json; tmp/xvla-google-gpu-20260924/quality-review.json; tmp/xvla-widowx-gpu-20260924/quality-review.json; tmp/xvla-robotwin2-gpu-20260924/quality-review.json; tmp/xvla-vlabench-gpu-20260924/quality-review.json; tmp/xvla-agiworld-gpu-20260924/quality-review.json; tmp/xvla-softfold-gpu-20260924/quality-review.json|
|abot-world-0-5b-lf|world_models|verified_configuration|passed|tmp/world-variants-gpu-20260921/abot-world/status.json|
|ac3d|world_models|blocked_external|passed|tmp/external-blockers-20260924/asset-preflight.json|
|adaworld|world_models|not_run|pending||
|alayaworld|world_models|verified_configuration|passed|tmp/causal-more-gpu-20260921/alaya-native/status.json|
|alayaworld-v1.1|world_models|verified_configuration|passed|tmp/causal-more-gpu-20260921/alaya-v11-ar/status.json; tmp/causal-more-gpu-20260921/alaya-v11-dmd-multiround/status.json|
|astra|world_models|verified_configuration|passed|tmp/video-more-gpu-20260921/astra/status.json|
|astronex-world|world_models|verified_configuration|passed|tmp/causal-more-gpu-20260921/astronex-bidirectional/status.json|
|ati-wan21-14b|world_models|verified_inference_only|passed|tmp/ati-gpu-20260924/status-40step.json|
|biwm-wan21|world_models|blocked_external|pending|/mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-hl02/hadoop-aipnlp/3A/multimodal/yangboxue/arena/WorldFoundry/tmp/biwm-integration-20260922/checkpoint-blocker.json|
|biwm-wan22|world_models|blocked_external|pending|/mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-hl02/hadoop-aipnlp/3A/multimodal/yangboxue/arena/WorldFoundry/tmp/biwm-integration-20260922/checkpoint-blocker.json|
|cameractrl|world_models|verified_inference_only|passed|tmp/cameractrl-gpu-20260924/quality-review.json|
|causal-forcing|world_models|verified_inference_only|passed|tmp/next-world-models-gpu-20260923/causal-forcing/status.json; tmp/next-world-models-gpu-20260923/causal-forcing-framewise/status.json; tmp/next-world-models-gpu-20260923/causal-forcing-2step/status.json; tmp/next-world-models-gpu-20260923/causal-forcing-1step/status.json|
|causal-rcm|world_models|verified_configuration|passed|tmp/causal-rcm-gpu-20260923/causal-rcm-teapot-41f/status.json|
|cosmos-predict-2|world_models|verified_configuration|passed|tmp/cosmos-wan-gpu-20260921/cosmos-predict2-2b/status.json|
|cosmos-predict-2.5|world_models|verified_configuration|passed|tmp/cosmos-wan-gpu-20260921/cosmos-predict25-2b/status.json|
|cosmos-transfer-2.5|world_models|verified_configuration|passed|tmp/cosmos-wan-gpu-20260921/cosmos-transfer25-edge/status.json|
|cosmos3|world_models|verified_configuration|passed|tmp/flagship-video-gpu-20260921/cosmos3-nano-fixed/status.json|
|ctrl-world|world_models|quality_concern|passed|tmp/next-world-models-gpu-20260923/ctrl-world/status.json|
|diamond|world_models|verified_inference_only|passed|tmp/diamond-atari-all-gpu-20260922/REVIEW_RECORDED.json; tmp/diamond-atari-regression-gpu-20260922/diamond-atari-all-parity/status.json|
|dino-wm|world_models|not_run|passed||
|dreamdojo|world_models|blocked_external|passed|tmp/dreamdojo-audit-20260923/asset-audit.json|
|dreamx-world-5b|world_models|verified_configuration|passed|tmp/world-variants-gpu-20260921/dreamx-world-ar/status.json|
|dreamx-world-5b-cam|world_models|verified_configuration|passed|tmp/world-followup-gpu-20260921/dreamx-world-cam/status.json|
|droid-w|world_models|not_run|passed||
|dualcamctrl|world_models|verified_inference_only|passed|tmp/dualcamctrl-gpu-20260924/status.json|
|echo-infinity|world_models|quality_concern|passed|tmp/world-variants-gpu-20260921/echo-infinity/status.json|
|echo-memory-context-k1|world_models|quality_concern|passed|tmp/echo-memory-context-k1-gpu-20260923/echo-memory-k1-two-chunks/status.json|
|egowm|world_models|verified_inference_only|passed|tmp/next-world-models-gpu-20260923/egowm/status.json|
|evoke|world_models|not_run|passed||
|gamma-world|world_models|verified_inference_only|passed|tmp/gamma-world-gpu-20260923/gamma-causal-9f/status.json|
|gen3c|world_models|verified_configuration|passed|tmp/world-followup-gpu-20260921/gen3c/status.json|
|genie-envisioner|world_models|verified_inference_only|passed|tmp/next-world-models-gpu-20260923/genie-envisioner/status.json|
|giga-world-0|world_models|verified_inference_only|passed|tmp/next-world-models-gpu-20260923/giga-world-0/status.json|
|happyoyster|world_models|not_run|passed||
|hma|world_models|verified_inference_only|passed|tmp/next-world-models-gpu-20260923/hma/status.json; tmp/next-world-models-gpu-20260923/hma-discrete-fixed/quality-review.json|
|hunyuan-game-craft|world_models|verified_configuration|passed|tmp/world-variants-gpu-20260921/hunyuan-gamecraft-final/status.json|
|hunyuanworld-1|world_models|not_run|passed||
|hunyuanworld-mirror|world_models|verified_inference_only|passed|tmp/hunyuan-mirror-preview-fixed-gpu-20260922/mirror-real-single/quality-review.json; tmp/hunyuan-mirror-preview-fixed-gpu-20260922/mirror-three-view/quality-review.json|
|hunyuanworld-voyager|world_models|failed_integration|passed|tmp/hunyuanworld-voyager-preflight-20260924/REPORT.md|
|hy-world-2.0|world_models|verified_inference_only|passed|tmp/hy-world2-worldmirror-gpu-20260924/output-check.json|
|hy-worldplay|world_models|not_run|passed||
|hydra|world_models|quality_concern|passed|tmp/hydra-gpu-20260924/validation-quality.json|
|hyworld-worldgen|world_models|not_run|passed||
|infinite-world|world_models|verified_configuration|passed|tmp/world-followup-gpu-20260921/infinite-world/status.json|
|inspatio-world|world_models|verified_configuration|passed|tmp/world-repair-gpu-20260921/inspatio-world/status.json|
|irasim|world_models|not_run|passed||
|joyai-echo-wm|world_models|verified_configuration|passed|tmp/causal-more-gpu-20260921/echo-base/status.json; tmp/causal-more-gpu-20260921/echo-flash/status.json|
|kairos-sensenova|world_models|verified_inference_only|passed|tmp/kairos-gpu-20260924/quality-review.json; tmp/kairos-robot-gpu-20260924/quality-review.json; tmp/kairos-720p-gpu-20260924/quality-review.json|
|leworldmodel|world_models|not_run|passed|tmp/leworldmodel-gpu-20260924/result.json|
|lingbot-world|world_models|verified_configuration|passed|tmp/world-followup-gpu-20260921/lingbot-world-base/status.json|
|lingbot-world-act|world_models|verified_inference_only|passed|tmp/next-world-models-gpu-20260923/lingbot-world-act/status.json|
|lingbot-world-v2|world_models|verified_inference_only|passed|tmp/causal-more-gpu-20260921/lingbot-world-v2-compact/status.json; tmp/lingbot-world-v2-fast-gpu-20260923/validation.json|
|liveworld|world_models|not_run|passed||
|magicworld|world_models|verified_configuration|passed|tmp/causal-more-gpu-20260921/magicworld-fast/status.json|
|matrix-game-1|world_models|verified_configuration|passed|tmp/scene-matrix-fix-gpu-20260921/matrix-game-1-fix/status.json|
|matrix-game-2|world_models|quality_concern|passed|tmp/scene-matrix-fix-gpu-20260921/matrix-game-2-fix/status.json; tmp/world-variants-gpu-20260921/matrix-game-2-templerun/status.json|
|matrix-game-3|world_models|verified_configuration|passed|tmp/matrix-all-gpu-20260921/matrix-game-3/status.json|
|matrix-game-3.5-first-person|world_models|verified_configuration|passed|tmp/matrix-all-gpu-20260921/matrix-game-35-first/status.json|
|matrix-game-3.5-third-person|world_models|verified_configuration|passed|tmp/world-variants-gpu-20260921/matrix-game-35-third/status.json|
|mineworld|world_models|not_run|passed||
|minwm-hy-action2v|world_models|verified_configuration|passed|tmp/world-variants-gpu-20260921/minwm-hy-final/status.json|
|minwm-wan-action2v|world_models|quality_concern|passed|tmp/causal-more-gpu-20260921/minwm-wan/status.json|
|mira|world_models|not_run|passed||
|mosaicmem|world_models|not_run|passed||
|motionbricks|world_models|not_run|passed||
|motionctrl|world_models|verified_configuration|passed|tmp/world-variants-gpu-20260921/motionctrl/status.json|
|moverse|world_models|verified_inference_only|passed|tmp/moverse-cachefix-gpu-20260923/moverse-full-image/quality-review.json|
|multiworld|world_models|verified_inference_only|passed|tmp/multiworld-gpu-20260923/demo-quality-result.json|
|nwm|world_models|blocked_external|passed|tmp/nwm-access-20260922/access.json|
|oasis-500m|world_models|verified_inference_only|passed|tmp/oasis-native-gpu-20260923/oasis-image/quality-review.json; tmp/oasis-native-gpu-20260923/oasis-video/quality-review.json|
|omniforcing|world_models|not_run|passed||
|open-dreamer|world_models|not_run|passed||
|pandora|world_models|not_run|passed||
|pointworld|world_models|not_run|passed||
|rolling-forcing|world_models|quality_concern|passed|tmp/rolling-forcing-gpu-20260923/rolling-forcing-126latent-turn-pathfix/status.json|
|sana-wm|world_models|verified_configuration|passed|tmp/world-followup-gpu-20260921/sana-wm/status.json|
|scope|world_models|quality_concern|passed|tmp/scope-gpu-20260923/quality-result.json|
|self-forcing|world_models|verified_configuration|passed|tmp/causal-more-gpu-20260921/self-forcing-dmd/status.json|
|shotstream|world_models|verified_inference_only|passed|tmp/next-world-models-gpu-20260923/shotstream/status.json|
|simworld|world_models|not_run|passed||
|solaris|world_models|verified_inference_only|passed|tmp/solaris-gpu-20260923/translation-status.json; tmp/solaris-gpu-20260923/rotation-status.json; tmp/solaris-gpu-20260923/one-looks-away-status.json; tmp/solaris-gpu-20260923/structure-status.json; tmp/solaris-gpu-20260923/turn_to_look-status.json; tmp/solaris-gpu-20260923/turn_to_look_opposite-status.json; tmp/solaris-gpu-20260923/both_look_away-status.json|
|solarwm|world_models|quality_concern|passed|tmp/video-quality-more-gpu-20260921/solarwm-dmd/status.json; tmp/causal-more-gpu-20260921/solarwm-dmd-237f/status.json|
|spatia|world_models|verified_inference_only|passed|tmp/spatia-gpu-20260924/demo-quality-result.json|
|starwm|world_models|verified_inference_only|passed|tmp/starwm-gpu-20260924/official-result.json|
|tesseract|world_models|verified_inference_only|passed|tmp/next-world-models-gpu-20260923/tesseract/status.json|
|uni3c|world_models|verified_inference_only|passed|tmp/uni3c-camera-gpu-20260923/uni3c-public-camera/quality-review.json|
|uwm|world_models|not_run|passed||
|versecrafter|world_models|quality_concern|passed|tmp/versecrafter-gpu-20260924/versecrafter-17f-30step-fixed-status.json|
|vggt-world|world_models|not_run|passed||
|vid2world|world_models|not_run|passed||
|viewcrafter|world_models|not_run|passed||
|wan21-fun-14b-cam|world_models|verified_configuration|passed|/mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-hl02/hadoop-aipnlp/3A/multimodal/yangboxue/arena/WorldFoundry/tmp/camera-fun-gpu-20260922/wan21-fun-14b-cam/status.json; tmp/camera-fun-fixed-gpu-20260922/wan21-fun-14b-cam/status.json|
|wan21-fun-1p3b-cam|world_models|quality_concern|passed|/mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-hl02/hadoop-aipnlp/3A/multimodal/yangboxue/arena/WorldFoundry/tmp/camera-fun-gpu-20260922/wan21-fun-1p3b-cam/status.json; /mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-hl02/hadoop-aipnlp/3A/multimodal/yangboxue/arena/WorldFoundry/tmp/camera-fun-gpu-20260922/wan21-fun-1p3b-cam-backward/status.json; tmp/camera-fun-fixed-gpu-20260922/wan21-fun-1p3b-cam/status.json|
|wan22-fun-5b-cam|world_models|verified_configuration|passed|/mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-hl02/hadoop-aipnlp/3A/multimodal/yangboxue/arena/WorldFoundry/tmp/camera-fun-gpu-20260922/wan22-fun-5b-cam/status.json; tmp/camera-fun-fixed-gpu-20260922/wan22-fun-5b-cam/status.json|
|wan22-fun-a14b-cam|world_models|verified_configuration|passed|/mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-hl02/hadoop-aipnlp/3A/multimodal/yangboxue/arena/WorldFoundry/tmp/camera-fun-a14-retry-gpu-20260922/resource-conflict.json; tmp/camera-fun-a14-retry-gpu-20260922/wan22-fun-a14b-cam/status.json|
|warp-as-history|world_models|verified_configuration|passed|tmp/world-repair-gpu-20260921/warp-as-history/status.json|
|wilddet3d|world_models|not_run|passed||
|wildworld|world_models|not_run|passed||
|worldcam|world_models|verified_configuration|passed|tmp/causal-more-gpu-20260921/worldcam/status.json|
|worldcrafter|world_models|verified_configuration|pending|/mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-hl02/hadoop-aipnlp/3A/multimodal/yangboxue/arena/WorldFoundry/tmp/worldcrafter-base-shift-fixed-gpu-20260922/worldcrafter-base-i2v/status.json; /mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-hl02/hadoop-aipnlp/3A/multimodal/yangboxue/arena/WorldFoundry/tmp/worldcrafter-base-shift-fixed-gpu-20260922/worldcrafter-base-t2v/status.json|
|worldcrafter-base|world_models|verified_configuration|pending|/mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-hl02/hadoop-aipnlp/3A/multimodal/yangboxue/arena/WorldFoundry/tmp/worldcrafter-base-shift-fixed-gpu-20260922/worldcrafter-base-i2v/status.json; /mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-hl02/hadoop-aipnlp/3A/multimodal/yangboxue/arena/WorldFoundry/tmp/worldcrafter-base-shift-fixed-gpu-20260922/worldcrafter-base-t2v/status.json|
|worldcrafter-fast|world_models|verified_configuration|pending|/mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-hl02/hadoop-aipnlp/3A/multimodal/yangboxue/arena/WorldFoundry/tmp/worldcrafter-preclamp-gpu-20260922/worldcrafter-fast-i2v/status.json; /mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-hl02/hadoop-aipnlp/3A/multimodal/yangboxue/arena/WorldFoundry/tmp/worldcrafter-preclamp-gpu-20260922/worldcrafter-fast-t2v/status.json|
|worldfm|world_models|verified_configuration|passed|tmp/worldfm-fix-gpu-20260921/worldfm/status.json|
|worldgrow|world_models|not_run|passed||
|worldlabs-marble-1.1|world_models|blocked_external|passed|tmp/api-validation-20260921/worldlabs-marble-1.1.json|
|worldmem|world_models|not_run|passed||
|wow|world_models|quality_concern|passed|tmp/wow-13b-gpu-20260923/quality-review.json|
|xgen-jing|world_models|not_run|pending||
|yume|world_models|verified_configuration|passed|tmp/world-variants-gpu-20260921/yume/status.json|
|zing|world_models|verified_configuration|passed|tmp/causal-more-gpu-20260921/zing-i2v/status.json; tmp/causal-more-gpu-20260921/zing-strafe/status.json|
|helios-distilled|catalog_variant|verified_configuration|passed|tmp/helios-longcat-gpu-20260921/helios-distilled/status.json|
|helios-base|catalog_variant|verified_configuration|passed|tmp/helios-longcat-gpu-20260921/helios-base/status.json|
|helios-mid|catalog_variant|verified_configuration|passed|tmp/helios-longcat-gpu-20260921/helios-mid/status.json|
|being-h05-2b|catalog_variant|not_run|pending||
|being-h05-2b-libero|catalog_variant|verified_inference_only|pending|tmp/being-gpu-20260921/being-h05-libero/status.json|
|being-h05-2b-robocasa|catalog_variant|not_run|pending||
|being-h05-2b-libero-robocasa|catalog_variant|verified_inference_only|pending|/mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-hl02/hadoop-aipnlp/3A/multimodal/yangboxue/arena/WorldFoundry/tmp/being-pathfix-gpu-20260921/being-h05-libero_robocasa/status.json|
|cogact-base|catalog_variant|verified_inference_only|pending|tmp/cogact-actions-gpu-20260921/cogact-base/status.json|
|db-cogact-libero|catalog_variant|verified_inference_only|pending|/mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-hl02/hadoop-aipnlp/3A/multimodal/yangboxue/arena/WorldFoundry/tmp/db-cogact-shape-gpu-20260921/db-cogact-libero/status.json|
|dreamzero-droid|catalog_variant|not_run|pending||
|dreamzero-agibot|catalog_variant|not_run|pending||
|giga-brain-0-3.5b-base|catalog_variant|verified_inference_only|pending|tmp/gigabrain-gpu-20260923/giga-brain-0/validation.json|
|giga-brain-0.1-3.5b-base|catalog_variant|verified_inference_only|pending|tmp/gigabrain-gpu-20260923/giga-brain-0.1/validation.json|
|gr00t-n1-2b|catalog_variant|not_run|pending||
|gr00t-n1.7-libero|catalog_variant|verified_inference_only|pending|tmp/gr00t-all-gpu-20260921/libero_10/status.json|
|hy-embodied-vla-umi|catalog_variant|verified_inference_only|pending|tmp/hy-embodied-vla-gpu-20260923/hy-vla-umi-synthetic-state-regexfix/status.json|
|hy-embodied-vla-robotwin|catalog_variant|verified_inference_only|pending|tmp/hy-embodied-vla-gpu-20260923/hy-vla-robotwin-synthetic-state-regexfix/status.json|
|lapa-7b-openx|catalog_variant|not_run|pending||
|lingbot-va-base|catalog_variant|verified_inference_only|pending|tmp/lingbot-va-gpu-20260923/base-synthetic/status.json|
|lingbot-va-posttrain-robotwin|catalog_variant|verified_inference_only|pending|tmp/lingbot-va-gpu-20260923/robotwin-synthetic/status.json|
|lingbot-va-posttrain-libero-long|catalog_variant|verified_inference_only|pending|tmp/lingbot-va-gpu-20260923/libero-long-synthetic/status.json|
|lingbot-vla-v2-6b|catalog_variant|verified_inference_only|pending|tmp/lingbot-vla-v2-gpu-20260923/lingbot-vla-v2-synthetic-state-defaultfix/status.json|
|lingbot-vla-4b|catalog_variant|verified_inference_only|pending|/mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-hl02/hadoop-aipnlp/3A/multimodal/yangboxue/arena/WorldFoundry/tmp/lingbot-vla-gpu-20260922/lingbot-vla-base/status.json; tmp/lingbot-vla-audited-gpu-20260922/lingbot-vla-base/status.json|
|lingbot-vla-4b-posttrain-robotwin|catalog_variant|verified_inference_only|pending|/mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-hl02/hadoop-aipnlp/3A/multimodal/yangboxue/arena/WorldFoundry/tmp/lingbot-vla-gpu-20260922/lingbot-vla-robotwin/status.json; tmp/lingbot-vla-audited-gpu-20260922/lingbot-vla-robotwin/status.json|
|lingbot-vla-4b-depth-posttrain-robotwin|catalog_variant|verified_inference_only|pending|tmp/lingbot-vla-audited-gpu-20260922/lingbot-vla-robotwin-depth/status.json|
|perceptual-framesamp-modul|catalog_variant|not_run|pending||
|molmoact2-droid|catalog_variant|verified_inference_only|pending|tmp/molmo-more-gpu-20260921/molmoact2-droid/status.json|
|molmoact2-bimanual-yam|catalog_variant|verified_inference_only|pending|tmp/molmo-more-gpu-20260921/molmoact2-bimanual-yam/status.json|
|molmoact2-so100-101|catalog_variant|verified_inference_only|pending|tmp/molmo-more-gpu-20260921/molmoact2-so100-101/status.json|
|molmoact2-libero|catalog_variant|verified_inference_only|pending|/mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-hl02/hadoop-aipnlp/3A/multimodal/yangboxue/arena/WorldFoundry/tmp/action-pathfix-gpu-20260921/molmoact2-libero/status.json|
|molmoact2-think-libero|catalog_variant|verified_inference_only|pending|tmp/molmo-more-gpu-20260921/molmoact2-think-libero/status.json|
|molmobot-droid|catalog_variant|verified_inference_only|pending|/mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-hl02/hadoop-aipnlp/3A/multimodal/yangboxue/arena/WorldFoundry/tmp/molmobot-gpu-20260922/molmobot-droid/status.json|
|molmobot-pi0-droid|catalog_variant|not_run|pending||
|octo-base-1.5|catalog_variant|verified_inference_only|pending|tmp/octo-gpu-20260922/octo-base/status.json|
|octo-small-1.5|catalog_variant|verified_inference_only|pending|tmp/octo-gpu-20260922/octo-small/status.json|
|pi0-base|catalog_variant|not_run|pending||
|pi05-libero|catalog_variant|not_run|pending||
|pi0-fast-base|catalog_variant|not_run|pending||
|pi0-fast-droid|catalog_variant|not_run|pending||
|openvla-oft-libero-spatial|catalog_variant|verified_inference_only|pending|tmp/action-more-gpu-20260921/openvla-oft-libero-spatial/status.json|
|openvla-oft-libero-object|catalog_variant|verified_inference_only|pending|/mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-hl02/hadoop-aipnlp/3A/multimodal/yangboxue/arena/WorldFoundry/tmp/action-pathfix-gpu-20260921/openvla-oft-libero-object/status.json|
|openvla-oft-libero-goal|catalog_variant|verified_inference_only|pending|/mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-hl02/hadoop-aipnlp/3A/multimodal/yangboxue/arena/WorldFoundry/tmp/action-pathfix-gpu-20260921/openvla-oft-libero-goal/status.json|
|openvla-oft-libero-10|catalog_variant|verified_inference_only|pending|/mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-hl02/hadoop-aipnlp/3A/multimodal/yangboxue/arena/WorldFoundry/tmp/action-pathfix-gpu-20260921/openvla-oft-libero-10/status.json|
|openvla-oft-libero-spatial-object-goal-10|catalog_variant|verified_inference_only|pending|tmp/action-more-gpu-20260921/openvla-oft-libero-spatial-object-goal-10/status.json|
|openvla-7b|catalog_variant|verified_inference_only|pending|tmp/action-all-gpu-20260921/openvla-7b/status.json; tmp/openvla-fixed-gpu-20260922/openvla-7b/status.json|
|openvla-libero-10|catalog_variant|verified_inference_only|pending|tmp/action-all-gpu-20260921/openvla-libero-10/status.json; tmp/openvla-fixed-gpu-20260922/openvla-7b-finetuned-libero-10/status.json|
|smolvla-libero|catalog_variant|verified_inference_only|pending|tmp/smolvla-camera-fixed-gpu-20260922/smolvla-libero/status.json|
|smolvla-base|catalog_variant|verified_inference_only|pending|tmp/smolvla-base-normalized-gpu-20260922/smolvla-base-normalized/status.json|
|spatial-forcing-libero-spatial|catalog_variant|verified_inference_only|pending|/mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-hl02/hadoop-aipnlp/3A/multimodal/yangboxue/arena/WorldFoundry/tmp/spatial-openvla-gpu-20260921/spatial-forcing-libero-spatial/status.json|
|spatial-forcing-libero-object|catalog_variant|verified_inference_only|pending|/mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-hl02/hadoop-aipnlp/3A/multimodal/yangboxue/arena/WorldFoundry/tmp/spatial-openvla-gpu-20260921/spatial-forcing-libero-object/status.json|
|spatial-forcing-libero-goal|catalog_variant|verified_inference_only|pending|/mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-hl02/hadoop-aipnlp/3A/multimodal/yangboxue/arena/WorldFoundry/tmp/spatial-openvla-gpu-20260921/spatial-forcing-libero-goal/status.json|
|spatial-forcing-libero-10|catalog_variant|verified_inference_only|pending|/mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-hl02/hadoop-aipnlp/3A/multimodal/yangboxue/arena/WorldFoundry/tmp/spatial-openvla-gpu-20260921/spatial-forcing-libero-10/status.json|
|starvla-qwen3-vl-oft-libero-4in1|catalog_variant|verified_inference_only|pending|tmp/star-xiaomi-gpu-20260922/starvla-qwen3-vl-oft-libero-4in1/status.json|
|starvla-wm4a-wan2d2-oft-libero-4in1|catalog_variant|verified_inference_only|pending|tmp/star-xiaomi-gpu-20260922/starvla-wm4a-wan2d2-oft-libero-4in1/status.json|
|vlanext-libero|catalog_variant|verified_inference_only|pending|tmp/vlanext-exact-gpu-20260922/vlanext-libero-10-exact256/status.json|
|x-wam-robocasa-sft|catalog_variant|verified_inference_only|pending|tmp/next-world-models-gpu-20260923/x-wam-robocasa/result.json|
|x-wam-robotwin-sft|catalog_variant|quality_concern|pending|tmp/next-world-models-gpu-20260923/x-wam-robotwin/result.json; tmp/xwam-world-gpu-20260923/quality-review.json|
|x-wam-pretrained|catalog_variant|blocked_external|pending|tmp/x-wam-pretrained-gpu-20260924/status.json|
|xiaomi-robotics-0-libero|catalog_variant|verified_inference_only|pending|tmp/star-xiaomi-gpu-20260922/xiaomi-robotics-0-libero/status.json|
|xiaomi-robotics-0-calvin-abcd|catalog_variant|verified_inference_only|pending|/mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-hl02/hadoop-aipnlp/3A/multimodal/yangboxue/arena/WorldFoundry/tmp/xiaomi-calvin-fixed-gpu-20260922/xiaomi-robotics-0-calvin-abcd/status.json|
|xiaomi-robotics-0-calvin-abc|catalog_variant|verified_inference_only|pending|/mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-hl02/hadoop-aipnlp/3A/multimodal/yangboxue/arena/WorldFoundry/tmp/xiaomi-calvin-fixed-gpu-20260922/xiaomi-robotics-0-calvin-abc/status.json|
|xiaomi-robotics-0-simplerenv-google|catalog_variant|verified_inference_only|pending|/mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-hl02/hadoop-aipnlp/3A/multimodal/yangboxue/arena/WorldFoundry/tmp/xiaomi-simpler-gpu-20260922/xiaomi-robotics-0-simplerenv-google/status.json|
|xiaomi-robotics-0-simplerenv-widowx|catalog_variant|verified_inference_only|pending|/mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-hl02/hadoop-aipnlp/3A/multimodal/yangboxue/arena/WorldFoundry/tmp/xiaomi-simpler-gpu-20260922/xiaomi-robotics-0-simplerenv-widowx/status.json|
|xiaomi-robotics-0-pretrain|catalog_variant|not_run|pending||
|cosmos-predict-2.5-2b|catalog_variant|verified_configuration|pending|tmp/cosmos-wan-gpu-20260921/cosmos-predict25-2b/status.json|
|cosmos-predict-2.5-14b|catalog_variant|quality_concern|pending|/mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-hl02/hadoop-aipnlp/3A/multimodal/yangboxue/arena/WorldFoundry/tmp/cosmos14-gpu-20260922/cosmos-predict25-14b/status.json|
|cosmos-transfer-2.5-2b|catalog_variant|verified_configuration|pending|tmp/cosmos-wan-gpu-20260921/cosmos-transfer25-edge/status.json|
|diamond-atari|catalog_variant|verified_inference_only|passed|tmp/diamond-atari-all-gpu-20260922/REVIEW_RECORDED.json; tmp/diamond-atari-regression-gpu-20260922/diamond-atari-all-parity/status.json|
|diamond-csgo|catalog_variant|verified_configuration|passed|tmp/diamond-csgo-preflight-20260922/result.json; tmp/diamond-csgo-integration-20260922/RECORDED.json|
|dreamdojo-2b-gr1-action-conditioned|catalog_variant|blocked_external|pending|tmp/dreamdojo-audit-20260923/asset-audit.json|
|dreamdojo-distilled-gr1-realtime|catalog_variant|blocked_external|pending|tmp/dreamdojo-audit-20260923/asset-audit.json|
|egowm-svd-3dof|catalog_variant|quality_concern|pending|tmp/next-world-models-gpu-20260923/egowm-3dof-fixed/quality-review.json|
|egowm-svd-25dof|catalog_variant|verified_inference_only|pending|tmp/next-world-models-gpu-20260923/egowm-25dof-regression/status.json|
|gamma-world-causal-few-step|catalog_variant|verified_inference_only|passed|tmp/gamma-world-gpu-20260923/gamma-few-step-9f/status.json|
|gamma-world-causal|catalog_variant|verified_inference_only|passed|tmp/gamma-world-gpu-20260923/gamma-causal-9f/status.json|
|gamma-world-bidirectional|catalog_variant|verified_inference_only|passed|tmp/gamma-world-gpu-20260923/gamma-bidirectional-actions-9f/status.json|
|hma-magvit-362m|catalog_variant|verified_inference_only|pending|tmp/next-world-models-gpu-20260923/hma-discrete-fixed/status.json|
|hma-mar-1b|catalog_variant|verified_inference_only|pending|tmp/next-world-models-gpu-20260923/hma-cont-regression/status.json|
|lingbot-world-base-cam|catalog_variant|not_run|pending||
|lingbot-world-base-act|catalog_variant|not_run|passed||
|lingbot-world-fast|catalog_variant|not_run|pending||
|matrix-game-2-universal-action-validation|catalog_variant|verified_configuration|pending|tmp/matrix-game2-actions-seedfix-gpu-20260923/quality-review.json|
|local-checkpoint|catalog_variant|not_run|pending||
|nwm-recon|catalog_variant|blocked_external|pending|tmp/nwm-access-20260922/access.json|
|nwm-scand|catalog_variant|blocked_external|pending|tmp/nwm-access-20260922/access.json|
|solarwm-wan2.2-5b|catalog_variant|verified_configuration|passed|tmp/next-world-models-gpu-20260923/solarwm-wan5b-regression/quality-review.json|
|solarwm-wan2.2-14b|catalog_variant|not_run|passed||
|solarwm-ltx-22b|catalog_variant|not_run|passed||
|solarwm-h3-33b|catalog_variant|not_run|passed||
|worldcrafter|world_models|verified_configuration|pending|/mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-hl02/hadoop-aipnlp/3A/multimodal/yangboxue/arena/WorldFoundry/tmp/worldcrafter-base-shift-fixed-gpu-20260922/worldcrafter-base-i2v/status.json; /mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-hl02/hadoop-aipnlp/3A/multimodal/yangboxue/arena/WorldFoundry/tmp/worldcrafter-base-shift-fixed-gpu-20260922/worldcrafter-base-t2v/status.json|
|worldcrafter-base|world_models|verified_configuration|pending|/mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-hl02/hadoop-aipnlp/3A/multimodal/yangboxue/arena/WorldFoundry/tmp/worldcrafter-base-shift-fixed-gpu-20260922/worldcrafter-base-i2v/status.json; /mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-hl02/hadoop-aipnlp/3A/multimodal/yangboxue/arena/WorldFoundry/tmp/worldcrafter-base-shift-fixed-gpu-20260922/worldcrafter-base-t2v/status.json|
|worldcrafter-fast|world_models|verified_configuration|pending|/mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-hl02/hadoop-aipnlp/3A/multimodal/yangboxue/arena/WorldFoundry/tmp/worldcrafter-preclamp-gpu-20260922/worldcrafter-fast-i2v/status.json; /mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-hl02/hadoop-aipnlp/3A/multimodal/yangboxue/arena/WorldFoundry/tmp/worldcrafter-preclamp-gpu-20260922/worldcrafter-fast-t2v/status.json|
|biwm-wan21|world_models|blocked_external|pending|/mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-hl02/hadoop-aipnlp/3A/multimodal/yangboxue/arena/WorldFoundry/tmp/biwm-integration-20260922/checkpoint-blocker.json|
|biwm-wan22|world_models|blocked_external|pending|/mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-hl02/hadoop-aipnlp/3A/multimodal/yangboxue/arena/WorldFoundry/tmp/biwm-integration-20260922/checkpoint-blocker.json|
|xgen-jing|world_models|not_run|pending||
