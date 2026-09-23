# World-model follow-up

SANA-WM bidirectional checkpoint passed60steps,33frames,1280x704/16fps,32 forward w actions. Full decode and contact sheet reviewed; room remains coherent,slight viewpoint motion,initial exposure transition. Other action patterns/checkpoints remain untested.

Other queued cases retain independent status and evidence; no previous-session inference counted as current success.

DreamX5B Cam passes30steps,32 exported newframes1280x704/16fps,forward scene movement. InfiniteWorld passes30steps81frames896x448/30fps,forward scene movement. Full decode and visual reviews completed. Warp-as-History failed missing jmespath and Inspatio failed an invalid skip_step2 flag without precomputed geometry;both are repaired and requeued in ../world-repair-gpu-20260921.

Gen3C passes35steps121frames1280x704/24fps;full decode and visually consistent forward camera approach.

## lingbot-world-base

verified_configuration: Base-Cam weights,40steps81frames1280x704/16fps,forward;full decode and coherent subtle approach;resize_H/W are input preprocessing hints,output contract to be examined

LingBotWorld尺寸契约复核：pipeline __call__中max_area默认720×1280，传给runtime.predict；resize_H/W传给traj_generator构建相机内参。1280×704输出符合输入图宽高比与空间步长约束，不是忽略输出尺寸参数。
