# Model variant validation

DreamX5B AR passes a9frame first-chunk inference at1280x704/16fps;full decode and visually coherent slight forward movement. Long rollouts remain untested.

MatrixGame2 TempleRun nomove->jump outputs57frames640x352/12fps and decodes,but character duplication and road deformation fail visual acceptance. Keep quality_concern. Prior-session official idle parity is only investigation context,not evidence for this run.

Remaining queued configurations pending independent review.

ABotWorld0.5B LF passes33frames1280x704/16fps with W action. MotionCtrl camera-motion passes50steps16frames256x256/10fps,temple camera movement visible. LongCat distilled passes16distill steps49frames832x480/15fps,coherent forest. All full decodes and contact-sheet reviews pass,limited to recorded configurations.

MatrixGame2 GTA drive checkpoint passes57frames640x352/12fps,forward then right turn;vehicle and road remain coherent. TempleRun concern remains independent.

MatrixGame3.5 third-person passed distinct weights,official street/car references and camera path,25steps/1block85frames1280x704/16fps. Full decode and coherent vehicle progression reviewed.

## echo-infinity

quality_concern: 81frames832x480/16fps;full decode and forward forest motion,but persistent irregular black upper band and boundary vegetation distortion;cause unlocalized

证据：tmp/world-variants-gpu-20260921/echo-infinity/status.json；tmp/world-variants-gpu-20260921/echo-infinity/demo.validation.json。接触图已人工查看。

## hunyuan-gamecraft-final

verified_configuration: 50steps33frames1216x704/24fps,forward interaction,cpu_offload;full decode and clear forward navigation through room;short rollout only

## yume

verified_configuration: Yume540P,50ODEsteps,32frames960x544/16fps,forward;full decode and reviewed coherent room with slight motion;short single-window only

## minwm-hy-final

verified_configuration: HY branch,4steps,77frames832x480/16fps,w*19;full decode and reviewed coherent first-person forward room movement
