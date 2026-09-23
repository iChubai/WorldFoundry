# Yume-1.5 local GPU validation

The local `Yume-5B-720P` checkpoint ran through the WorldFoundry pipeline on one H100 (GPU3) in `../envs/yume`. Both T2V and I2V produced decodable 29-frame, 1280×704, 16 FPS videos from a `forward` interaction and fixed seed 42. This is a short inference check, not an action-control benchmark.

| Case | Steps | Load / generation time | Evidence | Visual review |
| --- | ---: | ---: | --- | --- |
| T2V living room | 4 | 304.8 / 20.1 s | [video](t2v-4step/video.mp4), [metrics](t2v-4step/metrics.json), frames [0](t2v-4step/frame-000.png), [14](t2v-4step/frame-014.png), [28](t2v-4step/frame-028.png) | Room geometry and forward viewpoint remain coherent. |
| I2V from the generated first frame | 16 | 211.4 / 19.9 s | [video](i2v-16step/video.mp4), [metrics](i2v-16step/metrics.json), frames [0](i2v-16step/frame-000.png), [14](i2v-16step/frame-014.png), [28](i2v-16step/frame-028.png) | Source room remains recognizable while the camera advances. |

The runtime accepted a 32-frame segment request and decoded 29 frames from eight temporal latents. Both MP4s were checked with `ffprobe` and each reports 29 frames at 16 FPS. GPU3 reached about 56 GB during I2V sampling without OOM.

I fixed a T2V geometry bug in `worldfoundry_runtime.py`: the requested height-first size is now passed to Wan's width-first sampler. Previously, portrait T2V silently used the sampler's landscape default. The focused landscape/portrait regression test passed; the existing Yume-1.5 catalog suite also passed (11 tests in the combined run, one upstream autocast deprecation warning).

This validation does not cover V2V, long-horizon interaction chaining, the pipeline's 100-step quality default, or quantitative camera-control accuracy. The first T2V sample used the catalog's accelerated 4-step setting, and I2V used 16 steps.
