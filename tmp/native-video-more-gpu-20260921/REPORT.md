# Native视频本轮审核

Wan2.1I2V14B480P,81frames832x480/16fps,40steps;raw video/latents finite,full decode and reviewed coherent room with slight forward approach.

## wan21-i2v-720p

Wan2.1I2V14B720P,40steps81frames1280x720/16fps;raw tensors finite,full decode passes;reviewed coherent forward room approach with expected foreground objects moving out of frame;single prompt/seed.

## wan22-t2v-a14b

Wan2.2T2VA14B,40steps81frames832x480/16fps;raw tensors finite,full decode and visually coherent sunlit abandoned room with forward camera approach;single prompt/seed.

## wan2.1-t2v-14b

verified_configuration: Real14B weights,50steps,81frames1280x720/16fps;finite tensors and all frames decode;reviewed coherent sunlit abandoned room with forward camera motion. Single prompt/seed only.

## wan2.2-i2v-a14b

verified_configuration: Real A14B weights,40steps,81frames832x480/16fps;finite tensors/full decode;reviewed toy car and abandoned room from conditioning image,coherent slow advance. Single image/prompt/seed only.

## hunyuanvideo-t2v

quality_concern: 50steps129frames1280x720/24fps;DiT/VAE strict loads,HF text-encoder final audit pending;finite tensors,full decode. Contact sheet and full frame100 inspected:coherent dusty floor/sunlight,but low static floor view with unrequested human feet walking across instead of slow forward room traversal. Prompt/motion adherence concern on this seed,not numerical failure.

## hunyuanvideo-i2v

quality_concern: 50steps129frames1280x720/24fps;DiT/VAE strict0missing,HF text final loading review pending;finite raw/full decode. Actual contact sheet:conditioning room/car in frame0 abruptly becomes almost-static close-up wooden boards for rest;maxframe delta66.34. Temporal/conditioning failure,not quality pass.
