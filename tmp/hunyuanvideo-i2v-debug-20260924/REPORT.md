# HunyuanVideo I2V temporal continuity follow-up

The original 129-frame, 720p, 50-step image-to-video output is decodable and
finite, but its view changes abruptly from the complete abandoned room at
frame 8 to a close-up of wooden floorboards at frame 9. The mean absolute
8→9 decoded RGB difference is 67.22/255. Later frames mostly retain the
floorboard view. The result remains a quality concern.

A 17-frame rerun used the same public `HunyuanVideoI2VPipeline`, local
checkpoint, source image, prompt, seed 42, 720p, 50 steps, and 24 fps.
All 17 frames decode, stay in the source room, and change smoothly. Its
maximum adjacent-frame RGB difference is 2.28/255; the 8→9 difference is
1.17/255. This establishes that the failure depends on sequence length or
another length-dependent inference path, not merely MP4 decoding.

A 33-frame run with identical settings also keeps the complete room through
frame 32. All 33 frames decode; maximum adjacent-frame RGB MAE is 3.85/255,
and frame 8→9 is 2.44/255. The scene changes gradually, although the
requested forward camera motion is weak. The abrupt cut is therefore not
reproducible at 17 or 33 frames.

A 65-frame run crosses the VAE temporal-tiling threshold. All 65 frames
decode and the complete room persists through frame 64. The largest
adjacent-frame RGB MAE is 4.14/255, frame 8→9 is 1.80/255, and frame
63→64 is 1.19/255. The final video and latent tensors are finite with
shapes `[1,3,65,720,1280]` and `[1,16,17,90,160]`. The camera motion is
subtle and the exact prompt adherence is not established, but there is no
temporal break at this tested length.

An isolated causal VAE check used the actual I2V VAE checkpoint. The same
first five latent frames were decoded both alone (17 output frames) and as
the prefix of a 33-latent-frame input (129 output frames). The first 17
decoded frames agreed to 0.0133 maximum frame-wise MAE on the decoder's
approximately [-1, 1] scale. The VAE temporal tiling path did not reproduce
the abrupt frame-9 cut. This does not prove the VAE is exact or exclude an
interaction with generated latents at full 720p.

Evidence: `short17.json`, `short17/status.json`, `short17/tensor-validation.json`,
`short17/demo.mp4`, `short17-sheet.png`, `middle33.json`,
`middle33/status.json`, `middle33-quality.json`, `middle33-sheet.png`,
`middle65.json`, `middle65/status.json`, `middle65/tensor-validation.json`,
`middle65-quality.json`, `middle65-sheet.png`, `long129-sheet.png`, and
`vae-causality.json`. The 129-frame source artifact and original status are
under `tmp/native-video-more-gpu-20260921/hunyuanvideo-i2v/`.

The existing 129-frame quality concern remains. The abrupt cut is specific
to that tested long configuration; the 17-, 33-, and 65-frame tests and
isolated VAE comparison do not identify a deterministic implementation bug
to patch. No HunyuanVideo code was changed on the basis of these results.
