
## sana-1600m-1024px: sana-1600m-1024px

quality_concern: Architecture and shift3 corrected;20step native Euler,BF16;finite raw output but colored fragments/paintlike corruption remain.Visually reviewed.

## sana-1600m-1024px: sana-1600m-1024px-fp32

quality_concern: FP32 diagnostic after architecture/shift3 fix reproduces colored fragment artifacts;precision alone does not resolve problem.Visually reviewed;upstream scheduler parity pending.

## sana1p5-1600m-1024px: sana1p5-1600m-1024px

verified_configuration: Sana1.5 1.6B,20step,shift3,1024px;strict loading,finite raw output,reviewed red teapot and white cup by window;single prompt/seed only.

## sana-1600m-1024px-bf16: sana-1600m-1024px-bf16

quality_concern: 1024px BF16 checkpoint,20step shift3;finite and coherent red teapot/white cup but extra spout/connecting appendage;image quality concern,not numerical failure.

## sana-1600m-2k-bf16: sana-1600m-2k-bf16

quality_concern: 2048px BF16 checkpoint,20step shift3;finite image,extra white pot and inaccurate object shapes;count/geometry concern.

## sana-1600m-512px-multiling: sana-1600m-512px-multiling

quality_concern: 512px multilingual checkpoint,20step shift3;finite clear image but handle resembles second spout;object geometry concern.English prompt only;multilingual capability not established.

## sana: sana-1600m-1024px

quality_concern: Image family tested variants separately;some clear samples,standard1600M1024 has persistent corruption and several variants have object geometry errors;video/Sprint/ControlNet queued,no whole-family quality pass.
