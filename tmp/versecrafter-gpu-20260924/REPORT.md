# VerseCrafter GPU validation (2026-09-24)

**Status: `quality_concern`.** The public `VerseCrafterPipeline` completed a
checkpoint-backed 17-frame, 832×480, 16 fps, 30-step image-to-video run on one
H100. All frames decode. The courtyard, building, trees, and person remain
recognizable, while late frames contain visible blue/blocky disocclusion
artifacts on the grass and pavement. This is short-run inference evidence, not
measured camera-path accuracy or official output parity.

## Inputs and weights

- Input: `tmp/uni3c-validation-20260923/reference/data/demo_uni3c/reference.png`, SHA-256 `6e8a54d97a87e8ad94c1d390092183251b79c6b946a3de40402496b6b2010b48`.
- Local checkpoints passed explicitly to the public pipeline: `../ckpts/TencentARC--VerseCrafter` (four transformer shards), `../ckpts/Wan-AI--Wan2.1-T2V-14B` (base/T5/VAE), and `../ckpts/Ruicheng--moge-2-vitl-normal/model.pt`. The MoGe file's SHA-256 is `280741fd09bc3f403ccff9967784c2a391b52d2c0742ae3efdb21d9f90cc1a01`, matching the Hub LFS object digest returned by `hf download`. The large VerseCrafter and Wan files were structurally loaded by the runtime; no full official-hash claim is made for them.
- Default generated trajectory: 17 frames, Blender camera origin to +0.12 m X and −0.04 m Z. This is a synthetic validation trajectory, not an upstream camera fixture.

## Reproduced bug and repair

The original `_write_default_trajectory()` used an identity Blender camera
basis. The renderer first transforms the OpenCV depth cloud into Blender
world coordinates, then converts the trajectory to a PyTorch3D camera. With
the identity basis, the camera points away from the cloud: all nine original
`background_RGB.mp4` frames were uniform RGB 126, `background_depth.mp4` was
zero, and `merged_mask.mp4` was full 255. The initial nine-frame/eight-step
output exists, but its 4D conditioning was invalid and it is **not** counted
as a successful validation.

`worldfoundry_runtime.py` now sets the default Blender camera basis to match
the OpenCV-to-Blender cloud transform and rejects empty default control maps
before diffusion. A standalone replay using the same depth and image changed
the control video from uniform gray to a scene-bearing nine-frame clip with
RGB range 0–255 and mean first/last frame difference 18.28/255; depth is
nonzero and the unknown-region mask is no longer full. The new guard rejects
the original maps and accepts the repaired maps. The full 17-frame pipeline
run used the repaired trajectory and produced scene-bearing controls; its
`background_RGB.mp4` SHA-256 is
`2eb9cc15b3441b1c1379d75597ebd500d814f3e04993360e77d1329420eca8cb`,
and `background_depth.mp4` SHA-256 is
`7acb129cb9d852a0a33da47223099ba0055dcfb2ab0e75833419cf2b82f7fd9e`.

## Output check and limits

- Final MP4: `versecrafter-17f-30step-fixed.mp4`, SHA-256 `7bfb569399d1f1fe8a65d455b700d363e3274134447078608f8fbbcec72c5f40`.
- `ffprobe`: H.264, 832×480, 16 fps, 17 frames, 1.0625 s. Full `ffmpeg -f null -` decode passed. Decoded RGB spans 0–255; first-to-last mean absolute difference is 21.16/255 and adjacent differences range 2.14–4.73/255 (`video-metrics.json`).
- Visual inspection of frames 0, 8, and 16 shows a continuous courtyard and person with modest lateral view change. Frame 16 has obvious blue/blocky artifacts, consistent with holes and disocclusion visible in the last rendered point-cloud control frame. The run does not establish quantitative camera adherence, long-video quality, exact upstream output parity, or finite latent tensors before decoding.
- `py_compile` for the repaired runtime and `git diff --check` passed. The NumExpr thread-count message in the diffusion log did not stop generation.

Evidence: `status.json` and `versecrafter-17f-30step-fixed-status.json`,
`run.log` and `run-fixed.log`, the stage logs under each hidden input directory,
`render-fixed.log`, `fixed-control-first.png`, `fixed-control-last.png`, and
`fixed-output-{00,08,16}.png`.
