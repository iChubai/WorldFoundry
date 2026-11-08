# `worldfoundry.base_models.perception_core`

Inference-only perception models shared across WorldFoundry. Benchmark metrics are their
main consumer — detection, segmentation, tracking, optical flow, and quality models turn
generated videos into scores — and model pipelines reuse them for preprocessing (masks,
tracks, captions, embeddings).

## Layout

| Path | Models |
| --- | --- |
| `general_perception/` | Feature backbones and embeddings: DINO, DINOv2, DINOv3, OpenAI CLIP, OpenCLIP, UniCeption, Qwen3-VL entity features |
| `detection/` | Open-vocabulary detection: Grounding DINO, YOLO-World |
| `segment/` | SAM v1, SAM 2, SAM 3, MobileSAM, RepViT-SAM, Grounded-SAM, scene segmentation |
| `tracking/` | CoTracker, DOT, AOT, Track-Anything |
| `optical_flow/` | RAFT, SEA-RAFT, FlowFormer++ |
| `video_quality/` | DOVER, FAST-VQA, LPIPS, DreamSim, HPSv3, TransNetV2, FETV StyleGAN-V |
| `video_text/` | Video–text alignment: ViCLIP, VQAScore, BLIP-2 features, FETV BLIP, OpenS2V-Nexus |
| `captioning/` | Tag2Text, GRiT |
| `action_recognition/` | UMT, latent-action models |
| `frame_interpolation/` | AMT, VFIMamba |
| `face/` | ArcFace, RetinaFace |
| `human_anatomy/` | ViT-based body detector |
| `safety/` | NudeNet |
| `visualization/` | Colormaps and rendering helpers for perception outputs |

## Usage

These models are normally reached indirectly: through a benchmark metric
(`wf metric list`, `wf score ...`) or a pipeline that needs preprocessing. When importing
one directly, keep the import local to the call site — perception stacks are heavy, and the
package is designed so that importing `worldfoundry` does not pull them in.

## Conventions

- Inference only: keep transforms needed at inference time; no training datasets, losses, or experiment scripts.
- Checkpoints resolve through the Hugging Face cache or `WORLDFOUNDRY_CKPT_DIR`, not machine-local paths.
- Retain upstream attribution and license notices in vendored code.
