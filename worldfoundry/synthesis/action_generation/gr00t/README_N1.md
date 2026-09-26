# Original GR00T-N1 checkpoint route

The original nvidia/GR00T-N1-2B checkpoint has model_type=gr00t_n1, one
model.safetensors, and experiment_cfg/metadata.json. WorldFoundry dispatches
it to NVIDIA's original N1 implementation. N1.7 checkpoints continue through
the existing in-tree Gr00tN1d7 route.

The gr1 runtime variant selects N1 and GR1 when WORLDFOUNDRY_HFD_ROOT points
to the staged checkpoint directory. An explicit N1 model_path also takes
priority over the default libero_10 variant without setting that variable.

The N1 worker requires an Isaac-GR00T checkout at the n1-release commit
755876a9afdb41ca6eb6383b36f4a0adb085c73f. It rejects a different
revision and does not modify the official source directory. Run it in a separate
Python environment with the original release's dependencies and
n1_legacy_requirements.txt (Transformers 4.45.2, Tokenizers 0.20.3). The
worker checks the Transformers version before loading weights. It uses eager
attention in memory when FlashAttention2 is unavailable; it does not edit the
official source or checkpoint.

Pass n1_source_dir and n1_python to GR00TPipeline.from_pretrained, or set
WORLDFOUNDRY_GR00T_N1_SOURCE and WORLDFOUNDRY_GR00T_N1_PYTHON. Both are
required for inference. Missing values produce errors explaining what to
configure. A plan_only call inspects the checkpoint and writes a plan without
requiring the source checkout or worker environment.

For the staged GR1 checkpoint, the default N1 data configuration is
gr1_arms_only: provide an ego_view RGB camera, an instruction, and
left_arm, right_arm, left_hand, and right_hand state arrays under
gr00t_observation={"state": ...}. State arrays may be shaped [D], [1,D],
or [1,1,D]. The route emits four 16-step action groups. A different official
data configuration can be selected with n1_data_config_name, provided its
camera and state keys exist in the checkpoint metadata.
For named camera input, use `gr00t_observation={"camera_views":
{"video.ego_view": image, ...}, "state": ...}`. The worker selects the exact
camera keys required by the official data configuration. An unnamed image is
accepted only when that configuration requires one camera; multi-camera
configurations require a separate image for each configured view.

The local GPU probe used a real but out-of-domain LIBERO image and GR1
metadata-mean state. It established finite
offline action prediction, not GR1 control quality or task success. The
official loader leaves two old action_head.decode_layer checkpoint tensors
unused; all 802 model state keys matched the checkpoint shapes.
