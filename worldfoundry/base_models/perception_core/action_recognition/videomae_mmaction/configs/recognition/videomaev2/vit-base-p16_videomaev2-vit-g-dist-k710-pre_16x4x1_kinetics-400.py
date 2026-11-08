_base_ = ["../../_base_/default_runtime.py"]

# model settings
model = {
    "type": "Recognizer3D",
    "backbone": {
        "type": "VisionTransformer",
        "img_size": 224,
        "patch_size": 16,
        "embed_dims": 768,
        "depth": 12,
        "num_heads": 12,
        "mlp_ratio": 4,
        "qkv_bias": True,
        "num_frames": 16,
        "norm_cfg": {"type": "LN", "eps": 1e-6},
    },
    "cls_head": {"type": "TimeSformerHead", "num_classes": 400, "in_channels": 768, "average_clips": "prob"},
    "data_preprocessor": {
        "type": "ActionDataPreprocessor",
        "mean": [123.675, 116.28, 103.53],
        "std": [58.395, 57.12, 57.375],
        "format_shape": "NCTHW",
    },
}

# dataset settings
dataset_type = "VideoDataset"
data_root_val = "data/kinetics400/videos_val"
ann_file_test = "data/kinetics400/kinetics400_val_list_videos.txt"

test_pipeline = [
    {"type": "DecordInit"},
    {"type": "SampleFrames", "clip_len": 16, "frame_interval": 4, "num_clips": 5, "test_mode": True},
    {"type": "DecordDecode"},
    {"type": "Resize", "scale": (-1, 224)},
    {"type": "ThreeCrop", "crop_size": 224},
    {"type": "FormatShape", "input_format": "NCTHW"},
    {"type": "PackActionInputs"},
]

test_dataloader = {
    "batch_size": 1,
    "num_workers": 8,
    "persistent_workers": True,
    "sampler": {"type": "DefaultSampler", "shuffle": False},
    "dataset": {
        "type": dataset_type,
        "ann_file": ann_file_test,
        "data_prefix": {"video": data_root_val},
        "pipeline": test_pipeline,
        "test_mode": True,
    },
}

test_evaluator = {"type": "AccMetric"}
test_cfg = {"type": "TestLoop"}
