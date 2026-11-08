default_scope = "mmaction"

default_hooks = {
    "runtime_info": {"type": "RuntimeInfoHook"},
    "timer": {"type": "IterTimerHook"},
    "logger": {"type": "LoggerHook", "interval": 20, "ignore_last": False},
    "param_scheduler": {"type": "ParamSchedulerHook"},
    "checkpoint": {"type": "CheckpointHook", "interval": 1, "save_best": "auto"},
    "sampler_seed": {"type": "DistSamplerSeedHook"},
    "sync_buffers": {"type": "SyncBuffersHook"},
}

env_cfg = {
    "cudnn_benchmark": False,
    "mp_cfg": {"mp_start_method": "fork", "opencv_num_threads": 0},
    "dist_cfg": {"backend": "nccl"},
}

log_processor = {"type": "LogProcessor", "window_size": 20, "by_epoch": True}

vis_backends = [{"type": "LocalVisBackend"}]
visualizer = {"type": "ActionVisualizer", "vis_backends": vis_backends}

log_level = "INFO"
load_from = None
resume = False
