"""Lazy public facade for model-family training adapters."""

from __future__ import annotations

from importlib import import_module

_EXPORTS = {
    "ANYFLOW_BIDIRECTIONAL_WAN_SMALL_CHECKPOINT": ".anyflow",
    "ANYFLOW_FAR_WAN_SMALL_CHECKPOINT": ".anyflow",
    "AnyFlowArchitecture": ".anyflow",
    "AnyFlowCheckpoint": ".anyflow",
    "NativeAnyFlowModelMaterializer": ".anyflow",
    "SELF_FORCING_ODE_CHECKPOINT": ".causal_wan",
    "CausalWanTrainRole": ".causal_wan",
    "causal_wan_1p3b_config": ".causal_wan",
    "convert_self_forcing_causal_state_dict": ".causal_wan",
    "load_causal_wan_1p3b": ".causal_wan",
    "validate_causal_wan_dtype": ".causal_wan",
    "SANA_600M_512_TRAIN_FLOW_SHIFT": ".sana",
    "SANA_DEFAULT_TRAIN_TIMESTEPS": ".sana",
    "SanaTrainAdapter": ".sana",
    "build_cached_sana_train_adapter": ".sana",
    "build_sana_train_adapter": ".sana",
    "COSMOS1_DEFAULT_FPS": ".cosmos1",
    "Cosmos1TrainAdapter": ".cosmos1",
    "build_cached_cosmos1_train_adapter": ".cosmos1",
    "COSMOS2_DEFAULT_FPS": ".cosmos2",
    "Cosmos2TrainAdapter": ".cosmos2",
    "build_cached_cosmos2_train_adapter": ".cosmos2",
    "COSMOS2P5_DEFAULT_FPS": ".cosmos2p5",
    "COSMOS3_DEFAULT_FPS": ".cosmos3",
    "Cosmos3VideoTrainAdapter": ".cosmos3",
    "build_cached_cosmos3_video_train_adapter": ".cosmos3",
    "COSMOS2P5_DEFAULT_TRAIN_TIMESTEPS": ".cosmos2p5",
    "Cosmos2p5TrainAdapter": ".cosmos2p5",
    "build_cached_cosmos2p5_train_adapter": ".cosmos2p5",
    "ECHO_MEMORY_DEFAULT_TRAIN_TIMESTEPS": ".echo_memory",
    "GAMMA_WORLD_DEFAULT_TRAIN_TIMESTEPS": ".gamma_world",
    "GammaWorldBidirectionalTrainAdapter": ".gamma_world",
    "build_cached_gamma_world_bidirectional_train_adapter": ".gamma_world",
    "EchoMemoryTrainAdapter": ".echo_memory",
    "build_cached_echo_memory_train_adapter": ".echo_memory",
    "LTX_DEFAULT_FPS": ".ltx",
    "LTXVideoTrainAdapter": ".ltx",
    "build_cached_ltx_video_train_adapter": ".ltx",
    "MATRIX_GAME_35_DEFAULT_TRAIN_TIMESTEPS": ".matrix_game_3p5",
    "MatrixGame35TrainAdapter": ".matrix_game_3p5",
    "build_cached_matrix_game_35_train_adapter": ".matrix_game_3p5",
    "HUNYUAN_VIDEO_DEFAULT_TRAIN_TIMESTEPS": ".hunyuan_video",
    "HunyuanVideoTrainAdapter": ".hunyuan_video",
    "build_cached_hunyuan_video_train_adapter": ".hunyuan_video",
    "build_hunyuan_video_train_adapter": ".hunyuan_video",
    "HUNYUAN_VIDEO_I2V_DEFAULT_TRAIN_TIMESTEPS": ".hunyuan_video_i2v",
    "HunyuanVideo15TrainAdapter": ".hunyuan_video_i2v",
    "HunyuanVideoI2VTrainAdapter": ".hunyuan_video_i2v",
    "build_cached_hunyuan_video15_train_adapter": ".hunyuan_video_i2v",
    "build_cached_hunyuan_video_i2v_train_adapter": ".hunyuan_video_i2v",
    "DiffusersSanaConditioner": ".sana_sid",
    "DiffusersSanaDenoiser": ".sana_sid",
    "SanaSIDPredictionAdapter": ".sana_sid",
    "build_local_diffusers_sana_sid_adapter": ".sana_sid",
    "STEP_VIDEO_DEFAULT_FPS": ".step_video",
    "STEP_VIDEO_DEFAULT_MODEL_TIMESTEP_SCALE": ".step_video",
    "STEP_VIDEO_DEFAULT_TRAIN_TIMESTEPS": ".step_video",
    "StepVideoTrainAdapter": ".step_video",
    "build_cached_step_video_train_adapter": ".step_video",
    "build_step_video_train_adapter": ".step_video",
    "VCHITECT_DEFAULT_TRAIN_TIMESTEPS": ".vchitect",
    "VchitectTrainAdapter": ".vchitect",
    "build_cached_vchitect_train_adapter": ".vchitect",
    "build_vchitect_train_adapter": ".vchitect",
    "WAN_DEFAULT_CONTEXT_FEATURES": ".wan",
    "WAN_DEFAULT_TEXT_LENGTH": ".wan",
    "WAN_DEFAULT_TRAIN_TIMESTEPS": ".wan",
    "WanTrainAdapter": ".wan",
    "build_cached_wan_train_adapter": ".wan",
    "build_wan_train_adapter": ".wan",
    "wan_pixel_mask_to_latent": ".wan",
    "WanVaceTrainAdapter": ".wan_vace",
    "build_cached_wan_vace_train_adapter": ".wan_vace",
    "build_wan_vace_train_adapter": ".wan_vace",
}

__all__ = list(_EXPORTS)


def __getattr__(name: str) -> object:
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(name)
    value = getattr(import_module(module_name, __name__), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
