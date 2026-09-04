"""Factory that maps ``ModelRuntimeConfig.family`` to a concrete ``ModelAdapter``."""

from __future__ import annotations

from worldarena.models.adapters import (
    ABotWorldAdapter,
    AlayaWorldAdapter,
    CamI2VAdapter,
    Cosmos3Adapter,
    CosmosPredict25Adapter,
    DreamXWorldAdapter,
    EchoInfinityAdapter,
    EvokeAdapter,
    FlashWorldAdapter,
    Gen3CAdapter,
    HYWorldPlayAdapter,
    HunyuanGameCraftAdapter,
    HunyuanVideoAdapter,
    HunyuanWorldAdapter,
    HunyuanWorldVoyagerAdapter,
    InSpatioWorldAdapter,
    InfiniteWorldAdapter,
    LingBotVideoAdapter,
    LingBotWorldAdapter,
    LingBotWorldV2Adapter,
    LongCatVideoAdapter,
    LTX23Adapter,
    LucidDreamerAdapter,
    Lyra1Adapter,
    Lyra2Adapter,
    MatrixGame35Adapter,
    MatrixGameAdapter,
    MiniMaxH3Adapter,
    ModelAdapter,
    MotionCtrlAdapter,
    NeoVerseAdapter,
    RealCamI2VAdapter,
    SanaWMAdapter,
    StableVirtualCameraAdapter,
    VideoApiAdapter,
    Wan22Adapter,
    WarpAsHistoryAdapter,
    WonderJourneyAdapter,
    WonderWorldAdapter,
    WorldCamAdapter,
    WorldFMAdapter,
    YumeAdapter,
)
from worldarena.models.config import ModelRuntimeConfig


def build_model_adapter(config: ModelRuntimeConfig) -> ModelAdapter:
    """Instantiate the adapter class registered for ``config.family``."""
    if config.backend_type == "http_api":
        return VideoApiAdapter(config)
    if config.backend_type == "sdk_api":
        raise ValueError(
            "sdk_api adapters are not wired yet; use backend_type=http_api with submit_mode=http_polling for now"
        )

    family = config.family.lower()
    if family in {"abot", "abot_world", "abot-world"}:
        return ABotWorldAdapter(config)
    if family in {"alaya", "alaya_world", "alaya-world", "alayaworld"}:
        return AlayaWorldAdapter(config)
    if family in {"cami2v", "cam_i2v", "cam-i2v"}:
        return CamI2VAdapter(config)
    if family in {"realcam_i2v", "realcam-i2v", "realcami2v"}:
        return RealCamI2VAdapter(config)
    if family in {"sana_wm", "sana-wm", "sanawm", "sana"}:
        return SanaWMAdapter(config)
    if family in {
        "cosmos3",
        "cosmos-3",
        "cosmos3_super",
        "cosmos3-super",
        "cosmos3_nano",
        "cosmos3-nano",
    }:
        return Cosmos3Adapter(config)
    if family in {
        "cosmos_predict25",
        "cosmos-predict25",
        "cosmos_predict2.5",
        "cosmos-predict2.5",
        "cosmos_predict2_5",
    }:
        return CosmosPredict25Adapter(config)
    if family in {"dreamx_world", "dreamx-world", "dreamx"}:
        return DreamXWorldAdapter(config)
    if family in {"echo_infinity", "echo-infinity", "echoinfinity"}:
        return EchoInfinityAdapter(config)
    if family in {"evoke", "alaya_evoke", "alaya-evoke", "alayaevoke"}:
        return EvokeAdapter(config)
    if family in {"lingbot_world_v2", "lingbot-world-v2"}:
        return LingBotWorldV2Adapter(config)
    if family in {"lingbot", "lingbot_world", "lingbot-world"}:
        return LingBotWorldAdapter(config)
    if family in {"lingbot_video", "lingbot-video"}:
        return LingBotVideoAdapter(config)
    if family in {"gen3c"}:
        return Gen3CAdapter(config)
    if family in {"luciddreamer", "lucid_dreamer", "lucid-dreamer"}:
        return LucidDreamerAdapter(config)
    if family in {"hy_worldplay", "hy-worldplay", "worldplay"}:
        return HYWorldPlayAdapter(config)
    if family in {
        "hunyuan_gamecraft",
        "hunyuan-gamecraft",
        "hunyuan_gamecraft_1_0",
        "hunyuan-gamecraft-1.0",
    }:
        return HunyuanGameCraftAdapter(config)
    if family in {"hunyuan_video", "hunyuan-video", "hunyuanvideo"}:
        return HunyuanVideoAdapter(config)
    if family in {"hunyuanworld", "hunyuan_world", "hunyuanworld_1_0", "hunyuanworld-1.0"}:
        return HunyuanWorldAdapter(config)
    if family in {
        "hunyuanworld_voyager",
        "hunyuanworld-voyager",
        "hunyuan_world_voyager",
        "hunyuan-world-voyager",
        "voyager",
    }:
        return HunyuanWorldVoyagerAdapter(config)
    if family in {"longcat_video", "longcat-video", "longcat"}:
        return LongCatVideoAdapter(config)
    if family in {"ltx2.3", "ltx23", "ltx_2_3", "ltx-2.3", "ltx-23", "ltx_video"}:
        return LTX23Adapter(config)
    if family in {
        "matrix_game",
        "matrix-game",
        "matrix_game_1",
        "matrix-game-1",
        "matrix_game_2",
        "matrix-game-2",
        "matrix_game_3",
        "matrix-game-3",
    }:
        return MatrixGameAdapter(config)
    if family in {
        "matrix_game_35",
        "matrix-game-35",
        "matrix_game_3_5",
        "matrix-game-3.5",
    }:
        return MatrixGame35Adapter(config)
    if family in {"motionctrl", "motion_ctrl", "motion-ctrl"}:
        return MotionCtrlAdapter(config)
    if family in {"minimax_h3", "minimax-h3", "minimaxh3", "minimax_h3_t2va"}:
        return MiniMaxH3Adapter(config)
    if family in {"lyra1", "lyra_1", "lyra-1"}:
        return Lyra1Adapter(config)
    if family in {"lyra2", "lyra_2", "lyra-2"}:
        return Lyra2Adapter(config)
    if family in {"flashworld", "flash_world", "flash-world"}:
        return FlashWorldAdapter(config)
    if family in {"inspatio_world", "inspatio-world"}:
        return InSpatioWorldAdapter(config)
    if family in {"infinite_world", "infinite-world"}:
        return InfiniteWorldAdapter(config)
    if family in {"neoverse", "neo_verse", "neo-verse"}:
        return NeoVerseAdapter(config)
    if family in {"stable_virtual_camera", "stable-virtual-camera"}:
        return StableVirtualCameraAdapter(config)
    if family in {"wonderworld", "wonder_world", "wonder-world"}:
        return WonderWorldAdapter(config)
    if family in {"wonderjourney", "wonder_journey", "wonder-journey"}:
        return WonderJourneyAdapter(config)
    if family in {"worldcam"}:
        return WorldCamAdapter(config)
    if family in {"worldfm", "world_fm", "world-fm"}:
        return WorldFMAdapter(config)
    if family in {"yume", "yume1_5", "yume_1_5", "yume-1.5", "yume1.5"}:
        return YumeAdapter(config)
    if family in {"wan", "wan22", "wan2.2", "wan_2_2", "wan2_2"}:
        return Wan22Adapter(config)
    if family in {"warp_as_history", "warp-as-history", "warpas_history", "wah"}:
        return WarpAsHistoryAdapter(config)
    raise ValueError(f"Unsupported model family: {config.family}")
