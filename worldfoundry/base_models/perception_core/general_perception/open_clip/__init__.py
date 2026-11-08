"""Module for base_models -> perception_core -> general_perception -> open_clip -> __init__.py functionality."""

from .coca_model import CoCa
from .constants import OPENAI_DATASET_MEAN, OPENAI_DATASET_STD
from .version import __version__

# isort: split
from .factory import create_model, create_model_and_transforms, create_model_from_pretrained, get_tokenizer

# isort: split
# isort: split
from .factory import add_model_config, get_model_config, list_models, load_checkpoint

# isort: split
from .clap_model import CLAP
from .model import (
    CLIP,
    CLIPTextCfg,
    CLIPVisionCfg,
    CustomTextCLIP,
    convert_weights_to_fp16,
    convert_weights_to_lp,
    get_cast_dtype,
    get_input_dtype,
    get_model_preprocess_cfg,
    get_model_tokenize_cfg,
    set_model_preprocess_cfg,
)
from .pretrained import (
    download_pretrained,
    download_pretrained_from_url,
    get_pretrained_cfg,
    get_pretrained_url,
    is_pretrained_cfg,
    list_pretrained,
    list_pretrained_models_by_tag,
    list_pretrained_tags_by_model,
)

# isort: split
from .tokenizer import SimpleTokenizer, decode, tokenize

# isort: split
from .audio import AUDIO_AVAILABLE, AudioAugmentationCfg, AudioTower, CLIPAudioCfg, audio_transform_v2, require_audio
from .transform import AugmentationCfg, image_transform
from .zero_shot_classifier import build_zero_shot_classifier, build_zero_shot_classifier_legacy
from .zero_shot_metadata import IMAGENET_CLASSNAMES, OPENAI_IMAGENET_TEMPLATES, SIMPLE_IMAGENET_TEMPLATES
