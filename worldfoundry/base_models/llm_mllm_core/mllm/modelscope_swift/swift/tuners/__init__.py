# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import TYPE_CHECKING

from worldfoundry.core.utils.python.lazy_module import _LazyModule

if TYPE_CHECKING:
    from .adapter import Adapter, AdapterConfig, AdapterModule
    from .base import Swift, SwiftModel

    # isort: split
    from .lora import LoRA, LoRAConfig

    # isort: split
    # isort: split
    from .mapping import SWIFT_MAPPING, SwiftTuners

    # isort: split
    from .neftune import NEFTune, NEFTuneConfig
    from .side import Side, SideConfig, SideModule

    # isort: split
    from .longlora.longlora import LongLoRA, LongLoRAConfig, LongLoRAModelType

    # isort: split
    from .llamapro import LLaMAPro, LLaMAProConfig
    from .reft import Reft, ReftConfig
    from .restuning import ResTuning, ResTuningBypassModule, ResTuningConfig

    # isort: split
    from .peft import (
        AdaLoraConfig,
        BOFTConfig,
        LoftQConfig,
        LoHaConfig,
        LoKrConfig,
        LoraConfig,
        OFTConfig,
        PeftConfig,
        PeftModel,
        PeftModelForCausalLM,
        PeftModelForSeq2SeqLM,
        PeftModelForSequenceClassification,
        PeftModelForTokenClassification,
        PrefixTuningConfig,
        PromptEncoderConfig,
        PromptLearningConfig,
        PromptTuningConfig,
        VeraConfig,
        get_peft_config,
        get_peft_model,
        get_peft_model_state_dict,
    )

    # isort: split
    from .prompt import Prompt, PromptConfig, PromptModule
    from .scetuning.scetuning import SCETuning, SCETuningConfig
    from .utils import SwiftConfig, SwiftOutput, swift_to_peft_format
else:
    _import_structure = {
        "adapter": ["Adapter", "AdapterConfig", "AdapterModule"],
        "base": ["SwiftModel", "Swift"],
        "lora": ["LoRA", "LoRAConfig"],
        "longlora.longlora": ["LongLoRAModelType", "LongLoRAConfig", "LongLoRA"],
        "mapping": ["SWIFT_MAPPING", "SwiftTuners"],
        "side": ["Side", "SideConfig", "SideModule"],
        "reft": ["Reft", "ReftConfig"],
        "llamapro": ["LLaMAPro", "LLaMAProConfig"],
        "neftune": ["NEFTune", "NEFTuneConfig"],
        "restuning": ["ResTuning", "ResTuningConfig", "ResTuningBypassModule"],
        "peft": [
            "AdaLoraConfig",
            "LoftQConfig",
            "LoHaConfig",
            "LoKrConfig",
            "LoraConfig",
            "VeraConfig",
            "BOFTConfig",
            "OFTConfig",
            "PeftConfig",
            "PeftModel",
            "PeftModelForCausalLM",
            "PeftModelForSeq2SeqLM",
            "PeftModelForSequenceClassification",
            "PeftModelForTokenClassification",
            "PrefixTuningConfig",
            "PromptEncoderConfig",
            "PromptLearningConfig",
            "PromptTuningConfig",
            "get_peft_config",
            "get_peft_model",
            "get_peft_model_state_dict",
        ],
        "prompt": ["Prompt", "PromptConfig", "PromptModule"],
        "scetuning": ["SCETuning", "SCETuningConfig"],
        "utils": ["SwiftConfig", "SwiftOutput", "swift_to_peft_format"],
    }

    import sys

    sys.modules[__name__] = _LazyModule(
        __name__,
        globals()["__file__"],
        _import_structure,
        module_spec=__spec__,
        extra_objects={},
    )
