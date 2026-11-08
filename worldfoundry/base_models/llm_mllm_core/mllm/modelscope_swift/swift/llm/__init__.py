# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import TYPE_CHECKING

from worldfoundry.core.utils.python.lazy_module import _LazyModule

if TYPE_CHECKING:
    # Recommend using `xxx_main`
    from .argument import BaseArguments, DeployArguments, InferArguments
    from .infer import (
        AdapterRequest,
        BaseInferEngine,
        InferClient,
        InferEngine,
        LmdeployEngine,
        PtEngine,
        RequestConfig,
        VllmEngine,
        deploy_main,
        infer_main,
        prepare_model_template,
        run_deploy,
    )
    from .template import (
        TEMPLATE_MAPPING,
        InferRequest,
        MaxLengthError,
        Template,
        TemplateInputs,
        TemplateMeta,
        TemplateType,
        Word,
        draw_bbox,
        get_template,
        get_template_meta,
        load_file,
        load_image,
        register_template,
    )

    # isort: split
    from .model import (
        MODEL_ARCH_MAPPING,
        MODEL_MAPPING,
        HfConfigFactory,
        Model,
        ModelArch,
        ModelGroup,
        ModelInfo,
        ModelKeys,
        ModelMeta,
        ModelType,
        MultiModelKeys,
        get_llm_model,
        get_matched_model_meta,
        get_model_arch,
        get_model_info_meta,
        get_model_name,
        get_model_tokenizer,
        get_model_tokenizer_multimodal,
        get_model_tokenizer_with_flash_attn,
        git_clone_github,
        load_by_unsloth,
        register_model,
        register_model_arch,
        safe_snapshot_download,
    )

    # isort: split
    # isort: split
    from .dataset import (
        DATASET_MAPPING,
        DATASET_TYPE,
        AlpacaPreprocessor,
        AutoPreprocessor,
        DatasetMeta,
        EncodePreprocessor,
        HfDataset,
        LazyLLMDataset,
        MediaResource,
        MessagesPreprocessor,
        ResponsePreprocessor,
        RowPreprocessor,
        SubsetDataset,
        load_dataset,
        register_dataset,
        register_dataset_info,
        sample_dataset,
    )

    # isort: split
    # isort: split
    from .utils import (
        History,
        Messages,
        Processor,
        ProcessorMixin,
        deep_getattr,
        dynamic_gradient_checkpointing,
        get_cache_dir,
        get_temporary_cache_files_directory,
        history_to_messages,
        is_moe_model,
        messages_to_history,
        save_checkpoint,
        to_device,
        to_float_dtype,
    )

    # isort: split
    # isort: split
    from .base import SwiftPipeline

    # isort: split
    from .data_loader import BatchSamplerShard, DataLoaderDispatcher, DataLoaderShard
else:
    _import_structure = {
        "infer": [
            "deploy_main",
            "VllmEngine",
            "LmdeployEngine",
            "PtEngine",
            "infer_main",
            "InferClient",
            "run_deploy",
            "InferEngine",
            "AdapterRequest",
            "prepare_model_template",
            "BaseInferEngine",
        ],
        "infer.request_config": ["RequestConfig"],
        "argument": ["InferArguments", "DeployArguments", "BaseArguments"],
        "template": [
            "TEMPLATE_MAPPING",
            "Template",
            "Word",
            "get_template",
            "TemplateType",
            "register_template",
            "TemplateInputs",
            "TemplateMeta",
            "get_template_meta",
            "InferRequest",
            "load_image",
            "MaxLengthError",
            "load_file",
            "draw_bbox",
        ],
        "model": [
            "MODEL_MAPPING",
            "ModelType",
            "get_model_tokenizer",
            "safe_snapshot_download",
            "HfConfigFactory",
            "ModelInfo",
            "ModelMeta",
            "ModelKeys",
            "register_model_arch",
            "MultiModelKeys",
            "ModelArch",
            "MODEL_ARCH_MAPPING",
            "get_model_arch",
            "get_model_info_meta",
            "get_model_name",
            "register_model",
            "ModelGroup",
            "Model",
            "get_model_tokenizer_with_flash_attn",
            "get_model_tokenizer_multimodal",
            "load_by_unsloth",
            "git_clone_github",
            "get_matched_model_meta",
            "get_llm_model",
        ],
        "dataset": [
            "AlpacaPreprocessor",
            "MessagesPreprocessor",
            "AutoPreprocessor",
            "DATASET_MAPPING",
            "MediaResource",
            "register_dataset",
            "register_dataset_info",
            "EncodePreprocessor",
            "LazyLLMDataset",
            "load_dataset",
            "DATASET_TYPE",
            "sample_dataset",
            "RowPreprocessor",
            "ResponsePreprocessor",
            "DatasetMeta",
            "HfDataset",
            "SubsetDataset",
        ],
        "utils": [
            "deep_getattr",
            "to_device",
            "to_float_dtype",
            "History",
            "Messages",
            "history_to_messages",
            "messages_to_history",
            "Processor",
            "save_checkpoint",
            "ProcessorMixin",
            "get_temporary_cache_files_directory",
            "get_cache_dir",
            "is_moe_model",
            "dynamic_gradient_checkpointing",
        ],
        "base": ["SwiftPipeline"],
        "data_loader": ["DataLoaderDispatcher", "DataLoaderShard", "BatchSamplerShard"],
    }

    import sys

    sys.modules[__name__] = _LazyModule(
        __name__,
        globals()["__file__"],
        _import_structure,
        module_spec=__spec__,
        extra_objects={},
    )
