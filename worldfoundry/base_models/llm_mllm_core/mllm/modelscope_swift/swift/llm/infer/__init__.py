# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import TYPE_CHECKING

from worldfoundry.core.utils.python.lazy_module import _LazyModule

if TYPE_CHECKING:
    from .infer import SwiftInfer, infer_main

    # isort: split
    # isort: split
    from .rollout import rollout_main

    # isort: split
    # isort: split
    from .deploy import SwiftDeploy, deploy_main, run_deploy

    # isort: split
    from .request_config import RequestConfig

    # isort: split
    from .protocol import Function

    # isort: split
    # isort: split
    from .utils import prepare_model_template

    # isort: split
    from .infer_engine import (
        AdapterRequest,
        BaseInferEngine,
        InferClient,
        InferEngine,
        LmdeployEngine,
        PtEngine,
        VllmEngine,
        prepare_generation_config,
    )
else:
    _import_structure = {
        "rollout": ["rollout_main"],
        "infer": ["infer_main", "SwiftInfer"],
        "deploy": ["deploy_main", "SwiftDeploy", "run_deploy"],
        "request_config": ["RequestConfig"],
        "protocol": ["Function"],
        "utils": ["prepare_model_template"],
        "infer_engine": [
            "InferEngine",
            "VllmEngine",
            "LmdeployEngine",
            "PtEngine",
            "InferClient",
            "prepare_generation_config",
            "AdapterRequest",
            "BaseInferEngine",
        ],
    }

    import sys

    sys.modules[__name__] = _LazyModule(
        __name__,
        globals()["__file__"],
        _import_structure,
        module_spec=__spec__,
        extra_objects={},
    )
