from typing import TYPE_CHECKING

from transformers.utils import (
    OptionalDependencyNotAvailable,
    _LazyModule,
    is_torch_available,
)

_import_structure = {
    "configuration_otter": [
        "OtterConfig",
    ],
}

try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["modeling_otter"] = [
        "OtterModel",
        "OtterPreTrainedModel",
        "OtterForConditionalGeneration",
    ]

if TYPE_CHECKING:
    from .configuration_otter import OtterConfig as OtterConfig

    # from .processing_otter import OtterProcessor

    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .modeling_otter import (
            OtterForConditionalGeneration as OtterForConditionalGeneration,
        )
        from .modeling_otter import (
            OtterModel as OtterModel,
        )
        from .modeling_otter import (
            OtterPreTrainedModel as OtterPreTrainedModel,
        )

else:
    import sys

    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
