"""Builder integration for compiled LTX transformer blocks."""

from worldfoundry.base_models.diffusion_model.models.networks.ltx.model import LTXModel
from worldfoundry.base_models.diffusion_model.optimizations.ltx.compile import CompilationConfig, compile_transformer

from .module_ops import ModuleOps
from .sd_ops import SDOps


def build_compile_transformer_op(config: CompilationConfig) -> ModuleOps:
    """Build a ``ModuleOps`` that compiles transformer blocks with the given settings."""
    return ModuleOps(
        name="compile_transformer",
        matcher=lambda model: isinstance(model, LTXModel),
        mutator=lambda model: compile_transformer(model, config),
    )


def modify_sd_ops_for_compilation(original_sd_ops: SDOps, number_of_blocks: int = 48) -> SDOps:
    for i in range(number_of_blocks):
        original_sd_ops = original_sd_ops.with_replacement(
            f"transformer_blocks.{i}.", f"transformer_blocks.{i}._orig_mod."
        )
    return original_sd_ops
