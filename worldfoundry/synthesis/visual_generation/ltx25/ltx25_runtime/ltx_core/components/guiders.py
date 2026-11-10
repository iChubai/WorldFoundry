from dataclasses import dataclass, field


@dataclass(frozen=True)
class MultiModalGuiderParams:
    """
    Parameters for the multi-modal guider.
    """

    cfg_scale: float = 1.0
    "CFG (Classifier-free guidance) scale controlling how strongly the model adheres to the prompt."
    stg_scale: float = 0.0
    "STG (Spatio-Temporal Guidance) scale controls how strongly the model reacts to the perturbation of the modality."
    stg_blocks: list[int] | None = field(default_factory=list)
    "Which transformer blocks to perturb for STG."
    rescale_scale: float = 0.0
    "Rescale scale controlling how strongly the model rescales the modality after applying other guidance."
    modality_scale: float = 1.0
    "Modality scale controlling how strongly the model reacts to the perturbation of the modality."
    skip_step: int = 0
    "Skip step controlling how often the model skips the step."
