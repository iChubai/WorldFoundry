import functools
import math

import numpy as np
import torch


@functools.lru_cache(maxsize=5)
def generate_freq_grid_np(
    positional_embedding_theta: float, positional_embedding_max_pos_count: int, inner_dim: int
) -> torch.Tensor:
    theta = positional_embedding_theta
    start = 1
    end = theta

    n_elem = 2 * positional_embedding_max_pos_count
    pow_indices = np.power(
        theta,
        np.linspace(
            np.log(start) / np.log(theta),
            np.log(end) / np.log(theta),
            inner_dim // n_elem,
            dtype=np.float64,
        ),
    )
    # as_tensor (not tensor): under torch.compile the numpy ops above are traced as tensors,
    # so torch.tensor() would copy-construct from a tensor (warns); as_tensor casts in place.
    return torch.as_tensor(pow_indices * math.pi / 2, dtype=torch.float32)
