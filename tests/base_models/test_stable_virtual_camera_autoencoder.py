from unittest.mock import patch

import pytest

# This test module imports worldfoundry code that requires the optional
# "diffusers" dependency at import time; skip when it is unavailable.
pytest.importorskip("diffusers")
# The Stable Virtual Camera runtime is license-gated and exists only in local checkouts.
seva_autoencoder = pytest.importorskip(
    "worldfoundry.base_models.three_dimensions.general_3d.stable_virtual_camera."
    "stable_virtual_camera_runtime.seva.modules.autoencoder"
)


def test_autoencoder_prefers_local_checkpoint(tmp_path, monkeypatch):
    local_model = tmp_path / "stable-diffusion-2-1-base"
    local_model.mkdir()
    monkeypatch.setenv("WORLDFOUNDRY_CKPT_DIR", str(tmp_path))

    with patch.object(seva_autoencoder.AutoencoderKL, "from_pretrained") as load:
        load.return_value.eval.return_value.requires_grad_.return_value = None
        seva_autoencoder.AutoEncoder()

    load.assert_called_once_with(
        str(local_model),
        subfolder="vae",
        force_download=False,
        low_cpu_mem_usage=False,
    )
