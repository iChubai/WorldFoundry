# UniWorld-View

WorldFoundry's `uniworld-view` pipeline runs the official [UniWorld-View](https://github.com/PKU-YuanGroup/UniWorld-View) `inference.py` in a separate Python environment. The official source and weights stay outside this Git repository. This adapter currently has command and asset validation; a complete GPU inference result is still pending.

Set up the upstream runtime in its own environment using the upstream README (Python 3.10, Torch 2.4.0 CUDA 12.4, its `requirements.txt`, `carvekit`, Eigen, and PyTorch3D for hybrid/mesh rendering). Pin the source revisions used by this adapter:

```bash
git clone https://github.com/PKU-YuanGroup/UniWorld-View.git "$HOME/.cache/worldfoundry/official_runtime_repos/UniWorld-View"
cd "$HOME/.cache/worldfoundry/official_runtime_repos/UniWorld-View"
git checkout 660212c48bff7c91a2ea5c7d5f3785af581100e2
git submodule update --init extern/STream3R
bash checkpoints/download_hf.sh --stream3r
```

The last command downloads the UniView transformer, **Diffusers-format** Wan2.1-VACE-14B base, BLIP2, MoGe, SAM2, TracerB7, CausVid LoRA v2, and STream3R. These files are large. The adapter discovers the upstream `checkpoints/` layout and WorldFoundry's flat `ckpt`/`ckpts` layout; `checkpoint_root` explicitly selects an upstream-layout weight directory. The native Wan VACE `.pth` release cannot substitute for the Diffusers model.

Set `UNIWORLD_VIEW_REPO` if the source lives elsewhere. The adapter automatically uses `${WORLDFOUNDRY_CONDA_ENVS_ROOT}/uniworld-view-cu124/bin/python` when it exists; set `UNIWORLD_VIEW_PYTHON` to override it. Use the pipeline directly or select `uniworld-view` in WorldFoundry Studio/CLI:

```python
from worldfoundry.pipelines.uniworld_view import UniWorldViewPipeline

pipeline = UniWorldViewPipeline.from_pretrained(device="cuda:0")
result = pipeline(
    images="/path/to/reference.jpg",
    output_path="/path/to/novel-view.mp4",
    d_phi=50,
    num_frames=81,
    return_dict=True,
)
```

For a monocular source video, pass `video="/path/to/source.mp4"` and `mode="dynamic_view"`. The default `hybrid` renderer needs PyTorch3D; `render_method="warp"` selects the upstream BiSplat path. Camera options (`d_phi`, `d_theta`, offsets, trajectory type) and the default 81-frame, 8-step generation follow upstream `run_infer.sh`. The upstream CLI generates its own BLIP2 caption; it does not use a custom text prompt. Use `plan_only=True` to inspect the exact subprocess command without loading weights.

The pinned upstream commit leaves `UniScene.setup_stream3r()` empty while its dynamic-view inference calls `self.stream3r`. Its single-view final video also defaults to 8 fps despite the CLI `--fps` value. WorldFoundry's narrow entry point restores the intended STream3R load line and applies the requested frame rate to that final video before executing the unmodified official `inference.py`; this is not yet a claim of checkpoint-backed parity.
