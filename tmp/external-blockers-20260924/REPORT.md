# WorldGen / AC3D 资产预检（2026-09-24）

预检使用 `check_assets.py`，结果在 `asset-preflight.json`。只核查了当前主机的路径及代码实际加载要求，不将接口导入或历史目录描述计为推理通过。

- **WorldGen**：本地有 `LeoXie/WorldGen` 的 text2scene 与 img2scene LoRA（分别 34.6 MB、44.9 MB），但 `pano_gen.py` 需要的 FLUX.1-dev 底座在本地 checkpoint 根、HFD 镜像路径及 Hugging Face Hub 缓存中均不存在。`FLUX.1-Redux-dev` 不是同一底座。当前无法做真实 GPU 推理，记为 `blocked_external`。
- **AC3D**：项目内有官方推理适配代码，但默认 checkpoint 根和相邻 staged 根均无 2B/5B `checkpoint-10000.pt` ControlNet 权重；专用环境亦未找到。当前无法做真实 GPU 推理，记为 `blocked_external`。目录中先前声称本地权重已就位且 runner 已验证的文字已修正，避免与实况冲突。
