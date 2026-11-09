# Astronex-World inference runtime

Adapted from Astronex-Robotics/Astronex-World, revision
`27584bf1a89da01ef35a03b48d83bc818d2adf2b` (Apache License 2.0; see root
`THIRD-PARTY-NOTICES`). The `inference/`, `models/`, `utils/` and `post_train/data`
modules are the import closure of `inference/generate.py` and `inference/sample.py`.
Every file was modified by WorldFoundry: imports are package-relative, the code is
formatted to the repository style, `generate.py` launches the sampler as a package
module without editing `sys.path` or `PYTHONPATH`, and `astronex_env.child_env` only
locates the weights. No training entrypoints are included.
