# Model runtime configurations

Static model inference presets and supporting model configuration files live here,
organized by model family. This includes YAML and JSON settings, Python presets
loaded by upstream runtimes, and example inference inputs.

Related metadata under `worldfoundry/data/models/`:

| Directory | Contents |
| --- | --- |
| `catalog/` | Model metadata and capabilities |
| `bindings/` | Pipeline entrypoints and defaults |
| `runtime/profiles/` | Runtime requirements and execution profiles |
| `runtime/environments/` | Dependency environment specifications |
| `runtime/configs/<model>/` | Model inference presets and supporting configs |

Resolve bundled presets from Python without depending on the working directory:

```python
from worldfoundry.core.io.paths import package_data_path

config = package_data_path(
    "models", "runtime", "configs", "wonderworld", "inference.yaml"
)
```

Runtime manifests can use
`${WORLDFOUNDRY_DATA_ROOT}/models/runtime/configs/<model>/<file>`;
WorldFoundry expands the token before launching the runtime. Explicit external
configuration paths remain supported by their respective runners.

Keep relative configuration inheritance within each model directory. Configuration
classes and loaders remain in Python source modules; training and post-training
recipes stay separate from these inference presets. Checkpoints, datasets, and
generated job configurations belong in the configured cache or output directories,
not in the installed package data.
