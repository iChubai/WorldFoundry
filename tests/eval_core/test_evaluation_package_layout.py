from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_scripts_qa_tree_is_not_packaged() -> None:
    assert not any(path.name == "qa" for path in (REPO_ROOT / "scripts").iterdir() if path.is_dir())


def test_dataset_helpers_are_split_by_responsibility() -> None:
    tasks_root = REPO_ROOT / "worldfoundry/evaluation/tasks"
    datasets_facade = tasks_root / "datasets" / "__init__.py"

    assert sum(1 for _ in datasets_facade.open(encoding="utf-8")) < 180
    for retired_dataset_module in (
        "dataset_manifest.py",
        "dataset_manager.py",
        "dataset_readiness.py",
        "dataset_materialization.py",
        "datasets.py",
    ):
        assert not (tasks_root / retired_dataset_module).exists()
    for marker in ("# manifest.py", "# manager.py", "# readiness.py", "# materialization.py"):
        assert marker not in datasets_facade.read_text(encoding="utf-8")


def test_zoo_schemas_use_shared_json_contract_base() -> None:
    model_schema = REPO_ROOT / "worldfoundry/evaluation/models/catalog/schema.py"
    benchmark_schema = REPO_ROOT / "worldfoundry/evaluation/tasks/catalog/schema.py"

    for path in (model_schema, benchmark_schema):
        text = path.read_text(encoding="utf-8")
        assert "JsonSerializable = JsonContract" in text
        assert "def _to_plain(" not in text
        assert "def _json_dumps(" not in text
        assert "def _require_mapping(" not in text
        assert "class JsonSerializable" not in text


def test_embodied_contracts_use_shared_json_contract_base() -> None:
    contract_path = REPO_ROOT / "worldfoundry/evaluation/tasks/embodied/contracts.py"
    text = contract_path.read_text(encoding="utf-8")

    assert "JsonDataclass = JsonContract" in text
    assert "def _to_plain(" not in text
    assert "class JsonDataclass" not in text
    assert "hashlib" not in text
    assert "json.dumps" not in text


def test_embodied_task_modules_are_owned_by_embodied_package() -> None:
    tasks_root = REPO_ROOT / "worldfoundry/evaluation/tasks"

    import worldfoundry.evaluation.tasks.embodied as canonical
    import worldfoundry.evaluation.tasks.embodied.contracts as canonical_contracts
    import worldfoundry.evaluation.tasks.embodied.evaluate as canonical_evaluate
    import worldfoundry.evaluation.tasks.embodied.materialize as canonical_materialize
    import worldfoundry.evaluation.tasks.embodied.metrics as canonical_metrics
    import worldfoundry.evaluation.tasks.embodied.normalizer as canonical_normalizer

    for retired_module in (
        "run_embodied.py",
        "run_embodied_contracts.py",
        "run_embodied_evaluate.py",
        "run_embodied_materialize.py",
        "run_embodied_metrics.py",
        "run_embodied_normalizer.py",
    ):
        assert not (tasks_root / retired_module).exists()

    assert canonical_contracts.EmbodiedGenerationSpec is canonical.EmbodiedGenerationSpec
    assert canonical_evaluate.VlaVaWamRunRequest is canonical.VlaVaWamRunRequest
    assert canonical_materialize.materialize_vla_va_wam_requests is canonical.materialize_vla_va_wam_requests
    assert canonical_metrics.ResultFieldMetric is canonical.ResultFieldMetric
    assert canonical_normalizer.normalize_results is canonical.normalize_vla_va_wam_results


def test_external_benchmark_contract_registry_is_split_from_contract_data() -> None:
    tasks_root = REPO_ROOT / "worldfoundry/evaluation/tasks"
    contracts_module = tasks_root / "contracts" / "external.py"
    registry_module = tasks_root / "contracts" / "registry.py"

    contracts_text = contracts_module.read_text(encoding="utf-8")
    registry_text = registry_module.read_text(encoding="utf-8")

    assert not (tasks_root / "benchmark_contracts.py").exists()
    assert not (tasks_root / "benchmark_contract_registry.py").exists()
    assert not (tasks_root / "benchmark_contract_types.py").exists()
    assert not (tasks_root / "contracts" / "types.py").exists()
    assert not (tasks_root / "contracts" / "sources.py").exists()
    assert "class ExternalBenchmarkContract" in registry_text
    assert "class ExternalBenchmarkContractRegistry" in registry_text
    assert "ExternalBenchmarkContractRegistry(" in contracts_text
    assert "class ExternalBenchmarkContract" not in contracts_text
    assert "_CONTRACTS:" not in contracts_text
    assert 'known = ", ".join' not in contracts_text


def test_task_package_facade_is_lazy_and_bounded() -> None:
    tasks_init = REPO_ROOT / "worldfoundry/evaluation/tasks/__init__.py"
    text = tasks_init.read_text(encoding="utf-8")
    watched_modules = [
        "worldfoundry.evaluation.tasks.catalog.zoo_registry",
        "worldfoundry.evaluation.tasks.catalog.yaml",
        "worldfoundry.evaluation.tasks.catalog.registry",
        "worldfoundry.evaluation.tasks.execution.runners",
    ]
    script = (
        "import json, sys; "
        "import worldfoundry.evaluation.tasks; "
        f"watched = {watched_modules!r}; "
        "print(json.dumps([name for name in watched if name in sys.modules]))"
    )

    assert sum(1 for _ in tasks_init.open(encoding="utf-8")) < 90
    assert "from .catalog import" not in text
    assert "from .native_loader import" not in text
    assert "from .registry import" not in text
    assert "import_module" in text

    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=REPO_ROOT,
        check=True,
        text=True,
        capture_output=True,
    )
    assert json.loads(result.stdout) == []

    from worldfoundry.evaluation.tasks import load_benchmark_zoo_registry, load_task_registry_from_paths
    from worldfoundry.evaluation.tasks.catalog.registry import load_task_registry_from_paths as RegistryLoader
    from worldfoundry.evaluation.tasks.catalog.zoo_registry import load_benchmark_zoo_registry as ZooLoader

    assert load_benchmark_zoo_registry is ZooLoader
    assert load_task_registry_from_paths is RegistryLoader


def test_native_task_package_is_retired() -> None:
    tasks_root = REPO_ROOT / "worldfoundry/evaluation/tasks"

    assert not (tasks_root / "native").exists()
    for retired_native_shim in ("native_schema.py", "native_loader.py", "yaml.py"):
        assert not (tasks_root / retired_native_shim).exists()

    import importlib

    import pytest

    import worldfoundry.evaluation.tasks as facade
    import worldfoundry.evaluation.tasks.catalog.yaml as canonical_yaml

    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("worldfoundry.evaluation.tasks.native")
    assert facade.load_catalog_yaml is canonical_yaml.load_catalog_yaml
    assert facade.load_yaml_mapping_with_extends is canonical_yaml.load_yaml_mapping_with_extends


def test_model_pipeline_package_keeps_path_a_surface_only() -> None:
    models_root = REPO_ROOT / "worldfoundry/evaluation/models"
    pipeline_root = models_root / "pipelines"
    pipelines_facade = pipeline_root / "__init__.py"

    pipelines_facade_text = pipelines_facade.read_text(encoding="utf-8")
    for marker in (
        "_LEGACY",
        "UNSUPPORTED_GENERIC_INFER",
        "load_*_pipeline",
        "infer_*_pipeline",
    ):
        assert marker not in pipelines_facade_text

    lazy_script = (
        "import json, sys; "
        "import worldfoundry.evaluation.models.pipelines; "
        "blocked = {'torch', 'diffusers'}; "
        "print(json.dumps(sorted(name for name in sys.modules if name.split('.')[0] in blocked)))"
    )
    lazy_result = subprocess.run(
        [sys.executable, "-c", lazy_script],
        cwd=REPO_ROOT,
        check=True,
        text=True,
        capture_output=True,
    )
    assert json.loads(lazy_result.stdout) == []

    assert not (models_root / "compat").exists()
    for legacy_root_module in ("pipeline_runner.py", "resolver.py", "zoo.py"):
        assert not (models_root / legacy_root_module).exists()
    for module in (
        "lifecycle",
        "invocation",
        "results",
        "loading",
        "bindings",
        "aliases",
        "discovery",
    ):
        assert (pipeline_root / f"{module}.py").is_file()
    for removed_module in (
        "dispatch",
        "handlers",
        "components",
        "adapters",
        "media",
        "game_world",
        "named",
        "runtime_video",
        "variants",
        "world_video",
        "three" + "_dim",
        "loading" + "_maps",
    ):
        assert not (pipeline_root / f"{removed_module}.py").exists()
    for removed_module in (
        "pipeline_" + "loading.py",
        "pipeline_" + "inference.py",
        "reference_" + "runner.py",
    ):
        assert not (models_root / removed_module).exists()
    for module in ("manifest", "policy", "registry", "schema", "zoo_registry"):
        assert (models_root / "catalog" / f"{module}.py").is_file()
    for module in ("profiles", "environments", "assets"):
        assert (models_root / "runtime" / f"{module}.py").is_file()
    for module in ("builtins", "pipeline", "plugins", "registry", "resolver"):
        assert (models_root / "runners" / f"{module}.py").is_file()
    assert not (models_root / "runners" / "reference.py").exists()


def test_model_package_facade_is_lazy_and_bounded() -> None:
    models_init = REPO_ROOT / "worldfoundry/evaluation/models/__init__.py"
    text = models_init.read_text(encoding="utf-8")
    watched_modules = [
        "worldfoundry.evaluation.models.catalog.registry",
        "worldfoundry.evaluation.models.catalog.manifest",
        "worldfoundry.evaluation.models.integrations",
        "worldfoundry.evaluation.models.pipelines.lifecycle",
        "worldfoundry.evaluation.models.runners.builtins",
        "worldfoundry.evaluation.models.runners.resolver",
        "worldfoundry.evaluation.models.runners.registry",
    ]
    script = (
        "import json, sys; "
        "import worldfoundry.evaluation.models; "
        f"watched = {watched_modules!r}; "
        "print(json.dumps([name for name in watched if name in sys.modules]))"
    )

    assert "from .integrations import" not in text
    assert "from .pipeline_runner import" not in text
    assert "from .runtime_runners import" not in text
    assert "from .resolver import" not in text
    assert "import_module" in text

    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=REPO_ROOT,
        check=True,
        text=True,
        capture_output=True,
    )
    assert json.loads(result.stdout) == []

    from worldfoundry.evaluation.models import ModelRunnerRegistry, WorldFoundryPipelineRunner
    from worldfoundry.evaluation.models.runners.pipeline import WorldFoundryPipelineRunner as CanonicalPipelineRunner
    from worldfoundry.evaluation.models.runners.registry import ModelRunnerRegistry as CanonicalRegistry

    assert WorldFoundryPipelineRunner is CanonicalPipelineRunner
    assert ModelRunnerRegistry is CanonicalRegistry


def test_model_catalog_uses_canonical_pipeline_runner_target() -> None:
    catalog_root = REPO_ROOT / "worldfoundry/data/models/catalog"
    legacy_target = "worldfoundry.evaluation.models." + "pipeline_runner:WorldFoundryPipelineRunner"
    canonical_target = "worldfoundry.evaluation.models.runners.pipeline:WorldFoundryPipelineRunner"

    runner_targets = [
        line.strip()
        for path in catalog_root.rglob("*.yaml")
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip().startswith("runner_target:")
    ]

    assert runner_targets
    assert not any(legacy_target in line for line in runner_targets)
    assert any(canonical_target in line for line in runner_targets)


def test_retired_model_root_reexport_shims_are_not_reintroduced() -> None:
    models_root = REPO_ROOT / "worldfoundry/evaluation/models"
    retired = {
        "catalog_registry.py",
        "manifests.py",
        "pipeline_runner.py",
        "pipeline_" + "loading.py",
        "pipeline_" + "inference.py",
        "reference_" + "runner.py",
        "resolver.py",
        "runtime_runners.py",
        "zoo.py",
        "zoo_manifests.py",
        "zoo_policy.py",
        "zoo_registry.py",
        "zoo_variant_selection.py",
    }

    assert not sorted(path.name for path in models_root.glob("*.py") if path.name in retired)


def test_low_value_evaluation_subpackages_are_not_reintroduced() -> None:
    evaluation_root = REPO_ROOT / "worldfoundry/evaluation"

    forbidden_paths = [
        evaluation_root / "api" / "registry",
        evaluation_root / "cli",
        evaluation_root / "local_open_eval.py",
        evaluation_root / "mcp",
        evaluation_root / "models" / "pipeline_dispatch",
        evaluation_root / "reporting.py",
        evaluation_root / "reporting_format.py",
        evaluation_root / "reporting_markdown.py",
        evaluation_root / "reporting_run_browser.py",
        evaluation_root / "reporting_run_comparison.py",
        evaluation_root / "reporting_run_index.py",
        evaluation_root / "reporting_run_manifest.py",
        evaluation_root / "reporting_run_report.py",
        evaluation_root / "reporting_run_summary.py",
        evaluation_root / "reporting_runtime_evidence.py",
        evaluation_root / "reporting_compare.py",
        evaluation_root / "runtime",
        evaluation_root / "runtime.py",
        evaluation_root / "runtime_assets.py",
        evaluation_root / "runtime_conda.py",
        evaluation_root / "runtime_env.py",
        evaluation_root / "runtime_jobs.py",
        evaluation_root / "runtime_probes.py",
        evaluation_root / "runner",
        evaluation_root / "scorecard.py",
        evaluation_root / "tui",
        evaluation_root / "worldscore.py",
        evaluation_root / "runner" / "embodied",
        evaluation_root / "runner" / "metrics",
        evaluation_root / "tasks" / "zoo",
        evaluation_root / "tasks" / "zoo.py",
        evaluation_root / "utils",
        evaluation_root / "models" / "zoo",
    ]

    assert [str(path.relative_to(REPO_ROOT)) for path in forbidden_paths if path.exists()] == []


def test_old_evaluation_import_paths_are_not_reintroduced() -> None:
    old_markers = [
        "worldfoundry.evaluation.benchmarks",
        "worldfoundry.evaluation.cli",
        "worldfoundry.evaluation.local_open_eval",
        "worldfoundry.evaluation.mcp",
        "worldfoundry.evaluation.worldscore",
        "worldfoundry/evaluation/benchmarks",
        "worldfoundry/evaluation/cli",
        "worldfoundry/evaluation/mcp",
        "from ..benchmarks",
        "from .benchmarks",
        "worldfoundry.evaluation.api.registry.aliases",
        "worldfoundry.evaluation.api.registry.core",
        "worldfoundry.evaluation.models.pipeline_dispatch",
        "worldfoundry.evaluation.reporting_format",
        "worldfoundry.evaluation.reporting_markdown",
        "worldfoundry.evaluation.reporting_run_summary",
        "worldfoundry.evaluation.reporting_runtime_evidence",
        "worldfoundry.evaluation.runtime",
        "worldfoundry.evaluation.runtime.assets",
        "worldfoundry.evaluation.runtime.env",
        "worldfoundry.evaluation.tui",
        "worldfoundry.evaluation.tasks.run_embodied.normalizer",
        "worldfoundry.evaluation.tasks.run_embodied.metrics",
        "worldfoundry.evaluation.tasks.run_metrics.registry",
        "worldfoundry.evaluation.tasks." + "zoo.",
        "worldfoundry.evaluation.models." + "zoo.",
        "worldfoundry.evaluation.utils.paths",
    ]
    violations = []
    for path in (REPO_ROOT / "worldfoundry/evaluation").rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        for marker in old_markers:
            if marker in text:
                violations.append(f"{path.relative_to(REPO_ROOT)}: {marker}")

    assert violations == []


def test_docs_do_not_advertise_removed_evaluation_subpackages() -> None:
    forbidden_markers = [
        "worldfoundry/evaluation/runner/",
        "evaluation/runner/",
        "worldfoundry/evaluation/benchmark_zoo",
        "evaluation/benchmark_zoo",
        "worldfoundry/evaluation/cli",
        "worldfoundry/evaluation/mcp",
        "evaluation/cli",
        "evaluation/mcp",
        "worldfoundry/evaluation/model_zoo",
        "evaluation/model_zoo",
        "evaluation/models_schema.py",
        "evaluation/models_registry.py",
        "evaluation/models_manifests.py",
        "worldfoundry/evaluation/reporting.py",
        "worldfoundry/evaluation/reporting_run_",
        "worldfoundry/evaluation/runtime.py",
        "worldfoundry/evaluation/runtime_",
        "worldfoundry/evaluation/scorecard.py",
        "evaluation/reporting.py",
        "evaluation/reporting_compare.py",
        "evaluation/reporting_run_",
        "evaluation/runtime.py",
        "evaluation/runtime_",
        "evaluation/scorecard.py",
        '<Folder name="runner"',
        "worldfoundry/evaluation/tasks/" + "zoo.py",
        "evaluation/tasks/" + "zoo.py",
    ]
    docs_roots = [
        REPO_ROOT / "docs" / "fumadocs" / "content" / "docs",
        REPO_ROOT / "docs" / "design",
    ]
    docs_files = [
        REPO_ROOT / "README.md",
    ]

    violations = []
    for root in docs_roots:
        for path in sorted(root.rglob("*")):
            if path.suffix not in {".md", ".mdx"}:
                continue
            text = path.read_text(encoding="utf-8")
            for marker in forbidden_markers:
                if marker in text:
                    violations.append(f"{path.relative_to(REPO_ROOT)}: {marker}")
    for path in docs_files:
        text = path.read_text(encoding="utf-8")
        for marker in forbidden_markers:
            if marker in text:
                violations.append(f"{path.relative_to(REPO_ROOT)}: {marker}")

    assert violations == []
