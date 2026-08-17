#!/usr/bin/env python3
"""Build the docs model-recipe index from WorldFoundry's source-of-truth manifests.

The generated JSON deliberately keeps catalog support, runtime integration,
environment compatibility, checkpoint provenance, and runner evidence separate.
Missing data stays missing instead of being inferred into a stronger claim.
"""

from __future__ import annotations

import json
import re
import shlex
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import yaml


ROOT = Path(__file__).resolve().parents[3]
DOCS_ROOT = Path(__file__).resolve().parents[1]
OUT = DOCS_ROOT / "lib" / "model-recipes-data.json"
INDEX_OUT = DOCS_ROOT / "lib" / "model-recipes-index.json"
EDITORIAL_ROOT = DOCS_ROOT / "lib" / "model-editorial"

sys.path.insert(0, str(ROOT))
from worldfoundry.core.inference import get_model_inference_spec  # noqa: E402

CATALOG_ROOT = ROOT / "worldfoundry/data/models/catalog"
PROFILE_ROOT = ROOT / "worldfoundry/data/models/runtime/profiles"
ENVIRONMENT_ROOT = ROOT / "worldfoundry/data/models/runtime/environments"
BINDING_ROOT = ROOT / "worldfoundry/data/models/bindings/pipelines"

CATEGORY_META = {
    "video": {
        "label": "Video",
        "label_zh": "视频",
        "description": "Video, image, and audio-visual generation or editing runtimes.",
    },
    "world_models": {
        "label": "World models",
        "label_zh": "世界模型",
        "description": "Interactive worlds, prediction, navigation, and simulator-shaped systems.",
    },
    "three_d_four_d": {
        "label": "3D & 4D",
        "label_zh": "3D 与 4D",
        "description": "Reconstruction, geometry, point clouds, scenes, and dynamic representations.",
    },
    "vla_va_wam": {
        "label": "Embodied",
        "label_zh": "具身智能",
        "description": "VLA, vision-action, world-action, and robot-policy runtimes.",
    },
    "hosted_api": {
        "label": "Hosted API",
        "label_zh": "托管 API",
        "description": "Provider-backed models that require credentials or a hosted endpoint.",
    },
}

STATUS_ORDER = {
    "verified": 0,
    "integrated": 1,
    "runtime_ported": 2,
    "profile": 3,
    "planned": 4,
    "blocked": 5,
}

# Docs catalog hides blocked 3D/4D entries — upstream YAML stays for CLI inspection.
DOCS_EXCLUDED_RECIPE_IDS: set[str] = set()


def docs_catalog_visible(category_id: str, model_id: str, status_group: str) -> bool:
    if model_id in DOCS_EXCLUDED_RECIPE_IDS:
        return False
    if category_id == "three_d_four_d" and status_group == "blocked":
        return False
    return True


def load_yaml(path: Path) -> Any:
    return yaml.safe_load(path.read_text()) or {}


def as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def text(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (str, int, float)):
        result = str(value).strip()
        return result or None
    return None


def compact_text(value: Any, limit: int = 420) -> str | None:
    result = text(value)
    if not result:
        return None
    result = re.sub(r"\s+", " ", result).strip()
    if len(result) <= limit:
        return result
    return result[: limit - 1].rstrip() + "…"


def unique_strings(values: Iterable[Any], limit: int | None = None) -> list[str]:
    output: list[str] = []
    seen: set[str] = set()
    for value in values:
        candidate = compact_text(value)
        if not candidate or candidate in seen:
            continue
        seen.add(candidate)
        output.append(candidate)
        if limit is not None and len(output) >= limit:
            break
    return output


def humanize(value: str | None) -> str:
    if not value:
        return "Not recorded"
    return re.sub(r"\s+", " ", value.replace("_", " ").replace("-", " ")).strip().title()


def model_items(path: Path) -> list[dict[str, Any]]:
    data = load_yaml(path)
    if isinstance(data, dict) and isinstance(data.get("models"), list):
        return [item for item in data["models"] if isinstance(item, dict)]
    if isinstance(data, list):
        return [item for item in data if isinstance(item, dict)]
    if isinstance(data, dict):
        return [data]
    return []


def manifest_index(root: Path, id_keys: tuple[str, ...]) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for path in sorted(root.rglob("*.yaml")):
        if path.name.startswith("_"):
            continue
        data = load_yaml(path)
        if not isinstance(data, dict):
            continue
        identifiers = [path.stem]
        identifiers.extend(text(data.get(key)) for key in id_keys)
        for identifier in identifiers:
            if identifier:
                output.setdefault(identifier.removeprefix("runtime-profile:"), data)
    return output


def get_status(value: Any) -> str | None:
    if isinstance(value, dict):
        return text(value.get("status"))
    return text(value)


def status_data(item: dict[str, Any], profile: dict[str, Any] | None) -> dict[str, str]:
    integration = (
        get_status(item.get("integration"))
        or text(item.get("integration_status"))
        or text(item.get("status"))
    )
    if not integration and profile:
        execution = profile.get("execution") if isinstance(profile.get("execution"), dict) else {}
        integration = (
            get_status(profile.get("integration"))
            or text(profile.get("integration_status"))
            or text(execution.get("integration_status"))
        )

    runner = get_status(item.get("runner_parity"))
    demo = get_status(item.get("demo_parity"))
    normalized = (integration or "profile_only").lower().replace("-", "_")
    runner_normalized = (runner or "").lower().replace("-", "_")

    if runner_normalized in {"verified", "validated", "passed"}:
        group = "verified"
        label = "Runner verified"
    elif "blocked" in normalized or normalized in {"unavailable", "missing"}:
        group = "blocked"
        label = humanize(integration)
    elif normalized in {"planned", "todo", "proposed"}:
        group = "planned"
        label = "Planned"
    elif normalized in {"integrated", "verified", "ready", "supported"}:
        group = "integrated"
        label = "Integrated"
    elif "ported" in normalized or "runtime" in normalized:
        group = "runtime_ported"
        label = humanize(integration)
    else:
        group = "profile"
        label = humanize(integration) if integration else "Profile only"

    return {
        "group": group,
        "label": label,
        "integration": integration or "not_recorded",
        "runner": runner or "not_recorded",
        "demo": demo or "not_recorded",
    }


def profile_candidates(item: dict[str, Any], model_id: str) -> list[str]:
    variants = [variant for variant in as_list(item.get("variants")) if isinstance(variant, dict)]
    candidates: list[Any] = [item.get("runtime_profile")]
    candidates.extend(variant.get("runtime_profile") for variant in variants)
    candidates.extend([model_id, *as_list(item.get("aliases"))])
    return unique_strings(
        candidate.removeprefix("runtime-profile:") if isinstance(candidate, str) else candidate
        for candidate in candidates
    )


def select_manifest(index: dict[str, dict[str, Any]], candidates: list[str]) -> tuple[str | None, dict[str, Any] | None]:
    for candidate in candidates:
        if candidate in index:
            return candidate, index[candidate]
    return None, None


def github_owner(url: str | None) -> str | None:
    if not url:
        return None
    match = re.search(r"github\.com/([^/]+)", url)
    return match.group(1) if match else None


def github_readme_url(url: str | None) -> str | None:
    """Return the stable README anchor for a GitHub repository URL.

    Catalog source links occasionally point at a subdirectory or a revision.
    Model pages should still give readers one predictable starting point for
    the upstream usage instructions, so reduce those URLs to their repository
    root before adding the README anchor.
    """

    if not url:
        return None
    match = re.match(r"https?://(?:www\.)?github\.com/([^/]+)/([^/?#]+)", url)
    if not match:
        return None
    owner, repository = match.groups()
    repository = repository.removesuffix(".git")
    return f"https://github.com/{owner}/{repository}#readme"


def source_links(item: dict[str, Any], profile: dict[str, Any] | None) -> list[dict[str, str]]:
    links: list[dict[str, str]] = []
    seen: set[str] = set()

    def add(kind: str, label: str, url: Any, revision: Any = None) -> None:
        normalized = text(url)
        if not normalized or not normalized.startswith(("https://", "http://")) or normalized in seen:
            return
        seen.add(normalized)
        entry = {"kind": kind, "label": label, "url": normalized}
        normalized_revision = text(revision)
        if normalized_revision:
            entry["revision"] = normalized_revision
        links.append(entry)

    official = item.get("official_sources") if isinstance(item.get("official_sources"), dict) else {}
    for key, label, kind in [
        ("project_page", "Project", "project"),
        ("paper", "Paper", "paper"),
        ("docs", "Documentation", "docs"),
    ]:
        value = official.get(key)
        if isinstance(value, dict):
            add(kind, label, value.get("url"), value.get("revision"))
        else:
            add(kind, label, value)

    github = official.get("github")
    for record in as_list(github):
        if isinstance(record, dict):
            add("source", "GitHub", record.get("url"), record.get("revision") or record.get("sha"))
        else:
            add("source", "GitHub", record)

    source = item.get("source") if isinstance(item.get("source"), dict) else {}
    for key in ("official_repo_url", "repo_url", "github", "url"):
        value = source.get(key)
        if isinstance(value, dict):
            add("source", "GitHub", value.get("url"), value.get("revision") or value.get("sha"))
        else:
            add("source", "GitHub", value, source.get("revision"))
    add("weights", "Hugging Face", f"https://huggingface.co/{source['hf_repo_id']}" if source.get("hf_repo_id") else None)

    source_status = item.get("source_status") if isinstance(item.get("source_status"), dict) else {}
    github_status = source_status.get("github") if isinstance(source_status.get("github"), dict) else {}
    add("source", "GitHub", github_status.get("url"), github_status.get("head_sha"))

    for record in as_list(official.get("huggingface")):
        if isinstance(record, dict):
            repo_id = record.get("repo_id") or record.get("id")
            add("weights", "Hugging Face", f"https://huggingface.co/{repo_id}" if repo_id else record.get("url"), record.get("revision") or record.get("sha"))
        elif isinstance(record, str):
            add("weights", "Hugging Face", record if record.startswith("http") else f"https://huggingface.co/{record}")

    if profile:
        for record in as_list(profile.get("source_repos")):
            if isinstance(record, dict):
                add("source", "Source", record.get("url"), record.get("revision") or record.get("sha"))

    # Not every catalog manifest has a separately curated documentation URL,
    # but an upstream GitHub repository is still a first-class official usage
    # source. Surface its README explicitly instead of making readers infer
    # that the generic source link contains installation and inference details.
    for source_link in tuple(links):
        if source_link["kind"] != "source":
            continue
        readme_url = github_readme_url(source_link["url"])
        if readme_url:
            add("docs", "Upstream README", readme_url, source_link.get("revision"))

    return links


def checkpoint_data(item: dict[str, Any], profile: dict[str, Any] | None) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()

    def add(record: Any) -> None:
        if not isinstance(record, dict):
            return
        repo_id = text(record.get("repo_id") or record.get("id") or record.get("repo"))
        if not repo_id:
            return
        revision = text(record.get("revision") or record.get("sha")) or ""
        key = (repo_id, revision)
        if key in seen:
            return
        seen.add(key)
        entry: dict[str, Any] = {"id": repo_id}
        for source_key, target_key in [
            ("revision", "revision"),
            ("sha", "revision"),
            ("license", "license"),
            ("role", "role"),
            ("status", "status"),
        ]:
            value = text(record.get(source_key))
            if value and target_key not in entry:
                entry[target_key] = value
        for key_name in ("gated", "private"):
            if isinstance(record.get(key_name), bool):
                entry[key_name] = record[key_name]
        notes = unique_strings(as_list(record.get("notes")), limit=3)
        if notes:
            entry["notes"] = notes
        output.append(entry)

    for record in as_list(item.get("checkpoints")) + as_list(item.get("checkpoint_refs")):
        add(record)
    checkpoint = item.get("checkpoint") if isinstance(item.get("checkpoint"), dict) else {}
    for record in as_list(checkpoint.get("repos")):
        add(record)
    for variant in as_list(item.get("variants")):
        if isinstance(variant, dict):
            for record in as_list(variant.get("checkpoint_refs")):
                add(record)
    if profile:
        for record in as_list(profile.get("checkpoints")):
            add(record)
    source = item.get("source") if isinstance(item.get("source"), dict) else {}
    if source.get("hf_repo_id"):
        add({"repo_id": source.get("hf_repo_id"), "license": source.get("license")})
    return output


def provider_name(item: dict[str, Any], links: list[dict[str, str]], checkpoints: list[dict[str, Any]]) -> str:
    for key in ("developer", "organization", "family"):
        candidate = text(item.get(key))
        if candidate:
            return candidate
    raw_provider = text(item.get("provider"))
    if raw_provider and raw_provider not in {"official_repo", "hosted_api", "pipeline", "local"}:
        return humanize(raw_provider)
    for checkpoint in checkpoints:
        if "/" in checkpoint["id"]:
            return checkpoint["id"].split("/", 1)[0]
    for link in links:
        owner = github_owner(link["url"])
        if owner:
            return owner
    if raw_provider == "hosted_api":
        return "Hosted provider"
    return "Upstream project"


def task_data(item: dict[str, Any], profile: dict[str, Any] | None) -> list[str]:
    values = as_list(item.get("tasks"))
    values.extend(as_list(item.get("task")))
    values.extend(as_list(item.get("capabilities")))
    if not values and profile:
        values.extend(as_list(profile.get("groups")))
        values.extend(as_list(profile.get("task_family")))
    return unique_strings(values)


def variant_data(
    item: dict[str, Any],
    default_status: dict[str, str],
    profiles: dict[str, dict[str, Any]],
    environments: dict[str, dict[str, Any]],
    bindings: dict[str, dict[str, Any]],
    unified_environment: dict[str, Any],
) -> list[dict[str, Any]]:
    """Build variant records with the runtime route that actually serves each variant."""
    output: list[dict[str, Any]] = []
    for variant in as_list(item.get("variants")):
        if not isinstance(variant, dict):
            continue
        variant_id = text(variant.get("id") or variant.get("model_id"))
        if not variant_id:
            continue
        profile_candidates = unique_strings(
            (
                candidate.removeprefix("runtime-profile:") if isinstance(candidate, str) else candidate
                for candidate in (
                    variant.get("runtime_profile"),
                    variant_id,
                    item.get("runtime_profile"),
                    text(item.get("id") or item.get("model_id")),
                    *as_list(item.get("aliases")),
                )
            )
        )
        profile_id, profile = select_manifest(profiles, profile_candidates)
        profile_execution = profile.get("execution") if profile and isinstance(profile.get("execution"), dict) else {}
        profile_integration = get_status(profile.get("integration")) if profile else None
        integration = get_status(variant.get("integration")) or profile_integration
        integration = integration or text(profile_execution.get("integration_status")) or default_status["integration"]
        binding_candidates = unique_strings(
            (
                variant.get("pipeline_binding"),
                profile_execution.get("pipeline_binding"),
                variant_id,
            )
        )
        binding_id, binding = select_manifest(bindings, binding_candidates)
        environment_candidates = unique_strings(
            (
                variant.get("runtime_profile"),
                variant_id,
                profile_id,
                variant.get("pipeline_binding"),
                binding_id,
                item.get("runtime_profile"),
                text(item.get("id") or item.get("model_id")),
            )
        )
        environment_candidates = [candidate.removeprefix("runtime-profile:") for candidate in environment_candidates]
        environment_id, environment = select_manifest(environments, environment_candidates)
        if environment is None and profile is not None:
            environment_id, environment = "_unified", unified_environment
        variant_runtime = runtime_data(
            item,
            profile_id,
            profile,
            environment_id,
            environment,
            binding_id,
            binding,
        )
        pipeline = binding.get("pipeline") if binding and isinstance(binding.get("pipeline"), dict) else {}
        loading = pipeline.get("loading") if isinstance(pipeline.get("loading"), dict) else {}
        invocation = pipeline.get("invocation") if isinstance(pipeline.get("invocation"), dict) else {}
        variant_item = {**item, **variant}
        entry = {
            "id": variant_id,
            "label": text(variant.get("name") or variant.get("display_name")) or variant_id,
            "task": text(variant.get("task")) or "",
            "runtimeProfile": variant_runtime["profileId"] or "",
            "pipelineBinding": variant_runtime["bindingId"] or "",
            "status": integration,
            "pipelineTarget": variant_runtime["pipelineTarget"],
            "runner": variant_runtime["runner"] or variant_runtime["runnerTarget"],
            "loadingMethod": text(loading.get("method")),
            "invocationMode": text(invocation.get("mode")),
            "environmentName": variant_runtime["environmentName"],
            "environmentKind": variant_runtime["environmentKind"],
            "python": variant_runtime["python"],
            "cudaLabel": variant_runtime["cudaLabel"],
            "backendStage": variant_runtime["backendStage"],
            "runtimeStatus": variant_runtime["runtimeStatus"],
            "inputContract": input_contract(profile),
            "artifacts": artifact_data(variant_item, profile),
        }
        output.append(entry)
    return output


def cuda_label(value: str | None) -> str | None:
    if not value:
        return None
    normalized = value.lower()
    match = re.fullmatch(r"cu(\d{2,3})", normalized)
    if match:
        digits = match.group(1)
        return f"CUDA {digits[:-1]}.{digits[-1]}"
    if normalized.startswith("cuda"):
        return value.upper().replace("CUDA", "CUDA ").replace("  ", " ").strip()
    return humanize(value)


def package_versions(packages: list[str]) -> dict[str, str]:
    output: dict[str, str] = {}
    for package in packages:
        normalized = package.strip()
        name_match = re.match(r"([A-Za-z0-9_.-]+)", normalized)
        if not name_match:
            continue
        name = name_match.group(1).lower().replace("_", "-")
        if name in {"torch", "torchvision", "torchaudio", "diffusers", "transformers", "xfuser", "accelerate", "flash-attn"}:
            output.setdefault(name, normalized)
    return output


def runtime_data(
    item: dict[str, Any],
    profile_id: str | None,
    profile: dict[str, Any] | None,
    env_id: str | None,
    environment: dict[str, Any] | None,
    binding_id: str | None,
    binding: dict[str, Any] | None,
) -> dict[str, Any]:
    execution = profile.get("execution") if profile and isinstance(profile.get("execution"), dict) else {}
    pipeline = binding.get("pipeline") if binding and isinstance(binding.get("pipeline"), dict) else {}
    loading = pipeline.get("loading") if isinstance(pipeline.get("loading"), dict) else {}
    invocation = pipeline.get("invocation") if isinstance(pipeline.get("invocation"), dict) else {}
    pip_packages = unique_strings(as_list(environment.get("pip_packages")) if environment else [])
    conda_packages = unique_strings(as_list(environment.get("conda_packages")) if environment else [])
    env_name = text(environment.get("env_name")) if environment else None
    env_kind = "unrecorded"
    if env_name:
        env_kind = "unified" if "unified" in env_name.lower() else "dedicated"

    return {
        "profileId": profile_id,
        "bindingId": binding_id or text(item.get("pipeline_binding")),
        "runnerTarget": text(item.get("runner_target")),
        "runner": text(binding.get("runner")) if binding else None,
        "pipelineTarget": text(pipeline.get("target") or execution.get("pipeline_target")),
        "loadingMethod": text(loading.get("method")),
        "invocationMode": text(invocation.get("mode")),
        "backendStage": text(execution.get("backend_stage") or (profile.get("backend_stage") if profile else None)),
        "runtimeStatus": text(execution.get("runtime_status") or (profile.get("runtime_status") if profile else None)),
        "environmentId": env_id,
        "environmentName": env_name,
        "environmentKind": env_kind,
        "python": text(environment.get("python")) if environment else None,
        "cudaProfile": text(environment.get("cuda_profile")) if environment else None,
        "cudaLabel": cuda_label(text(environment.get("cuda_profile"))) if environment else None,
        "driverStatus": text(environment.get("driver_status")) if environment else None,
        "condaPackages": conda_packages,
        "pipPackages": pip_packages,
        "packageVersions": package_versions(pip_packages),
        "validationImports": unique_strings(as_list(environment.get("validation_imports")) if environment else []),
        "notes": unique_strings(
            [
                *as_list(execution.get("notes")),
                *as_list(profile.get("notes") if profile else None),
                *as_list(environment.get("notes") if environment else None),
            ],
            limit=12,
        ),
    }


def input_contract(profile: dict[str, Any] | None) -> list[dict[str, str]]:
    schema = profile.get("input_schema") if profile else None
    if not isinstance(schema, dict):
        return []
    output: list[dict[str, str]] = []
    for key, value in schema.items():
        if isinstance(value, bool):
            detail = "Required" if value else "Optional"
        elif isinstance(value, list):
            detail = ", ".join(str(item) for item in value)
        elif isinstance(value, dict):
            detail = json.dumps(value, ensure_ascii=False, sort_keys=True)
        else:
            detail = text(value) or "Recorded"
        output.append({"field": str(key), "detail": detail})
    return output


def artifact_data(item: dict[str, Any], profile: dict[str, Any] | None) -> list[dict[str, str]]:
    output: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()

    def add(kind: Any, filename: Any = None) -> None:
        normalized_kind = text(kind)
        normalized_filename = text(filename) or ""
        if not normalized_kind or (normalized_kind, normalized_filename) in seen:
            return
        seen.add((normalized_kind, normalized_filename))
        output.append({"kind": normalized_kind, "filename": normalized_filename})

    for artifact in as_list(item.get("output_artifacts")):
        if isinstance(artifact, dict):
            add(artifact.get("kind") or artifact.get("type"), artifact.get("path") or artifact.get("filename"))
        else:
            add(artifact)
    if profile:
        add(profile.get("artifact_kind"), profile.get("artifact_filename"))
    for area in (item.get("demo_parity"), item.get("runner_parity")):
        if isinstance(area, dict):
            for artifact in as_list(area.get("expected_artifacts")):
                if isinstance(artifact, dict):
                    add("expected artifact", artifact.get("path"))
    return output


_WORKSPACE_ROOT = str(ROOT.parent)
# Machine-local absolute paths that are NOT inside the repo — a sibling
# ``…/arena/ckpt/diamond`` or a ``$HOME/.cache/worldfoundry/…`` cache path — get
# their internal prefix collapsed to a portable ``<local>`` token. Covers both
# ``/mnt/…`` mounts and ``/home/…`` home dirs (each leaks the internal username).
_LOCAL_PATH_RE = re.compile(r"(?:/[^\s\"']*)?/(?:mnt|home)/[^\s\"']*")
# Tails after these anchors are portable and worth keeping after ``<local>``.
_PORTABLE_ANCHORS = (".cache/worldfoundry", str(ROOT.parent.name) + "/")


def sanitize_paths(value: Any) -> Any:
    """Strip machine-local absolute paths from published docs data.

    Inference-spec field defaults can carry absolute paths to example inputs
    (condition images, pose files, local checkpoints, cache dirs). Those must
    never leak into the public docs, so: paths inside the repo become
    repo-relative, and any other machine-local (``/mnt/…`` or ``/home/…``) path
    has its internal prefix replaced with a ``<local>`` token, keeping a portable
    tail when one is recognizable. The structure is walked recursively.
    """

    root_str = str(ROOT)
    if isinstance(value, str):
        result = value
        if root_str in result:
            result = result.replace(root_str + "/", "").replace(root_str, "")

        def _scrub(match: re.Match) -> str:
            path = match.group(0)
            for anchor in _PORTABLE_ANCHORS:
                if anchor in path:
                    tail = path.split(anchor, 1)[1].lstrip("/")
                    return f"<local>/{tail}" if tail else "<local>"
            return "<local>"

        return _LOCAL_PATH_RE.sub(_scrub, result)
    if isinstance(value, dict):
        return {k: sanitize_paths(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(sanitize_paths(v) for v in value)
    return value


def task_field_detail(field: Any) -> str:
    required = "Required" if getattr(field, "required", False) else "Optional"
    pieces = [required]
    default = sanitize_paths(getattr(field, "default", None))
    if default is not None:
        if isinstance(default, (dict, list, tuple)):
            rendered = json.dumps(default, ensure_ascii=False, sort_keys=True)
        else:
            rendered = str(default)
        pieces.append(f"default={rendered}")
    choices = list(getattr(field, "choices", ()) or ())
    if choices:
        pieces.append(f"choices={', '.join(str(choice) for choice in choices)}")
    return "; ".join(pieces)


def core_inference_task_data(spec: Any, variant_ids: list[str]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for task in getattr(spec, "tasks", ()):
        inputs = []
        for field in task.inputs:
            inputs.append(
                {
                    "field": field.field_id,
                    "detail": task_field_detail(field),
                    "kind": field.kind,
                    "target": field.target,
                    "required": bool(field.required),
                    "default": sanitize_paths(field.default),
                    "choices": list(field.choices),
                    "description": field.description,
                }
            )
        artifacts = [
            {
                "kind": artifact.kind,
                "filename": artifact.artifact_id,
                "description": artifact.description,
            }
            for artifact in task.outputs
        ]
        output.append(
            {
                "id": task.task_id,
                "label": task.label,
                "description": task.description,
                "source": "inference_spec",
                "variantIds": list(variant_ids),
                "inputs": inputs,
                "artifacts": artifacts,
            }
        )
    return output


def catalog_inference_task_data(
    item: dict[str, Any],
    profile: dict[str, Any] | None,
    variant_records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Expose task profiles for models without a curated Python inference spec.

    Catalog ``tasks`` are capabilities, while a variant's ``task`` is the
    closest executable profile when variants exist. Keep the raw catalog
    values for fallback model-run schemas, but bind them to the variants that
    declare them so the docs cannot pair an arbitrary task with a variant.
    """
    declared_tasks = (
        unique_strings(record.get("task") for record in variant_records if record.get("task"))
        if variant_records
        else task_data(item, profile)
    )
    if not declared_tasks:
        declared_tasks = ["default"]

    output: list[dict[str, Any]] = []
    for task_id in declared_tasks:
        matching = [record for record in variant_records if record.get("task") == task_id]
        source_record = matching[0] if matching else (variant_records[0] if variant_records else None)
        fields = source_record.get("inputContract", []) if source_record else input_contract(profile)
        inputs = [
            {
                "field": field.get("field", ""),
                "detail": field.get("detail", "Recorded"),
                "kind": "string",
                "target": "input",
                "required": field.get("detail") == "Required",
                "description": "Recorded input field from the runtime profile.",
            }
            for field in fields
            if field.get("field")
        ]
        artifacts = source_record.get("artifacts", []) if source_record else artifact_data(item, profile)
        output.append(
            {
                "id": task_id,
                "label": humanize(task_id),
                "description": "Catalog task profile used by the model runtime.",
                "source": "catalog",
                "variantIds": [record.get("id") for record in matching if record.get("id")],
                "inputs": inputs,
                "artifacts": [
                    {
                        "kind": artifact.get("kind", ""),
                        "filename": artifact.get("filename", ""),
                    }
                    for artifact in artifacts
                    if artifact.get("kind") or artifact.get("filename")
                ],
            }
        )
    return output


def inference_task_data(
    item: dict[str, Any],
    profile: dict[str, Any] | None,
    variant_records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    model_id = text(item.get("id") or item.get("model_id")) or ""
    spec = get_model_inference_spec(model_id)
    if spec is None:
        alias_specs = {
            candidate.model_family_id: candidate
            for alias in as_list(item.get("aliases"))
            if alias
            for candidate in [get_model_inference_spec(str(alias))]
            if candidate is not None
        }
        if len(alias_specs) == 1:
            spec = next(iter(alias_specs.values()))
    if spec is not None:
        return core_inference_task_data(spec, [record["id"] for record in variant_records])
    return catalog_inference_task_data(item, profile, variant_records)


def recipe_notes(item: dict[str, Any], profile: dict[str, Any] | None) -> list[str]:
    integration = item.get("integration") if isinstance(item.get("integration"), dict) else {}
    runner = item.get("runner_parity") if isinstance(item.get("runner_parity"), dict) else {}
    demo = item.get("demo_parity") if isinstance(item.get("demo_parity"), dict) else {}
    checkpoint = item.get("checkpoint") if isinstance(item.get("checkpoint"), dict) else {}
    return unique_strings(
        [
            *as_list(item.get("notes")),
            *as_list(integration.get("notes")),
            *as_list(runner.get("notes")),
            *as_list(demo.get("notes")),
            *as_list(checkpoint.get("notes")),
            *as_list(profile.get("notes") if profile else None),
        ],
        limit=18,
    )


def load_editorial() -> dict[str, dict[str, Any]]:
    """Read the hand-authored editorial layer (lib/model-editorial/<id>.json).

    The generator only reads and merges these files; it never rewrites them, so
    re-running the generator preserves the prose.
    """
    output: dict[str, dict[str, Any]] = {}
    if not EDITORIAL_ROOT.is_dir():
        return output
    for path in sorted(EDITORIAL_ROOT.glob("*.json")):
        try:
            data = json.loads(path.read_text())
        except json.JSONDecodeError as error:
            print(f"skipping malformed editorial file {path.name}: {error}", file=sys.stderr)
            continue
        if not isinstance(data, dict):
            continue
        model_id = text(data.get("id")) or path.stem
        entry = {key: value for key, value in data.items() if key != "id"}
        if entry:
            output[model_id] = entry
    return output


def _video_probe_specs(probe: dict[str, Any]) -> list[dict[str, str]]:
    specs: list[dict[str, str]] = []
    width = probe.get("expected_width")
    height = probe.get("expected_height")
    if width and height:
        specs.append({"label": "Resolution", "detail": f"{width}×{height}"})
    frames = probe.get("expected_frames")
    if frames:
        specs.append({"label": "Frames", "detail": str(frames)})
    fps = probe.get("expected_fps")
    if fps:
        specs.append({"label": "FPS", "detail": str(fps)})
    duration = probe.get("expected_duration_seconds")
    if duration:
        specs.append({"label": "Duration", "detail": f"{duration}s"})
    return specs


def output_specs_for(item: dict[str, Any]) -> list[dict[str, str]]:
    """Precise output specs recorded by demo/runner parity video probes."""
    seen: set[tuple[str, str]] = set()
    specs: list[dict[str, str]] = []

    def collect(area: Any) -> None:
        if not isinstance(area, dict):
            return
        for artifact in as_list(area.get("expected_artifacts")):
            if not isinstance(artifact, dict):
                continue
            probe = artifact.get("video_probe")
            if isinstance(probe, dict):
                for spec in _video_probe_specs(probe):
                    key = (spec["label"], spec["detail"])
                    if key in seen:
                        continue
                    seen.add(key)
                    specs.append(spec)

    collect(item.get("demo_parity"))
    collect(item.get("runner_parity"))
    for variant in as_list(item.get("variants")):
        if isinstance(variant, dict):
            collect(variant.get("demo_parity"))
            collect(variant.get("runner_parity"))
    return specs


def facts_for(item: dict[str, Any]) -> dict[str, Any]:
    """Surface manifest facts the older pipeline left buried in raw YAML.

    Every field is optional: missing stays missing rather than being inferred
    into a stronger claim.
    """
    native = item.get("native_diffusion")
    if not isinstance(native, dict):
        # A family manifest may carry native_diffusion on its first variant.
        for variant in as_list(item.get("variants")):
            if isinstance(variant, dict) and isinstance(variant.get("native_diffusion"), dict):
                native = variant["native_diffusion"]
                break
    native = native if isinstance(native, dict) else {}

    components = unique_strings(as_list(native.get("components")))
    key_validation: dict[str, int] = {}
    raw_keys = native.get("checkpoint_key_validation")
    if isinstance(raw_keys, dict):
        for key, value in raw_keys.items():
            if isinstance(value, int):
                key_validation[str(key)] = value

    min_vram = item.get("min_vram_gb")
    if not isinstance(min_vram, (int, float)):
        min_vram = None

    return {
        "minVramGb": min_vram,
        "family": text(item.get("family")),
        "developer": text(item.get("developer")),
        "releaseType": text(item.get("release_type")),
        "modalities": unique_strings(as_list(item.get("modalities"))),
        "architectureComponents": components,
        "checkpointKeyValidation": key_validation,
        "outputSpecs": output_specs_for(item),
    }


def summary_for(item: dict[str, Any], tasks: list[str], notes: list[str]) -> str:
    for candidate in [item.get("description"), item.get("summary")]:
        normalized = compact_text(candidate, 240)
        if normalized:
            return normalized
    if notes:
        return compact_text(notes[0], 240) or notes[0]
    if tasks:
        labels = ", ".join(humanize(task).lower() for task in tasks[:3])
        return f"Manifested for {labels}. Open the recipe to inspect runtime and provenance records."
    return "Cataloged in WorldFoundry. Open the recipe to inspect the available runtime and provenance records."


def command_placeholder(field: dict[str, Any]) -> str | None:
    if not field.get("required") or field.get("default") not in (None, ""):
        return None
    field_id = str(field.get("field") or "")
    if field_id in {"prompt", "instruction", "text", "caption"}:
        return '"Describe the desired output."'
    if field_id in {"input_path", "image", "video", "audio", "images", "input"}:
        return "/path/to/input"
    if field.get("kind") in {"json", "interaction_tokens"}:
        return "'{}'"
    return "VALUE"


def direct_run_command(runtime_model_id: str, task: dict[str, Any]) -> str:
    lines = [
        "worldfoundry-eval run \\",
        f"  {shlex.quote(runtime_model_id)} \\",
        f"  --pipeline.task-profile {shlex.quote(str(task['id']))} \\",
    ]
    for field in task.get("inputs", []):
        placeholder = command_placeholder(field)
        if placeholder is None:
            continue
        option = str(field.get("field") or "").replace("_", "-")
        lines.append(f"  --pipeline.{option} {placeholder} \\")
    lines.append("  --json")
    return "\n".join(lines)


def command_data(model_id: str, runtime_model_id: str, default_task: dict[str, Any]) -> dict[str, str]:
    return {
        "prepare": f"bash scripts/inference/prepare_model_infer.sh {runtime_model_id}",
        "install": f"bash scripts/setup/model_env_install.sh --model {runtime_model_id}",
        "inspect": f"worldfoundry-eval zoo model-show --model-id {model_id} --include-manifest --json",
        "check": f"worldfoundry-eval zoo model-download --model-id {model_id} --check-local --json",
        "run": direct_run_command(runtime_model_id, default_task),
    }


def main() -> None:
    profiles = manifest_index(PROFILE_ROOT, ("model_id", "id"))
    environments = manifest_index(ENVIRONMENT_ROOT, ("model_id", "id"))
    bindings = manifest_index(BINDING_ROOT, ("binding_id", "model_id", "id"))
    unified_environment = load_yaml(ENVIRONMENT_ROOT / "_unified.yaml")

    recipes: list[dict[str, Any]] = []
    category_counts: Counter[str] = Counter()

    for category_id, category in CATEGORY_META.items():
        category_root = CATALOG_ROOT / category_id
        for path in sorted(category_root.glob("*.yaml")):
            if path.name.startswith("_"):
                continue
            for item in model_items(path):
                model_id = text(item.get("id") or item.get("model_id"))
                if not model_id:
                    continue
                name = text(item.get("name") or item.get("display_name")) or model_id
                candidates = profile_candidates(item, model_id)
                profile_id, profile = select_manifest(profiles, candidates)

                variants = [variant for variant in as_list(item.get("variants")) if isinstance(variant, dict)]
                env_candidates = unique_strings(
                    [
                        *(variant.get("runtime_profile") for variant in variants),
                        *(variant.get("id") for variant in variants),
                        *candidates,
                    ]
                )
                env_candidates = [candidate.removeprefix("runtime-profile:") for candidate in env_candidates]
                env_id, environment = select_manifest(environments, env_candidates)
                if environment is None and profile is not None:
                    env_id, environment = "_unified", unified_environment

                binding_candidates = unique_strings(
                    [
                        item.get("pipeline_binding"),
                        *(variant.get("pipeline_binding") for variant in variants),
                        model_id,
                    ]
                )
                binding_id, binding = select_manifest(bindings, binding_candidates)

                status = status_data(item, profile)
                tasks = task_data(item, profile)
                links = source_links(item, profile)
                checkpoints = checkpoint_data(item, profile)
                notes = recipe_notes(item, profile)
                variant_records = variant_data(item, status, profiles, environments, bindings, unified_environment)
                inference_tasks = inference_task_data(item, profile, variant_records)
                runtime_model_id = variant_records[0]["id"] if variant_records else model_id
                aliases = unique_strings(as_list(item.get("aliases")))

                recipe = {
                    "id": model_id,
                    "name": name,
                    "category": category_id,
                    "categoryLabel": category["label"],
                    "categoryLabelZh": category["label_zh"],
                    "provider": provider_name(item, links, checkpoints),
                    "summary": summary_for(item, tasks, notes),
                    "aliases": aliases,
                    "tasks": tasks,
                    "status": status,
                    "runtime": runtime_data(item, profile_id, profile, env_id, environment, binding_id, binding),
                    "sources": links,
                    "checkpoints": checkpoints,
                    "variants": variant_records,
                    "inferenceTasks": inference_tasks,
                    "inputContract": input_contract(profile),
                    "artifacts": artifact_data(item, profile),
                    "notes": notes,
                    "facts": facts_for(item),
                    "commands": command_data(model_id, runtime_model_id, inference_tasks[0]),
                    "catalogPath": str(path.relative_to(ROOT)),
                }
                if not docs_catalog_visible(category_id, model_id, status["group"]):
                    continue
                recipes.append(recipe)
                category_counts[category_id] += 1

    editorial = load_editorial()
    for recipe in recipes:
        entry = editorial.get(recipe["id"])
        if not entry:
            continue
        recipe["editorial"] = entry
        tagline = entry.get("tagline")
        if isinstance(tagline, dict):
            english = compact_text(tagline.get("en"), 240)
            if english:
                recipe["summary"] = english

    recipes.sort(
        key=lambda recipe: (
            STATUS_ORDER.get(recipe["status"]["group"], 99),
            recipe["name"].lower(),
            recipe["id"],
        )
    )
    payload = {
        "total": len(recipes),
        "categories": [
            {
                "id": category_id,
                **meta,
                "count": category_counts[category_id],
            }
            for category_id, meta in CATEGORY_META.items()
        ],
        "recipes": recipes,
    }
    # Final choke point: strip any machine-local absolute paths from the whole
    # payload regardless of which code path produced a field, so the published
    # docs data is portable and leaks no internal filesystem layout.
    payload = sanitize_paths(payload)
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    index_payload = {
        "total": payload["total"],
        "categories": payload["categories"],
        "recipes": [
            {
                "id": recipe["id"],
                "name": recipe["name"],
                "category": recipe["category"],
                "categoryLabel": recipe["categoryLabel"],
                "categoryLabelZh": recipe["categoryLabelZh"],
                "provider": recipe["provider"],
                "summary": recipe["summary"],
                "aliases": recipe["aliases"],
                "tasks": recipe["tasks"],
                "status": recipe["status"],
                "runtime": {
                    key: recipe["runtime"].get(key)
                    for key in (
                        "profileId",
                        "environmentName",
                        "environmentKind",
                        "python",
                        "cudaLabel",
                    )
                },
                "checkpoint": recipe["checkpoints"][0] if recipe["checkpoints"] else None,
            }
            for recipe in recipes
        ],
    }
    INDEX_OUT.write_text(json.dumps(index_payload, ensure_ascii=False, separators=(",", ":")) + "\n")
    print(f"wrote {OUT} and {INDEX_OUT} recipes={len(recipes)}")


if __name__ == "__main__":
    main()
