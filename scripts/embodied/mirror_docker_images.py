#!/usr/bin/env python3
"""Mirror embodied benchmark Docker images into the WorldFoundry release namespace."""

from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from worldfoundry.evaluation.tasks.embodied.image_refs import (  # noqa: E402
    assert_image_ref_pinned,
    repository_name,
    require_pinned_images,
    resolve_docker_image,
    resolve_docker_platform,
)

DEFAULT_PROFILE_DIR = Path("worldfoundry/data/benchmarks/runtime_profiles/official")


@dataclass(frozen=True)
class ImageMapping:
    profile_id: str
    source_image: str
    target_image: str
    platform: str | None = None


def _repo_root() -> Path:
    return REPO_ROOT


def _replace_tag(image: str, tag: str | None) -> str:
    if not tag:
        return image
    return f"{repository_name(image)}:{tag}"


def _target_from_prefix(source_image: str, target_prefix: str, *, tag: str | None) -> str:
    source_repository = repository_name(source_image)
    leaf = source_repository.rsplit("/", 1)[-1]
    unpinned_source = source_image.split("@", 1)[0]
    source_leaf = unpinned_source.rsplit("/", 1)[-1]
    source_tag = source_leaf.rsplit(":", 1)[-1] if ":" in source_leaf else "latest"
    return f"{target_prefix.rstrip('/')}/{leaf}:{tag or source_tag}"


def _load_mapping(path: Path, *, tag: str | None, target_prefix: str | None) -> ImageMapping | None:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        return None
    docker = payload.get("docker") or {}
    if not isinstance(docker, dict):
        return None
    source = docker.get("source_image")
    target = docker.get("image")
    if not source or not target:
        return None
    resolved_target = resolve_docker_image(docker, require_pinned=False)
    platform = resolve_docker_platform(docker)
    target_image = (
        _target_from_prefix(str(source), target_prefix, tag=tag)
        if target_prefix
        else _replace_tag(resolved_target, tag)
    )
    return ImageMapping(
        profile_id=str(payload.get("id") or path.stem),
        source_image=str(source),
        target_image=target_image,
        platform=platform,
    )


def load_image_mappings(
    profile_dir: Path,
    profiles: Iterable[str],
    *,
    tag: str | None = None,
    target_prefix: str | None = None,
) -> list[ImageMapping]:
    requested = {item.strip() for item in profiles if item.strip()}
    include_all = not requested or requested == {"all"}
    mappings: list[ImageMapping] = []
    for path in sorted(profile_dir.glob("*.yaml")):
        mapping = _load_mapping(path, tag=tag, target_prefix=target_prefix)
        if mapping is None:
            continue
        if include_all or mapping.profile_id in requested or path.stem in requested:
            mappings.append(mapping)
    missing = requested - {"all"} - {item.profile_id for item in mappings} - {Path(item.profile_id).stem for item in mappings}
    if missing:
        raise ValueError(f"no Docker source_image mapping found for profiles: {', '.join(sorted(missing))}")
    return mappings


def _run(cmd: list[str], *, plan_only: bool) -> None:
    print("+ " + " ".join(cmd))
    if plan_only:
        return
    subprocess.run(cmd, check=True)


def mirror_images(mappings: Iterable[ImageMapping], *, push: bool, plan_only: bool = False) -> None:
    require_pinned = require_pinned_images()
    for mapping in mappings:
        assert_image_ref_pinned(mapping.source_image, require=require_pinned, what="source_image")
        assert_image_ref_pinned(mapping.target_image, require=require_pinned, what="image")
        pull_cmd = ["docker", "pull"]
        if mapping.platform:
            pull_cmd.extend(["--platform", mapping.platform])
        _run([*pull_cmd, mapping.source_image], plan_only=plan_only)
        if mapping.source_image == mapping.target_image:
            continue
        _run(["docker", "tag", mapping.source_image, mapping.target_image], plan_only=plan_only)
        if push:
            _run(["docker", "push", mapping.target_image], plan_only=plan_only)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile",
        action="append",
        default=[],
        help="Runtime profile id to mirror. Repeatable. Defaults to all profiles with docker.source_image.",
    )
    parser.add_argument("--profile-dir", type=Path, default=_repo_root() / DEFAULT_PROFILE_DIR)
    parser.add_argument("--tag", help="Override the target image tag while preserving the configured repository.")
    parser.add_argument(
        "--target-prefix",
        help=(
            "Retag images under this registry/repository prefix, e.g. "
            "registry.cn-wulanchabu.aliyuncs.com/worldfoundry."
        ),
    )
    parser.add_argument("--push", action="store_true", help="Push the retagged WorldFoundry images.")
    parser.add_argument("--plan", action="store_true", help="Print docker commands without executing them.")
    args = parser.parse_args(argv)

    mappings = load_image_mappings(
        args.profile_dir,
        args.profile or ["all"],
        tag=args.tag,
        target_prefix=args.target_prefix,
    )
    if not mappings:
        print("No embodied Docker image mappings found.", file=sys.stderr)
        return 1
    mirror_images(mappings, push=bool(args.push), plan_only=bool(args.plan))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
