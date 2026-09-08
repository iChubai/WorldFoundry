#!/usr/bin/env python3
"""Resolve immutable GHCR digests for official embodied Docker profiles."""

from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import yaml

_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROFILE_DIR = _REPO_ROOT / "worldfoundry/data/benchmarks/runtime_profiles/official"
DEFAULT_DIGEST_MAP = DEFAULT_PROFILE_DIR / "docker_image_digests.json"
DEFAULT_PLATFORM = "linux/amd64"
_DIGEST_RE = re.compile(r"sha256:[0-9a-fA-F]{64}\Z")


def normalize_resolved_digest(digest: str) -> str:
    """Return a canonical registry digest or reject malformed registry data."""

    value = str(digest or "").strip()
    if not value:
        raise ValueError("digest must be non-empty")
    if not value.startswith("sha256:"):
        value = f"sha256:{value}"
    if _DIGEST_RE.fullmatch(value) is None:
        raise ValueError(f"invalid Docker digest {digest!r}; expected sha256:<64 hex>")
    return value.lower()


def repository_and_tag(image: str) -> tuple[str, str]:
    ref = str(image or "").strip()
    if not ref:
        raise ValueError("image must be non-empty")
    if not ref.startswith("ghcr.io/"):
        raise ValueError(f"only ghcr.io refs are supported: {image!r}")
    rest = ref[len("ghcr.io/") :]
    if "@" in rest:
        raise ValueError(f"expected a tag, not a digest reference: {image!r}")
    leaf = rest.rsplit("/", 1)[-1]
    if ":" in leaf:
        repository, tag = rest.rsplit(":", 1)
    else:
        repository, tag = rest, "latest"
    if not repository or not tag:
        raise ValueError(f"invalid GHCR image reference: {image!r}")
    return repository, tag


def fetch_ghcr_digest(image: str, *, timeout: float = 30.0) -> str:
    """Return a strictly validated ``sha256:...`` digest from GHCR."""

    repository, tag = repository_and_tag(image)
    scope = urllib.parse.quote(f"repository:{repository}:pull", safe="")
    token_url = f"https://ghcr.io/token?service=ghcr.io&scope={scope}"
    with urllib.request.urlopen(token_url, timeout=timeout) as token_response:
        token = json.load(token_response)["token"]
    request = urllib.request.Request(
        f"https://ghcr.io/v2/{repository}/manifests/{tag}",
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": (
                "application/vnd.docker.distribution.manifest.list.v2+json,"
                "application/vnd.oci.image.index.v1+json,"
                "application/vnd.docker.distribution.manifest.v2+json,"
                "application/vnd.oci.image.manifest.v1+json"
            ),
        },
        method="HEAD",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        digest = response.headers.get("Docker-Content-Digest")
    if digest is None:
        raise RuntimeError(f"no Docker-Content-Digest for {image}")
    return normalize_resolved_digest(digest)


def _validate_pinned_reference(image: str) -> None:
    if "@" not in image:
        return
    if image.count("@") != 1:
        raise ValueError(f"invalid Docker digest reference: {image!r}")
    repository, digest = image.split("@", 1)
    if not repository:
        raise ValueError(f"invalid Docker digest reference: {image!r}")
    normalize_resolved_digest(digest)


def collect_floating_images(profile_dir: Path) -> dict[str, list[str]]:
    """Map every floating image reference to the profiles that use it."""

    images: dict[str, list[str]] = {}
    for path in sorted(profile_dir.glob("*.yaml")):
        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        docker = payload.get("docker") if isinstance(payload, dict) else None
        if not isinstance(docker, dict):
            continue
        configured_digest = str(docker.get("digest") or docker.get("image_digest") or "").strip()
        if configured_digest:
            normalize_resolved_digest(configured_digest)
        for key in ("image", "source_image"):
            image = str(docker.get(key) or "").strip()
            if not image:
                continue
            _validate_pinned_reference(image)
            if "@" in image or (key == "image" and configured_digest):
                continue
            leaf = image.rsplit("/", 1)[-1]
            if ":" not in leaf or leaf.rsplit(":", 1)[-1] == "latest":
                images.setdefault(image, []).append(path.stem)
    return images


def resolve_images(
    images: dict[str, list[str]],
    *,
    fetch_digest: Callable[[str], str] = fetch_ghcr_digest,
) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    resolved: dict[str, dict[str, Any]] = {}
    errors: dict[str, str] = {}
    for image, profiles in sorted(images.items()):
        try:
            digest = normalize_resolved_digest(fetch_digest(image))
            resolved[image] = {
                "digest": digest,
                "platform": DEFAULT_PLATFORM,
                "profiles": sorted(set(profiles)),
            }
        except Exception as exc:  # noqa: BLE001 - report auth/network/validation failures per image
            errors[image] = str(exc)
    return resolved, errors


def _upsert_docker_field(text: str, field: str, value: str) -> str:
    pattern = rf"^(\s*{re.escape(field)}:\s*)\S+\s*$"
    if re.search(pattern, text, flags=re.MULTILINE):
        return re.sub(pattern, rf"\g<1>{value}", text, count=1, flags=re.MULTILINE)
    lines = text.splitlines(keepends=True)
    output: list[str] = []
    inserted = False
    for line in lines:
        output.append(line)
        if not inserted and re.match(r"^\s*image:\s*\S+", line):
            indent = re.match(r"^(\s*)", line).group(1)
            output.append(f"{indent}{field}: {value}\n")
            inserted = True
    if not inserted:
        raise ValueError("docker.image line not found while applying digest")
    return "".join(output)


def _repository_name(image: str) -> str:
    ref = image.split("@", 1)[0]
    parent, slash, leaf = ref.rpartition("/")
    if ":" in leaf:
        leaf = leaf.rsplit(":", 1)[0]
    return f"{parent}{slash}{leaf}"


def apply_pins_to_profile(path: Path, *, image_digests: dict[str, str]) -> bool:
    """Apply resolved digests and the verified platform to one profile."""

    original = path.read_text(encoding="utf-8")
    payload = yaml.safe_load(original) or {}
    docker = payload.get("docker") if isinstance(payload, dict) else None
    if not isinstance(docker, dict):
        return False
    image = str(docker.get("image") or "").strip()
    source = str(docker.get("source_image") or "").strip()
    text = original
    changed = False
    if image in image_digests:
        digest = normalize_resolved_digest(image_digests[image])
        text = _upsert_docker_field(text, "digest", digest)
        text = _upsert_docker_field(text, "platform", DEFAULT_PLATFORM)
        changed = True
    if source in image_digests:
        digest = normalize_resolved_digest(image_digests[source])
        pinned_source = f"{_repository_name(source)}@{digest}"
        text, count = re.subn(
            r"^(\s*source_image:\s*)\S+\s*$",
            rf"\g<1>{pinned_source}",
            text,
            count=1,
            flags=re.MULTILINE,
        )
        if count:
            text = _upsert_docker_field(text, "platform", DEFAULT_PLATFORM)
            changed = True
    if changed and text != original:
        path.write_text(text, encoding="utf-8")
        return True
    return False


def write_digest_map(path: Path, resolved: dict[str, dict[str, Any]]) -> None:
    normalized: dict[str, dict[str, Any]] = {}
    for image, metadata in resolved.items():
        entry = dict(metadata)
        entry["digest"] = normalize_resolved_digest(str(entry.get("digest") or ""))
        entry["platform"] = str(entry.get("platform") or DEFAULT_PLATFORM)
        normalized[image] = entry
    payload: dict[str, Any] = {
        "schema_version": 1,
        "note": "Source digests resolved from GHCR Docker-Content-Digest. Do not invent digests.",
        "resolved_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "images": normalized,
    }
    if path.is_file():
        existing = json.loads(path.read_text(encoding="utf-8"))
        merged = dict(existing.get("images") or {})
        merged.update(normalized)
        payload["images"] = merged
        if existing.get("note"):
            payload["note"] = existing["note"]
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile-dir", type=Path, default=DEFAULT_PROFILE_DIR)
    parser.add_argument("--digest-map", type=Path, default=DEFAULT_DIGEST_MAP)
    parser.add_argument("--write", action="store_true", help="Update YAML profiles and the digest map")
    parser.add_argument("--json", action="store_true", help="Print a machine-readable resolution report")
    args = parser.parse_args(argv)

    floating = collect_floating_images(args.profile_dir)
    resolved, errors = resolve_images(floating)
    report = {
        "resolved": resolved,
        "errors": errors,
        "floating_count": len(floating),
        "resolved_count": len(resolved),
        "error_count": len(errors),
    }
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        for image, metadata in resolved.items():
            print(f"OK  {image} -> {metadata['digest']} [{metadata['platform']}]")
        for image, error in errors.items():
            print(f"ERR {image}: {error}", file=sys.stderr)
        print(
            f"summary: resolved={len(resolved)} errors={len(errors)} floating={len(floating)}",
            file=sys.stderr,
        )
    if args.write and resolved:
        image_digests = {image: str(metadata["digest"]) for image, metadata in resolved.items()}
        changed = sum(
            apply_pins_to_profile(path, image_digests=image_digests)
            for path in sorted(args.profile_dir.glob("*.yaml"))
        )
        write_digest_map(args.digest_map, resolved)
        print(f"wrote digest map {args.digest_map}; updated {changed} profiles", file=sys.stderr)
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
