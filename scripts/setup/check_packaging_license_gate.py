#!/usr/bin/env python3
"""Audit the Apache-2.0 package boundary against license-gated source trees.

``MANIFEST.in`` controls sdists, while setuptools package discovery and
``include-package-data`` control wheels.  This checker keeps all three paths
aligned and can additionally inspect the contents of a built wheel.
"""

from __future__ import annotations

import argparse
import fnmatch
import sys
import zipfile
from pathlib import Path
from typing import Iterable, Mapping, Sequence

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 only
    import tomli as tomllib  # type: ignore[no-redef]


REPO_ROOT = Path(__file__).resolve().parents[2]
PYPROJECT_PATH = REPO_ROOT / "pyproject.toml"
MANIFEST_PATH = REPO_ROOT / "MANIFEST.in"
LICENSE_GATE_MARKER = "License-gated upstream runtimes"

# These first-party integration packages sit above gated upstream code and
# must remain installable.  In particular, excluding hunyuan_world wholesale
# silently removed its checkpoint/config/wrapper modules from the wheel.
REQUIRED_WHEEL_PACKAGES = (
    "worldfoundry.synthesis.visual_generation.hunyuan_world",
)


def load_pyproject(path: Path = PYPROJECT_PATH) -> dict:
    with path.open("rb") as handle:
        return tomllib.load(handle)


def load_find_config(path: Path = PYPROJECT_PATH) -> dict:
    """Return the ``[tool.setuptools].packages.find`` table."""
    return load_pyproject(path)["tool"]["setuptools"]["packages"]["find"]


def _matches_any(value: str, patterns: Iterable[str]) -> bool:
    return any(fnmatch.fnmatchcase(value, pattern) for pattern in patterns)


def discover_packages(find_config: Mapping[str, object], *, repo_root: Path = REPO_ROOT) -> list[str]:
    """Discover the pre-exclude namespace package set used by setuptools.

    The project includes only ``worldfoundry*`` from the repository root.  On
    BeeGFS, asking setuptools to walk ``.`` also traverses unrelated docs and
    generated trees before applying that include filter.  The bounded branch
    below walks only ``worldfoundry/`` and then applies the identical dotted
    include semantics.
    """
    from setuptools import find_namespace_packages, find_packages

    where_values = find_config.get("where", ["."])
    if not isinstance(where_values, (list, tuple)) or len(where_values) != 1:
        raise ValueError("packages.find.where must contain exactly one search root")
    where = str(where_values[0])
    include = tuple(str(pattern) for pattern in find_config.get("include", ("*",)))
    namespaces = bool(find_config.get("namespaces", True))
    finder = find_namespace_packages if namespaces else find_packages
    search_root = repo_root / where

    if where == "." and include == ("worldfoundry*",) and (repo_root / "worldfoundry").is_dir():
        relative = finder(where=str(repo_root / "worldfoundry"))
        root_is_package = namespaces or (repo_root / "worldfoundry" / "__init__.py").is_file()
        candidates = [
            *(["worldfoundry"] if root_is_package else []),
            *(f"worldfoundry.{name}" for name in relative),
        ]
        return sorted(name for name in candidates if _matches_any(name, include))

    return sorted(finder(where=str(search_root), include=include))


def selected_packages(all_packages: Sequence[str], exclude: Sequence[str]) -> list[str]:
    """Apply setuptools' dotted exclude patterns to a discovered package set."""
    return sorted(name for name in all_packages if not _matches_any(name, exclude))


def dead_exclude_patterns(all_packages: Sequence[str], exclude: Sequence[str]) -> list[str]:
    """Return exclude patterns that match no discoverable package."""
    return [pattern for pattern in exclude if not _matches_any_package(all_packages, pattern)]


def _matches_any_package(packages: Sequence[str], pattern: str) -> bool:
    return any(fnmatch.fnmatchcase(name, pattern) for name in packages)


def license_gated_paths(manifest_path: Path = MANIFEST_PATH) -> list[str]:
    """Read worldfoundry prune paths from the manifest's license-gate block."""
    paths: list[str] = []
    in_block = False
    for line in manifest_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if LICENSE_GATE_MARKER in stripped:
            in_block = True
            continue
        if in_block and not stripped:
            break
        if in_block and stripped.startswith("prune worldfoundry/"):
            paths.append(stripped.removeprefix("prune ").rstrip("/"))
    if not paths:
        raise RuntimeError(
            f"no gated paths found below the '{LICENSE_GATE_MARKER}' marker in {manifest_path}"
        )
    return paths


def license_gated_prefixes(manifest_path: Path = MANIFEST_PATH) -> list[str]:
    return [path.replace("/", ".") for path in license_gated_paths(manifest_path)]


def leaked_packages(kept_packages: Sequence[str], prefixes: Sequence[str]) -> list[str]:
    """Return retained packages located at or below a gated prefix."""
    return sorted(
        name
        for name in kept_packages
        if any(name == prefix or name.startswith(prefix + ".") for prefix in prefixes)
    )


def _nearest_kept_parent(path: str, kept_packages: set[str]) -> tuple[str, str] | None:
    parts = path.split("/")
    for index in range(len(parts) - 1, 0, -1):
        parent = ".".join(parts[:index])
        if parent in kept_packages:
            return parent, "/".join(parts[index:])
    return None


def _pattern_covers_relative_path(pattern: str, relative_path: str) -> bool:
    literal_prefix = pattern.split("*", 1)[0].rstrip("/")
    return bool(literal_prefix) and (
        relative_path == literal_prefix or relative_path.startswith(literal_prefix + "/")
    )


def package_data_exclusion_gaps(
    gated_paths: Sequence[str],
    kept_packages: Sequence[str],
    exclude_package_data: Mapping[str, Sequence[str]],
) -> list[str]:
    """Return gated paths lacking a per-parent include-package-data barrier."""
    kept = set(kept_packages)
    gaps: list[str] = []
    for path in gated_paths:
        owner = _nearest_kept_parent(path, kept)
        if owner is None:
            gaps.append(f"{path} (no retained parent package)")
            continue
        parent, relative = owner
        patterns = exclude_package_data.get(parent, ())
        if not any(_pattern_covers_relative_path(pattern, relative) for pattern in patterns):
            gaps.append(f"{path} (parent {parent!r}, relative path {relative!r})")
    return gaps


def audit_wheel(wheel_path: Path, gated_paths: Sequence[str]) -> list[str]:
    """Return files inside a wheel located at or below a gated path."""
    prefixes = tuple(path.rstrip("/") + "/" for path in gated_paths)
    with zipfile.ZipFile(wheel_path) as wheel:
        return sorted(name for name in wheel.namelist() if name.startswith(prefixes))


def missing_required_wheel_packages(
    wheel_path: Path,
    required_packages: Sequence[str] = REQUIRED_WHEEL_PACKAGES,
) -> list[str]:
    """Return first-party wrapper packages absent from a built wheel."""
    with zipfile.ZipFile(wheel_path) as wheel:
        entries = set(wheel.namelist())
    return sorted(
        package
        for package in required_packages
        if f"{package.replace('.', '/')}/__init__.py" not in entries
    )


def _print_items(header: str, items: Sequence[str], *, limit: int | None = None) -> None:
    print(header)
    visible = items if limit is None else items[:limit]
    for item in visible:
        print(f"  {item}")
    if limit is not None and len(items) > limit:
        print(f"  ... and {len(items) - limit} more")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--wheel", type=Path, help="also audit one freshly built .whl artifact")
    args = parser.parse_args(argv)

    pyproject = load_pyproject()
    find_config = pyproject["tool"]["setuptools"]["packages"]["find"]
    exclude = [str(pattern) for pattern in find_config.get("exclude", ())]
    all_packages = discover_packages(find_config)
    kept_packages = selected_packages(all_packages, exclude)
    gated_paths = license_gated_paths()
    gated_prefixes = [path.replace("/", ".") for path in gated_paths]
    exclude_package_data = pyproject["tool"]["setuptools"].get("exclude-package-data", {})

    failures = 0
    dead = dead_exclude_patterns(all_packages, exclude)
    if dead:
        failures += 1
        _print_items(f"FAIL: {len(dead)} dead package exclude pattern(s):", dead)

    leaks = leaked_packages(kept_packages, gated_prefixes)
    if leaks:
        failures += 1
        _print_items(f"FAIL: {len(leaks)} gated package(s) remain in the wheel set:", leaks)

    gaps = package_data_exclusion_gaps(gated_paths, kept_packages, exclude_package_data)
    if gaps:
        failures += 1
        _print_items(f"FAIL: {len(gaps)} gated path(s) lack per-parent data excludes:", gaps)

    missing = sorted(set(REQUIRED_WHEEL_PACKAGES) - set(kept_packages))
    if missing:
        failures += 1
        _print_items("FAIL: required first-party wrapper package(s) excluded:", missing)

    if args.wheel is not None:
        if not args.wheel.is_file() or args.wheel.suffix != ".whl":
            parser.error(f"--wheel must name an existing .whl file: {args.wheel}")
        wheel_leaks = audit_wheel(args.wheel, gated_paths)
        if wheel_leaks:
            failures += 1
            _print_items(
                f"FAIL: {len(wheel_leaks)} gated file(s) found in {args.wheel}:",
                wheel_leaks,
                limit=50,
            )
        missing_from_wheel = missing_required_wheel_packages(args.wheel)
        if missing_from_wheel:
            failures += 1
            _print_items("FAIL: first-party wrapper package(s) absent from wheel:", missing_from_wheel)

    if failures:
        return 1

    artifact = f", wheel {args.wheel.name} clean" if args.wheel is not None else ""
    print(
        "packaging license gate OK: "
        f"{len(all_packages)} namespace packages discovered, "
        f"{len(kept_packages)} retained, {len(exclude)} live excludes, "
        f"{len(gated_paths)} gated paths protected{artifact}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
