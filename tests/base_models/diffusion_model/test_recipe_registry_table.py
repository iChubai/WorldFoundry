# SPDX-FileCopyrightText: Copyright (c) 2026 WorldFoundry contributors
# SPDX-License-Identifier: Apache-2.0
"""Weight-free checks that lazy native-diffusion recipe tables name real factories.

Recipes register lazily, so a table entry whose module or factory is missing
only fails when someone resolves that model ID. These checks parse the tables
without importing model stacks.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit
RECIPES = Path(__file__).resolve().parents[3] / "worldfoundry/base_models/diffusion_model/recipes"


def _module_path(module: str) -> Path:
    """Resolve a module name relative to the recipes package to its source file."""
    name = module.lstrip(".")
    base = RECIPES
    for _ in range(len(module) - len(name) - 1):
        base = base.parent
    path = base.joinpath(*name.split(".")) if name else base
    return path.with_suffix(".py") if path.with_suffix(".py").is_file() else path / "__init__.py"


def _defined_names(module: str) -> set[str] | None:
    """Top-level names a module defines or imports, or None when it is missing."""
    path = _module_path(module)
    if not path.is_file():
        return None
    names: set[str] = set()
    for node in ast.parse(path.read_text(encoding="utf-8")).body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            names.update(target.id for target in node.targets if isinstance(target, ast.Name))
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            names.update((alias.asname or alias.name).split(".")[0] for alias in node.names)
    return names


def _registry_references() -> list[tuple[str, str | None]]:
    """Return the (module, attribute) pairs the lazy registry resolves on demand."""
    references: list[tuple[str, str | None]] = []
    for node in ast.walk(ast.parse((RECIPES / "registry.py").read_text(encoding="utf-8"))):
        if isinstance(node, ast.Tuple) and len(node.elts) == 4:
            values = [item.value if isinstance(item, ast.Constant) else None for item in node.elts[:3]]
            model_id, module, factory = values
            if isinstance(model_id, str) and isinstance(module, str) and isinstance(factory, str):
                if module.startswith("."):
                    references.append((module, factory))
        elif isinstance(node, ast.ImportFrom) and node.level == 1:
            references.extend((f".{node.module or ''}", alias.name) for alias in node.names)
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "import_module":
            target = node.args[0].value if node.args and isinstance(node.args[0], ast.Constant) else None
            if isinstance(target, str) and target.startswith("."):
                references.append((target, None))
    return references


def _unresolved(pairs: list[tuple[str, str | None]]) -> list[str]:
    missing = []
    for module, attribute in pairs:
        names = _defined_names(module)
        if names is None or (attribute is not None and attribute not in names):
            missing.append(f"{module}:{attribute}" if attribute else module)
    return sorted(set(missing))


def test_lazy_registry_entries_name_existing_factories() -> None:
    references = _registry_references()
    assert references, "no lazy recipe entries were found in recipes/registry.py"
    missing = _unresolved(references)
    assert not missing, f"lazy recipe registrations point at missing modules or factories: {missing}"


def test_recipe_exports_name_existing_definitions() -> None:
    tree = ast.parse((RECIPES / "__init__.py").read_text(encoding="utf-8"))
    exports = next(
        node.value
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "_EXPORTS" for target in node.targets)
    )
    assert isinstance(exports, ast.Dict)
    pairs: list[tuple[str, str | None]] = []
    for key, value in zip(exports.keys, exports.values, strict=True):
        name = key.value if isinstance(key, ast.Constant) else None
        module = value.value if isinstance(value, ast.Constant) else None
        if isinstance(name, str) and isinstance(module, str):
            pairs.append((module, name))
    assert pairs, "recipes/__init__.py declares no lazy exports"
    missing = _unresolved(pairs)
    assert not missing, f"recipe exports point at missing modules or names: {missing}"
