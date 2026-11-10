# SPDX-FileCopyrightText: Copyright (c) 2026 WorldFoundry contributors
# SPDX-License-Identifier: Apache-2.0

"""Validate repository skill discovery metadata and canonical references."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pytest
import yaml

pytestmark = pytest.mark.unit
REPO_ROOT = Path(__file__).resolve().parents[2]
SKILLS_ROOT = REPO_ROOT / "skills"
SKILL_PATHS = sorted(SKILLS_ROOT.glob("*/SKILL.md"))
REFERENCE_PATHS = sorted(SKILLS_ROOT.glob("*/references/*.md"))
SKILL_DOCUMENTS = [*SKILL_PATHS, *REFERENCE_PATHS, SKILLS_ROOT / "index.md"]

# Commands and paths are read from inline and fenced code. A placeholder such as
# ``<model-id>`` or ``*`` ends a path; the directory before it must still exist.
_CODE_SPANS = re.compile(r"```.*?```|`[^`\n]+`", re.S)
_REPO_PATH = re.compile(r"(?<![\w./$-])((?:worldfoundry|tests|scripts|docs|benchmarks|skills|\.github)/[\w./-]*)")
_MAKE_TARGET = re.compile(r"(?<![\w-])make ([a-z][a-z0-9-]*)")
_MODULE = re.compile(r"(?<![\w-])-m ((?:worldfoundry|benchmarks)(?:\.\w+)+)")
_MAKE_TARGETS = frozenset(
    re.findall(r"^([a-z][a-z0-9-]*):", (REPO_ROOT / "Makefile").read_text(encoding="utf-8"), re.M)
)


def _load_skill(path: Path) -> tuple[dict[str, Any], str]:
    text = path.read_text(encoding="utf-8")
    assert text.startswith("---\n"), path
    frontmatter, body = text.removeprefix("---\n").split("\n---\n", 1)
    metadata = yaml.safe_load(frontmatter)
    assert isinstance(metadata, dict), path
    return metadata, body


def _assert_file_references(path: Path, text: str) -> None:
    for target in re.findall(r"\]\(([^)]+)\)", text):
        if target.startswith(("https://", "http://", "#")):
            continue
        assert (path.parent / target.split("#", 1)[0]).is_file(), (path, target)


def _code_references(text: str) -> tuple[set[str], set[str], set[str]]:
    """Collect repository paths, make targets, and ``-m`` modules named in code."""
    paths: set[str] = set()
    targets: set[str] = set()
    modules: set[str] = set()
    for span in _CODE_SPANS.findall(text):
        for match in _REPO_PATH.finditer(span):
            path = match.group(1)
            if span[match.end() : match.end() + 1] in {"<", "*", "{", "$"}:
                path = path.rpartition("/")[0]
            paths.add(path.rstrip("."))
        targets.update(_MAKE_TARGET.findall(span))
        modules.update(_MODULE.findall(span))
    return paths, targets, modules


def _module_exists(module: str) -> bool:
    base = REPO_ROOT.joinpath(*module.split("."))
    return base.with_suffix(".py").is_file() or (base / "__init__.py").is_file()


@pytest.mark.parametrize("path", SKILL_PATHS, ids=[path.parent.name for path in SKILL_PATHS])
def test_skill_metadata_and_references(path: Path) -> None:
    metadata, body = _load_skill(path)
    name = metadata["name"]
    description = metadata["description"]
    assert isinstance(name, str) and 1 <= len(name) <= 64
    assert name == path.parent.name
    assert re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", name)
    assert isinstance(description, str) and description.strip() and len(description) <= 1024
    if "compatibility" in metadata:
        compatibility = metadata["compatibility"]
        assert isinstance(compatibility, str) and compatibility.strip() and len(compatibility) <= 500
    assert metadata["license"] == "Apache-2.0"
    assert body.strip()
    assert len(path.read_text(encoding="utf-8").splitlines()) < 500

    _assert_file_references(path, body)


@pytest.mark.parametrize("path", REFERENCE_PATHS, ids=[path.stem for path in REFERENCE_PATHS])
def test_skill_reference_links_resolve(path: Path) -> None:
    skill = path.parent.parent / "SKILL.md"
    assert f"](references/{path.name})" in skill.read_text(encoding="utf-8")
    text = path.read_text(encoding="utf-8")
    assert text.strip(), path
    _assert_file_references(path, text)


@pytest.mark.parametrize(
    "path", SKILL_DOCUMENTS, ids=[path.relative_to(SKILLS_ROOT).as_posix() for path in SKILL_DOCUMENTS]
)
def test_skill_commands_name_existing_repository_entries(path: Path) -> None:
    """A renamed path, make target, or module must not leave stale skill instructions."""
    paths, targets, modules = _code_references(path.read_text(encoding="utf-8"))
    missing_paths = sorted(item for item in paths if not (REPO_ROOT / item).exists())
    assert not missing_paths, f"{path.relative_to(REPO_ROOT)} names missing paths: {missing_paths}"
    missing_targets = sorted(targets - _MAKE_TARGETS)
    assert not missing_targets, f"{path.relative_to(REPO_ROOT)} names missing make targets: {missing_targets}"
    missing_modules = sorted(item for item in modules if not _module_exists(item))
    assert not missing_modules, f"{path.relative_to(REPO_ROOT)} names missing modules: {missing_modules}"


def test_skill_index_covers_all_published_skills() -> None:
    assert SKILL_PATHS, "The skill directory must contain at least one skill."
    index = (SKILLS_ROOT / "index.md").read_text(encoding="utf-8")
    targets = re.findall(r"\]\(([^)]+/SKILL\.md)\)", index)
    assert len(targets) == len(set(targets))
    assert set(targets) == {path.relative_to(SKILLS_ROOT).as_posix() for path in SKILL_PATHS}


def test_repository_guidance_links_to_skills() -> None:
    for name in ("README.md", "AGENTS.md", "CONTRIBUTING.md"):
        assert "(skills/index.md)" in (REPO_ROOT / name).read_text(encoding="utf-8"), name
    for name in ("contributing.mdx", "contributing.zh.mdx"):
        text = (REPO_ROOT / "docs/fumadocs/content/docs/maintainers" / name).read_text(encoding="utf-8")
        assert "skills/index.md" in text
        assert ".agents/skills" in text
        assert ".claude/skills" in text
        assert ".cursor/skills" in text


def test_source_manifest_preserves_skill_documents() -> None:
    from setuptools._distutils.filelist import FileList

    documents = {path.relative_to(REPO_ROOT).as_posix() for path in SKILLS_ROOT.rglob("*.md")}
    files = FileList()
    files.allfiles = sorted(documents)
    files.files = sorted(documents)
    for line in (REPO_ROOT / "MANIFEST.in").read_text(encoding="utf-8").splitlines():
        if line.strip() and not line.lstrip().startswith("#"):
            files.process_template_line(line)
    assert documents <= set(files.files)
