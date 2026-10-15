# SPDX-FileCopyrightText: Copyright (c) 2026 WorldFoundry contributors
# SPDX-License-Identifier: Apache-2.0

"""Keep repository policies discoverable and included in source releases."""

from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = pytest.mark.unit
REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("filename", ["CODE_OF_CONDUCT.md", "SECURITY.md"])
def test_policies_are_linked_and_included_in_source_manifest(filename: str) -> None:
    assert (REPO_ROOT / filename).read_text(encoding="utf-8").strip()
    for page in ("README.md", "CONTRIBUTING.md"):
        assert f"]({filename})" in (REPO_ROOT / page).read_text(encoding="utf-8")
    for language in ("", ".zh"):
        page = REPO_ROOT / "docs/fumadocs/content/docs/maintainers" / f"contributing{language}.mdx"
        assert f"/blob/main/{filename})" in page.read_text(encoding="utf-8")
    manifest = (REPO_ROOT / "MANIFEST.in").read_text(encoding="utf-8")
    included = [item for line in manifest.splitlines() if line.startswith("include ") for item in line.split()[1:]]
    assert filename in included


@pytest.mark.parametrize("filename", ["CODE_OF_CONDUCT.md", "SECURITY.md"])
def test_private_reporting_uses_existing_maintainers_and_community_channels(filename: str) -> None:
    text = (REPO_ROOT / filename).read_text(encoding="utf-8")
    assert "](.github/CODEOWNERS)" in text
    assert "](README.md#-community)" in text
    assert (REPO_ROOT / ".github/CODEOWNERS").is_file()
    assert "## 🤝 Community" in (REPO_ROOT / "README.md").read_text(encoding="utf-8")
