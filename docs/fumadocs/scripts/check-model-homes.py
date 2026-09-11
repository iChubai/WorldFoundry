#!/usr/bin/env python3
"""Audit (and optionally patch) model home MDX pages for global polish structure."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from model_paper_figures import has_paper_figures

ROOT = Path(__file__).resolve().parents[1]
PAGES_DIR = ROOT / "content/docs/guides/supported-models"
RESERVED = {"index", "meta"}
CUSTOM_STRUCTURE = {"hunyuanvideo"}

SECTIONS = {
    "en": [
        "Model introduction",
        "Typical use cases",
        "Technical characteristics",
        "Running in WorldFoundry",
        "Limitations and requirements",
    ],
    "zh": [
        "模型介绍",
        "典型使用场景",
        "技术特点",
        "在 WorldFoundry 中运行",
        "限制与要求",
    ],
}

TECH_HEADING = {"en": "Technical characteristics", "zh": "技术特点"}
HERO_PAPER_RE = re.compile(
    r'<figure className="wf-recipe-media is-diagram">\s*\n\s*'
    r'<img src="/models/[^"]+/(?:paper|architecture|arch)\.png"[^>]*>\s*\n'
    r'(?:\s*<figcaption>[^<]*</figcaption>\s*\n)?\s*</figure>\s*\n?',
    re.MULTILINE,
)
ARCH_COMPONENT_RE = re.compile(r"<ModelArchDiagram\b")
PAPER_FIGURES_RE = re.compile(r"<ModelPaperFigures\b")
LEAD_SPLIT_RE = re.compile(r"\A(\n*.+?)(\n{2,})(.*)\Z", re.DOTALL)


def is_paper_cover_src(src: str) -> bool:
    return str(src or "").lower().endswith("/paper.png")


def load_arch_registry() -> dict[str, dict]:
    registry_path = ROOT / "lib/model-arch-diagrams.json"
    if not registry_path.is_file():
        return {}
    payload = json.loads(registry_path.read_text(encoding="utf-8"))
    diagrams = payload.get("diagrams") or {}
    return diagrams if isinstance(diagrams, dict) else {}


def has_real_arch_asset(model_id: str, diagrams: dict[str, dict]) -> bool:
    src = (diagrams.get(model_id) or {}).get("src")
    return bool(src) and not is_paper_cover_src(str(src))


def locale_for(path: Path) -> str:
    return "zh" if path.name.endswith(".zh.mdx") else "en"


def model_id_for(path: Path) -> str:
    name = path.name
    if name.endswith(".zh.mdx"):
        return name[: -len(".zh.mdx")]
    return name[: -len(".mdx")]


def inject_paper_figures(text: str, model_id: str, locale: str) -> tuple[str, bool]:
    if PAPER_FIGURES_RE.search(text):
        return text, False
    if not text.startswith("---"):
        return text, False
    parts = text.split("---", 2)
    if len(parts) < 3:
        return text, False
    body = parts[2]
    block = f'<ModelPaperFigures modelId="{model_id}" locale="{locale}" />'
    match = LEAD_SPLIT_RE.match(body)
    if match:
        lead, gap, rest = match.group(1), match.group(2), match.group(3)
        updated_body = f"{lead}{gap}{block}\n\n{rest.lstrip()}"
    else:
        builder = re.search(r"<ModelCommandBuilder\b", body)
        if not builder:
            return text, False
        updated_body = f"{body[: builder.start()].rstrip()}\n\n{block}\n\n{body[builder.start() :]}"
    if has_paper_figures(model_id):
        updated_body = re.sub(
            r"\n*<ModelArchDiagram\b[^>]*/>\n*",
            "\n\n",
            updated_body,
            count=1,
        )
    return f"---{parts[1]}---{updated_body}", True


def inject_arch_diagram(text: str, model_id: str, locale: str) -> tuple[str, bool]:
    if ARCH_COMPONENT_RE.search(text):
        return text, False

    heading = TECH_HEADING[locale]
    marker = f"## {heading}"
    idx = text.find(marker)
    if idx == -1:
        return text, False

    insert_at = idx + len(marker)
    while insert_at < len(text) and text[insert_at] in "\r\n":
        insert_at += 1

    block = f'\n\n<ModelArchDiagram modelId="{model_id}" locale="{locale}" />\n\n'
    updated = text[:insert_at] + block + text[insert_at:]
    updated = HERO_PAPER_RE.sub("", updated, count=1)
    return updated, True


def audit_page(path: Path) -> dict[str, object]:
    locale = locale_for(path)
    text = path.read_text(encoding="utf-8")
    sections = SECTIONS[locale]
    missing = [heading for heading in sections if f"## {heading}" not in text]
    if model_id_for(path) in CUSTOM_STRUCTURE:
        missing = []
    return {
        "path": path,
        "model_id": model_id_for(path),
        "locale": locale,
        "missing_sections": missing,
        "has_arch_component": bool(ARCH_COMPONENT_RE.search(text)),
        "has_paper_figures": bool(PAPER_FIGURES_RE.search(text)),
        "has_command_builder": "<ModelCommandBuilder" in text,
        "has_related": "<ModelRelatedRecipes" in text,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fix", action="store_true", help="Inject ModelPaperFigures / ModelArchDiagram where missing")
    parser.add_argument(
        "--paper-only",
        action="store_true",
        help="With --fix, only inject ModelPaperFigures (do not add ModelArchDiagram)",
    )
    args = parser.parse_args()

    pages = sorted(
        path
        for path in PAGES_DIR.glob("*.mdx")
        if path.name not in {"index.mdx", "index.zh.mdx"} and model_id_for(path) not in RESERVED
    )

    reports = [audit_page(path) for path in pages]
    diagrams = load_arch_registry()
    missing_sections = [item for item in reports if item["missing_sections"]]
    needs_paper = reports
    missing_paper = [item for item in needs_paper if not item["has_paper_figures"]]
    needs_arch = [
        item
        for item in reports
        if has_real_arch_asset(str(item["model_id"]), diagrams)
        and str(item["model_id"]) not in CUSTOM_STRUCTURE
        and not has_paper_figures(str(item["model_id"]))
    ]
    missing_arch = [item for item in needs_arch if not item["has_arch_component"]]
    fixed = 0

    if args.fix:
        for item in missing_paper:
            path = item["path"]
            locale = str(item["locale"])
            model_id = str(item["model_id"])
            text = path.read_text(encoding="utf-8")
            updated, changed = inject_paper_figures(text, model_id, locale)
            if changed:
                path.write_text(updated, encoding="utf-8")
                item["has_paper_figures"] = True
                fixed += 1
        if not args.paper_only:
            for item in missing_arch:
                path = item["path"]
                locale = str(item["locale"])
                model_id = str(item["model_id"])
                text = path.read_text(encoding="utf-8")
                updated, changed = inject_arch_diagram(text, model_id, locale)
                if changed:
                    path.write_text(updated, encoding="utf-8")
                    fixed += 1
        missing_paper = [item for item in needs_paper if not item["has_paper_figures"]]
        if not args.paper_only:
            missing_arch = [item for item in needs_arch if not item["has_arch_component"]]

    en_pages = [item for item in reports if item["locale"] == "en"]
    zh_pages = [item for item in reports if item["locale"] == "zh"]
    arch_assets = sum(
        1
        for item in diagrams.values()
        if isinstance(item, dict) and item.get("src") and not is_paper_cover_src(str(item.get("src")))
    )
    paper_thumbs = sum(
        1
        for item in diagrams.values()
        if isinstance(item, dict)
        and (item.get("paperSrc") or is_paper_cover_src(str(item.get("src") or "")))
    )

    print(f"model home pages: {len(pages)} ({len(en_pages)} en + {len(zh_pages)} zh)")
    print(f"five-section structure complete (en): {len(en_pages) - len([i for i in en_pages if i['missing_sections']])}/{len(en_pages)}")
    print(
        f"ModelPaperFigures present: "
        f"{len(needs_paper) - len(missing_paper)}/{len(needs_paper)}"
    )
    print(
        f"ModelArchDiagram present where real arch exists: "
        f"{len(needs_arch) - len(missing_arch)}/{len(needs_arch)}"
    )
    print(f"registry body architecture assets: {arch_assets}")
    print(f"registry paper thumbnails: {paper_thumbs}")
    if missing_sections:
        print(f"missing sections: {len(missing_sections)} pages")
        for item in missing_sections[:8]:
            print(f"  - {item['path'].name}: {', '.join(item['missing_sections'])}")
    if missing_paper and not args.fix:
        print(f"missing ModelPaperFigures: {len(missing_paper)} pages")
    if missing_arch and not args.fix:
        print(f"missing ModelArchDiagram: {len(missing_arch)} pages")
    if args.fix:
        print(f"patched paper/arch figures into {fixed} pages")

    if missing_sections or ((missing_paper or missing_arch) and not args.fix):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
