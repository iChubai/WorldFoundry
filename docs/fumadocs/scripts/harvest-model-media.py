#!/usr/bin/env python3
"""Collect paper titles and teaser media for model homepages.

Reads the generated recipe JSON, pulls arXiv metadata, attaches in-repo demo
clips, and downloads Hugging Face paper thumbnails when they exist. Writes
``lib/model-paper-media.json`` and patches recipe/index JSON in place so the
pages pick up figures without a full catalog regenerate.
"""

from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

DOCS_ROOT = Path(__file__).resolve().parents[1]
RECIPES = DOCS_ROOT / "lib" / "model-recipes-data.json"
INDEX = DOCS_ROOT / "lib" / "model-recipes-index.json"
OUT = DOCS_ROOT / "lib" / "model-paper-media.json"
PUBLIC = DOCS_ROOT / "public"
THUMB_DIR = PUBLIC / "models"

USER_AGENT = "WorldFoundryDocs/1.0 (model homepage media harvest)"
ARXIV_NS = {"a": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}
ARXIV_ID_RE = re.compile(
    r"(?:arxiv\.org/(?:abs|pdf|html)|huggingface\.co/papers|hf\.co/papers)/(\d{4}\.\d{4,5})(?:v\d+)?",
    re.I,
)
GITHUB_RE = re.compile(r"https?://(?:www\.)?github\.com/([^/]+)/([^/?#]+)", re.I)

# In-tree Studio / demo clips already checked into public/.
LOCAL_MEDIA: dict[str, list[dict[str, str]]] = {
    "ac3d": [{"kind": "video", "src": "/demos/ac3d_01.mp4", "caption": "Camera-controlled video", "captionZh": "相机控制视频"}],
    "astra": [{"kind": "video", "src": "/demos/astra_01.mp4", "caption": "Interactive camera control", "captionZh": "交互式相机控制"}],
    "cogvideox": [{"kind": "video", "src": "/demos/cogvideo_01.mp4", "caption": "Text-to-video sample", "captionZh": "文本生成视频"}],
    "cosmos-predict-2.5": [{"kind": "video", "src": "/demos/cosmos_01.mp4", "caption": "World-video prediction", "captionZh": "世界视频预测"}],
    "hunyuan-game-craft": [{"kind": "video", "src": "/demos/studio/hunyuan-game-craft-village.mp4", "caption": "Village-scale game world", "captionZh": "村落尺度游戏世界"}],
    "hunyuanvideo": [
        {
            "kind": "video",
            "src": "/demos/studio/hunyuanvideo-t2v-cat-grass-official.mp4",
            "poster": "/models/hunyuanvideo/video_poster.png",
            "caption": "Official HunyuanVideo text-to-video demo (README teaser poster)",
            "captionZh": "HunyuanVideo 官方文生视频演示（README 作品展示海报）",
        },
        {
            "kind": "video",
            "src": "/demos/studio/hunyuanvideo-i2v-firework-official.mp4",
            "poster": "/models/hunyuanvideo/i2v_video_poster.jpg",
            "caption": "HunyuanVideo-I2V image-to-video demo",
            "captionZh": "HunyuanVideo-I2V 图生视频演示",
        },
        {
            "kind": "image",
            "src": "/models/hunyuanvideo/overall.png",
            "caption": "HunyuanVideo overall architecture — Causal 3D VAE, LLM text encoder, and latent diffusion",
            "captionZh": "HunyuanVideo 整体架构：因果 3D VAE、大语言模型文本编码与潜空间扩散",
        },
        {
            "kind": "image",
            "src": "/models/hunyuanvideo/backbone.png",
            "caption": "Dual-stream to single-stream Transformer backbone for unified image and video generation",
            "captionZh": "双流到单流 Transformer 骨干，用于统一的图视频生成",
        },
        {
            "kind": "image",
            "src": "/models/hunyuanvideo/text_encoder.png",
            "caption": "MLLM text encoder with a bidirectional token refiner",
            "captionZh": "带双向 token 优化器的 MLLM 文本编码器",
        },
        {
            "kind": "image",
            "src": "/models/hunyuanvideo/3dvae.png",
            "caption": "CausalConv3D VAE — temporal 4×, spatial 8×, 16-channel latent",
            "captionZh": "CausalConv3D VAE：时间 4 倍、空间 8 倍压缩，16 通道潜空间",
        },
        {
            "kind": "image",
            "src": "/models/hunyuanvideo/i2v_backbone.png",
            "caption": "HunyuanVideo-I2V architecture — token replace of the reference image into the video generator",
            "captionZh": "HunyuanVideo-I2V 架构：用 token replace 将参考图写入视频生成过程",
        },
    ],
    "hunyuanvideo-1.5": [{"kind": "video", "src": "/demos/studio/hunyuanvideo-1-5-t2v-cat.mp4", "caption": "HunyuanVideo 1.5 text-to-video", "captionZh": "HunyuanVideo 1.5 文本生成视频"}],
    "hy-worldplay": [{"kind": "video", "src": "/demos/studio/hy-worldplay-official-8gpu.mp4", "caption": "Interactive world play", "captionZh": "交互式世界漫游"}],
    "leworldmodel": [{"kind": "video", "src": "/demos/studio/leworldmodel-pusht.mp4", "caption": "PushT world-model rollout", "captionZh": "PushT 世界模型滚动"}],
    "longvie-1": [{"kind": "video", "src": "/demos/studio/longvie-1-control-video.mp4", "caption": "Long controllable video", "captionZh": "可控长视频"}],
    "longvie-2": [{"kind": "video", "src": "/demos/studio/longvie-2-control-video.mp4", "caption": "Queued long-video segments", "captionZh": "分段衔接的长视频"}],
    "ltx-video": [{"kind": "video", "src": "/demos/studio/ltx-video-i2v-penguin.mp4", "caption": "Image-to-video", "captionZh": "图生视频"}],
    "ltx-2.x": [{"kind": "video", "src": "/demos/studio/ltx2-3-i2v-penguin.mp4", "caption": "LTX-2.3 image-to-video", "captionZh": "LTX-2.3 图生视频"}],
    "matrix-game-2": [
        {
            "kind": "video",
            "src": "/demos/studio/matrix-game-2-official-universal.mp4",
            "poster": "/images/hero/matrix-game-2.webp",
            "caption": "Keyboard-controlled interactive world",
            "captionZh": "键鼠控制的交互世界",
        }
    ],
    "matrix-game-3": [{"kind": "video", "src": "/demos/studio/matrix-game-3-cityscape.mp4", "caption": "Cityscape interactive world", "captionZh": "城市场景交互世界"}],
    "modelscope-t2v": [{"kind": "video", "src": "/demos/studio/modelscope-t2v.mp4", "caption": "Text-to-video", "captionZh": "文本生成视频"}],
    "neoverse": [
        {
            "kind": "video",
            "src": "/demos/studio/neoverse-robot-tabletop.mp4",
            "poster": "/images/hero/neoverse.webp",
            "caption": "Tabletop manipulation trajectories",
            "captionZh": "桌面操作轨迹",
        }
    ],
    "open-sora-plan": [{"kind": "video", "src": "/demos/studio/open-sora-plan-tokyo-street.mp4", "caption": "Open-Sora Plan sample", "captionZh": "Open-Sora Plan 样例"}],
    "skyreels-v3": [{"kind": "video", "src": "/demos/studio/skyreels-v3-reference-to-video.mp4", "caption": "Reference-to-video", "captionZh": "参考图生成视频"}],
    "unianimate-dit": [{"kind": "video", "src": "/demos/studio/unianimate-dit-human-animation.mp4", "caption": "Human animation", "captionZh": "人物动画"}],
    "videocrafter1-i2v": [{"kind": "video", "src": "/demos/studio/videocrafter1-i2v.mp4", "caption": "Image-to-video", "captionZh": "图生视频"}],
    "videocrafter1-t2v": [{"kind": "video", "src": "/demos/studio/videocrafter1-t2v.mp4", "caption": "Text-to-video", "captionZh": "文本生成视频"}],
    "videocrafter2-t2v": [{"kind": "video", "src": "/demos/studio/videocrafter2-t2v.mp4", "caption": "Text-to-video", "captionZh": "文本生成视频"}],
    "wan2.1-vace": [{"kind": "video", "src": "/demos/studio/wan2-1-vace-girl-snake.mp4", "caption": "VACE-controlled generation", "captionZh": "VACE 可控生成"}],
    "worldcam": [{"kind": "video", "src": "/demos/studio/worldcam-industrial.mp4", "caption": "Industrial camera-controlled world", "captionZh": "工业场景相机控制"}],
    "yume": [{"kind": "video", "src": "/demos/studio/yume-1p5-jungle-castle.mp4", "caption": "First-person world exploration", "captionZh": "第一人称世界探索"}],
}


def request(url: str, timeout: int = 20) -> bytes | None:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            if response.status >= 400:
                return None
            return response.read()
    except (urllib.error.URLError, TimeoutError, ValueError):
        return None


def arxiv_id_from(url: str) -> str | None:
    match = ARXIV_ID_RE.search(url or "")
    return match.group(1) if match else None


def fetch_arxiv(ids: list[str]) -> dict[str, dict[str, object]]:
    papers: dict[str, dict[str, object]] = {}
    for start in range(0, len(ids), 16):
        batch = ids[start : start + 16]
        url = "http://export.arxiv.org/api/query?id_list=" + ",".join(batch) + "&max_results=16"
        raw = request(url, timeout=30)
        if not raw:
            time.sleep(1.2)
            continue
        root = ET.fromstring(raw)
        for entry in root.findall("a:entry", ARXIV_NS):
            entry_id = (entry.findtext("a:id", default="", namespaces=ARXIV_NS) or "")
            match = re.search(r"(\d{4}\.\d{4,5})", entry_id)
            if not match:
                continue
            title = re.sub(r"\s+", " ", (entry.findtext("a:title", default="", namespaces=ARXIV_NS) or "")).strip()
            abstract = re.sub(r"\s+", " ", (entry.findtext("a:summary", default="", namespaces=ARXIV_NS) or "")).strip()
            published = entry.findtext("a:published", default="", namespaces=ARXIV_NS) or ""
            comment = entry.findtext("arxiv:comment", default="", namespaces=ARXIV_NS) or ""
            venue = None
            venue_match = re.search(
                r"\b(CVPR|ICCV|ECCV|NeurIPS|ICML|ICLR|AAAI|IJCAI|SIGGRAPH|ACL|EMNLP|IROS|ICRA|CoRL|RSS|3DV|WACV)[^.;,]{0,24}",
                comment,
                re.I,
            )
            if venue_match:
                venue = venue_match.group(0).strip(" ;,")
            papers[match.group(1)] = {
                "title": title,
                "year": int(published[:4]) if published[:4].isdigit() else None,
                "venue": venue,
                "arxivId": match.group(1),
                "abstract": abstract[:900] if abstract else None,
            }
        time.sleep(0.8)
    return papers


def download_thumb(arxiv_id: str, model_id: str) -> str | None:
    url = f"https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/{arxiv_id}.png"
    raw = request(url, timeout=15)
    if not raw or len(raw) < 4000:
        return None
    dest_dir = THUMB_DIR / model_id
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / "paper.png"
    dest.write_bytes(raw)
    return f"/models/{model_id}/paper.png"


def teaser_src(figures: list[dict[str, str]]) -> str | None:
    if not figures:
        return None
    first = figures[0]
    return first.get("poster") or (first.get("src") if first.get("kind") != "video" else None)


def apply_payload(media: dict[str, dict[str, object]]) -> None:
    recipes = json.loads(RECIPES.read_text(encoding="utf-8"))
    for recipe in recipes["recipes"]:
        extra = media.get(recipe["id"]) or {}
        docs = recipe.setdefault("docs", {})
        if extra.get("paper"):
            docs["paper"] = extra["paper"]
        else:
            docs.setdefault("paper", None)
        if extra.get("figures"):
            docs["figures"] = extra["figures"]
        else:
            docs.setdefault("figures", [])
    RECIPES.write_text(json.dumps(recipes, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    index = json.loads(INDEX.read_text(encoding="utf-8"))
    for recipe in index["recipes"]:
        extra = media.get(recipe["id"]) or {}
        recipe["teaser"] = teaser_src(extra.get("figures") or [])
    INDEX.write_text(json.dumps(index, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8")


def main() -> None:
    recipes = json.loads(RECIPES.read_text(encoding="utf-8"))["recipes"]
    arxiv_for: dict[str, str] = {}
    for recipe in recipes:
        for source in recipe.get("sources") or []:
            found = arxiv_id_from(source.get("url") or "")
            if found:
                arxiv_for[recipe["id"]] = found
                break

    unique_ids = sorted(set(arxiv_for.values()))
    print(f"arxiv ids: {len(unique_ids)} across {len(arxiv_for)} models", flush=True)
    papers = fetch_arxiv(unique_ids)
    print(f"arxiv metadata: {len(papers)}", flush=True)

    media: dict[str, dict[str, object]] = {}
    downloaded = 0
    for recipe in recipes:
        model_id = recipe["id"]
        entry: dict[str, object] = {}
        arxiv_id = arxiv_for.get(model_id)
        if arxiv_id and arxiv_id in papers:
            entry["paper"] = papers[arxiv_id]
        figures = [dict(item) for item in LOCAL_MEDIA.get(model_id, [])]
        if arxiv_id and not any(item.get("kind") == "image" or item.get("poster") for item in figures):
            thumb = download_thumb(arxiv_id, model_id)
            if thumb:
                downloaded += 1
                paper = entry.get("paper") if isinstance(entry.get("paper"), dict) else {}
                title = str(paper.get("title") or "Paper figure")
                figures.insert(
                    0,
                    {
                        "kind": "image",
                        "src": thumb,
                        "caption": title,
                        "captionZh": title,
                    },
                )
                time.sleep(0.15)
        if figures:
            entry["figures"] = figures
        if entry:
            media[model_id] = entry

    OUT.write_text(json.dumps(media, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    apply_payload(media)
    print(f"wrote {len(media)} media records, {downloaded} paper thumbs -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
