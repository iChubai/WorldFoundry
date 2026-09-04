#!/usr/bin/env python3
"""SVG generator for architecture flow diagrams.

Text uses native SVG <text> so labels still render when the file is shown as
an <img> (Next.js / markdown images). HTML foreignObject is not used.

The docs site prefers React diagrams via `components/arch-diagram.tsx`. Keep
this script for the static SVGs under public/diagrams/.

Run:  python3 scripts/gen_arch_flow_diagrams.py
Writes SVGs to public/diagrams/.
"""
from __future__ import annotations
import html
from pathlib import Path

# ── palette (matches app/styles/tokens.css light theme) ──────────────────
# Light values are the defaults; dark values apply via prefers-color-scheme
# inside each SVG so the diagram adapts when the browser is in dark mode.
LIGHT = {
    "bg": "#f7f5ef",
    "card": "#ffffff",
    "ink": "#111111",
    "muted": "#65645f",
    "line": "#d8d4c9",
    "accent": "#9a8568",
    "band": "#efe9df",  # boundary-band tint, between bg and line
}
DARK = {
    "bg": "#141311",
    "card": "#22201c",
    "ink": "#f3efe6",
    "muted": "#b3ada3",
    "line": "#3a3733",
    "accent": "#c4ad92",
    "band": "#2a2722",
}

# ── layout ────────────────────────────────────────────────────────────────
NODE_W = 212
NODE_H = 104
GAP = 48      # arrow region width
PAD = 16

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "public" / "diagrams"


FONT = (
    "ui-sans-serif, system-ui, -apple-system, 'Segoe UI', "
    "'Noto Sans SC', 'PingFang SC', 'Microsoft YaHei', sans-serif"
)


def esc(s: str) -> str:
    return html.escape(s, quote=False)


def _char_width(ch: str, size: float) -> float:
    code = ord(ch)
    if ch == " ":
        return size * 0.34
    if 0x4E00 <= code <= 0x9FFF or 0x3000 <= code <= 0x303F or 0xFF00 <= code <= 0xFFEF:
        return size
    if ch in "ilI.,:'/":
        return size * 0.32
    if ch.isupper() or ch in "MW@":
        return size * 0.68
    return size * 0.56


def _text_width(text: str, size: float) -> float:
    return sum(_char_width(ch, size) for ch in text)


def wrap_lines(text: str, max_width: float, size: float) -> list[str]:
    tokens: list[str] = []
    buf = ""
    for ch in text:
        if ch == " ":
            if buf:
                tokens.append(buf)
                buf = ""
            tokens.append(" ")
        else:
            buf += ch
    if buf:
        tokens.append(buf)

    lines: list[str] = []
    current = ""
    current_w = 0.0
    for tok in tokens:
        token_w = _text_width(tok, size)
        if current and current_w + token_w > max_width:
            lines.append(current.rstrip())
            current = tok.lstrip()
            current_w = _text_width(current, size)
            if current_w > max_width and len(current) > 1:
                overflow = current
                current = ""
                current_w = 0.0
                for ch in overflow:
                    ch_w = _char_width(ch, size)
                    if current and current_w + ch_w > max_width:
                        lines.append(current)
                        current = ch
                        current_w = ch_w
                    else:
                        current += ch
                        current_w += ch_w
        else:
            current += tok
            current_w += token_w
    if current.strip():
        lines.append(current.rstrip())
    return lines or [text]


def svg_text(
    x: float,
    y: float,
    content: str,
    *,
    size: float,
    fill: str,
    weight: int = 400,
    anchor: str = "start",
    italic: bool = False,
) -> str:
    style = (
        f"font-family:{FONT};font-size:{size}px;font-weight:{weight};"
        f"fill:{fill};font-style:{'italic' if italic else 'normal'}"
    )
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}" style="{style}">'
        f"{esc(content)}</text>"
    )


def role_chip(x: float, y: float, role: str, *, size: float = 9) -> str:
    width = 8 + _text_width(role, size)
    height = size + 5
    return (
        f'<rect x="{x:.1f}" y="{y:.1f}" width="{width:.1f}" height="{height:.1f}" '
        f'rx="3" ry="3" style="fill:none;stroke:var(--c-line);stroke-width:1"/>'
        + svg_text(x + 4, y + size + 1.2, role, size=size, fill="var(--c-muted)", weight=500)
    )


def node_svg(
    x: float,
    y: float,
    w: float,
    h: float,
    role: str | None,
    title: str,
    detail: str | None,
    *,
    compact: bool = False,
) -> str:
    radius = 8 if compact else 10
    pad_x = 10 if compact else 12
    pad_y = 7 if compact else 12
    title_size = 12 if compact else 13
    detail_size = 10 if compact else 11
    parts = [
        f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" '
        f'rx="{radius}" ry="{radius}" style="fill:var(--c-card);stroke:var(--c-line);stroke-width:1"/>'
    ]
    cursor = y + pad_y
    if role:
        parts.append(role_chip(x + pad_x, cursor, role, size=9 if compact else 10))
        cursor += 16 if compact else 20
    parts.append(
        svg_text(x + pad_x, cursor + title_size, title, size=title_size, fill="var(--c-ink)", weight=650)
    )
    cursor += title_size + (4 if compact else 6)
    if detail:
        for line in wrap_lines(detail, w - pad_x * 2, detail_size):
            parts.append(
                svg_text(x + pad_x, cursor + detail_size, line, size=detail_size, fill="var(--c-muted)")
            )
            cursor += detail_size + 2
    return "".join(parts)


def palette_style(*, include_band: bool = False) -> str:
    extra_light = f"; --c-band: {LIGHT['band']}" if include_band else ""
    extra_dark = f"; --c-band: {DARK['band']}" if include_band else ""
    return f"""  <style>
    :root {{
      --c-bg: {LIGHT['bg']}; --c-card: {LIGHT['card']}; --c-ink: {LIGHT['ink']};
      --c-muted: {LIGHT['muted']}; --c-line: {LIGHT['line']}; --c-accent: {LIGHT['accent']}{extra_light};
    }}
    @media (prefers-color-scheme: dark) {{
      :root {{
        --c-bg: {DARK['bg']}; --c-card: {DARK['card']}; --c-ink: {DARK['ink']};
        --c-muted: {DARK['muted']}; --c-line: {DARK['line']}; --c-accent: {DARK['accent']}{extra_dark};
      }}
    }}
  </style>"""


def render(steps: list[dict], out_path: Path, aria: str) -> None:
    n = len(steps)
    width = 2 * PAD + n * NODE_W + (n - 1) * GAP
    height = 2 * PAD + NODE_H

    nodes_svg: list[str] = []
    arrows_svg: list[str] = []
    for i, st in enumerate(steps):
        x = PAD + i * (NODE_W + GAP)
        y = PAD
        nodes_svg.append(
            node_svg(x, y, NODE_W, NODE_H, st.get("role"), st["title"], st.get("detail"))
        )
        if i < n - 1:
            ax1 = x + NODE_W
            ax2 = x + NODE_W + GAP
            ay = y + NODE_H / 2
            arrows_svg.append(
                f'<line x1="{ax1}" y1="{ay}" x2="{ax2 - 7}" y2="{ay}" '
                f'style="stroke:var(--c-accent);stroke-width:1.5"/>'
            )
            arrows_svg.append(
                f'<polygon points="{ax2 - 7},{ay - 5} {ax2 - 7},{ay + 5} {ax2},{ay}" '
                f'style="fill:var(--c-accent)"/>'
            )

    style = palette_style()

    svg = f'''<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-label="{esc(aria)}">
  <rect x="0" y="0" width="{width}" height="{height}" style="fill:var(--c-bg)"/>
  {''.join(arrows_svg)}
  {''.join(nodes_svg)}
{style}
</svg>
'''
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(svg, encoding="utf-8")


def render_boundary(data: dict, out_path: Path, aria: str) -> None:
    """Two-column boundary diagram: model execution | artifact boundary | benchmark.

    Left and right columns each hold a vertical flow of node cards. A tinted
    central band marks the artifact boundary; one bold arrow crosses it to
    show that only GenerationResult / artifact refs cross, never checkpoints
    or evaluators. Node text uses the same native SVG labels as the
    linear flow diagrams.
    """
    PAD = 20
    COL_W = 372
    BAND_W = 130
    HEADER_H = 44
    NODE_H = 54
    NODE_VGAP = 14
    CAPTION_H = 40

    left = data["left_nodes"]
    right = data["right_nodes"]
    n_max = max(len(left), len(right))
    nodes_h = n_max * NODE_H + (n_max - 1) * NODE_VGAP
    inner_top = PAD + HEADER_H + 12
    caption_top = inner_top + nodes_h + 18
    col_bottom = caption_top + CAPTION_H
    col_h = col_bottom - PAD
    height = col_bottom + PAD
    width = 2 * PAD + 2 * COL_W + BAND_W

    lx = PAD
    band_x = lx + COL_W
    rx = band_x + BAND_W
    mid_y = inner_top + nodes_h / 2
    node_w = COL_W - 32
    node_x_left = lx + 16
    node_x_right = rx + 16

    def node_span(nodes: list[dict]) -> float:
        cnt = len(nodes)
        return cnt * NODE_H + (cnt - 1) * NODE_VGAP

    def col_nodes(nodes: list[dict], x0: float) -> list[str]:
        start_y = inner_top + (nodes_h - node_span(nodes)) / 2
        out: list[str] = []
        for i, st in enumerate(nodes):
            y = start_y + i * (NODE_H + NODE_VGAP)
            out.append(
                node_svg(
                    x0,
                    y,
                    node_w,
                    NODE_H,
                    st.get("role"),
                    st["title"],
                    st.get("detail"),
                    compact=True,
                )
            )
            if i < len(nodes) - 1:
                ax = x0 + node_w / 2
                ay1 = y + NODE_H
                ay2 = y + NODE_H + NODE_VGAP
                out.append(
                    f'<line x1="{ax}" y1="{ay1}" x2="{ax}" y2="{ay2 - 5}" '
                    f'style="stroke:var(--c-accent);stroke-width:1.5"/>'
                )
                out.append(
                    f'<polygon points="{ax - 5},{ay2 - 5} {ax + 5},{ay2 - 5} {ax},{ay2}" '
                    f'style="fill:var(--c-accent)"/>'
                )
        return out

    parts: list[str] = [
        f'<rect x="0" y="0" width="{width}" height="{height}" style="fill:var(--c-bg)"/>'
    ]

    # column containers (border only, nodes sit inside)
    for cx in (lx, rx):
        parts.append(
            f'<rect x="{cx}" y="{PAD}" width="{COL_W}" height="{col_h}" rx="12" ry="12" '
            f'style="fill:none;stroke:var(--c-line);stroke-width:1"/>'
        )
    for cx, title in ((lx, data["left_title"]), (rx, data["right_title"])):
        parts.append(
            svg_text(cx + 16, PAD + 28, title, size=13, fill="var(--c-ink)", weight=700)
        )

    parts.append(
        f'<rect x="{band_x}" y="{PAD}" width="{BAND_W}" height="{col_h}" rx="12" ry="12" '
        f'style="fill:var(--c-band);stroke:var(--c-line);stroke-width:1;stroke-dasharray:4 3"/>'
    )
    band_cx = band_x + BAND_W / 2
    for i, line in enumerate(wrap_lines(data["boundary_label"].upper(), BAND_W - 16, 10)):
        parts.append(
            svg_text(
                band_cx,
                PAD + 22 + i * 13,
                line,
                size=10,
                fill="var(--c-muted)",
                weight=700,
                anchor="middle",
            )
        )

    parts.extend(col_nodes(left, node_x_left))
    parts.extend(col_nodes(right, node_x_right))

    parts.append(
        svg_text(
            band_cx,
            mid_y - 16,
            data["cross_main"],
            size=11,
            fill="var(--c-accent)",
            weight=700,
            anchor="middle",
        )
    )
    parts.append(
        f'<line x1="{band_x}" y1="{mid_y}" x2="{rx - 9}" y2="{mid_y}" '
        f'style="stroke:var(--c-accent);stroke-width:2.5"/>'
    )
    parts.append(
        f'<polygon points="{rx - 9},{mid_y - 6} {rx - 9},{mid_y + 6} {rx},{mid_y}" '
        f'style="fill:var(--c-accent)"/>'
    )
    parts.append(
        svg_text(
            band_cx,
            mid_y + 18,
            data["cross_sub"],
            size=9,
            fill="var(--c-muted)",
            weight=500,
            anchor="middle",
        )
    )

    for cx, cap in ((lx, data["left_caption"]), (rx, data["right_caption"])):
        for i, line in enumerate(wrap_lines(cap, COL_W - 32, 10)):
            parts.append(
                svg_text(
                    cx + 16,
                    caption_top + 14 + i * 13,
                    line,
                    size=10,
                    fill="var(--c-muted)",
                    italic=True,
                )
            )

    style = palette_style(include_band=True)

    svg = f'''<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-label="{esc(aria)}">
  {''.join(parts)}
{style}
</svg>
'''
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(svg, encoding="utf-8")


# ── diagram data ──────────────────────────────────────────────────────────

EVAL = {
    "aria": "Evaluation run flow: request to scorecard",
    "steps": [
        {"title": "Request", "role": "Entry", "detail": "EvaluateRunRequest / ModelBenchmarkRunRequest"},
        {"title": "Facade", "role": "Orchestration", "detail": "run_evaluate: validate + pick delegate"},
        {"title": "Delegate runner", "role": "Execution", "detail": "ContractRunner / ExistingResults / ModelBenchmark"},
        {"title": "Generate", "role": "Model", "detail": "model runner → GenerationResult"},
        {"title": "Metrics", "role": "Metrics", "detail": "per-sample + aggregate"},
        {"title": "Scorecard + report", "role": "Report", "detail": "scorecard.json, report.md"},
    ],
}
RUNTIME = {
    "aria": "Model runtime assembly chain",
    "steps": [
        {"title": "Catalog binding", "role": "Control plane", "detail": "pipeline_target / runtime_profile"},
        {"title": "PipelineABC", "role": "Runtime", "detail": "load / process / stream"},
        {"title": "Operator", "role": "Input", "detail": "validate, media, camera/action, interaction"},
        {"title": "Synthesis", "role": "Inference", "detail": "native runtime or hosted API"},
        {"title": "Representation", "role": "Geometry", "detail": "depth / point cloud / 3DGS"},
        {"title": "Memory", "role": "State", "detail": "streaming / multi-turn"},
        {"title": "GenerationResult", "role": "Artifact", "detail": "normalized outputs + metadata"},
    ],
}
SURF = {
    "aria": "Surfaces to orchestration to execution flow",
    "steps": [
        {"title": "Surface", "role": "Entry", "detail": "CLI / TUI / MCP / Studio"},
        {"title": "Intent", "role": "Intent", "detail": "ModelBenchmark / ScoreArtifacts / GenerateAndScore / ScoreResults / Reproduce"},
        {"title": "prepare_evaluation", "role": "Orchestration", "detail": "validate + preflight diagnostics"},
        {"title": "PreparedEvaluation", "role": "Plan", "detail": "claim policy + readiness"},
        {"title": "execute_prepared", "role": "Compile", "detail": "→ EvaluateRunRequest / ModelBenchmarkRunRequest"},
        {"title": "Delegate runner", "role": "Execution", "detail": "see Workflow"},
    ],
}
DIFF = {
    "aria": "Native diffusion assembly flow",
    "steps": [
        {"title": "Catalog / profile", "role": "Control plane", "detail": "model ID + runtime policy"},
        {"title": "NativeDiffusionPipeline", "role": "Boundary", "detail": "public entrypoint (pipeline.py:16)"},
        {"title": "Recipe", "role": "Recipe", "detail": "immutable component + checkpoint specs"},
        {"title": "Assembler", "role": "Assembly", "detail": "resolve + validate (assembly.py:31)"},
        {"title": "Runner", "role": "Execution", "detail": "condition / init / schedule / denoise / decode"},
        {"title": "Artifact", "role": "Output", "detail": "normalized WorldFoundry artifact"},
    ],
}

EVAL_ZH = {
    "aria": "评测 run 流程：request 到 scorecard",
    "steps": [
        {"title": "Request", "role": "入口", "detail": "EvaluateRunRequest / ModelBenchmarkRunRequest"},
        {"title": "Facade", "role": "编排", "detail": "run_evaluate：校验 + 选 delegate"},
        {"title": "Delegate runner", "role": "执行", "detail": "ContractRunner / ExistingResults / ModelBenchmark"},
        {"title": "Generate", "role": "模型", "detail": "model runner → GenerationResult"},
        {"title": "Metrics", "role": "指标", "detail": "逐样本 + 聚合"},
        {"title": "Scorecard + report", "role": "报告", "detail": "scorecard.json、report.md"},
    ],
}
RUNTIME_ZH = {
    "aria": "模型 runtime 组装链",
    "steps": [
        {"title": "Catalog 绑定", "role": "控制平面", "detail": "pipeline_target / runtime_profile"},
        {"title": "PipelineABC", "role": "运行时", "detail": "load / process / stream"},
        {"title": "Operator", "role": "输入", "detail": "校验、媒体、camera/action、interaction"},
        {"title": "Synthesis", "role": "推理", "detail": "native runtime 或 hosted API"},
        {"title": "Representation", "role": "几何", "detail": "深度 / 点云 / 3DGS"},
        {"title": "Memory", "role": "状态", "detail": "流式 / 多轮"},
        {"title": "GenerationResult", "role": "产物", "detail": "归一化输出 + metadata"},
    ],
}
SURF_ZH = {
    "aria": "使用界面到编排到执行的流程",
    "steps": [
        {"title": "界面", "role": "入口", "detail": "CLI / TUI / MCP / Studio"},
        {"title": "Intent", "role": "意图", "detail": "ModelBenchmark / ScoreArtifacts / GenerateAndScore / ScoreResults / Reproduce"},
        {"title": "prepare_evaluation", "role": "编排", "detail": "校验 + preflight 诊断"},
        {"title": "PreparedEvaluation", "role": "计划", "detail": "claim 策略 + 就绪状态"},
        {"title": "execute_prepared", "role": "编译", "detail": "→ EvaluateRunRequest / ModelBenchmarkRunRequest"},
        {"title": "Delegate runner", "role": "执行", "detail": "见工作流"},
    ],
}
DIFF_ZH = {
    "aria": "原生扩散装配流程",
    "steps": [
        {"title": "Catalog / profile", "role": "控制平面", "detail": "model ID + runtime policy"},
        {"title": "NativeDiffusionPipeline", "role": "边界", "detail": "公开入口（pipeline.py:16）"},
        {"title": "Recipe", "role": "Recipe", "detail": "不可变 component + checkpoint spec"},
        {"title": "Assembler", "role": "装配", "detail": "解析 + 校验（assembly.py:31）"},
        {"title": "Runner", "role": "执行", "detail": "condition / init / schedule / denoise / decode"},
        {"title": "Artifact", "role": "产物", "detail": "归一化 WorldFoundry artifact"},
    ],
}

DIAGRAMS = [
    ("flow-evaluation-core", EVAL, EVAL["aria"]),
    ("flow-model-runtime", RUNTIME, RUNTIME["aria"]),
    ("flow-surfaces-orchestration", SURF, SURF["aria"]),
    ("flow-native-diffusion", DIFF, DIFF["aria"]),
    ("flow-evaluation-core.zh", EVAL_ZH, EVAL_ZH["aria"]),
    ("flow-model-runtime.zh", RUNTIME_ZH, RUNTIME_ZH["aria"]),
    ("flow-surfaces-orchestration.zh", SURF_ZH, SURF_ZH["aria"]),
    ("flow-native-diffusion.zh", DIFF_ZH, DIFF_ZH["aria"]),
]


BOUNDARY = {
    "aria": "Model execution and benchmark evaluation separated by the artifact boundary",
    "left_title": "Model execution",
    "right_title": "Benchmark / reporting",
    "boundary_label": "Artifact boundary",
    "cross_main": "GenerationResult",
    "cross_sub": "artifacts.jsonl",
    "left_caption": "model side never imports a benchmark evaluator",
    "right_caption": "benchmark side never loads a model checkpoint",
    "left_nodes": [
        {"title": "GenerationRequest", "role": "Input", "detail": "sample id, media, text/action, params"},
        {"title": "Pipeline + Operator", "role": "Runtime", "detail": "checkpoint load + native call"},
        {"title": "Native inference", "role": "Inference", "detail": "local runtime or hosted API"},
        {"title": "GenerationResult", "role": "Artifact", "detail": "status, timing, outputs, metadata"},
    ],
    "right_nodes": [
        {"title": "Metrics / benchmark runner", "role": "Evaluation", "detail": "reads artifacts, not checkpoints"},
        {"title": "per_sample + summary", "role": "Metrics", "detail": "per-sample rows + aggregates"},
        {"title": "scorecard.json + report.md", "role": "Report", "detail": "coverage, blockers, eligibility"},
    ],
}
BOUNDARY_ZH = {
    "aria": "模型执行与 benchmark 评测以 artifact 边界分离",
    "left_title": "模型执行",
    "right_title": "评测 / 报告",
    "boundary_label": "Artifact 边界",
    "cross_main": "GenerationResult",
    "cross_sub": "artifacts.jsonl",
    "left_caption": "模型侧不 import 任何 benchmark evaluator",
    "right_caption": "benchmark 侧不加载任何模型 checkpoint",
    "left_nodes": [
        {"title": "GenerationRequest", "role": "输入", "detail": "样本 id、媒体、text/action、参数"},
        {"title": "Pipeline + Operator", "role": "运行时", "detail": "checkpoint 加载 + 原生调用"},
        {"title": "Native inference", "role": "推理", "detail": "本地 runtime 或托管 API"},
        {"title": "GenerationResult", "role": "产物", "detail": "状态、耗时、输出、metadata"},
    ],
    "right_nodes": [
        {"title": "Metrics / benchmark runner", "role": "评测", "detail": "读 artifact，不读 checkpoint"},
        {"title": "per_sample + summary", "role": "指标", "detail": "逐样本行 + 聚合"},
        {"title": "scorecard.json + report.md", "role": "报告", "detail": "覆盖、blocker、合格性"},
    ],
}

BOUNDARY_DIAGRAMS = [
    ("boundary", BOUNDARY, BOUNDARY["aria"]),
    ("boundary.zh", BOUNDARY_ZH, BOUNDARY_ZH["aria"]),
]


def main() -> int:
    for name, data, aria in DIAGRAMS:
        out = OUT / f"{name}.svg"
        render(data["steps"], out, aria)
        print(f"wrote {out.relative_to(ROOT)}  ({len(data['steps'])} steps)")
    for name, data, aria in BOUNDARY_DIAGRAMS:
        out = OUT / f"{name}.svg"
        render_boundary(data, out, aria)
        print(f"wrote {out.relative_to(ROOT)}  (boundary)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
