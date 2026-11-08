# SPDX-FileCopyrightText: 2026 OpenEnvision
# SPDX-License-Identifier: Apache-2.0
"""Terminal UI for browsing the catalog and building ``wf`` commands.

This package root stays import-light: ``wf --help`` registers the ``tui``
subcommand without loading Textual, which is imported only when the TUI runs.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def register_tui_subparser(subparsers: argparse._SubParsersAction) -> None:
    """Register ``wf tui`` and its generation flags."""
    tui_parser = subparsers.add_parser(
        "tui",
        help="Browse models and launch runs in the terminal UI",
    )
    tui_parser.add_argument("--model-manifest-dir", type=Path)
    tui_parser.add_argument("--benchmark-manifest-dir", type=Path)
    tui_parser.add_argument("--runtime-profile-dir", type=Path)
    tui_parser.add_argument("--suite", action="append", dest="suite_ids", default=None)
    tui_parser.add_argument("--model-id")
    tui_parser.add_argument("--benchmark-id")
    tui_parser.add_argument("--metric", action="append", default=None)
    tui_parser.add_argument("--output-dir", type=Path)
    tui_parser.add_argument("--input", type=Path)
    tui_parser.add_argument("--input-dir", "--input_dir", dest="input_dir", type=Path)
    tui_parser.add_argument("--video", type=Path)
    tui_parser.add_argument("--trajectory-file", type=Path)
    tui_parser.add_argument("--prompt")
    tui_parser.add_argument("--negative-prompt", dest="negative_prompt")
    tui_parser.add_argument("--task")
    tui_parser.add_argument("--mode")
    tui_parser.add_argument("--resize-mode")
    tui_parser.add_argument("--size")
    tui_parser.add_argument("--frames")
    tui_parser.add_argument("--steps")
    tui_parser.add_argument("--frames-per-generation", "--chunk-frames", dest="frames_per_generation")
    tui_parser.add_argument("--guidance-scale", "--cfg-scale", dest="guidance_scale")
    tui_parser.add_argument("--seed")
    tui_parser.add_argument("--fps")
    tui_parser.add_argument("--dtype")
    tui_parser.add_argument("--max-sequence-length", dest="max_sequence_length")
    tui_parser.add_argument("--cam-type", "--camera-type", dest="cam_type")
    tui_parser.add_argument("--interactions", "--actions", dest="interactions")
    tui_parser.add_argument("--output-formats", "--output_formats", dest="output_formats")
    tui_parser.add_argument("--trajectory", "--camera-trajectory", dest="trajectory")
    tui_parser.add_argument("--angle")
    tui_parser.add_argument("--distance")
    tui_parser.add_argument("--orbit-radius", dest="orbit_radius")
    tui_parser.add_argument("--zoom-ratio", dest="zoom_ratio")
    tui_parser.add_argument("--alpha-threshold", dest="alpha_threshold")
    tui_parser.add_argument("--static-scene", action="store_true", default=None)
    tui_parser.add_argument("--low-vram", action="store_true", default=None)
    tui_parser.add_argument("--disable-lora", action="store_true", default=None)
    tui_parser.add_argument("--vis-rendering", action="store_true", default=None)
    tui_parser.add_argument("--offload-t5", "--offload_t5", dest="offload_t5", action="store_true", default=None)
    tui_parser.add_argument(
        "--offload-transformer-during-vae",
        "--offload_transformer_during_vae",
        dest="offload_transformer_during_vae",
        action="store_true",
        default=None,
    )
    tui_parser.add_argument("--offload-vae", "--offload_vae", dest="offload_vae", action="store_true", default=None)
    tui_parser.add_argument("--output-path", type=Path)
    tui_parser.add_argument("--ckpt-type", "--weight-type", dest="ckpt_type")
    tui_parser.add_argument("--ckpt-root", type=Path)
    tui_parser.add_argument("--ckpt-path", type=Path)
    tui_parser.add_argument("--conda-envs-root", type=Path)
    tui_parser.add_argument("--gpu")
    tui_parser.add_argument("--fallback", action="store_true")
    tui_parser.add_argument("--catalog-json", action="store_true")
    tui_parser.add_argument("--print-command", action="store_true")
    tui_parser.add_argument("--write-suite-plan", action="store_true")
    tui_parser.set_defaults(func=handle_tui)


def handle_tui(args: argparse.Namespace) -> int:
    """Translate root CLI arguments into the TUI command namespace."""
    from .launcher import main as tui_main

    # TUI accepts the same high-level options as the CLI entrypoint; this bridge
    # normalizes generation flags into one argument list.
    argv: list[str] = []
    if args.model_manifest_dir is not None:
        argv.extend(["--model-manifest-dir", str(args.model_manifest_dir)])
    if args.benchmark_manifest_dir is not None:
        argv.extend(["--benchmark-manifest-dir", str(args.benchmark_manifest_dir)])
    if args.runtime_profile_dir is not None:
        argv.extend(["--runtime-profile-dir", str(args.runtime_profile_dir)])
    for suite_id in args.suite_ids or ():
        argv.extend(["--suite", str(suite_id)])
    if args.model_id is not None:
        argv.extend(["--model-id", str(args.model_id)])
    if args.benchmark_id is not None:
        argv.extend(["--benchmark-id", str(args.benchmark_id)])
    for metric in getattr(args, "metric", None) or ():
        argv.extend(["--metric", str(metric)])
    if args.output_dir is not None:
        argv.extend(["--output-dir", str(args.output_dir)])
    if getattr(args, "input", None) is not None:
        argv.extend(["--input", str(args.input)])
    if getattr(args, "input_dir", None) is not None:
        argv.extend(["--input-dir", str(args.input_dir)])
    if getattr(args, "video", None) is not None:
        argv.extend(["--video", str(args.video)])
    if getattr(args, "trajectory_file", None) is not None:
        argv.extend(["--trajectory-file", str(args.trajectory_file)])
    for attr, flag in (
        ("prompt", "--prompt"),
        ("negative_prompt", "--negative-prompt"),
        ("task", "--task"),
        ("mode", "--mode"),
        ("resize_mode", "--resize-mode"),
        ("size", "--size"),
        ("frames", "--frames"),
        ("steps", "--steps"),
        ("frames_per_generation", "--frames-per-generation"),
        ("guidance_scale", "--guidance-scale"),
        ("seed", "--seed"),
        ("fps", "--fps"),
        ("dtype", "--dtype"),
        ("max_sequence_length", "--max-sequence-length"),
        ("cam_type", "--cam-type"),
        ("interactions", "--interactions"),
        ("output_formats", "--output-formats"),
        ("trajectory", "--trajectory"),
        ("angle", "--angle"),
        ("distance", "--distance"),
        ("orbit_radius", "--orbit-radius"),
        ("zoom_ratio", "--zoom-ratio"),
        ("alpha_threshold", "--alpha-threshold"),
    ):
        value = getattr(args, attr, None)
        if value:
            argv.extend([flag, str(value)])
    for attr, flag in (
        ("static_scene", "--static-scene"),
        ("low_vram", "--low-vram"),
        ("disable_lora", "--disable-lora"),
        ("vis_rendering", "--vis-rendering"),
        ("offload_t5", "--offload-t5"),
        ("offload_transformer_during_vae", "--offload-transformer-during-vae"),
        ("offload_vae", "--offload-vae"),
    ):
        if getattr(args, attr, False):
            argv.append(flag)
    if getattr(args, "output_path", None) is not None:
        argv.extend(["--output-path", str(args.output_path)])
    if getattr(args, "ckpt_type", None):
        argv.extend(["--ckpt-type", str(args.ckpt_type)])
    if getattr(args, "ckpt_root", None) is not None:
        argv.extend(["--ckpt-root", str(args.ckpt_root)])
    if getattr(args, "ckpt_path", None) is not None:
        argv.extend(["--ckpt-path", str(args.ckpt_path)])
    if getattr(args, "conda_envs_root", None) is not None:
        argv.extend(["--conda-envs-root", str(args.conda_envs_root)])
    if getattr(args, "gpu", None):
        argv.extend(["--gpu", str(args.gpu)])
    if args.fallback:
        argv.append("--fallback")
    if args.catalog_json:
        argv.append("--catalog-json")
    if args.print_command:
        argv.append("--print-command")
    if getattr(args, "write_suite_plan", False):
        argv.append("--write-suite-plan")
    return tui_main(argv)
