#!/usr/bin/env python3
"""Run full-quality Workspace video models that require one exclusive GPU gang.

The ordinary video matrix intentionally assigns one independent model to each
GPU.  This companion driver handles models whose official runtime launches a
coordinated torchrun job across every selected GPU.  It waits for a stably idle
gang, preserves the catalog's quality defaults, overrides only the process
topology, and atomically records the submitted payload and terminal result.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

TERMINAL_STATUSES = {"completed", "failed", "cancelled"}
DEFAULT_MODELS = ("lingbot-world-v2", "wan2.1-vace")
MODEL_CHECKPOINT_DIRS = {
    "lingbot-world-v2": "lingbot-world-v2-14b-causal-fast",
    "wan2.1-vace": "Wan-AI--Wan2.1-VACE-14B",
}


def _request_json(
    base_url: str,
    method: str,
    path: str,
    payload: dict[str, Any] | None = None,
) -> Any:
    body = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}{path}",
        data=body,
        method=method,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=900) as response:
            return json.load(response)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"Workspace {method} {path} returned HTTP {exc.code}: {detail}"
        ) from exc


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _gpu_activity() -> dict[int, dict[str, Any]]:
    gpu_result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
    )
    app_result = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
    )
    activity: dict[int, dict[str, Any]] = {}
    uuid_to_index: dict[str, int] = {}
    for line in gpu_result.stdout.splitlines():
        index_text, uuid, memory_text, utilization_text = (
            part.strip() for part in line.split(",", 3)
        )
        index = int(index_text)
        activity[index] = {
            "memory_mib": int(memory_text),
            "utilization_percent": int(utilization_text),
            "compute_pids": [],
        }
        uuid_to_index[uuid] = index
    for line in app_result.stdout.splitlines():
        if not line.strip():
            continue
        uuid, pid_text = (part.strip() for part in line.split(",", 1))
        index = uuid_to_index.get(uuid)
        if index is not None:
            activity[index]["compute_pids"].append(int(pid_text))
    return activity


def _is_idle(
    state: dict[str, Any] | None,
    *,
    memory_limit_mib: int,
    utilization_limit_percent: int,
) -> bool:
    return bool(state) and (
        int(state.get("memory_mib") or 0) <= memory_limit_mib
        and int(state.get("utilization_percent") or 0)
        <= utilization_limit_percent
        and not state.get("compute_pids")
    )


def _wait_for_idle_gang(
    gpus: list[int],
    *,
    samples: int,
    poll_seconds: float,
    memory_limit_mib: int,
    utilization_limit_percent: int,
) -> None:
    consecutive = 0
    last_report = 0.0
    while consecutive < samples:
        activity = _gpu_activity()
        if all(
            _is_idle(
                activity.get(gpu),
                memory_limit_mib=memory_limit_mib,
                utilization_limit_percent=utilization_limit_percent,
            )
            for gpu in gpus
        ):
            consecutive += 1
        else:
            consecutive = 0
        if time.monotonic() - last_report >= 30:
            states = ", ".join(f"gpu={gpu} {activity.get(gpu, {})}" for gpu in gpus)
            print(
                f"waiting for exclusive gang: samples={consecutive}/{samples}; {states}",
                flush=True,
            )
            last_report = time.monotonic()
        if consecutive < samples:
            time.sleep(max(1.0, poll_seconds))


def _checkpoint_path(checkpoint_root: Path, model_id: str) -> Path:
    path = checkpoint_root / MODEL_CHECKPOINT_DIRS[model_id]
    if not path.is_dir():
        raise FileNotFoundError(f"checkpoint directory is unavailable: {path}")
    # Preserve the user-facing checkpoint-root path in manifests even when a
    # particular model directory is a symlink to the persistent weight store.
    return path.absolute()


def _call_kwargs(model_id: str, world_size: int) -> dict[str, Any]:
    if world_size <= 1 or 40 % world_size:
        raise ValueError(
            f"the selected world size must be a divisor of 40 and greater than one; got {world_size}"
        )
    if model_id == "lingbot-world-v2":
        return {"nproc_per_node": world_size}
    if model_id == "wan2.1-vace":
        return {
            "nproc_per_node": world_size,
            "ulysses_size": world_size,
            "ring_size": 1,
        }
    raise ValueError(f"unsupported gang model: {model_id}")


def _submission_payload(
    model_id: str,
    *,
    checkpoint_root: Path,
    world_size: int,
) -> dict[str, Any]:
    return {
        "job_type": "inference",
        "model_id": model_id,
        # The Workspace subprocess launcher exposes the complete selected node
        # topology to torchrun when the request uses the unpinned CUDA device.
        "device": "cuda",
        "model_ref": str(_checkpoint_path(checkpoint_root, model_id)),
        "call_kwargs": _call_kwargs(model_id, world_size),
    }


def _compact_job(job: dict[str, Any]) -> dict[str, Any]:
    return {
        key: job.get(key)
        for key in (
            "id",
            "job_id",
            "model_id",
            "status",
            "created_at",
            "started_at",
            "completed_at",
            "finished_at",
            "elapsed",
            "error",
            "result",
        )
    }


def _validate_manifest(
    manifest_path: Path,
    *,
    output_dir: Path,
    model_id: str,
) -> list[dict[str, Any]]:
    validation_path = output_dir / f"strict_video_validation_{model_id}.jsonl"
    command = [
        os.fspath(Path(sys.executable)),
        os.fspath(Path(__file__).with_name("validate_generated_videos.py")),
        "--output",
        os.fspath(validation_path),
        os.fspath(manifest_path),
    ]
    completed = subprocess.run(command, check=False, text=True, timeout=1800)
    if completed.returncode:
        raise RuntimeError(
            f"strict video validation exited with status {completed.returncode}"
        )
    return [
        json.loads(line)
        for line in validation_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace-url", default="http://127.0.0.1:7870")
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", action="append", choices=DEFAULT_MODELS)
    parser.add_argument("--gpu", action="append", type=int, dest="gpus")
    parser.add_argument("--poll-seconds", type=float, default=10.0)
    parser.add_argument("--job-timeout-seconds", type=float, default=14400.0)
    parser.add_argument("--gpu-idle-samples", type=int, default=3)
    parser.add_argument("--gpu-idle-memory-mib", type=int, default=32)
    parser.add_argument("--gpu-idle-utilization-percent", type=int, default=0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    models = list(dict.fromkeys(args.model or DEFAULT_MODELS))
    gpus = list(dict.fromkeys(args.gpus or [0, 1, 2, 3]))
    checkpoint_root = args.checkpoint_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "gang.json"
    report: dict[str, Any] = {
        "schema_version": "worldfoundry-workspace-video-gang-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "workspace_url": args.workspace_url,
        "checkpoint_root": str(checkpoint_root),
        "gpus": gpus,
        "models": [],
    }
    _atomic_json(report_path, report)

    catalog_ids = {
        str(row.get("id") or "")
        for row in _request_json(args.workspace_url, "GET", "/api/models")
    }
    missing_models = [model_id for model_id in models if model_id not in catalog_ids]
    if missing_models:
        raise RuntimeError(
            f"models are unavailable from the Workspace catalog: {', '.join(missing_models)}"
        )

    failed = False
    for model_id in models:
        _wait_for_idle_gang(
            gpus,
            samples=max(1, args.gpu_idle_samples),
            poll_seconds=args.poll_seconds,
            memory_limit_mib=max(0, args.gpu_idle_memory_mib),
            utilization_limit_percent=max(
                0, args.gpu_idle_utilization_percent
            ),
        )
        payload = _submission_payload(
            model_id,
            checkpoint_root=checkpoint_root,
            world_size=len(gpus),
        )
        job = _request_json(args.workspace_url, "POST", "/api/jobs", payload)
        row: dict[str, Any] = {
            "model_id": model_id,
            "status": job.get("status"),
            "job_id": job.get("id"),
            "payload": payload,
            "peak_memory_mib_by_gpu": {str(gpu): 0 for gpu in gpus},
        }
        report["models"].append(row)
        report["updated_at"] = datetime.now(timezone.utc).isoformat()
        _atomic_json(report_path, report)
        print(f"model={model_id} job={job['id']} submitted on gpus={gpus}", flush=True)

        started = time.monotonic()
        last_status = ""
        while True:
            activity = _gpu_activity()
            for gpu in gpus:
                memory_mib = int((activity.get(gpu) or {}).get("memory_mib") or 0)
                key = str(gpu)
                row["peak_memory_mib_by_gpu"][key] = max(
                    int(row["peak_memory_mib_by_gpu"][key]), memory_mib
                )
            job = _request_json(
                args.workspace_url,
                "GET",
                f"/api/jobs/{row['job_id']}",
            )
            status = str(job.get("status") or "")
            row["status"] = status
            row["elapsed_seconds_observed"] = round(time.monotonic() - started, 3)
            report["updated_at"] = datetime.now(timezone.utc).isoformat()
            _atomic_json(report_path, report)
            if status != last_status:
                print(
                    f"model={model_id} job={row['job_id']} status={status}",
                    flush=True,
                )
                last_status = status
            if status in TERMINAL_STATUSES:
                break
            if time.monotonic() - started > args.job_timeout_seconds:
                _request_json(
                    args.workspace_url,
                    "POST",
                    f"/api/jobs/{row['job_id']}/stop",
                    {},
                )
                row["status"] = "timeout"
                row["error"] = (
                    f"job exceeded {args.job_timeout_seconds:.0f} seconds"
                )
                break
            time.sleep(max(1.0, args.poll_seconds))

        row["job"] = _compact_job(job)
        row["error"] = row.get("error") or job.get("error")
        row["peak_memory_mib"] = max(row["peak_memory_mib_by_gpu"].values())
        if row["status"] == "completed":
            result = job.get("result") or {}
            manifest_path = Path(str(result.get("manifest_path") or ""))
            if not manifest_path.is_file():
                row["status"] = "failed_validation"
                row["error"] = f"manifest is unavailable: {manifest_path}"
            else:
                try:
                    row["strict_validation"] = _validate_manifest(
                        manifest_path,
                        output_dir=output_dir,
                        model_id=model_id,
                    )
                    if not row["strict_validation"] or not all(
                        item.get("ok") for item in row["strict_validation"]
                    ):
                        row["status"] = "failed_validation"
                        row["error"] = "strict video validation did not pass"
                except Exception as exc:  # noqa: BLE001 - persist every failure.
                    row["status"] = "failed_validation"
                    row["error"] = f"{type(exc).__name__}: {exc}"
        if row["status"] != "completed":
            failed = True
        report["updated_at"] = datetime.now(timezone.utc).isoformat()
        _atomic_json(report_path, report)
        print(
            f"model={model_id} final_status={row['status']} "
            f"peak_memory_mib={row['peak_memory_mib']}",
            flush=True,
        )

    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
