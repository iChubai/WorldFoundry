from __future__ import annotations

import argparse
import json
from pathlib import Path

from worldfoundry.core.io.paths import checkpoint_root_path, resolve_worldfoundry_path


DEFAULT_OUTPUT = "${WORLDFOUNDRY_CKPT_DIR}/leworldmodel/pusht_worldfoundry_smoke.h5"


def _public_path(path: Path) -> str:
    checkpoint_root = checkpoint_root_path().resolve()
    try:
        suffix = path.resolve().relative_to(checkpoint_root)
    except ValueError:
        return path.name
    return f"${{WORLDFOUNDRY_CKPT_DIR}}/{suffix.as_posix()}"


def prepare_dataset(*, output: Path, steps: int, seed: int, overwrite: bool) -> dict[str, object]:
    if steps < 31:
        raise ValueError("steps must be at least 31 for the default 25-step evaluation budget")
    if output.exists() and not overwrite:
        return {
            "status": "reused",
            "dataset": _public_path(output),
            "bytes": output.stat().st_size,
        }
    if output.exists():
        output.unlink()

    import stable_worldmodel as swm

    output.parent.mkdir(parents=True, exist_ok=True)
    world = swm.World(
        env_name="swm/PushT-v1",
        num_envs=1,
        image_shape=(224, 224),
        max_episode_steps=steps,
    )
    try:
        world.set_policy(swm.policy.RandomPolicy(seed=seed))
        world.collect(
            path=output,
            episodes=1,
            seed=seed,
            format="hdf5",
            progress=False,
        )
    finally:
        world.close()

    return {
        "status": "created",
        "dataset": _public_path(output),
        "bytes": output.stat().st_size,
        "episodes": 1,
        "steps": steps,
        "seed": seed,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect a deterministic PushT trajectory for LeWorldModel smoke evaluation")
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--steps", type=int, default=60)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = prepare_dataset(
        output=resolve_worldfoundry_path(args.output),
        steps=args.steps,
        seed=args.seed,
        overwrite=args.overwrite,
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
