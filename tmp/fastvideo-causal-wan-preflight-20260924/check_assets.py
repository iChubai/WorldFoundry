"""Read checkpoint indexes and actual safetensors payload slices."""

import json
from pathlib import Path

import torch
from safetensors import safe_open


ROOT = Path(__file__).resolve().parents[2]
CKPT = ROOT.parent / "ckpts/FastVideo--CausalWan2.2-I2V-A14B-Preview-Diffusers"
metadata = json.loads((CKPT / ".hfd/repo_metadata.json").read_text())
official = {entry["rfilename"]: entry for entry in metadata["siblings"]}
components = ["text_encoder", "transformer", "transformer_2", "vae"]
result = {"repo_id": metadata["id"], "revision": metadata["sha"], "components": {}}


def sample(handle, name):
    sl = handle.get_slice(name)
    shape = sl.get_shape()
    if not shape:
        values = sl[:]
    else:
        ranges = []
        for size in shape[:-1]:
            ranges.append(slice(size // 2, size // 2 + 1))
        ranges.append(slice(max(0, shape[-1] // 2 - 4), min(shape[-1], shape[-1] // 2 + 4)))
        values = sl[tuple(ranges)]
    values = values.float().reshape(-1)
    assert values.numel() > 0 and bool(torch.isfinite(values).all()), name
    return {"name": name, "shape": list(shape), "sample_count": values.numel(),
            "sample_abs_mean": float(values.abs().mean()), "sample_max": float(values.max())}


for component in components:
    folder = CKPT / component
    index_files = sorted(folder.glob("*.safetensors.index.json"))
    index = json.loads(index_files[0].read_text()) if index_files else None
    weights = sorted(folder.glob("*.safetensors"))
    observed = set()
    reports = []
    for path in weights:
        rel = str(path.relative_to(CKPT))
        expected = official.get(rel, {}).get("size")
        assert expected is not None, rel
        actual = path.stat().st_size
        assert actual == expected, (rel, actual, expected)
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            keys = list(handle.keys())
            assert keys, rel
            chosen = list(dict.fromkeys((keys[0], keys[len(keys) // 2], keys[-1])))
            samples = [sample(handle, key) for key in chosen]
            observed.update(keys)
        reports.append({"path": rel, "size": actual, "sidecar": Path(str(path) + ".aria2").exists(),
                        "tensor_count": len(keys), "samples": samples})
    if index:
        weight_map = index["weight_map"]
        assert set(weight_map) == observed, (component, len(set(weight_map)), len(observed))
        index_filenames_match = {p.name for p in weights} == set(weight_map.values())
    else:
        index_filenames_match = None
    result["components"][component] = {"index": str(index_files[0].relative_to(CKPT)) if index else None,
                                      "index_tensor_count": len(index["weight_map"]) if index else None,
                                      "index_filenames_match": index_filenames_match,
                                      "index_referenced_files": sorted(set(index["weight_map"].values())) if index else [],
                                      "files": reports}

dest = Path(__file__).with_name("actual-read.json")
dest.write_text(json.dumps(result, indent=2), encoding="utf-8")
print(dest)
