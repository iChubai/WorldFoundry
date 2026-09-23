"""Check the numeric and structural outputs of the real WorldMirror run."""

import json
from pathlib import Path

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "output"


def ply_summary(path):
    dtype_map = {"float": "<f4", "double": "<f8", "uchar": "u1", "int": "<i4"}
    properties = []
    with path.open("rb") as stream:
        assert stream.readline() == b"ply\n"
        assert stream.readline() == b"format binary_little_endian 1.0\n"
        count = None
        while True:
            line = stream.readline().decode("ascii").strip()
            if line.startswith("element vertex "):
                count = int(line.split()[-1])
            elif line.startswith("property "):
                _, kind, name = line.split()
                properties.append((name, dtype_map[kind]))
            elif line == "end_header":
                offset = stream.tell()
                break
    assert count is not None and count > 0
    dtype = np.dtype(properties)
    assert path.stat().st_size == offset + count * dtype.itemsize
    vertices = np.memmap(path, dtype=dtype, mode="r", offset=offset, shape=(count,))
    for name in ("x", "y", "z"):
        assert np.isfinite(vertices[name]).all(), (path, name)
    return {"vertices": count, "size_bytes": path.stat().st_size,
            "xyz_min": [float(vertices[name].min()) for name in ("x", "y", "z")],
            "xyz_max": [float(vertices[name].max()) for name in ("x", "y", "z")]}


def main():
    depth = []
    for path in sorted((OUTPUT / "depth").glob("*.npy")):
        values = np.load(path)
        assert values.shape == (294, 518) and np.isfinite(values).all()
        assert float(values.min()) > 0 and float(values.std()) > 0.01
        depth.append({"frame": path.stem, "min": float(values.min()),
                      "median": float(np.median(values)), "max": float(values.max()),
                      "std": float(values.std())})
    normal = []
    for path in sorted((OUTPUT / "normal").glob("*.png")):
        values = np.asarray(Image.open(path))
        assert values.shape == (294, 518, 3) and float(values.std()) > 1
        normal.append({"frame": path.stem, "std": float(values.std())})
    camera = json.loads((OUTPUT / "camera_params.json").read_text())
    assert camera["num_cameras"] == len(depth) == len(normal) == 4
    assert len(camera["extrinsics"]) == len(camera["intrinsics"]) == 4
    for item in camera["extrinsics"]:
        matrix = np.asarray(item["matrix"])
        assert matrix.shape == (4, 4) and np.isfinite(matrix).all()
        assert abs(np.linalg.det(matrix[:3, :3]) - 1) < 0.01
        assert np.allclose(matrix[3], [0, 0, 0, 1])
    for item in camera["intrinsics"]:
        matrix = np.asarray(item["matrix"])
        assert matrix.shape == (3, 3) and np.isfinite(matrix).all()
        assert matrix[0, 0] > 0 and matrix[1, 1] > 0
    summary = {"depth": depth, "normal": normal, "num_cameras": 4,
               "points": ply_summary(OUTPUT / "points.ply"),
               "gaussians": ply_summary(OUTPUT / "gaussians.ply")}
    (ROOT / "output-check.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
