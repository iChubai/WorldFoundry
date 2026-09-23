"""Decode all ATI output frames and record structural/motion statistics."""

import json
import sys
from pathlib import Path

import cv2
import numpy as np


def main(path):
    capture = cv2.VideoCapture(str(path))
    assert capture.isOpened(), path
    frames = []
    while True:
        okay, frame = capture.read()
        if not okay:
            break
        assert frame.ndim == 3 and frame.shape[2] == 3
        assert np.isfinite(frame).all()
        frames.append(frame)
    capture.release()
    assert len(frames) == 81, len(frames)
    assert all(frame.shape == frames[0].shape for frame in frames)
    differences = [float(np.abs(a.astype(np.float32) - b.astype(np.float32)).mean())
                   for a, b in zip(frames, frames[1:])]
    summary = {
        "artifact": str(path), "frames_decoded": len(frames),
        "height": frames[0].shape[0], "width": frames[0].shape[1],
        "first_last_mae": float(np.abs(frames[0].astype(np.float32) - frames[-1].astype(np.float32)).mean()),
        "adjacent_mae_min": min(differences), "adjacent_mae_median": float(np.median(differences)),
        "adjacent_mae_max": max(differences),
        "pixel_std_first": float(frames[0].std()), "pixel_std_middle": float(frames[40].std()),
        "pixel_std_last": float(frames[-1].std()),
    }
    output = Path(path).with_suffix(".quality.json")
    output.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main(Path(sys.argv[1]))
