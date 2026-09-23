# SolarWM Wan 5B short-route regression

The public WorldFoundry SolarWMPipeline loaded the pinned upstream source, local SolarWM-5B base and stage2-81f weights, then completed an 81-frame 864×480 video on GPU0. All frames decode; mean adjacent-frame RGB absolute difference is 4.148 (maximum 5.988). Viewed frames 0, 40 and 80 preserve the illustrated room, cat, person and stairway while the camera advances. This validates the short Wan DMD route only; the separately observed 237-frame live route still has a quality concern near its end.

Evidence: `status.json`, `preflight.json`, `result.json`, `demo.mp4`, `quality-review.json`, and exact frame samples in this directory.
