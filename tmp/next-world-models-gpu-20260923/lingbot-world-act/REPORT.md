# LingBot-World Base-Act GPU validation

The public `LingBotPipeline` selected `model_id=lingbot-world-act` and loaded the local Base-Act preview checkpoint in the dedicated LingBot environment on GPU 1. A real abandoned-room image, a forward camera trajectory, and an explicit 48×4 action matrix (first channel +0.05) drove a 49-frame, 40-step, seed-42 run. The returned raw video tensor was finite with shape `[49,464,848,3]`, and all 49 MP4 frames decoded. The SHA-256 is in `status.json`.

The viewed five-frame contact shows a coherent room and clear forward viewpoint change. Some foreground objects change as the camera advances. Because camera trajectory and nonzero action were supplied together, this run establishes that Base-Act accepts both controls and produces video, but does not isolate or measure action adherence. Evidence: `status.json`, `generated.mp4`, `contact.jpg`, and the run log in the parent directory.
