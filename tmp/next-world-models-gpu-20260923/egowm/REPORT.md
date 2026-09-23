# EgoWM local GPU validation

The public EgoWM pipeline used the pinned official source checkout, local 25-DoF navigation checkpoint, and Stable Video Diffusion base assets on GPU 3. It generated a decodable nine-frame 512×512 MP4 from the official real-world hallway image with a small synthetic navigation action scale. The sampled frames in `contact.jpg` keep the doors and wall coherent and show a modest viewpoint change. The direction is not clearly the requested forward motion, so action adherence remains unverified.

Evidence: `status.json`, `generated.mp4`, `generated.json`, `result-summary.json`, and `contact.jpg`. The source checkout is `tmp/official-sources-20260923/egowm` at revision `4a24b6fb917b8b86cea892312971f39691c532e8`. This is a short inference smoke, not a navigation benchmark.
