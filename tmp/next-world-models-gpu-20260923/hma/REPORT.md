# HMA local GPU validation

The public HMA pipeline loaded the pinned official source checkout, local continuous checkpoint, and SVD temporal VAE on GPU 0. A real LangTable prompt image and a small synthetic rightward action produced a decodable five-frame 256×256 MP4. The sampled `contact.jpg` retains the colored blocks and robot gripper, with modest change across frames. This establishes short inference and output continuity; the tiny sample does not establish accurate action response or task success.

Evidence: `status.json`, `generated.mp4`, `generated.json`, `result-summary.json`, and `contact.jpg`. The source checkout is `tmp/official-sources-20260923/HMA` at revision `e3c89088fa82f2c208af6796485b4ec65ab318c5`. The missing `mup` Python dependency was staged under ignored `tmp/official-sources-20260923/deps` for this validation run.
