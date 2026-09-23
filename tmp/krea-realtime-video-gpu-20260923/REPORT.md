# Krea Realtime Video local GPU validation

The public `KreaRealtimeVideoPipeline` loaded the local release checkpoint and completed a two-block text-to-video generation on GPU 3. `krea-realtime-toy-2blocks/status.json` and `result.json` record the public pipeline call and successful official runtime invocation. The output `generated.mp4` has 18 decodable frames at 832×480 and 16 fps; the runner recomputed its SHA-256 as `58a2480370eefc531dbac6910f217b5170e68cc83bca6ac79a8c91d701f3368b`.

Frames 0, 9, and 17 in `review/` were visually inspected. They show the requested turquoise plush frog on a white tray over a red tablecloth. The frog grows slightly in frame with coherent motion, consistent with the requested slow camera move. This supports the tested prompt and short duration, not general quality across prompts or longer streams.

The 28,577,102,160-byte local `krea-realtime-video-14b.safetensors` hashes to `792f6042a645e227f5d71851845dc4098a7338fec9ec55279f88ebbc3527eec7`, matching the public `krea/krea-realtime-video` LFS SHA-256. The in-tree release server now loads its auxiliary PyTorch state dictionaries with `weights_only=True`; this run exercised the release inference path after that fix.
