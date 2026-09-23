
## wan22-fun-a14b-cam

After Wan VAE root-offload fix: all81frames832x48024fps,50scheduler steps; TeaCache default0.1 after5steps, not50uncached DiT calls. Actual final4components have zero missing/unexpected keys; raw decoder finite before clamp; complete decode matches dimensions/fps. Actual contact sheet reviewed: room/car identity retained, coherent forward travel with foreground expansion and steady scene lighting, consistent with explicitly synthetic camera-to-world positive-z trajectory. Single input/seed/trajectory only; no quantitative pose accuracy. Original GPU1 attempt ran out of memory while external training held56.3GiB. This independent full retry succeeded on GPU3; previous resource-conflict evidence is preserved. Two-stage high/low DiTs both loaded exactly.
