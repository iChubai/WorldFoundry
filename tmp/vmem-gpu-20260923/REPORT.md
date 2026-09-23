# VMem local GPU validation

The public VMem pipeline generated a six-frame 576×576 forward-navigation clip on GPU 0 from a real RGB photo and a forward command. All frames decoded, the output pixel range was `[0,255]`, and the sampled contact sheet shows a coherent plush toy and plate that grow in frame as the view advances. There is some visual drift and background distortion. Evidence: `vmem-forward-fast/{status.json,generated.mp4,contact.jpg}` and `vmem-forward-fast-retest5.log`.

This run exposed and drove fixes for component option forwarding, a missing CUT3R pose-distance helper, and fp16 camera geometry operations. The final run completed generation after those fixes. It establishes this one short navigation configuration, not long-horizon consistency.
