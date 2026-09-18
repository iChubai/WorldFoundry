# AlayaWorld v1.1 inference integration

Upstream: https://github.com/AlayaLab/AlayaWorld

Revision: `ea03cfbb2e4c4e9102ed8ea8562e0b5370ca9b79`

The inference-only rollout methods and three small configuration adapters are extracted from the pinned upstream and paired with WorldFoundry checkpoint/LoRA loaders, LTX VAE, Gemma encoding and the existing Alaya history encoder. An inference-only entry skips trainer setup, optimizers, critics and training data.

No training loops, losses, optimizers, datasets or training configuration files are vendored by this integration. GPU artifact validation is pending.
