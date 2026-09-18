# Zing-0.5 inference integration

Upstream: https://github.com/seedleap/zing-world-model

Revision: `11212da06f63290f4d354cdc0324241ba6eac356`

Native Wan2.2 TI2V-5B layers, UMT5, VAE38 and attention are reused. Only the keyboard convolution and prompt-switching causal cache are model-specific.

No training loops, losses, optimizers, datasets or training configuration files are vendored by this integration. GPU artifact validation is pending.
