# AlayaLab Evoke (bundled runtime)

WorldFoundry bundles the official Evoke source needed by its inference adapter under this
directory. Runtime source is never downloaded or resolved from an external checkout.

- Upstream: <https://github.com/AlayaLab/Evoke>
- Revision: `74d268516d95c8fceadd2378f91a73f9f187042b`
- License: Apache-2.0; see `LICENSE` in this directory
- Vendored on: 2026-08-18

The copy excludes upstream demonstration media and example assets, which are not needed for
inference. It retains the inference/training Python source, launch scripts, configuration files,
requirements capture, README, and all upstream license/provenance records. WorldFoundry-owned
adapter code lives one directory above and invokes `scripts/inference/infer_single.py` without
modifying the vendored entrypoint.

## Nested third-party source

Evoke itself vendors several components. Their complete terms and provenance remain beside the
source under `evoke/third_party/`:

- Depth Anything 3 code: Apache-2.0, upstream commit `41736238`
- ViGeo code: Apache-2.0, upstream commit `78100ce`
- Pi3/Pi3X code: BSD-3-Clause, upstream commit `b56ef4b`
- VideoAlign/VideoReward code: MIT

No model weights are included in the WorldFoundry source distribution. The main
`AlayaLab/Evoke` snapshot is resolved locally at revision
`7fa34ecef85754fde6f08996b1ece9d195dcd2f4`. Camera-trajectory inference additionally uses
`pkqbajng/ViGeo1.1` at revision `49103e6eeab888bae974251d3578b496bec711d7`.
The optional Depth Anything 3 checkpoint is `depth-anything/da3-giant` at revision
`7cd62ae9315b9dff094d2d300e4ad012640607dd`.

The ViGeo1.1 and DA3 weights are licensed under CC-BY-NC-4.0, independently of their
Apache-2.0 implementation source. Those non-commercial weight terms remain applicable to users
who stage and run either checkpoint.
