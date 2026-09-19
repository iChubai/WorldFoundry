# Warp-as-History inference runtime

WorldFoundry packages the already integrated inference modules from
https://github.com/yyfz/Warp-as-History and its Helios dependencies from
https://github.com/PKU-YuanGroup/Helios. Both use Apache-2.0; see
LICENSE-Warp-as-History and LICENSE-Helios. Source-file notices also apply.

Pi3X source and checkpoints remain external. Camera-pose inputs require a Pi3
checkout available through WORLDFOUNDRY_PI3_SOURCE_ROOT (a directory containing
pi3/models/pi3x.py) or the ignored local point_clouds/pi3 directory. Precomputed
warp-video inputs do not require a Pi3 checkout. This distribution includes no
Pi3 source, training entrypoints, training configs, or checkpoints.
