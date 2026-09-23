# GR00T N1.7 LIBERO offline validation

All four local LIBERO weights(libero_10,goal,object,spatial) completed fresh CUDA inference using the public GR00TPipeline. Actual checkpoint paths are checked. Each emits7 action groups ofshape(1,16,1),finite and within checkpoint q01/q99 bounds for this observation. Complete HuggingFace loading_info has zero missing/unexpected/mismatched keys for each checkpoint; low-level per-shard load_state_dict logs show intermediate incomplete loads and must not be mistaken for final missing weights.

Fixed two stale preprocessing imports from absent config.py to canonical types.py. Fixed the runtime camera mapping so image/wrist_image remain distinct,and accept canonical state mapping alongside joint_state/proprio. CPU regression checks exact camera/state preservation,missing-camera rejection and legacy state support. GPU action traces confirm both distinct image sources.

Input is real lerobot/libero episode0 frame0. Pinned official GR00T LIBERO adapter specifies both cameras rotated180 and xyz+axis-angle+two finger positions. The LIBERO10 checkpoint is in-domain;goal/object/spatial probes use that same OOD observation and verify integration only. None establishes simulator task success. Current runtime validation covers single-timestep LIBERO observations; history-dependent embodiments remain untested.
