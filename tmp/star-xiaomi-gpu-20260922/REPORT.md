
## starvla-qwen3-vl-oft-libero-4in1

Real LIBERO10 observation,180degree camera rotation then224x224 resize,front/wrist order;no state follows upstream LIBERO example_dict;franka action normalization,8x7 chunk;offline integration only. Final backbone loading zero missing/unexpected/mismatched/errors,no meta parameters;outer policy strict load zero missing/unexpected. Regression outputs can exceed normalized range;no action clipping or robot execution.

## starvla-wm4a-wan2d2-oft-libero-4in1

Real LIBERO10 observation,180degree camera rotation then224x224 resize,front/wrist order;no state follows upstream LIBERO example_dict;franka action normalization,8x7 chunk;offline integration only. Final backbone loading zero missing/unexpected/mismatched/errors,no meta parameters;outer policy strict load zero missing/unexpected. Regression outputs can exceed normalized range;no action clipping or robot execution.

## xiaomi-robotics-0-libero

Real LIBERO10 dual-camera180degree rotated observations;raw8D xyz/axis-angle/two-finger state padded32 per official eval;5steps and10-frame32D padded actions;robot_type libero_all.Offline integration only,no closed-loop success. Final HF loading all-zero missing/unexpected/mismatched/errors,no meta parameters;32D tensor has7active dimensions and25padding dimensions below1e-5. Small gripper overshoot about0.021 recorded;no robot execution.
