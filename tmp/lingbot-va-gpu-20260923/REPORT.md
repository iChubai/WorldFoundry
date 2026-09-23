# LingBot-VA local GPU validation

The posttrained LIBERO-Long and RoboTwin checkpoints loaded through the public pipeline and in-tree official websocket server/client on GPU 1. Both returned finite action arrays: LIBERO-Long `[7,4,4]` (112 scalars) and RoboTwin `[16,2,16]` (512 scalars). Evidence: `{libero-long-synthetic,robotwin-synthetic}/{status.json,action_trace.json,lingbot_va_server.log}`.

The input used the same real RGB photo in each camera slot and a synthetic zero robot state. These checks establish server/client integration and action structure only. They do not demonstrate a successful robot rollout or task instruction following.

The base checkpoint also loaded through the same public/server path using the RoboTwin server configuration and produced a finite `[16,2,16]` action array (512 scalars). Its first attempt used the `robotwin_i2av` standalone config, which tried to read an unstaged `example/robotwin` image. The validation harness selected the server config for the successful retest. Evidence: `base-synthetic/{status.json,action_trace.json,lingbot_va_server.log}`. This remains an integration and structure check with synthetic state, not a robot-task result.
