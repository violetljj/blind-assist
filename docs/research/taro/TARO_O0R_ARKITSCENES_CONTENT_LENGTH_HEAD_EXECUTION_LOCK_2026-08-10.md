# TARO O0R ARKitScenes Content-Length HEAD Execution Lock

状态：`WILD_LAB / AUTHORIZED_UNCONSUMED / HEAD_NOT_RUN / SOURCE_UNOPENED`

本锁只授权一次精确的 HEAD-only preflight：24 个冻结 Training video ×
`upsampling.zip / lowres_wide_intrinsics.zip / lowres_wide.traj`，共 72 个 URL。request-plan SHA-256
为 `C0F0D41E381333BEFF4C2C0EC4678BC75A22C86173E11B3150F7F7FFBCB18927`。

执行固定使用 8 workers、20 秒/attempt、最多 3 attempts；redirect 与 response body 均禁止，压缩总量上限
20 GiB。exclusive HEAD root 一旦创建即消费本 evidence version；之后即使 transport、写盘或 timeout 失败，
也必须记为 `ONE_SHOT_CONSUMED`，不得写成未启动、不得覆盖或重跑。

本锁已绑定实现提交 `9c5251035944505705bbae332847ad3123105dfb`、preflight、用户授权 receipt、
implementation lock、materializer 与 HEAD runner。它必须先提交到版本历史，之后才允许执行冻结 argv。

本锁不授权 GET、source body、truth、uncertainty fit、DepthART、factorial、训练、设备、产品或 safety。
HEAD PASS 后仍须另冻并提交 truth-only one-shot execution lock；HEAD 非 PASS 时不得替换 parent 或重跑。
