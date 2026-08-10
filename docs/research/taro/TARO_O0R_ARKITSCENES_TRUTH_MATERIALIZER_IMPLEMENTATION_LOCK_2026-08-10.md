# TARO O0R ARKitScenes truth materializer implementation lock

状态：`IMPLEMENTATION_LOCK_PASS / SYNTHETIC_ONLY / HEAD_NOT_RUN / SOURCE_UNOPENED / SCIENTIFIC_NOT_RUN / EXECUTION_NOT_AUTHORIZED`

机器真源：
[TARO_O0R_ARKITSCENES_TRUTH_MATERIALIZER_IMPLEMENTATION_LOCK_2026-08-10.json](TARO_O0R_ARKITSCENES_TRUTH_MATERIALIZER_IMPLEMENTATION_LOCK_2026-08-10.json)

本锁把用户授权的精确 24 个 Training video × 3 asset 绑定到可静态复验的 fail-closed 实现，但没有
调用任何正式 runner：

- HEAD-only runner 固定 72 URL、零 body、retry/Content-Length/redirect receipt；
- truth runner 在 root 消费后才下载，并先冻结全部 frame denominator，再用 8 个 fit parent 的全部
  exact frame 封存 uncertainty；封存前 eval decode 必须为零；
- 9 个 query 各自从 bound FARO/confidence corridor 推导 confidence/range，不接受 caller scalar；
- official `{video_id}_{timestamp}` 原始 member 与 timestamp-only adapter path 同时留存；
- uncertainty cell 与 factor ndarray 写为 content-addressed gzip blob，写后 hydrate 并重算 canonical SHA；
- truth root 一旦创建即消费，不覆盖、不 resume、不重跑；DepthART/factorial root 全程必须不存在。

24/24 synthetic focused tests 通过，其中包含 HEAD transport/write failure 与 truth root-creation 窗口的 one-shot consumed 注入。该结果不说明 72 个远端 asset 可用、真实 gate 可通过或 O0R 有因果
headroom。当前唯一 successor 只是另冻
`TARO_O0R_ARKITSCENES_CONTENT_LENGTH_HEAD_EXECUTION_LOCK`；在该锁提交前仍不得发送 HEAD，更不得 GET、
打开 source 或物化 truth。
