# TARO O0R materializer input and persistence amendment lock

状态：`LOCKED / SCIENTIFIC_NOT_RUN / EXECUTION_NOT_AUTHORIZED`

机器真源：
[TARO_O0R_ARKITSCENES_MATERIALIZER_INPUT_AND_PERSISTENCE_AMENDMENT_LOCK_2026-08-10.json](TARO_O0R_ARKITSCENES_MATERIALIZER_INPUT_AND_PERSISTENCE_AMENDMENT_LOCK_2026-08-10.json)

该 outcome-blind amendment 只关闭 materializer implementation 前的四个接口歧义：

- frame denominator 使用所有四 modality 同 stem、exact-stem K、bounded pose 的帧；不得 earliest-N、
  outcome selection、换 parent 或限额抽样；
- uncertainty 必须在 8 个 ADAPTER_FIT parent 全部 exact frames 上先封存，eval payload 在此前不得 decode；
- 每个 query 的 confidence/range 从同一 query capsule 内、FARO-at-Apple-centers 与 bound confidence 的
  common support 内推；along 上界严格为 inclusive `horizon_m`（不加 endpoint tolerance），横向距离在
  support plane 上投影，但不另加 height gate；caller scalar 和九 query 共用一个 frame scalar 均禁止；
- official `{video_id}_{timestamp}` member 的原始 path/SHA/CRC 必须由外层 envelope 绑定，再显式映射到
  frozen adapter 所需的 canonical `{timestamp}` path；数组必须写 content-addressed blob 并通过 hydrate
  round-trip hash，不能只存 ndarray hash receipt。

本锁没有发送 HEAD/GET、打开 selected source、创建 root、物化 truth、拟合 uncertainty、运行 DepthART
或执行 O0R。它不改变 roster、gate、预算或任何科学/产品/安全权限。
