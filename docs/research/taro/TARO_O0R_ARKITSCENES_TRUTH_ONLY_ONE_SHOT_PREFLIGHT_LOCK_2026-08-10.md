# TARO O0R ARKitScenes truth-only one-shot preflight lock

状态：`PREFLIGHT_LOCKED / HEAD_NOT_RUN / ONE_SHOT_UNCONSUMED / EXECUTION_NOT_AUTHORIZED`

机器真源：
[TARO_O0R_ARKITSCENES_TRUTH_ONLY_ONE_SHOT_PREFLIGHT_LOCK_2026-08-10.json](TARO_O0R_ARKITSCENES_TRUTH_ONLY_ONE_SHOT_PREFLIGHT_LOCK_2026-08-10.json)

本锁从 hash-bound O0R contract 重算并冻结 `8 ADAPTER_FIT + 16 O0R_EVAL_CANDIDATE`
parent，以及每个 parent 的 `upsampling.zip`、`lowres_wide_intrinsics.zip`、
`lowres_wide.traj`，共 72 个精确 HEAD target。展开后的 canonical request-plan SHA-256 为
`C0F0D41E381333BEFF4C2C0EC4678BC75A22C86173E11B3150F7F7FFBCB18927`。

静态 validator 会重算 roster、URL、request digest、binding、预算、权限矩阵和四个 future root 的
不存在性。该 argv 只用于离线验证；本次没有发送 HEAD/GET/Range，没有读取响应体或 Content-Length，
没有创建/消费 one-shot root，也没有物化 truth、拟合 selected-source uncertainty、运行 DepthART 或
执行 factorial arm。

## 已冻结 blocker

- 绑定的 Assistive Geometry B0 回执只覆盖 6 个旧视频，且未列 trajectory；它不足以授权 TARO
  24-parent body access。HEAD 或 body access 前必须另签精确覆盖 24 × 3 assets 的 TARO receipt。
- `47333152` 位于官方 downloader 的 missing-3dod 列表，download helper 会抑制其
  `lowres_wide.traj`。未来 HEAD 若不是 `200 + Content-Length`，R0 必须 `NOT_EVALUABLE`，不得换 parent。
- 72 个 target 尚无 HEAD receipt；Content-Length budget gate 仍未运行。
- source/truth materializer、atomic writer 与正式 execution argv 尚不存在，不能把本锁写成
  `AUTHORIZED_UNCONSUMED` 或 truth admission PASS。

## 唯一 successor

`TARO_O0R_ARKITSCENES_TRUTH_ONLY_MATERIALIZER_IMPLEMENTATION_LOCK`（`execution=false`）。只有新的
TARO-specific signed receipt 已存在后，才允许实现并静态测试 fail-closed HEAD/source/truth
materializer 和 atomic evidence writer；仍不得执行网络/source/truth、创建 root、运行 DepthART 或
O0R factorial。

Claim ceiling 仅为预检对象、静态复验入口、预算、授权缺口、availability risk 和 root absence 已冻结；
它不建立远端可用性、真实 truth、模型、因果 headroom、穿戴式/主动观测、设备、产品或 safety 权限。
