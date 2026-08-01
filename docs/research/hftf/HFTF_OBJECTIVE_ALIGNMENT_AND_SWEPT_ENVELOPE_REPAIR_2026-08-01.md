# HFTF objective alignment and swept-envelope repair

日期：2026-08-01

状态：`OBJECTIVE_MISMATCH_FOUND / STAGE_B_LABEL_MECHANICS_REOPENED`

## 1. 结论

R2 的正式终点保持
`H1_MULTI_HEIGHT_PROXY_NOT_SUPPORTED_STOP`，但它关闭的是当前
`angular-cell point-support` teacher，不足以关闭原始 HFTF 的
human-envelope-conditioned traversability hypothesis。

原始构想要求：

- 放入具有宽度、高度和安全余量的人体碰撞包络；
- 沿全部候选方向滚动短时轨迹；
- 分别表达脚下落差/台阶、身体碰撞和头部悬空障碍；
- Stage B 身体包络监督没有增量时，不进入 Stage C 时序预测。

R0–R2 runner 实际做的是：

- 将障碍点按 `theta × distance × height` angular cells 计数；
- `risk=min(1, point_count/8)`；
- 没有人体横向宽度、安全余量或 swept trajectory collision；
- 没有 ground continuity、step/drop teacher；
- three-height max 只验证 point-support 分层，不验证人体能否通过。

因此直接删除 height 进入 single-height future R3 会把研究问题缩水成更容易通过的
表示，违反原始 Stage B→C 纪律。

## 2. 当前证据能说明什么

R2 可以支持：

- source-specific metric geometry 与 local-ground proxy 可执行；
- history-causal rolling origin 在四个新 sessions 上使 current/near/far field
  coverage 可评价；
- 当前 three-height point-support representation 未达到 4/4 非冗余门。

R2 不能支持：

- swept human envelope 没有增量；
- foot/body/head collision supervision 不可行；
- 应当删除人本身体条件；
- single-height future 已获准进入 H2。

## 3. 修复路线

先在 R2 burned sessions 上执行 Development-only Stage B label-mechanics canary：

1. 把 theta bin 改解释为候选 path direction；
2. 每个 distance bin 是该 path 的 longitudinal segment；
3. 对每个 height layer 使用显式 lateral half-width + safety margin 的 swept prism；
4. 障碍点按 along-track、cross-track、height 与 prism 碰撞；
5. foot layer 同时检查 ground continuity 和冻结 step/drop threshold；
6. known/UNKNOWN 继续由可见 probe 与 ground support fail closed；
7. dynamic label只作 provenance，并修正为
   `opening-door/opening-gate/pedestrian/rider/animal/vehicle`。

这一步只检验标签生成器是否 faithful、非退化、满足单调性与高度特异性。它不训练
student，不产生 fresh evidence，也不授权 H2。

只有 synthetic structural tests 和 burned-source mechanics audit 都可信，才允许在
全新 sessions 上冻结 formal H1 R3 swept-envelope teacher canary。
