# RCLE periodic self-motion counterfactual R2 P3 R0 结果

日期：2026-07-29
阶段：`P3_TRANSPORT_ANALYSIS_AND_RUNTIME_PREFLIGHT`
终态：`PERFORMANCE_QUALIFIED / VALID / P4_NOT_ACTIVATED`

## 结论

轻量 P3 已按冻结边界及用户授权的 scheduler successor 完成。R3 synthetic
transport 与冻结 pair core 精确等价；
统计实现和 mutation tests 有效；固定 8 个 `PREFLIGHT` identities 已分别完成
4-worker 与 8-worker guarded-host qualification。两 profile 的 4816 个 frame
identity、4808 个 ordered pair numeric identity 逐项一致，资源门、heartbeat、
分页和残余进程门均通过。

初始 R0 receipt 的均匀 `8 -> 496` 外推错误地把 PREFLIGHT 中 `2/8`
guardrail 比例放大到 formal；其 `24.1952 / 20.1591 h` 不再作为当前性能结论。
successor 改为按 `480 factorial + 16 guardrail` 的逐 arm 分项时间外推，并在不缓存
R3、不减少 pair 的前提下，将三个静态 identity 的同 pose frame 从重复渲染 602 次
改为渲染 1 次、复用 601 次。用户另行授权 W8 复现 predecessor 的实际线程行为：
每 worker `OpenCV=1`、实测 `OpenBLAS=18`。

successor W8 完整实测为 `677.5074 s`；含独立列示的 10% retry reserve 后，
496-sequence 投影为 `7.1575 h`，低于 `12 h` ceiling。全 8 identities 的四类
transport hash 与 predecessor W4/W8 精确一致。因此 W8 被选择为 scheduler，
但 formal execution authority 仍为 false，P4 仍未激活。

## R3 transport equivalence

新 adapter 只提供 generator-native RGB/valid mask、`K`、timestamp 和
world-from-camera pose，并使用：

```text
K @ (R_current.T @ R_previous) @ inv(K)
```

等价 fixture 覆盖 RGB/BGR sentinel、partial valid mask、identity/nonzero rotation
和连续 `PairState`。同一 4 个 pair 分别经 Pillow PNG reference transport 与
in-memory RGB transport 后，pair row SHA-256 和 state SHA-256 全部相同。

终态：

```text
TRANSPORT_EQUIVALENCE_PASS / VALID / PREFLIGHT_ONLY
```

未修改 R3、Sparse LK、support manager、local affine、strict `>0.01/s`、三-pair、
reset 或 PairState。

## Analysis implementation 与 mutation tests

analysis lock 固定：

- 80 个 `scene_seed × motion_block` cluster，而不是 frame/pair 伪样本；
- 每 cluster 精确六 arm，缺失、重复、换 block/seed/arm 均 fail-close；
- fixed denominator `601`，abstention 保留在分母并 reset streak；
- 九个 terminal-driving contrast 共用同一 block-stratified cluster draw matrix；
- bootstrap seed `20260728`、`20,000` replicates、sample SD `ddof=1`；
- equal-four-block weighting、type-7 95% max-t simultaneous interval；
- nonfinite 与 inconsistent zero-SD 均为 `INVALID`。

定向 suite 覆盖 pair 增删/乱序、伪造 trigger、abstention response、threshold equality、
六 arm/cluster identity、guardrail 混入、bootstrap seed/count、共享重采样绑定、
block weighting 和 nonfinite mutation。P3 全部定向测试最终为 `23/23 PASS`；
完整 RCLE periodic R2 module tests 为 `132/132 PASS`。

analysis fixture 仅为非科学 mutation fixture。没有读取 formal input，也没有解释任何
PREFLIGHT response 或 trigger outcome。

## 固定 PREFLIGHT identities

P3 在任何 runtime output 前冻结 uppercase literal：

```text
RCLE_PERIODIC_SELF_MOTION_COUNTERFACTUAL_R2|PREFLIGHT|ADVIO_14|FACTORIAL|00
RCLE_PERIODIC_SELF_MOTION_COUNTERFACTUAL_R2|PREFLIGHT|ADVIO_14|GUARDRAIL|00
```

对应 numeric seed：

- `FACTORIAL`: `1727242067111453576`
- `GUARDRAIL`: `18409799703140433944`

identity set SHA-256：

```text
819845c2cd7e766477d8fcb3bf963e7b3a9d940ae78e226cdf432d08953f9c65
```

固定集合为 6 条 factorial arm 加 2 条 guardrail arm，每条精确
`602 frames / 601 pairs`。12/16 workers 保持禁止。

## Guarded-host qualification

| 项目 | predecessor W4 | successor W8 |
| --- | ---: | ---: |
| 完整 identities | 8/8 | 8/8 |
| frames / pairs | 4816 / 4808 | 4816 / 4808 |
| measured wall | 1277.166 s | 677.507 s |
| launch available RAM | 9.80 GiB | 8.73 GiB |
| minimum available RAM | 7.58 GiB | 7.42 GiB |
| max heartbeat interval | 20.065 s | 20.058 s |
| swap in / out delta | 0 / 0 | 0 / 0 |
| residual workers | 0 | 0 |
| scheduler eligibility | comparator only | selected |
| corrected projected 496 + retry reserve | not selection-eligible | 7.1575 h |

W4/W8 的每条 sequence `scene_geometry_sha256`、`frame_manifest_sha256`、
`ordered_pair_numeric_sha256` 与 `transport_identity_sha256` 全部匹配。
telemetry 只含 wall time、RAM、I/O、swap、heartbeat 与完成计数，不含 response 或
trigger value。

successor W8 的分项投影为：

```text
render: 2.7511 h
R3: 3.7458 h
validation_and_receipt: 0.0099 h
retry reserve (10%): 0.6507 h
total: 7.1575 h
selected_profile: W8
```

predecessor W4 只作为 numeric equivalence comparator；其有效 OpenBLAS 线程数没有
被观测，因此不参与 successor scheduler selection。

## 独立验证与证据入口

独立 validator 不导入 producer、transport adapter、analysis implementation、R3 pair
core 或 runtime runner。它独立重建 8 identities，验证 lock/file hashes，复核两 profile
的 identity/count/order/firewall/resource/heartbeat/paging/residual-process 证据，
逐 identity 比较四类 transport hash，并按冻结 wall ceiling 形成唯一 receipt。

- [R3 transport equivalence lock](RCLE_PERIODIC_SELF_MOTION_COUNTERFACTUAL_R2_R3_TRANSPORT_EQUIVALENCE_LOCK_R0_2026-07-29.json)
- [analysis implementation lock](RCLE_PERIODIC_SELF_MOTION_COUNTERFACTUAL_R2_ANALYSIS_IMPLEMENTATION_LOCK_R0_2026-07-29.json)
- [PREFLIGHT identity lock](RCLE_PERIODIC_SELF_MOTION_COUNTERFACTUAL_R2_PREFLIGHT_IDENTITY_LOCK_R0_2026-07-29.json)
- [current scheduler successor independent receipt R2](RCLE_PERIODIC_SELF_MOTION_COUNTERFACTUAL_R2_TRANSPORT_ANALYSIS_RUNTIME_PREFLIGHT_R0_INDEPENDENT_RECEIPT_R2_2026-07-29.json)
- runtime evidence:
  `artifacts.local/evidence/rcle_periodic_self_motion_counterfactual_r2/p3_transport_analysis_runtime_preflight_r3/w8/`

independent receipt SHA-256：

```text
db66704384b1cf02ae752f554faa8e22e03c53db66ef47f7efc7bd6a7b79269b
```

初始 receipt 与 R1 保留为不可覆盖 predecessor。R2 独立绑定 predecessor W4、
successor W8、优化后的 runner、validator 和 tests；它明确废止错误的均匀比例外推，
但不删除历史失败证据。

## 权限与禁止项复核

- 正式 `480+16`：未运行；
- formal seed：未访问；
- 科学 outcome：未读取、未解释；
- quality strength：未调整；
- R3 / threshold / three-pair：未修改；
- sequence16 / Android / realtime：未访问或启动；
- P4 activation：`false`；
- formal execution authority：`false`。

本 successor 只解除 runtime performance gate；不授权 formal execution，也不激活
P4。任何后续阶段仍须独立授权。
