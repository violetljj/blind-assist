# RCLE Phase B real positive-approach role admission R1 result

日期：2026-07-27

## 终态

`HOLD_ALGORITHM_CANARY_APPROACH_ROLE_INCOMPLETE / VALID`

本次 R1 没有得到 `REAL_POSITIVE_APPROACH_ROLE_ADMITTED / VALID`。因此：

- 不创建、不授权 performance qualification；
- 不运行 RGB algorithm，不读取 RGB pixels，不调参；
- 不换 `sofa_4`、其他 ETH3D sequence、mirror 或 repack；
- 不重试、不续传、不另选窗口；
- ETH3D sofa scene capture family 因本次 payload access 永久失去
  confirmation 资格，只能保留为 source characterization、counterexample
  或 regression。

## 唯一正式访问

唯一候选为 ETH3D SLAM `sofa_3` RGB-D：

`https://www.eth3d.net/data/slam/datasets/sofa_3_rgbd.zip`

guarded canonical run：

- guard process：`pid-53872`；
- guard exit：`0 / COMPLETE`；
- claim SHA-256：
  `1ef030cfea14719d573701c903d79a69f1fd0aff5136f67d1b044ab8e7a4bee2`；
- request：`1 GET`，`0 HEAD`，`0 retry`，`0 mirror`，`0 fallback`，
  `0 replacement`；
- redirect chain：空；
- requested/final URL：均为冻结 official URL；
- response artifact bytes：`84,363,952`；
- response artifact SHA-256：
  `9ea6fcc823072dc913c765c35de8ddff91d3c6cdd8bffb4c4e1647c92e73c5d1`。

claim 创建时间为 `2026-07-27T04:00:32.114659+00:00`，唯一 GET 开始时间为
`2026-07-27T04:00:32.137982+00:00`；claim 严格先于 request。

## HOLD 原因

冻结 source terminal code：

`R1_OFFICIAL_CONTAINER_IDENTITY_INCOMPLETE`

response artifact 可作为 ZIP inventory 打开，但不满足冻结的单一
`sofa_3` RGB-D container identity / required-member 合同。按照预注册，
这属于普通 source-container identity 不闭合，因此终态为有效 HOLD，
不是 geometry negative，也不是数据角色准入。

geometry producer 没有执行；不存在窗口、pair ledger 或 approach 指标，
不得从本结果推断 `sofa_3` 是否真实接近，也不得推断 RCLE algorithm 性能。

## 独立验证

独立 validator 终态：

`VALID`

验证范围仅为：

`ONE_SHOT_SOURCE_ACCESS_PROCEDURE_ONLY`

validator 复核：

- contract、source authority、burned manifest、implementation lock 与 claim
  hash binding；
- claim-before-request 时间顺序；
- exact GET URL、final URL 与空 redirect chain；
- response artifact bytes/hash；
- `1 / 0 / 0 / 0 / 0` request/retry/fallback/mirror/replacement 计数；
- source terminal code allowlist；
- performance qualification 仍为 closed。

关键冻结 SHA-256：

- burned/exclusion manifest：
  `0b54cecc1f3908264f3d4bd06a37b7c27b6f149c05e92e5b3949c0a6ef201593`；
- source authority/candidate lock：
  `7fc127f42ab50516d198b36938c396d9a1d3bcbbf219c02a72b991853ed7eccf`；
- contract：
  `e2a3dfdecfbfb660a6c708e8f1146e7c3652c3192c34fdb19b9f13c47f92dc38`；
- implementation lock：
  `0e8e146cbadb97cadc69cc7ee1f083901ce94b2d05489bd9eb6a0a8bcba4e69d`；
- source receipt：
  `690392dbe333e57564fcac39c320da642ef4cb636f0ddd34f04cb26b35ff8b34`；
- success terminal：
  `df66ab26988512999e71af6010df194888264b9e198aa52ca95f8bcaa4577660`；
- validation receipt：
  `01794c03f75630bb3234872bde63b5738b0be05f26a3afd3857aa467cef861e0`。

## Preclaim launcher 记录

在 canonical claim 前有两次 host-launcher 启动失败：

1. script-mode import path 未包含 repository root；
2. guard 默认 Python 不含 NumPy。

两次均发生在 claim 创建前，且 `claim/request/payload bytes/progress/terminal`
全部为零；它们不是 R1 source retry。修复后重新冻结 runner SHA、
implementation lock 与 preflight，再由 canonical guarded run 独占创建唯一 claim。

## 权限边界

本结果只证明：

`frozen one-shot source-access procedure was valid, but the candidate source
identity was not admitted`

它不授权 performance qualification、confirmation、Kill Gate B、Android、
human、safety 或 production 结论。
