# RCLE unseen external confirmation Source Discovery R1 candidate-lock review

日期：2026-07-27

## 结论

独立 review 终态：

`CANDIDATE_LOCK_REVIEW_PASS_GEOMETRY_ONLY_ACQUISITION_AUTHORIZED`

候选锁：
[RCLE_UNSEEN_EXTERNAL_CONFIRMATION_SOURCE_DISCOVERY_R1_CANDIDATE_LOCK_2026-07-27.json](RCLE_UNSEEN_EXTERNAL_CONFIRMATION_SOURCE_DISCOVERY_R1_CANDIDATE_LOCK_2026-07-27.json)

- candidate-lock SHA-256：
  `c1a0ea53dc698b1f12107db9951f7ecf88def7aa4aed2d2edd0e186093cb5a3c`
- review receipt SHA-256：
  `fbc727609a4c760c1f038e3941250c9f2319b8e6468add8eadeae25b4fcf8585`
- review errors：`[]`
- review 前新 payload root bytes：`0`

## 锁定候选

只允许两个 ancestry-independent source families：

1. OpenLORIS corridor：`corridor1-1`、`corridor1-2`；
2. MultiScan：`scene_00000_00`、`scene_00000_01`。

禁止添加、替换或重排来源；路径、自然语言描述、trajectory 或 metadata 均不授予
positive/below 角色。

## Review 覆盖

独立 stdlib validator 不 import geometry producer、不发网络请求，重新检查：

- 两个 source-authority validation receipt 均为 `PASS / errors=[]`；
- 所有 contract、audit、validation 与 geometry formula binding 的当前 SHA-256；
- exact source/capture 集合、顺序、bytes、LFS identity 与 ancestry independence；
- acquisition allowlist 不含 RGB、MP4、JPG、texture 或 color member；
- 原 R0 的 `10 s` 非重叠网格、同 capture `20 s` 选中窗间隔、至少 `90` pairs、
  `dt <= 0.1 s`、geometry coverage `>=0.8`；
- positive gate 仍为 `>=0.80` fixed denominator at `>=0.05/s` 且连续 `>=5 s`；
- below gate 仍为 `>=0.80` fixed denominator below `0.01/s` 且连续 `>=5 s`；
- 每来源必须恰好 `1 positive + 1 below`，禁止 pooled rescue；
- download budget 恰为 `40 GiB`，locked outer objects 合计
  `34,926,553,912` bytes（`32.527888 GiB`）；
- review 前指定 acquisition root 不存在或为 `0 bytes`；
- candidate expansion、replacement、threshold relaxation、RGB 与 Android 均为
  `FORBIDDEN`。

## 授权边界

本 review PASS 后只允许：

- 获取 lock allowlist 中必要的 pose/calibration/timestamp/depth geometry members；
- 在固定网格上运行绑定 SHA 的 geometry-only formula；
- 生成完整 window ledger，并按固定 tie-break 选每来源最早可行的
  `1 positive + 1 below` tuple。

仍不允许：

- 读取任何 RGB member、preview 或运行 RCLE RGB algorithm；
- 修改算法、公式、阈值、窗口或候选；
- pooled source rescue；
- Android、设备、产品或安全 authority。

若任一来源角色不完整、transport/sync 失败或预算耗尽，终态必须保持
`EXTERNAL_COHORT_NOT_EVALUABLE`。
