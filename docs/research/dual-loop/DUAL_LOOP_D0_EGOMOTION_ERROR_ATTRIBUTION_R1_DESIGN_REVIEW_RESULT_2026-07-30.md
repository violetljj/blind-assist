# D0 Ego-motion Error Attribution R1 设计复核结果

日期：2026-07-30（Asia/Hong_Kong）

```text
STATISTICAL_DESIGN_REVIEW: PASS
IMPLEMENTABILITY_REVIEW: PASS
CONTRACT_STATUS: FROZEN
IMPLEMENTATION_STATUS: NOT_IMPLEMENTED
SCIENTIFIC_OUTCOME: NOT_RUN
EXECUTION_AUTHORITY: NONE
```

协议 SHA-256：
`87931369f912fdd054783db9decb2a1813080d0a961c3526b83ce686d1a48183`

依赖收据 SHA-256：
`0377944df2abdeb6044d49182e1f4bc1908b4bf8ba40eb632a091b4d2d10dc7f`

## 结论

[D0 R1 协议](DUAL_LOOP_D0_EGOMOTION_ERROR_ATTRIBUTION_R1_PROTOCOL_2026-07-30.json)
已修复 R0 的统计与可识别性缺口，可以进入实现、测试、implementation lock 与独立
实现复核。它不能直接进入 activation 或正式执行。

R1 的三个出口只表示 burned single-capture 内的 operational canary priority：

- `EGO_CANARY_PRIORITY`
- `TEMPORAL_TREND_PRIORITY`
- `NO_PRIORITY_IDENTIFIED`

它们不表示 causal dominance、总体机制、算法有效性、泛化、Confirmation、产品或
安全结论。

## 前序与防火墙

Production temporal geometry factorial A/B R0 已得到
`VALID / NO_INCREMENT`，只作为启动 D0 R1 的 eligibility binding。协议绑定其
result、independent validation 与 seal 的哈希，但明确禁止读取 production trace，
也不允许把 CrowdBot 指标接入 REveL D0、选择路由指标或裁决平局。

旧 F-1B decision、Confirmation、EVIMO2v2、JRDB、新 REveL 数据和 RCLE 均保持关闭。

## 依赖与统计复核

dependency preflight 对冻结的 1,660 个 natural-event rows 复算 469 个 primary
events，得到：

- 159 个闭区间跨 target overlap pairs；
- 0 个同 target overlap pairs；
- 310 个 transitive overlap components；
- component size distribution：
  `1×222, 2×48, 3×24, 4×9, 5×3, 6×3, 10×1`；
- 从最早事件起点固定的六个 60 秒块，事件数
  `69 / 38 / 52 / 101 / 98 / 111`。

协议分别冻结 event-weighted、component-balanced 和 60-second-block-balanced
Cliff delta；所有权重归一化、finite denominator、missingness fraction、candidate
block 与 material reverse 均唯一。469 个 event 仍不是 469 个独立总体样本。

路由指标在 output 前唯一预选为：

- ego：`median_abs_sensor_approach_component_mps`；
- competing person motion：`median_abs_person_approach_component_mps`；
- temporal direct：`median_flow_score_mad_per_s`；
- temporal support：负向 `median_surviving_tracks`。

其他 camera、ROI、flow、support 与 reference-arm 字段只有 diagnostic 权限。

## 可复算性修复

独立复核迭代要求并已确认：

- production A/B trace firewall 与 R2/natural/truth exact join；
- person bracket、nearest sensor tie-left、sync、continuity、speed、origin 与
  sensor index delta；
- source/person/sensor/camera 公式、`1e-6 m/s` truth closure、pair/event/share
  三层 support；
- 独立 `roi_dt_s`、target order、BBOX finite inclusion、`1e-12/s` rate closure、
  ROI jitter 与 flow/quality inclusion；
- type-7 quantile、raw MAD、exact Cliff ties、全部 required-cell missingness；
- global route evaluability、person competing、material contradiction 与严格互斥
  三出口；
- prestart 可解析 burned JSON，但 formal marker 必须在首次读取 Vicon bag message
  或计算 D0 metric 前 exclusive-create + fsync；
- producer receipt 不得自证有效；independent validator 必须独立重算 event table、
  dependence、motion、ROI/flow、missingness、Cliff、route 与 exit，之后才可发布
  `VALID` execution receipt；
- canonical UTF-8/LF/sorted JSON、exclusive success/failure receipt 与 post-start
  invalid/consumed/no-rerun 语义。

## 验证

dependency preflight tests `6/6 PASS`，覆盖：

- 闭区间 shared endpoint；
- synthetic deterministic components/blocks；
- exclusive output；
- transitive A-B-C overlap component；
- midpoint 精确落在 60 秒边界；
- 真实冻结 input golden receipt、159 pairs、310 components 与固定 block counts。

JSON parser、hash binding、所有冻结输入存在性与 SHA-256、project structure 和
`git diff --check` 均通过。`run-r1` 正式 namespace 仍不存在。

## 权限

本复核只把 `d0_r1_implementation` 提升为 `AUTHORIZED`。允许实现 frozen producer、
analysis、independent validator、focused tests 与 implementation lock。

以下仍不授权：

- D0 R1 activation 或正式执行；
- 修改或重跑 LITE R2；
- production A/B trace、旧 F-1B decision 或 Confirmation access；
- 新算法、阈值/window search、EVIMO2v2/JRDB 下载或 canary；
- Android 产品接线、默认提醒变更、真人、产品或安全主张。
