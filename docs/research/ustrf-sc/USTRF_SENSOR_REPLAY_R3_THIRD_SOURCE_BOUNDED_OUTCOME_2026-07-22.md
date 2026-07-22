# USTRF Sensor Replay R3 第三来源有界检索结论（2026-07-22）

状态：`FORMAL_DATA_LIMITATION_2_OF_3 / EVALUATOR_NOT_RUN / DO_NOT_SELECT_HARDWARE`

## 结论

按预先冻结的第三来源范围和顺序，IDSIA MSMPT `s9 -> s12 -> s13 -> s14` 已全部走到各自的 fail-closed 终点，没有得到第 3 条可准入来源。当前合集仍只有 LILocBench `dynamics_0` 与 `lt_changes_dynamics_0`，正式记录为 `2/3` 数据局限。

本轮没有降低 `24/12/0.03/0.50`，没有扩大 LILocBench 或 Bonn，没有把 Bonn 负样本计作来源，也没有运行 evaluator。因此五项事件指标及 worst-source 结果均未生成；这不是零分，而是未满足冻结的三来源执行前提。

## 冻结合同

- R3 prereg SHA-256：`3aa3fdb460c697d6d669e2174f7f7c9d17f1fc06b6f6392d35ba1ac5b2b73eaa`。
- third-source discovery v1 manifest：`configs/ustrf_sensor_replay_r3_third_source_discovery_v1.json`，SHA-256 `00e3646f730e509fe1b4ebd3e1215948a2cde4fa27fe3ba3fdb82dd7dc9e20cd`。
- 限定候选：非 Bonn、移动机器人/身体绑定、前向 RGB-D、公开 OptiTrack 6D 位姿真值、场景含动态行人；数据集为 IDSIA MSMPT，Zenodo DOI `10.5281/zenodo.20044662`，许可声明 CC-BY-4.0。
- GT-only 拒绝门：24-frame truth horizon、12-frame causal history、最小位移 `0.03 m`、truth/causal unknown 最大 `.50`、RGB-pose 最大时间差 `40 ms`、对齐率至少 `.95`。
- 完整适配几何门：对齐深度最低有效比例至少 `.50`；review 必须是 candidate-hidden 的两模型隔离审核，来源准入取 AND。

## 有界序列结果

| 序列 | GT RGB 帧 | pose 对齐率 | truth unknown | causal unknown | GT-only | 完整 RGB-D | 双模型路线审核 | 终点 |
|---|---:|---:|---:|---:|---|---|---|---|
| `s9` | 522 | `.710728` | `.513410` | `.507663` | 拒绝 | 未运行 | 未运行 | 对齐率和两个 unknown 门均失败 |
| `s12` | 1463 | `.966507` | `.097061` | `.245386` | 通过 | 1460 帧；最低有效深度 `.719080` | A 准入、B 拒绝 | AND 门拒绝，禁止事件裁决 |
| `s13` | 1120 | `.999107` | `.023214` | `.100000` | 通过 | 1118 帧；最低有效深度 `.724681` | A 准入、B 拒绝 | AND 门拒绝，禁止事件裁决 |
| `s14` | 1258 | `.981717` | `.065978` | `.220986` | 通过 | 1254 帧；最低有效深度 `.440792` | A/B 均拒绝 | 几何硬门失败且路线双拒绝 |

`s12`、`s13`、`s14` 的完整准备都使用 camera-1 连续 RGB-D、20 ms 内 RGB-depth 关联、`chair -> color/depth optical` 静态外参、Brown 畸变投影、nearest-z 冲突处理和纯注册深度；没有填洞。原始 `0/65535`、非有限、负值以及无法表示为 `uint16` 毫米的投影深度保持 unknown。GT pose 在 estimator 输入中物理隐藏，candidate 在两位 reviewer 输入中隐藏。

## Hash-bound 拒绝证据

- `s9` GT-only：`artifacts.local/evidence/ustrf-sensor-replay-r3/third-source-discovery-v1/idsia-s9-gt-prescreen-v1.json`，SHA-256 `b65c22b79852bd4d2e869a35d531dade5eb4faba2039fe451eb5ccc20c183b86`。
- `s12` GT-only：`f287b318a9ce3e49143e131acd8a961d2c16cf55f1282d5beb785a3c47422277`；review consensus：`a057bdb228b073ad25a8dc4ec2448da7cf980f5edb50e6d38140ea7990f8c4a7`；terminal rejection：`39f73f7d3096c14551933ee7f1decdf0389fe47c45326ec09691eec246bf953a`。
- `s13` GT-only：`3eab3da994f184566f95d9a1e40aaf007244868ed89a02dfe1dc465f3cf6c61c`；review consensus：`1db1adbba95542260ace13a017db95ec0a258a569cf8fac13cc6e0e933a72bab`；terminal rejection：`a13eb5270ab4d865f9ba78cb730d8a465f2ab7e8ff4daa24fe960ae07245e888`。
- `s14` GT-only：`cc091286b930e91571c7b850d0568bf262954fe23766fca1b6940059fe3d47a6`；RGB-D preparation：`30bc5aac2dde56ac04ddc73c1d82eebfb529e81f059399b24377c55285c01c1e`；review consensus：`5ab08adae0bd7d79e638008570d842382b3e50920a39915db621dd2b8608b9ad`；terminal rejection：`7b3fb26de1574acf2b3b3165f227f9c858c4db901e76e7090d021149182ace4f`。

每份 terminal rejection 都绑定 GT-only、完整准备、冻结 candidate、reviewer A/B 和 consensus 的 SHA-256，并显式写入 `source_count_credit=0`、`evaluator_ran=false`、`production_authority=false`。

## 已有两源的评价前一致性修复

`lt_changes_dynamics_0` 旧 consensus 的 critical 字段不足以支持将来的 critical-miss 门。本轮在 candidate-hidden、原始 A/B/anchor 裁决和旧 consensus 全部 hash-bound 的独立输入上完成一次 criticality-only 裁决，生成 `review-consensus-v2.json`：12 个 canonical events 中 11 个 critical，SHA-256 `1b79c73c037496663339025b13037196b0d1c5ffdded164c07ea2f74ba5de0c6`。该修复只消除将来评价的定义缺口，不授予第三来源、event truth 或 evaluator 权限。

## 磁盘回收

所有已终止序列的原始 bag、压缩分片和失败的中间目录都只在 successor receipt 哈希核验后删除；随后删除已完成研究、可从公开源重取的 OpenLORIS/Bonn/TUM 缓存及两份已有完整 prepared/evidence 的 LILocBench 官方 ZIP。保留配置、GT 报告、完整适配结果、review sheet、模型原始回执、consensus 与 terminal rejection。累计回收约 `62 GiB`：早期第三来源批次约 `21.68 GiB`，S14 原始 bag `5,180,620,800` bytes（约 `4.82 GiB`），完成态 replay 下载缓存 `38,237,337,405` bytes（约 `35.61 GiB`）。清理后该 replay 下载目录约 `6.44 GiB`，E 盘可用约 `71.64 GiB`；删除内容均可从官方来源重取，hash-bound 本地证据不受影响。

## 决策与后续边界

1. 冻结当前两源合集为 `2/3` 数据局限，不伪造“三源合集”。
2. 不运行唯一一次 evaluator，五项事件指标和 worst-source 保持 `not produced`。
3. 不以 Bonn 负样本、同族追加搜索、门槛变化或事后解释补齐来源。
4. 如未来开启新的第三来源研究，必须新建独立版本的 discovery manifest 和预注册，不得改写本轮 v1 结论。
5. App、U0、设备、训练、硬件选择与生产权限全部不变。
