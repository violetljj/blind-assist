# Conditional Segmentation Gating R0 Development Result

状态：`R0_PRIMARY_EXECUTION_COMPLETE / VALID /
CONDITIONAL_GATING_NO_ROBUST_INCREMENT_STOP_GATING_ROUTE /
PRIMARY_CANDIDATE_ONLY_AFTER_SCOPE_CORRECTION /
R0_1_SHADOW_FAMILY_TERMINAL_PENDING /
DEVELOPMENT_ONLY / FINAL_CONFIRMATION_NOT_ACTIVATED /
DEFAULT_APP_UNCHANGED`

日期：2026-08-01（Asia/Hong_Kong）

## 结论

冻结的单一 `CLASS_CONDITIONED_MULTI_NEGATIVE` 候选没有形成稳健增量，该 primary
在 R0 内到此停止。候选通过 overall、`boundary_step_curb` 和 `obstacle` 三项
recall-retention 门，但 false-positive reduction 只有 `0.092572 < 0.30`，且最低
source-session recall retention 为 `0.774580 < 0.80`。五项冻结门没有全部通过，
终态只能是：

```text
CONDITIONAL_GATING_NO_ROBUST_INCREMENT_STOP_GATING_ROUTE
```

这不是模型、语义分割方向、双环概念或所有条件门的总体否定。它只否定当前 DDRNet
输出之上的这一个预冻结组合门。

### 2026-08-01 post-result scope correction

用户指出，只执行一个 primary 会降低选优风险，却可能因主候选刚好较差而错杀整个门控
家族。该纠正成立。R0 的 machine result、指标、哈希和 terminal 均保持不可变，但
`STOP_GATING_ROUTE` 的科学范围收窄为：不在 R0 内以未执行候选救援 primary；它不表示
候选 1、2 已被评价。前向
[R0.1 shadow protocol](DUAL_LOOP_SEGMENTATION_CONDITIONAL_GATING_R0_1_SHADOW_PROTOCOL_2026-08-01.md)
将两个 pre-R0 conceptual、但当时未 hash-bound 的方案冻结为 diagnostic-only shadows。
在 R0.1 收口前，bounded family terminal 与 residual-aware 训练顺序保持 pending。

## 冻结、执行与证据身份

- 协议：`DUAL_LOOP_SEGMENTATION_CONDITIONAL_GATING_R0`
- stage：`DEVELOPMENT_STANDARD`
- 冻结 implementation Git：`2e46d76057becb1f85c22bf0c9ea4e8b59d26c31`
- 配置：
  [`configs/dual_loop_segmentation_conditional_gating_r0/default.json`](../../../configs/dual_loop_segmentation_conditional_gating_r0/default.json)
- Module：
  [`scripts/research/dual_loop_segmentation_conditional_gating/`](../../../scripts/research/dual_loop_segmentation_conditional_gating/)
- 候选数：`1`；执行中 fit、选择与阈值修改：均为 `false`
- 输入：`520` 帧、`11,757` 个 raw components、`10` 个互不重叠的 source sessions
- 数据角色：4 个 `r1_consumed_fresh`、4 个 `dev`、2 个
  `consumed_old_blind`，全部只保留为已消费 Development

十个 session 的 frame/image/view identity 均不重叠，但输入没有 participant、route 或
parent-capture 标识，因此现实独立性为
`NOT_EVALUABLE_MISSING_IDENTIFIERS`。逐 session 留出只作 fit-free stress reporting；
不是 cross-validation、独立验证或 Confirmation。

## 运行时真值防火墙

Atlas 的 `UPPER_FIELD_BACKGROUND_ACTIVATION_PROXY` 依赖
`dominant_truth_class`，不能成为运行时 gate 输入。本轮在运行前将它替换为纯几何
`INTERSECTS_UPPER_HEAD_BAND`：raw component 与 `y < 0.35H` 任一像素相交即为真。
temporal history 只读 raw predicted class mask，按 class 隔离，并在
sequence/session 边界重置。候选 callable 不接收 truth、mechanism、session、scene、
role 或 YOLO attribution。

新同类历史合同还删除了旧 union probe 可获得的跨类 temporal credit：

| 预测类 | 被移除的跨类 credit pixels | 覆盖 session |
|---|---:|---:|
| `boundary_step_curb` | 152,080 | 10 |
| `obstacle` | 112,013 | 10 |

## 全量结果

| arm | TP | FP | recall retention | FP reduction | class-strict false components/frame |
|---|---:|---:|---:|---:|---:|
| `BASELINE_UNFILTERED` | 3,357,954 | 3,623,403 | 1.000000 | 0.000000 | 14.4750 |
| `REFERENCE_CAUSAL_2_OF_3_UNION` | 2,607,955 | 2,543,902 | 0.776650 | 0.297925 | 13.4231 |
| `REFERENCE_CONFIDENCE_GE_0_65` | 2,670,989 | 2,520,836 | 0.795422 | 0.304290 | 0.9192 |
| `CLASS_CONDITIONED_MULTI_NEGATIVE` | 3,164,532 | 3,287,977 | 0.942399 | 0.092572 | 8.6308 |

候选 precision 为 `0.490434`，recall 为 `0.316831`；baseline 分别为 `0.480989` 与
`0.336196`。它明显保留了更多 baseline true-positive pixels，但只删除了少量
false-positive pixels，没有达到预声明的误报降幅。

### 冻结决定门

| 检查 | 阈值 | 结果 | 判定 |
|---|---:|---:|---|
| overall recall retention | `>= 0.90` | `0.942399` | PASS |
| false-positive reduction | `>= 0.30` | `0.092572` | **FAIL** |
| minimum session recall retention | `>= 0.80` | `0.774580` | **FAIL** |
| `boundary_step_curb` recall retention | `>= 0.80` | `0.945451` | PASS |
| `obstacle` recall retention | `>= 0.80` | `0.946764` | PASS |

class-strict false components/frame 从 `14.4750` 降至 `8.6308`，通过只作诊断的
non-increase 检查；它不替代五项主门中的 pixel FP reduction。

## Source-session stress

| source session | frames | FP reduction | recall retention |
|---|---:|---:|---:|
| `5LlqRK-hWoDLSW5MmoLjKj6uQtZMKjb9` | 60 | 0.069447 | 0.951824 |
| `972O8sd5HpUbGeEE_UAb1g0z1OZUtfHl` | 50 | 0.175819 | 0.847295 |
| `CCG-oMqgJf90LzEhlqj0WkP3xN0dHlXk` | 50 | 0.098585 | 0.961814 |
| `GxMb4zhAvoM5jbF54kfcs8wxTL4fqNnT` | 50 | 0.149741 | 0.898927 |
| `LRWTaGC62fB3NflMQaOPXf0HJR1m5Ypp` | 50 | 0.021291 | 0.982514 |
| `eHxtA669WpN381O4ZjVAmG3-3ZUewuXr` | 50 | 0.082155 | 0.940571 |
| `i2jglnBfoIqIIA7ojQGe-4vK07hUm4T3` | 60 | 0.023835 | 0.985840 |
| `ic_BpoiSOIW-7_mffGenT6yissRNiPzT` | 50 | 0.176406 | 0.831519 |
| `lmkIchCJ1RIKsZvbb4HjCDl85B2nOicv` | 50 | 0.283640 | **0.774580** |
| `yQ5Ij3w49RMUxxIai6ZVsvPHssZJ13FO` | 50 | 0.053251 | 0.951193 |

所有 held-out rows 与同 session 的 direct aggregation 逐项相等，且
`fit_used=false / training_used=false / candidate_selection_used=false`。最弱 session
`lmk...` 说明 aggregate scene 指标不能替代 session floor：合并后的 `step_curb` scene
recall retention 为 `0.948592`，但该 source session 仍只有 `0.774580`。

## 组件与 Pareto 解释

11,757 个 raw components 中，4,605 个完整保留、2,054 个部分保留、5,098 个完全删除；
518 个 source components 被切分，门后共有 7,762 个 fragments
（`14.9269/frame`）。any-hazard false components/frame 从 `9.2135` 降至 `6.3000`。

候选与 `REFERENCE_CONFIDENCE_GE_0_65` 同在二维 Pareto frontier，但：

- `dominates_predecessor_references=[]`
- `new_frontier_point=false`

因此 frontier membership 不能提升为“新改进点”。候选只是以较低误报降幅换取较高召回；
它没有同时解决 Atlas 暴露的误报规模与最弱 session 稳健性。

## 校验与可复算输出

独立 validator 从逐帧和逐组件账本重算输入身份、aggregation、五项决定门、
held-out/direct 等价与 terminal，共通过 `85,235` 项检查，错误数为 `0`。
随后写入独立输出目录的第二次确定性复算再次通过 `85,235` 项检查；`result.json`、
`frame_metrics.jsonl` 与 `component_decisions.jsonl` 的 SHA-256 均与首次执行逐字节一致。

| 输出 | SHA-256 |
|---|---|
| `result.json` | `c5fca4c69b5f346a01ccfab55ecc4b88ce69e81829cfc6c6f20e64172a34554c` |
| `frame_metrics.jsonl` | `032b1a6a261bb19b981c4d3870753bba4b7db31bf2591f5e6bea668594f8703a` |
| `component_decisions.jsonl` | `0cc4a12f092f2555f0d5977c835765539ac6640b4c911c71d59ba30fe46cf8c6` |
| candidate definition | `2aade52f1a690b7494a6259f1698354fa7ce925f4829682c646cb94e3e943ec9` |

本地可复算 evidence 位于
`artifacts.local/evidence/dual-loop-segmentation-conditional-gating-r0/`；该目录被忽略，
不作为仓库 source 提交。

## 后继边界

R0 本身不运行未冻结候选、不改门、不按 session 选 gate，也不以 shadow 覆盖 primary
失败。两个概念方案只能在新的 R0.1 协议、definition hash、runner、validator 和独立
output root 下作 post-primary shadow ablation。R0.1 不选择候选；其结果只决定有限三臂
家族是否存在反例，以及 residual-aware DDRNet 是否成为唯一下一主边界。

本结果没有改变模型、Android、QNN/A568、risk/feedback、TTS、振动、提醒或默认 App。
`drives_alerts=false`、`FINAL_CONFIRMATION_NOT_ACTIVATED` 与所有产品/安全禁令保持不变。
