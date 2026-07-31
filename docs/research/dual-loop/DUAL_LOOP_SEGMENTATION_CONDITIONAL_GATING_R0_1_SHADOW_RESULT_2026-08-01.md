# Conditional Segmentation Gating R0.1 Shadow Result

状态：`EXECUTION_COMPLETE / VALID /
POST_TERMINAL_SHADOW_ABLATION_COMPLETE_DIAGNOSTIC_ONLY /
NO_MATERIAL_SHADOW_SIGNAL / NO_SESSION_HETEROGENEITY /
TWO_SHADOWS_WEAK_FIXED_HANDCRAFTED_GATING_FAMILY_STOP /
PRIMARY_R0_UNCHANGED / NO_SELECTION_AUTHORITY /
RESIDUAL_AWARE_DDRNET_DEVELOPMENT_DESIGN_AUTHORIZED_NOT_EXECUTED /
FINAL_CONFIRMATION_NOT_ACTIVATED / DEFAULT_APP_UNCHANGED`

日期：2026-08-01（Asia/Hong_Kong）

## 结论

用户关于单 primary 假阴性的纠正已得到完整回答：R0.1 将另外两个预先提出、但当时未
repo-freeze 的方案作为 diagnostic-only shadows 全量执行。两个 shadow 均没有形成
预声明的 material signal，也没有 `H_min` 或跨 session 双向 Pareto winner inversion。
因此可以关闭下列精确三臂、固定阈值、静态手工门家族：

```text
CLASS_CONDITIONED_MULTI_NEGATIVE
CLASS_CONDITIONAL_TEMPORAL
MULTI_NEGATIVE
```

这个结论不能扩大成所有 conditional gating、learned gating、postprocess 或语义分割
方法失败。它说明当前 raw DDRNet 输出上的固定手工规则只能交换 FP 与 recall，不能同时
保护最弱 session 和两个 hazard classes。下一主边界为 residual-aware DDRNet
Development；本轮没有训练模型。

R0 primary 的 machine terminal、指标与哈希保持不可变。R0.1 不救援 primary、不选
shadow，也不恢复任何 Android、提醒、产品或安全权限。

## 执行与验证身份

- protocol：`DUAL_LOOP_SEGMENTATION_CONDITIONAL_GATING_R0_1`
- mode：`POST_R0_FORWARD_SHADOW_DIAGNOSTIC`
- 成功执行 Git：`827dcda976394cd4d2a0c6f5bc29993ada9d9d5d`
- shadow definition：
  `f664da109e5d1784f5883eca38fda559ba9deef7c37ab4bb4288130b00a84057`
- 输入：520 帧、11,757 raw components、10 个 consumed Development sessions
- 输出：520 行 frame metrics、23,514 行 shadow component decisions
- validator recovery Git：`dd0daacc3d847e94fae1e0000179ffbb796ce33d`
- 独立 validator：`VALID`，167,327 项检查，错误数 0

初始 implementation Git `6ef3014` 在 raw input 读取前因 list/single-binding loader
类型错误停止，无 output、mask 或指标。V2 只修 input routing。V2 结果产生后，初始
validator 又因 primary-summary schema 不一致在 0 项 aggregation checks 后停止；
validator recovery 只修摘要字段匹配，不修改已有 result/frame/component evidence。

## 两条 shadow 的全量结果

| arm | FP reduction | overall recall retention | minimum-session retention | boundary retention | obstacle retention | material |
|---|---:|---:|---:|---:|---:|---|
| R0 primary reference | 0.092572 | 0.942399 | 0.774580 | 0.945451 | 0.946764 | false |
| `CLASS_CONDITIONAL_TEMPORAL` | 0.284667 | 0.781123 | 0.612024 | 0.945451 | 0.791604 | false |
| `MULTI_NEGATIVE` | 0.109286 | 0.922445 | 0.629324 | 0.612015 | 0.946764 | false |

两条 shadow 的最弱 session 都是
`sanpo_real_v0:lmkIchCJ1RIKsZvbb4HjCDl85B2nOicv`。Shadow A 接近 FP 门，却同时失败
overall、minimum-session 与 obstacle recall；Shadow B 保住 overall 与 obstacle，
却把 boundary/step/curb recall retention 从 primary 的 `0.945451` 降到 `0.612015`。

| arm | FP `>=.30` | overall `>=.90` | min session `>=.80` | boundary `>=.80` | obstacle `>=.80` |
|---|---|---|---|---|---|
| `CLASS_CONDITIONAL_TEMPORAL` | FAIL | FAIL | FAIL | PASS | FAIL |
| `MULTI_NEGATIVE` | FAIL | PASS | FAIL | FAIL | PASS |

`R4` 对两臂均为 false；两臂也未在各自的 fixed-reference comparison 中严格支配任一
reference，因此 `N=false`、`MATERIAL=false`。

## Per-session stress

| source session | C1 FP reduction | C1 recall retention | C2 FP reduction | C2 recall retention |
|---|---:|---:|---:|---:|
| `5Llq...Kjb9` | 0.215916 | 0.722037 | 0.085968 | 0.904608 |
| `972O...UtfHl` | 0.419423 | 0.709208 | 0.175440 | 0.847803 |
| `CCG-...HlXk` | 0.348978 | 0.766792 | 0.143465 | 0.947473 |
| `eHxt...ewuXr` | 0.343115 | 0.719318 | 0.081680 | 0.940636 |
| `GxMb...qNnT` | 0.330748 | 0.729581 | 0.148944 | 0.899282 |
| `i2jg...m4T3` | 0.158760 | 0.835272 | 0.043366 | 0.975459 |
| `ic_B...iPzT` | 0.300415 | 0.701236 | 0.218325 | 0.799127 |
| `lmkI...Oicv` | 0.495485 | **0.612024** | 0.438097 | **0.629324** |
| `LRWT...5Ypp` | 0.251189 | 0.844539 | 0.023541 | 0.982239 |
| `yQ5I...13FO` | 0.231118 | 0.806275 | 0.068713 | 0.947966 |

C1 在每个 session 都更激进地降 FP，同时更伤 recall；C2 在每个 session 都更保守。
没有 session 由任一 arm 在两个维度同时严格支配另一 arm，所以 `H_cross=false`。
两臂也都不满足“只失败 minimum-session、其余四门全过”，所以 `H_min=false`。

## 组件、单变量与 family assessment

| arm | fully retained | partially retained | removed | split source components | post fragments |
|---|---:|---:|---:|---:|---:|
| `CLASS_CONDITIONAL_TEMPORAL` | 2,739 | 3,540 | 5,478 | 1,220 | 9,441 |
| `MULTI_NEGATIVE` | 4,406 | 2,499 | 4,852 | 618 | 8,188 |

C1 的 3,114 个 boundary components 与 primary 逐字段一致；C2 的 8,643 个 obstacle
components 与 primary 逐字段一致，mismatch 均为 0。

机器 result 的 generic summary 为 `MECHANISM_ONLY`；其底层冻结字段是
`any_material=false / H_min=false / H_cross=false`。按结果前已冻结的映射，科学 family
terminal 为：

```text
TWO_SHADOWS_WEAK_FIXED_HANDCRAFTED_GATING_FAMILY_STOP
```

这支持停止继续堆叠静态手工门；不得用这些 burned sessions 训练按 session 路由器。

## 可复算证据

| 文件 | SHA-256 |
|---|---|
| `result/result.json` | `2c2dfb0264bb329323d8b95cf4321f43fb7a18d900bb8994bb61012465d0b5d1` |
| `result/shadow_frame_metrics.jsonl` | `98b33cc62627a2099653f4c488eba6c5efdde5529657c18380b07cc5927cfef3` |
| `result/shadow_component_decisions.jsonl` | `deebf8d3d1828757a7acf69b4e175b1f613498a19125d5d80d47d84dfbeab4a5` |
| `validation.json` | `c376fa74064c120ef7330df3ffa6fb2042b3c76afa61770a77e4460a3ea3fe33` |

独立目录的第二次复算再次通过 167,327 项检查；frame 与 component 两个核心输出逐字节
一致。排除预期变化的 `git_head` 与 preflight `output_root` 后，规范化 result 也一致。

本地 evidence 位于
`artifacts.local/evidence/dual-loop-segmentation-conditional-gating-r0-1-shadow/`，被
Git 忽略。

## 下一边界

下一步不再试新的阈值、latch、固定类别规则或 oracle session routing。唯一科学主线是
另立 residual-aware DDRNet Development：保持同一 backbone 与四类全图任务，只改变
residual/FP-aware target、loss、hard-negative sampling 和 boundary/curb protection，
与冻结 baseline 做单变量比较。设计授权不等于训练已完成。

本结果不改变模型、Android、QNN/A568、risk/feedback、TTS、振动、提醒或默认 App；
`drives_alerts=false`、Confirmation 未激活，产品与安全 authority 均为 none。
