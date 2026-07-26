# RCLE Phase B 小型 real-data geometry canary R0 预注册

状态：`DESIGN_FROZEN / EXECUTION_NOT_AUTHORIZED`

日期：2026-07-26

机器合同：
[RCLE_PHASE_B_REAL_DATA_GEOMETRY_CANARY_R0_CONTRACT_2026-07-26.json](RCLE_PHASE_B_REAL_DATA_GEOMETRY_CANARY_R0_CONTRACT_2026-07-26.json)

## 结论先行

本轮只冻结一个 **F1 implementation canary**：在已经烧掉的 TUM
`fr2/rpy` 上，用四个固定 10 s 窗检验 real-data geometry producer 与独立
validator 能否逐 pair 保持相同的输入身份、分母、字段结构、弃权语义和 float64
数值。

它不是 RCLE RGB 算法实验，也不重新证明 `fr2/rpy` 是 rotation source。四窗来自
已经查看过 geometry outcome 的同一 sequence，pair 之间还有时间相关性，因此本轮
没有独立科学样本量、没有统计功效声明、没有 p 值，也没有 confirmation 权限。
唯一可取得的正面结论是：

`VALID / IMPLEMENTATION_DEBUGGED`

即 real-data geometry interface 已基本调通，足以让后续任务**审查是否值得另立**
RCLE RGB algorithm canary；它不自动授权实现或运行该后继。

## 为什么现在做这个 canary

前一轮 TUM source-native audit 已确认：

- 固定 10 窗中 `9/10` 可评价；
- 窗 `0/3/6` 是已查看过的 rotation-dominant geometry；
- 窗 `4` 是唯一 window-level `SOURCE_DEPTH_COVERAGE_LT_0P50`；
- 全序列 `2852/2990` pair 可评价；
- 来源、注册深度、color-camera pose、时间戳和 PB-H1 geometry 公式可对齐。

但 Bonn B1A 的旧失败说明，连续几何数字看似合理仍不等于 execution
ready：其 24 个弃权 pair 出现了 216 个 blank-grid key-set mismatch，并连带
破坏 ledger identity。所以下一最小高信息量问题不是再下载一个来源，也不是直接
运行 RGB 算法，而是让一个小型真实 cohort 同时覆盖成功分支和 fail-closed 分支，
专门检验 producer/validator interface。

## 阶段、问题与非目标

- `protocol_id`：`RCLE-PHASE-B-REAL-DATA-GEOMETRY-CANARY-R0`
- `stage`：`CANARY`
- `freeze`：`F1`
- 主要问题：独立实现能否在真实 RGB-D/pose 输入上确定性复现 pair identity、
  geometry 和弃权序列化？
- 允许声明：`IMPLEMENTATION_DEBUGGED` 或 `NOT_EVALUABLE`。

本轮明确不回答：

- 旋转补偿后的 RCLE RGB signal 是否优于 raw flow；
- `fr2/rpy` rotation role 是否获得新的独立支持；
- 是否存在 real approach role；
- 跨来源泛化、confirmation、Kill Gate B；
- Replay、Android、真人、助行安全或产品能力。

## 数据角色与泄漏披露

唯一数据为官方 TUM `rgbd_dataset_freiburg2_rpy.tgz`：

```text
SHA-256
3a35b7999af8631b6421ed9087a5325ffea1564c23d52ba640f0c239af62b51f
```

它在前一 Discovery audit 中已发生全十窗 `GEOMETRY_ONLY` outcome access。本轮
角色明确降为：

`CANARY / GEOMETRY_ONLY / BURNED`

它不能再作为同一 geometry role 或未来 RCLE 命题的 outcome-blind confirmation
数据。以后即使换 runner、目录或协议名称，也不能恢复独立性。

canary producer 不得读取前一 audit 的 `result.json`，不得把旧 metric 值作为运行
输入。旧结果只用于在设计期公开选择来源：取全部三个 rotation-dominant 窗，加上
唯一 window-level depth abstention，避免只覆盖成功路径。

## 固定 cohort

| 窗 | Unix 半开区间 | 既有访问角色 | candidate pair | 既有 pair 可评价 | 既有 window disposition |
| ---: | --- | --- | ---: | ---: | --- |
| 0 | `[1311867719, 1311867729)` | rotation-dominant | 299 | 299 | evaluable |
| 3 | `[1311867749, 1311867759)` | rotation-dominant | 299 | 299 | evaluable |
| 4 | `[1311867759, 1311867769)` | depth-coverage stress | 299 | 293 | `SOURCE_DEPTH_COVERAGE_LT_0P50` |
| 6 | `[1311867779, 1311867789)` | rotation-dominant | 299 | 299 | evaluable |

总分母固定为 `1196` 个 pair record；既有可计算数为 `1190`。这些数字是 branch
coverage 基线，不是独立统计样本量。禁止增加、替换、滑动窗口，也禁止在失败后用
窗 `1/2/5/7/8/9` 救结果。

## 冻结 interface

### Producer

后续若获单独实现授权，只能新增版本隔离的
`real_data_geometry_canary_r0/` 与单一 runner：

- 只读本合同绑定的 TUM archive、source-native index/pose/depth；
- 可复用已绑定的 PB-H1 geometry primitive；
- 不得 import 前一 TUM audit runner/module；
- 不得读取前一 `result.json`；
- 不得 decode RGB image 或运行 RCLE、LK、optical flow、local affine、score。

### Independent validator

validator 必须从绑定的 raw archive 独立复算 association、pose interpolation、
depth sampling、geometry、pair record 和 window summary：

- 不 import producer；
- 不 import 前一 TUM audit 实现；
- 不以 producer 的中间量作为真值；
- 对所有 `1196` 个 denominator pair 逐项比较，而不是只比较 summary/hash。

“独立”在这里指实现依赖隔离，不代表数据独立或 confirmation 独立。

### Pair schema

每个 denominator pair 都必须输出同一组键。成功分支写满可用 metric；弃权分支保留
同一 key set、一个明确 `abstention_reason`，不可用 metric 写 `null`。任何条件
分支增加、遗漏或改名字段都计为 schema mismatch，不能用总体成功率掩盖。

主要连续量保持 PB-H1 定义：

- raw translation speed，`m/s`；
- angular speed，`deg/s`；
- signed / absolute translation-induced radial expansion，`s^-1`；
- positive expansion fraction，`[0,1]`；
- Q90 time-normalized parallax，`rad/s`；
- valid depth fraction，`[0,1]`。

## 判定规则

所有 gate 同时通过，才可得到
`VALID_IMPLEMENTATION_DEBUGGED_GEOMETRY_INTERFACE_ONLY`：

1. archive、上游 contract/result 和 PB-H1 geometry SHA-256 全部匹配；
2. 只输出固定 `4` 窗；
3. producer 和 validator 各输出 `1196` 个 pair record；
4. ordered pair identity mismatch 为 `0`；
5. pair key-set mismatch 为 `0`；
6. pair abstention 与 window disposition mismatch 为 `0`；
7. float64 metric parity violation 为 `0`；
8. 窗 `0/3/6` 各为 `299/299` evaluable，窗 `4` 为 `293/299` pair-evaluable
   且 window-level 保持 `SOURCE_DEPTH_COVERAGE_LT_0P50`。

数值 parity 对每个 metric 使用：

```text
abs(error) <= 1e-12 OR relative_error <= 1e-10
```

该容差来自 PB-H1 解析 fixture 的 `<1e-12` 误差与当前固定 Windows/Python
float64 环境的 deterministic replay。另报告
`abs<=1e-10 OR rel<=1e-8` 的放宽敏感性，但它只作诊断，不能救 R0。

任何有效建立的 gate mismatch 都令本 implementation version 得到
`VALID / NOT_EVALUABLE` 并关闭 R0；它不是 RCLE 科学失败。source、contract 或
执行证据身份无法验证时为
`INVALID / NOT_EVALUABLE_DUE_TO_EXECUTION`，只关闭 evidence version。

## 统计与报告

本 canary 的 pair 是高频、同序列、时间相关记录；四窗又是依据既有 geometry
outcome 选择的 branch-complete cohort。因此：

- 不把 `n=1196` 写成 1196 个独立样本；
- 不运行 t 检验、秩和检验或显著性检验；
- 不做 seed/frame bootstrap 来制造精度感；
- 不以 pooled coverage 代替逐窗 disposition；
- 连续量只报告每窗 min、Q10、Q25、median、Q75、Q90、Q95、max；
- 主要证据是逐记录 deterministic parity 和 exact branch coverage。

这个样本预算来自软件 canary 的分支覆盖，不来自统计功效分析。未来
confirmation 必须使用未访问的独立 sequence/partition 和另行冻结的样本量依据。

## 修改、停止和合法后继

geometry outcome 已在前一 audit 中访问，所以 R0 从冻结起只能
`NEW_VERSION_ONLY`：

- 不允许原地改 cohort、阈值、key set、association、pose 或 null 语义；
- 不允许失败后加第二个来源或删除窗 `4`；
- 允许查看 canary 输入/输出定位错误，但修复后必须保留 R0 并新建 R1；
- INVALID 只影响本 evidence version；
- NOT_EVALUABLE 只影响本 implementation version。

R0 若未来 `VALID / IMPLEMENTATION_DEBUGGED`，唯一自动开放内容是：可以在独立任务
中**审查是否值得设计** RCLE RGB algorithm canary。它不自动授权实现、运行、调参、
confirmation、Kill Gate B 或产品工作。

## 当前交付边界

本轮没有：

- 新增 producer、validator 或测试；
- 解码任何新 RGB/depth；
- 运行这四窗 canary；
- 下载新来源；
- 读取或运行 RCLE RGB algorithm outcome；
- 生成 formal evidence receipt。

当前终态保持：

`DESIGN_FROZEN / EXECUTION_NOT_AUTHORIZED`
