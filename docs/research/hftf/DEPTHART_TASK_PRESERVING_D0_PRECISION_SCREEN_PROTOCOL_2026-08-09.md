# DepthART task-preserving D0 precision screen

状态：`CLOSED / NO_ELIGIBLE_ARM / R2_NOT_ACTIVATED / DEVELOPMENT_ONLY`

机器合同：[`DEPTHART_TASK_PRESERVING_D0_PRECISION_SCREEN_PROTOCOL_2026-08-09.json`](DEPTHART_TASK_PRESERVING_D0_PRECISION_SCREEN_PROTOCOL_2026-08-09.json)
公共源图与诊断 control：[`DEPTHART_TASK_PRESERVING_D0_SOURCE_CONTROL_LOCK_2026-08-09.json`](DEPTHART_TASK_PRESERVING_D0_SOURCE_CONTROL_LOCK_2026-08-09.json)

当前执行进度：[`D0_FP16_R0` technical preflight](DEPTHART_TASK_PRESERVING_D0_FP16_TECHNICAL_PREFLIGHT_RESULT_2026-08-09.md)
· [W8A16/INT8 shared calibration roster](DEPTHART_TASK_PRESERVING_D0_TUM_CALIBRATION_ROSTER_2026-08-09.json)
· [D0 terminal result](DEPTHART_TASK_PRESERVING_D0_PRECISION_SCREEN_RESULT_2026-08-09.md)

## 决策

D0 先在 Development 数据上比较三条预声明硬件友好路线，再只冻结一个候选进入独立 R2：

| arm | 标准算子表示 | custom island | 当前含义 |
|---|---|---|---|
| `D0_FP16_R0` | FP16 | SelectiveScan/LayerNorm 仍要求 FP32 | 先验证 QAIRT 是否能合法插入固定边界 cast 并生成 v75 context |
| `D0_W8A16_R0` | W8A16 | 同上 | 8-bit weights、16-bit activations；量化校准 roster 与 INT8 完全相同 |
| `D0_INT8_R0` | W8A8 | 同上 | 作为最高压缩 arm；不能以性能掩盖 false-clear 或 UNKNOWN 退化 |

既有 G4-C fixed-mixed 图只作为公共 converter source 与诊断 control，不是已经选中的 R2
候选。strict-repair PatchConv/BatchNorm/GELU 及近完整 custom-float32 engine 均不属于 D0。

## 工具与技术前门

已从本机 QAIRT 2.47.0.260601 CLI 核准：converter 支持 `float_bitwidth=16`；quantizer
支持 `weights_bitwidth=8`、`act_bitwidth=8/16`、同一 raw input calibration list 及
per-channel Conv weight quantization。当前 custom op definition 则把 SelectiveScan 和
LayerNorm tensor 明确限制为 `QNN_DATATYPE_FLOAT_32`。

因此每臂先过技术前门：转换、SM8650/v75 context、saved-context 真机执行、finite output、
完整 HTP 且无 CPU fallback、custom family 数量与 FP32 dtype 不漂移。任何 arm 在冻结 recipe
下失败，只记 `D0_ARM_TECHNICALLY_INELIGIBLE_NO_RECIPE_REPAIR`；不得临场修改 custom kernel、
量化例外或 graph 来救该 arm。技术不可行不等于算法质量 FAIL。

## 数据与质量顺序

- D0 只用专门冻结为 calibration/selection 的 Development 数据；既有 consumed R0 exact rows
  与独立 R2 cohort 都不得复用，结果也没有独立确认 authority。
- 量化 calibration 使用另行冻结的 16-frame TUM RGB roster；它先排除了既有 120-frame
  consumed R0 rows，两个 sequence 各 8 帧，W8A16/INT8 必须共用其精确 SHA-256 身份。
- 新锁定的 ARKit 8-video R2 roster 保持 metadata-only、payload 未读取；它不得参与 calibration、
  threshold 选择、arm 排名或任何 D0 调试。
- reference 与三臂使用相同 frame、intrinsics、ground/clearance/risk 后处理和 UNKNOWN 规则。
- 每臂先按 R2 已冻结的全部绝对门与非劣门计算任务质量。质量不合格，不测该臂性能。
- 只有质量合格臂才测 bounded device p50/p90、peak RSS 与 context bytes；thermal 留给最终单一
  候选，不在三臂筛选时扩大成本。

## 单一候选选择

候选集合是同时通过技术前门和全部任务门的 arms。选择顺序在 outcome 前固定为：

1. false-clear/all-known 增量最小；
2. clearance MAE 增量最小；
3. temporal clearance-delta MAE 增量最小；
4. device p90 latency 最小；
5. peak RSS 最小；
6. arm id 字典序，仅作完全相同结果的确定性终结规则。

筛选后必须生成 selection receipt 和新的单一 R2 candidate lock，随后 pre-outcome validator
才可绑定该身份。若无任何 arm 合格，终态为
`D0_NO_TASK_PRESERVING_CANDIDATE_R2_NOT_ACTIVATED`，不能访问独立 R2 outcome。

## 权限边界

D0 只产生候选选择证据。它不改写 strict G4-D，不证明 R2 task equivalence，不授权替换 DA2、
默认 Android、生产或 safety。R2 独立 cohort 只能确认一个已经由 D0 选定并锁定的候选。
