# Central obstruction Agent label readiness D0-A successor R0 result

状态：`COMPLETE / VALID / CENTRAL_OBSTRUCTION_AUXILIARY_FEATURE_ONLY /
D0_A2_NOT_AUTHORIZED / D0_A3_A4_STOPPED / DEFAULT_APP_UNCHANGED`

时间：2026-07-31（Asia/Hong_Kong）

## 结论

本 successor 完成了一个小而硬的 fresh calibration：3 个 session、6 个固定 clip、24
个 observation slot，由两个 `fork_context=false` 的独立 `luna_reader` Agent 分别完成
observation-level review。D0-A1 已烧毁的 11 clips 没有进入本轮验证，只用于设计固定单位
规则；没有读取 candidate output、YOLO、truth、risk、feedback 或旧 effect output；没有
启动第三 Agent。

固定 clip 的边界由冻结的 source timestamp、1 秒左闭右开窗口和四个固定 slot offset
直接生成，`fixed_unit_boundary_reproducibility=1.0`。这一步成功切掉了主观
parent-natural-event 分组：本轮没有 event onset/clearance 推断，也没有 event-match
指标。

但 observation 语义本身在全新来源上没有过门：两遍共匹配 `16/24=0.6667`，8 个分歧；
claim-critical agreement 同为 `0.6667`，低于 `0.80`。固定 clip summary 的 unit-state
match 为 `4/6=0.6667`，低于 `0.80`；unresolved fraction 为 `8/24=0.3333`，高于
`0.10`。因此终态不是 READY，而是正式将中央阻塞降为
`AUXILIARY_FEATURE_ONLY`，停止 D0-A3/A4，不扩展本路线。

## Frozen calibration 与 review

| 项目 | 结果 |
| --- | ---: |
| fresh session | 3 |
| fixed clip unit | 6 |
| observation slot | 24 |
| burned D0-A1 source overlap | 0 |
| production frame overlap | 0 |
| natural-event grouping | 未使用 |
| isolated Agent pass | 2 |
| third-Agent adjudication | 未使用 |
| `NOT_EVALUABLE` union | `0/24=0.0000` |
| fixed boundary reproducibility | `1.0000` |
| overall / claim-critical observation agreement | `16/24=0.6667` |
| unresolved fraction | `8/24=0.3333` |
| fixed-unit state match | `4/6=0.6667` |

8 个 disagreement 不是边界漂移，而是两个完整 fixed clip 的 observation 语义冲突：

- Matoaka `fixed_clip_01`：primary 的 4 个 slot 均为 `PRESENT`，isolated 的 4 个 slot
  均为 `NO_EVIDENCE`；争点是画面右侧的外围车辆是否构成中央视线阻塞。
- Shanghai `fixed_clip_00`：primary 的 4 个 slot 均为 `NO_EVIDENCE`，isolated 的 4 个
  slot 均为 `PRESENT`；争点是中央广告展示墙是否属于会阻断背景的 obstruction。

这说明固定单位已经把“事件边界不可复现”从转换函数中移除，但没有证明中央阻塞
observation 定义在新来源上可稳定复现。继续增加 D0-A3/A4 或再做第三 Agent 裁决只会
重新把语义争议包装成治理流程，不改变本轮的失败信息。

## Gate 与权限

所有门必须同时通过；本轮只有固定边界和样本量/来源量通过。

| gate | 结果 | threshold | pass |
| --- | ---: | ---: | --- |
| minimum fixed units | `6` | `>=6` | 是 |
| minimum source count | `3` | `>=3` | 是 |
| fixed unit boundary reproducibility | `1.0000` | `=1.00` | 是 |
| overall observation agreement | `0.6667` | `>=0.80` | 否 |
| claim-critical agreement | `0.6667` | `>=0.80` | 否 |
| unresolved fraction | `0.3333` | `<=0.10` | 否 |
| union `NOT_EVALUABLE` fraction | `0.0000` | `<=0.40` | 是 |
| fixed-unit state match | `0.6667` | `>=0.80` | 否 |

本结果明确写入：`d0a2_authorized=false`、`d0a3_authorized=false`、
`d0a4_authorized=false`。不启动 D0-AT、D0-B、模型效果评价、Android、默认生产、
真人助行、产品或安全结论。默认 App 行为不变。

## 与 D0-A1 的关系

D0-A1 的有效 observation 信息保持不变：raw observation agreement 为
`47/55=0.8545`，claim-critical union agreement 为 `39/47=0.8298`。这些数字仍由
[D0-A1 result](CENTRAL_OBSTRUCTION_AGENT_LABEL_READINESS_D0_A1_RESULT_2026-07-31.md)
负责，本 successor 不覆盖、不重算、不降格它们。

本轮新增的结论更窄：固定 clip 转换函数可单测、边界可复现，但 fresh source 上的
observation semantics 未过 `0.80 / 0.10` readiness family。中央阻塞因此只能保留为
observation-level 的非生产辅助特征候选；不能成为 D0-B 主阻塞算子或事件效果评价输入。

## Evidence identity

本地 evidence root：
`artifacts.local/evidence/central-obstruction-agent-label-readiness-d0-a-successor-r0/`

| artifact | SHA-256 |
| --- | --- |
| successor protocol | `70e786c705a161958beedc24f0990c713bafbf6490244a6d483c0d6e4dab8440` |
| calibration input manifest | `a7dad2e424373ff79b2abc506ee9ffb49c8e1e99c62d7899cf645f970149f0a4` |
| calibration input receipt | `fb1bc98b8b37c74a5cc1f36bd3fe4a145aea11dfde15ff690ffeadaba3c126b3` |
| primary review | `75de31bb091063881e2d5d0082c10f880be91fb57c50784dc6a728cccbd73518` |
| isolated review | `4e7a6395a672ed13e5e4d0fcb8d81ac30b3c1fa806bf25a124c19df28874120` |
| review capture receipt | `03024d2cf676922c544f3703a81ac8189f05e8aabc59f3161027a4201c208830` |
| calibration result | `9efed081d595e91cb68d7870666278b4e1006be432b279442d02813eb58137e6` |
| calibration validation | `79e46d9eb5e98c1fe011b3ad2fae1fe6917bfdcb15e6884dc142c93d3aebf164` |

## Statistical status and claim ceiling

这是小样本、重复 slot 嵌套于 clip 的描述性 `CANARY_LITE` calibration；没有把 24 个
slot 当作独立总体样本，没有运行推断检验、p 值、置信区间或效应外推。失败表示当前
Agent observation workflow 不足以支撑这条路线的下一门，不表示 RGB 中央阻塞概念在
所有来源上永远不可观察，也不构成客观真值、人类真值、模型增量、可通行性、产品或
安全证据。
