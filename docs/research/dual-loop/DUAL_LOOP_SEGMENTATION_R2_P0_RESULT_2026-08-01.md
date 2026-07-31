# DUAL_LOOP_SEGMENTATION_R2_P0 结果

状态：`COMPLETE / VALID / R2_NOT_WORTH_BURNING_FRESH_HOLDOUT`

证据层级：`DEVELOPMENT_STANDARD / READINESS_ONLY / NO_NEW_FORMAL_TRUTH`

## 结论

R2-P0 没有产生值得消耗新鲜 holdout 的候选。DDRNet-23-Slim 基线仍在
false activation 和增量 false-positive area 上失败；SegFormer-B0 还同时在 host
runtime 上失败。唯一一次预冻结的 36 点 DDRNet pipeline refinement 虽将 false
components 降至 `0.905/frame`，最接近候选的 delta false-positive area 仍为
`0.072513 > 0.05` 正式硬门和 `> 0.04` readiness margin，故未选择候选，也未运行该
候选的 runtime 或任何新 holdout 输出。

终态固定为：

`R2_NOT_WORTH_BURNING_FRESH_HOLDOUT`

这只淘汰当前候选与本次有界后处理搜索，不淘汰语义分割双环问题本身；R2、Android、
QNN、device benchmark、risk/event、主动提醒和路线 B 均未获授权，默认 App 不变。

## Current 入口与 R1 角色

`docs/research/dual-loop/README.md` 已同步为：

- `SEGMENTATION_MODEL_SELECTION_R1_BLOCKED`
- `MODEL_SELECTION_NOT_EVALUABLE`
- `R2_NOT_AUTHORIZED`
- `DEVICE_BENCHMARK_NOT_AUTHORIZED`
- `DEFAULT_APP_UNCHANGED`

R1 四个已消费 fresh session 通过独立 role amendment 永久降级为
regression/rehearsal/validator/canonicalizer-canary only。复制、重命名、重映射、
重新打包或 manifest alias 均不能恢复 fresh/unseen 身份。原 R1 role ledger、result、
failure receipt、closeout validator 和冻结身份未修改；readiness lock 独立重算并核验
formal freeze receipt 内 `22` 个身份。

## R2 协议草案

新增的 R2 草案保持 `DRAFT_NOT_AUTHORIZED_FOR_FORMAL`。问题改为单候选
qualification：只有一个新 pipeline identity 在 Development 上以 margin 通过全部门，
且 canonical/rehearsal/runtime validators 全部为 `VALID`，未来才可能另行授权 R2
formal；草案本身不构成执行授权。

## Canonicalization contract

source mask decoder 固定如下：

- `L`：像素值即 native ID；
- `P`：palette index 即 native ID，忽略 palette color；
- `RGB`：red channel 为 native class ID，忽略 green/blue instance metadata；
- `RGBA`：同 RGB，并忽略 alpha；
- 其他 mode、未知 native ID、canonical passthrough 中的非 `0..3` ID 全部 fail closed；
- 输出固定 `L / 256×256 / nearest-neighbor / canonical IDs 0..3`。

完整 native → canonical 映射为：

```text
0→3, 1→0, 2→1, 3→0, 4→2, 5→0, 6→0, 7→3,
8→2, 9→2, 10→2, 11→2, 12→2, 13→2, 14→2, 15→1,
16→2, 17→0, 18→2, 19→2, 20→2, 21→2, 22→2, 23→2,
24→2, 25→2, 26→2, 27→3, 28→2, 29→3, 30→3
```

冻结 SHA256：

| identity | SHA256 |
| --- | --- |
| canonicalizer code | `142ba3683270a85bad33d5f5e3773cd263c28fa9801823a6a8785655d309d1aa` |
| canonicalization config | `a1818b8b52ece55cc046c424defc4451a800790e15905c9c8f4b18a630fe67e1` |
| canonical-view schema | `7c2baca23ef7627819531e32247080cb51d1e0d3c2cdebb862dfbe76c60f47aa` |

## Materialized canonical view 与 rehearsal

materialized view 每行绑定 source/session/frame、source mask SHA、decoder、canonical mask
SHA、canonicalizer code/config/schema SHA。后续 evaluator 只能读取该 view manifest，不得
直接解释 source-native mask。

独立 validator 对 `924` 行全量复算：

- train `400`、dev `200`、consumed old blind `120`、R1 consumed fresh `200`、
  synthetic canary `4`；
- native `0..30` 全覆盖，canonical IDs 仅 `0..3`；
- manifest identity、行数、role/source/session 聚合、零行 fail-closed、原子发布和
  中断恢复全部通过；
- manifest SHA256：
  `7e8eb7de6eb74f57e35a9666a48ceb83bd817455557ee8ed32f56320dbb11d61`。

synthetic canary 覆盖 `L/P/RGB/RGBA`、全部 native ID、非法 native `31`、非法
canonical `4` 和不接受的 PNG mode；非法输入全部被拒绝。

R1 consumed fresh rehearsal 为 `200` frame、`5,043` component rows、4 source/session，
独立 validator 为 `VALID`。全量复算得到 delta recall `0.245424`、delta FP area
`0.100176`、component recall `0.669643`、false components `11.865/frame`。该结果仅为
consumed rehearsal，不是新 formal。

## Per-frame runtime

runtime row schema 固定 frame identity 和六阶段：

`preprocess / tflite_inference / output_dequantize_argmax /
component_extraction / fusion_operator / total_increment`

共保存 `200` 条 immutable rows。独立 validator 从底层 rows 重算 count、mean、P50、
P90、P95、min、max，而非复信 aggregate receipt。关键 P95 为：

- TFLite inference：`4.947520 ms`
- total increment：`25.795245 ms`

runtime rows SHA256：
`1001bafe73af3bf3ad15748a05bb675b8966504938951af85aa30c048ec4cc43`。

## Candidate gate matrix

正式硬门未放宽；readiness 另要求更大的通过 margin。

| candidate | delta recall | delta FP area | component recall | false comp/frame | seg P95 | total P95 | 结论 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| DDRNet R1 baseline | 0.248641 | 0.093471 | 0.776559 | 7.885 | 4.948 ms | 25.795 ms | FP、false components 失败 |
| SegFormer R1 baseline | 0.198221 | 0.068322 | 0.625086 | 3.645 | 60.707 ms | 74.139 ms | FP、false components、runtime 失败 |
| DDRNet bounded near-best | 0.199425 | 0.072513 | 0.556546 | 0.905 | 未运行 | 未运行 | FP 先行失败，不再花费 runtime |

SegFormer 的旧 runtime 只有 aggregate receipt，且 `74.139 ms` 已远超 `30 ms`，不得原样
进入 R2。DDRNet 搜索空间固定为 area `8/24/64` × confidence `0.50/0.65/0.80` ×
margin `0.05/0.15` × bottom-region `0.35/0.50`，共 `36` 组，选择规则在执行前冻结。
qualified count 为 `0`，selected candidate 为 `null`，搜索到此停止。

## 剩余 holdout 可用性与访问边界

metadata-only audit 覆盖官方 test split 的 `141` 个 session，排除六个已消费 session 后
剩余 `135` 个；其中 `24` 个仅从 GCS object-list metadata 可证明至少有 50 个对齐
RGB/mask object，错误数 `0`。审计下载 mask object `0`、读取 mask pixel `0`、运行候选
输出 `0`，且没有选择 holdout。

## 冻结身份与证据

在任何可能的新 fresh access 之前，readiness lock 已冻结 dev frames、dev components、
dev report、逐帧 runtime rows、YOLO trace、checkpoint、TFLite、postprocess config、
rehearsal evaluator/validator、runtime harness/validator、refinement config/report、
gate matrix 和三份独立 validation。主要身份见
`artifacts.local/evidence/dual-loop-segmentation-r2-p0/readiness-lock.json`；完整新增 tracked
文件、测试与本地证据 SHA 清单见
`artifacts.local/evidence/dual-loop-segmentation-r2-p0/artifact-inventory.json`。

readiness lock SHA256：
`99d364d38169da4bd6f2537dffd7baf43c6305d6ff13dd7ce28f2cad429cba73`。

## 验证

- R2-P0 module unit tests：`17 passed`；
- canonical-view、rehearsal、runtime independent validator：全部 `VALID`；
- closeout validator：显式要求 `14/14 PASS / VALID`；
- R1 formal freeze identities：`22/22` 重算一致；
- 最终仓库卫生、文档索引、diff、R1 不可变性和 Git remote parity 见提交交付记录。
