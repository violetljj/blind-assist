# BA-Clear/Q-Plane O0-A 表示上界结果

状态：`FAIL / CLOSE_QPLANE / NO_O0B / NO_TRAINING`

## 结论

query-local inverse-depth ray-plane residual 没有形成可晋级的表示优势。A4 的 parent-macro
clearance MAE 为 `0.17284 m`，差于 A1 global scale 的 `0.14590 m`、A2 global affine 的
`0.15373 m` 和 A3 global ray-plane 的 `0.15405 m`；A3→source-oracle gap closure 为
`-12.20%`。因此冻结门失败，Q-Plane 关闭，不进入 O0-B runtime identifiability，更不授权 learned head。

这不是“完全没有局部信号”：A4 比 shuffled-query 好 `0.00691 m`，center/left 两个 band 与总体
false-block 有改善，coverage 与 A0 同为 `1.0`。但这些局部收益不能覆盖 right band、长 horizon 与
逐 parent false-clear 退化，不能包装成成功。

## 主臂 parent-macro

| arm | clearance MAE (m) | bias (m) | false-block | false-clear | coverage |
|---|---:|---:|---:|---:|---:|
| A0 Frozen DepthART | 0.14807 | 0.08106 | 1.111% | 9.722% | 1.000 |
| A1 global scale | **0.14590** | 0.07772 | 1.944% | **8.148%** | 1.000 |
| A2 global affine | 0.15373 | 0.09717 | 0.926% | 11.389% | 1.000 |
| A3 global ray-plane | 0.15405 | 0.11326 | 1.123% | 10.475% | 0.972 |
| A4 query-local ray-plane | **0.17284** | 0.10138 | 0.926% | **13.241%** | 1.000 |
| A5 source-depth oracle | 0.00000 | 0.00000 | 0.000% | 0.000% | 1.000 |

## A4 相对 A3 的分解

逐 parent 只有 `pose-000` 的 MAE 改善；false-block rate 改善为 `2/4` parent，而冻结门要求 `3/4`。
`pose-000`、`pose-016` 和 walking-xyz 的 false-clear 分别增加 `4.07 / 5.14 / 2.59 pp`，超过逐 parent
`0.5 pp` 上限。

| parent | Δ MAE (m) | Δ false-block | Δ false-clear | Δ coverage |
|---|---:|---:|---:|---:|
| walking-static pose-000 | -0.03324 | 0.000 pp | +4.074 pp | 0.000 |
| walking-static pose-016 | +0.05559 | -0.046 pp | +5.139 pp | +0.111 |
| walking-static pose-020 | +0.02663 | -0.741 pp | -0.741 pp | 0.000 |
| walking-xyz pose-024 | +0.02618 | 0.000 pp | +2.593 pp | 0.000 |

按 band，left/center 的 MAE 从 `0.13107/0.26656` 降到 `0.12731/0.21268 m`，但 right 从
`0.06850` 恶化到 `0.17853 m`。按 horizon，只有 `1.0 m` 从 `0.15220` 小幅降到 `0.14807 m`；
`1.5/2.0 m` 分别恶化到 `0.15737/0.21308 m`，且 `2.0 m` false-clear 从 `19.43%` 升到 `26.67%`。

逐 query reporting replay 进一步定位到 `5/9` query 的 MAE 改善全部集中在 center 三个 horizon 和
left `1.0/1.5 m`；left `2.0 m` 与 right 三个 query 全部退化：

| query | A3 MAE (m) | A4 MAE (m) | A4−A3 (m) | A3→A4 false-clear |
|---|---:|---:|---:|---:|
| center@1.0m | 0.26656 | 0.20501 | -0.06155 | 0.00% → 0.00% |
| center@1.5m | 0.26656 | 0.20780 | -0.05876 | 18.18% → 16.67% |
| center@2.0m | 0.26656 | 0.22523 | -0.04133 | 30.91% → 31.67% |
| left@1.0m | 0.13107 | 0.08244 | -0.04863 | 0.00% → 0.00% |
| left@1.5m | 0.13107 | 0.08822 | -0.04285 | 12.50% → 13.33% |
| left@2.0m | 0.13107 | 0.21128 | +0.08021 | 28.33% → 30.00% |
| right@1.0m | 0.06850 | 0.15677 | +0.08828 | 0.00% → 0.00% |
| right@1.5m | 0.06850 | 0.17609 | +0.10759 | 5.83% → 9.17% |
| right@2.0m | 0.06850 | 0.20273 | +0.13424 | 0.00% → 18.33% |

## 负控与机制读数

- shuffled-query MAE `0.17975 m`，A4 好 `0.00691 m`，通过冻结的 shuffled advantage 门；
- wrong-gravity MAE `0.17195 m`，反而略好于 A4 的 `0.17284 m`；
- wrong-`K` MAE `0.17300 m`，与 A4 几乎相同；
- shared/globalized theta 即 A3，MAE `0.15405 m`，明显好于 A4。

因此结果更像“局部权重产生少量 query specificity，但 query-local plane 参数总体不稳、对正确
gravity/K 缺少正向辨识”，不是一个值得进入 runtime predictor 的 oracle 表示。

## 防火墙与运行回执

- 最终 `candidate-plan.json` 在 task evaluation 前写入并固定 SHA-256：
  `BA8D9845630C65F4976F73050F8E97D8598C9AF2B0D66A288447A09FF0B4BAF6`；
- `120` 帧、`1080` 个 query parameter vector、每 query `3` 参数、每帧 `27` DoF；
- 最少/最多拟合像素 `1650 / 23893`，拟合像素与 obstacle evaluation cell 最大交集 `0`；
- 未用 task label 优化，未持久化 corrected dense depth，未读取 fresh outcome；
- host CPU Phase A `83.20 s`、Phase B `211.72 s`、总计 `295.01 s`；仅为运行诊断。
- query decomposition 是绑定原 result/candidate SHA 的 reporting-only replay，未 refit candidate、未改参数
  或 gate；CPU `255.76 s`，回执 SHA-256
  `AEA0C8DB4896AB4F434A9AB268D1887CF9486371F192DED4E3176C2FC9107034`。

正式 candidate freeze 前有两次 Phase-A-only 中止：一次暴露 wrong-gravity 支持消失，一次暴露
`1.0 m` query 的相机视场支持不足；二者均未生成 candidate plan、未进入 task evaluation。最终协议在
任何 task outcome 打开前冻结为 source-support horizon-weighted ridge，并固定负控拟合像素 ID。

机器摘要见同名 JSON；完整本地结果为
`artifacts.local/experiments/ba-clear-qplane-o0a-headroom-r0/result.json`，SHA-256
`8C8487D05C2F831EA51019EB680E1107AF78BD59EE9C4A6E920365BB86745B04`。

## 决策

`NONE_CLOSE_QPLANE_NO_TRAINING`。禁止结果后重调 horizon 权重、epsilon、ridge、query mask 或门限；
禁止创建 O0-B、selector、UNKNOWN/deferral head、teacher/student training、Android/QNN/HTP successor。

证据上限：consumed Development representation-headroom negative；不证明 runtime identifiability、训练、
跨数据泛化、部署、产品或助行安全。
