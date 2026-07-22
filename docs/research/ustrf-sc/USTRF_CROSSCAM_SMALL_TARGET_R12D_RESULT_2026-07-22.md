# USTRF stride-4/P2 小目标检测器 R1.2d 受控研究结果（2026-07-22）

## 结论

R1.2d 的预注册 P2 假设不成立，停止该候选，不进入 R1.3、INT8、Android 或生产替换。三组共享骨干初始化的 P2/P3 配对训练全部完成；两臂在三个 seed 上都只命中正事件 `4/6`，London 都是事件 `0/1`、可见锚点 `0/22`、帧 `0/22`，且都稳定漏掉 Bridge bollard。现有 YOLOE-768 外部参考仍为 `5/6`，只漏 London。

P2 在 17 来源离线验证上确实提高了总体、小目标和 London-like 几何召回，配对均值分别为 `+1.95pp / +2.20pp / +2.54pp`；但没有带来任何事件收益，同时离线假检测增加 `+0.236/图`，三个 seed 的路线内未分配检测压力为 `640 / 560 / 486`，稳定性也弱于 P3。这是“离线小框代理改善，但真实关键事件不闭合”的负结果，不能挑选单 seed 或调阈值回救。

## 冻结比较

- P2：官方 `yolo26-p2.yaml`，输出 stride `[4, 8, 16, 32]`，`2,516,504` 参数、`7.5 GFLOPs`。
- P3 因果对照：同系列 `yolo26.yaml`，输出 stride `[8, 16, 32]`，`2,504,970` 参数、`5.8 GFLOPs`。
- 两臂使用同一 YOLO26n 预训练骨干层 `0..10`，每个 seed 在训练首 batch 前的骨干张量 SHA-256 都是 `725c82530c1e3e7820faa83d9f20539fa5ffb64a5d21b049107cda27617d949d`；neck/head 才做 seed 化随机初始化。
- seed：`2026072201 / 2026072202 / 2026072203`；训练固定 `640`、30 epoch、batch 8、SGD、AMP、deterministic，未后验改变超参。
- 推理固定 confidence `0.05`、NMS IoU `0.45`、目标锚点 IoU `0.30`、max detections `100`；事件固定 500 ms 采样、2 次 miss 清除和既有 12 事件路线/关联协议。
- YOLOE-768 仅为外部参考；因架构、训练来源和输入尺度不同，不参与 P2 的因果归因。

P2 配置来自 [Ultralytics 官方 YOLO26 P2 YAML](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/models/26/yolo26-p2.yaml)，训练参数按冻结矩阵执行，而不是按本轮结果优化。

## 跨来源离线指标

括号内为三个 seed 的最差值至最好值；阈值保持 `0.05`。

| 指标 | P2 stride-4 | P3 control | P2−P3 配对均值 |
| --- | ---: | ---: | ---: |
| 总体 recall | `73.13%`（`71.71–74.34%`） | `71.18%`（`70.84–71.35%`） | `+1.95pp` |
| small recall | `71.17%`（`69.55–72.57%`） | `68.97%`（`68.68–69.20%`） | `+2.20pp` |
| London-like recall | `68.29%`（`66.55–69.97%`） | `65.76%`（`65.53–65.92%`） | `+2.54pp` |
| precision | `26.66%`（`25.68–27.49%`） | `26.74%`（`26.40–27.05%`） | `-0.07pp` |
| 假检测/图 | `7.588`（`7.374–8.000`） | `7.352`（`7.252–7.495`） | `+0.236`，更差 |
| 每 seed 的最差来源 recall | `60.31%`（`57.21–62.79%`） | `59.32%`（`56.28–61.40%`） | `+0.99pp` |

P2 的总体/small/London-like recall 标准差分别为 `1.08pp / 1.25pp / 1.39pp`，P3 为 `0.24pp / 0.22pp / 0.17pp`。P2 的增益方向在三组配对上相同，但波动显著更大。P2 的最差来源始终是 Minneapolis；P3 的全局最差 seed 也在 Minneapolis，另一个 seed 为 Indianapolis。

## 固定 12 事件结果

| 指标 | P2 stride-4，3 seed | P3 control，3 seed | YOLOE-768 外部参考 |
| --- | ---: | ---: | ---: |
| 正事件召回 | 每次 `4/6` | 每次 `4/6` | `5/6` |
| 关键漏检 | 每次 `2`：London、Bridge | 每次 `2`：London、Bridge | `1`：London |
| London 事件/锚点/帧 | `0/1 · 0/22 · 0/22` | `0/1 · 0/22 · 0/22` | `0/1 · 0/22 · 0/22` |
| 目标条件实际假告警 | 每次 `0` | 每次 `0` | `0` |
| 重复交付 / identity switch | 每次 `0 / 0` | 每次 `0 / 0` | `0 / 0` |
| 路线内未分配检测压力 | 均值 `562`（`486–640`） | 均值 `615.3`（`607–625`） | `344` |
| 关联覆盖 | `21.16%`（`17.35–26.03%`） | `15.98%`（`11.42–20.09%`） | `47.95%` |
| 关联歧义帧率 | `12.02%`（`8.68–15.07%`） | `12.48%`（`11.87–13.70%`） | `3.65%` |
| 最差来源关联覆盖 | 每次 `0%`，London | 每次 `0%`，London | `0%`，London |
| 最差来源歧义率 | 均值 `36.36%`（`27.27–45.45%`），Thailand | 均值 `39.04%`（`35.29–45.45%`），Thailand/Sidewalk | `19.05%`，Edmonton |

事件告警生成不读取 `expected_class`；该字段只在事后计分时读取。因此 `0` 次目标条件实际假告警不等于全局无误报。路线内未分配检测压力显示三种模型仍产生大量未归因检测，P2 的压力标准差/范围为 `62.89 / 154`，明显大于 P3 的 `7.41 / 18`，不能把 P2 的较低压力均值解释为稳定优势。

## 清除延迟

- P2/P3 每个 seed 都有 `3` 个可观察清除、`1` 个删失清除；可观察样本 p50/p95 都为 `0/0 ms`。
- YOLOE 同样有 `3` 个可观察清除、`1` 个删失清除；p50/p95 为 `0/500 ms`。
- London 从未建立目标轨迹，按协议记为 `10,000 ms` 删失，而不是“快速清除”。Bridge 在 P2/P3 中也从未命中，但该片段没有可观察的离场清除窗。
- 可观察清除数量很少，且包括 truth-blind 负事件轨迹；`0 ms` 只能说明这些有限轨迹在冻结采样下及时消失，不能抵消 London/Bridge 的关键漏检。

## 数据与证据强度

准入数据收据为 `small-target-r12d-dataset-v2/dataset_receipt.json`，SHA-256 `cd3bb7707d88fa34d804214fb44ef50c658473e9ae95442d9a58d5d26dc1d540`：

- `2,106` 个唯一图像；训练 `1,440`、跨来源验证 `666`，每轮训练含 bollard 重复采样共 `1,720` draws；验证覆盖 17 个非 Pittsburgh 来源。
- 唯一框为 cone `7,109`、delineator `2,186`、bollard `140`；事件帧、synthetic 和 provisional 标签均未进入训练。
- RoadWork 使用 Pittsburgh 训练、其他城市验证。其官方页面描述了自动/人工标注后人工复核；本轮只使用收据绑定的 COCO 几何和许可。来源见 [CMU ROADWork](https://www.cs.cmu.edu/~roadwork/) 与 [官方仓库](https://github.com/anuragxel/roadwork-dataset)。
- bollard 数据来自 [Mendeley 数据集 v2](https://data.mendeley.com/datasets/3psr2g4s7r/2)，许可为 CC BY 4.0；150 个原始图像/标签只形成 40 个唯一图像，39 组存在标签版本分歧。它仅进入训练，Bridge 为不相交事件泛化；三次 P2/P3 都漏掉 Bridge，说明 bollard 证据仍很弱。

数据 v1 在训练 cache 前暴露 679 行完全重复的序列化标签，本轮在看任何事件结果前改为按 9 位归一化行去重、保留最小 annotation ID，并重建 v2 收据。该处理显式化了 Ultralytics 原本会隐式执行的重复行移除。

## 受控性审计与失效回路

只有 `small-target-r12d-training-v4` 被接纳：

1. training-v1 因数据重复行未在收据中披露而在结果前拒绝。
2. training-v2 暴露 Ultralytics 从 YAML 启动 trainer 时会重建随机模型、丢失预先转移骨干；在事件评测前拒绝。
3. runner 改为先保存全精度初始化 checkpoint，再由 trainer 重载，并在首 batch 前回调核对骨干张量哈希。
4. training-v3 因旧训练进程残留造成显存竞争而拒绝；清理后保持原 batch，不改变矩阵。
5. training-v4 六份回执全部通过权重、矩阵、数据收据与三组骨干配对哈希校验。

评测器另补了模型族分辨率回归门：R1.2d 候选只能是 `640`，YOLOE 外部参考只能是 `768`；错误覆盖在推理前 fail closed。

## 决策与下一步

- `p2_hypothesis_supported_under_preregistered_rule = false`。
- 不选择任何 seed，不调 confidence/NMS/tracker，不继续同一 P2 路线的分辨率搜索。
- 不运行 R1.3，不执行 INT8、Android benchmark、App 默认模型替换或生产 promotion。
- 若未来重新打开 detector 研究，必须是新的、独立可归因预注册路线，优先解决 London 真实外观域和 Bridge bollard 数据质量/覆盖；仍需以固定事件和最差 seed/source 为门，不能只刷 detector AP。
- 即使后续 seen 事件改善，也仍不构成人类 route-conditioned event truth 或设备米制 route/pose/depth 几何授权。

## 可复现证据

- 冻结矩阵：`configs/ustrf_crosscam_small_target_detector_r12d_matrix_v1.json`，SHA-256 `f2c33aa891013da8e363dd6b981587a03a630531152dd2e911f4e832c20fe469`。
- 数据：`artifacts.local/evidence/ustrf-crosscam-codex/small-target-r12d-dataset-v2/`。
- 六个训练：`artifacts.local/evidence/ustrf-crosscam-codex/small-target-r12d-training-v4/`。
- 七个统一评测与汇总：`artifacts.local/evidence/ustrf-crosscam-codex/small-target-r12d-results-v2/`。
- YOLOE TFLite SHA-256：`aa274c986ec6e360717b07efda06eb3ebe045cdd73c0ff71e1a1329bec1fc407`。
