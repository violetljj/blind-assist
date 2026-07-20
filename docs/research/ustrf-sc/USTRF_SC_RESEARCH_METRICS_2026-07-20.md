# USTRF-SC 研究型离线量化基线（2026-07-20）

本记录按研究原型目标组织：优先比较算法表征和数据量化指标，不把 Android 上线或用户反馈接入作为本轮完成条件。数据仍保留可追溯来源、哈希和可复现命令；这些是实验可解释性的基础，而不是产品授权声明。

## 数据与协议

- 数据根：`artifacts.local/evidence/datasets/sanpo-v3-canonical-evidence-v4-20260713`
- 训练/dev：200 / 100 帧，4 / 2 个 session；blind holdout 未由探针读取。
- 权重：`artifacts.local/evidence/segmentation-candidate/p1-sigmoid-no-pooled-bn-20260713/candidate.weights.h5`
- 模型：MobileNetV3Small + LR-ASPP，4 类；`256×256`，每类均衡抽取 128 个冻结特征点，float64 closed-form ridge，重复 2 次。

## 结果

| 冻结特征 | global mIoU | 最差 scene mIoU | obstacle IoU | unknown IoU | boundary/step/curb IoU | 可分离门 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| raw `activation_1` | 0.1133 | 0.0391 | 0.1270 | 0.0598 | 0.000093 | 否 |
| decoder `lraspp_fuse` | 0.2403 | 0.1564 | 0.3244 | 0.3609 | 0.000089 | 否 |

两次重复的系数 SHA256 完全一致，故上述差异是该确定性设置下的实际表征差异，而非随机训练波动。完整 receipt 分别见：

- `artifacts.local/evidence/ustrf-sc/sanpo-linear-probe-20260720/probe_report.json`
- `artifacts.local/evidence/ustrf-sc/sanpo-linear-probe-fuse-20260720/probe_report.json`

## 结论与下一实验

1. `lraspp_fuse` 明显优于 raw backbone：当前弱点不只是线性读出，decoder 语义融合确实携带更可分的 obstacle/unknown 信息。
2. boundary/step/curb 在两路均接近零，不能将通用 segmentation 的提升解读为台阶风险能力。
3. 下一优先级应是 boundary/step 的目标化数据增广、距离/边界辅助头和跨 scene 最差分位指标；基线扩训前先要求该类别的独立消融优于当前近零 IoU。

## 边界数据诊断

对同一 canonical 数据根的 256×256 mask 做 signed distance 诊断后：

| split | 帧数 | 含 boundary 帧数 | boundary 像素占比 | `abs(distance)<1` 像素占比 | 平均 distance-loss 权重 |
| --- | ---: | ---: | ---: | ---: | ---: |
| train | 200 | 77 | 0.2901% | 2.1226% | 0.06375 |
| dev | 100 | 45 | 0.0468% | 1.8806% | 0.05825 |

dev 的 boundary 像素密度约为 train 的 16%，而非仅仅少几个样本。这是当前 boundary IoU 接近零的重要数据分布解释，也说明下一轮应以 session-balanced boundary sampling / boundary-distance auxiliary 为变量，配对比较，而不是把现有 global mIoU 当作台阶能力。

诊断 receipt：`artifacts.local/evidence/ustrf-sc/sanpo-boundary-distance-diagnostic-20260720.json`。

## GPU boundary-distance 配对消融

训练使用 NVIDIA RTX 5060 Laptop GPU / PyTorch CUDA 13.0 / Keras-Torch。为避免读取 canonical dev 或 blind，训练只从 canonical train 的四个 session 中留出 `center_obstacle` 与 `parallel_boundary` 两个 session；留出/优化 boundary 像素覆盖比为 0.496，低于理想均衡但在本轮明确记录并以相同划分做 paired baseline 对照。

| 步数 | 路径 | mIoU | obstacle IoU | unknown IoU | boundary IoU | 结论 |
| ---: | --- | ---: | ---: | ---: | ---: | --- |
| 20 | baseline | 0.1702 | 0.1636 | 0.0129 | 0 | 低步数基线 |
| 20 | `+ distance auxiliary` | 0.1643 | 0.1440 | 0.0182 | 0 | 无改善 |
| 100 | baseline | 0.2502 | 0.2840 | 0.1007 | 0 | 通用类收敛，边界仍未出现 |
| 100 | `+ distance auxiliary` | 0.2479 | 0.2852 | 0.0898 | 0 | distance weighted MAE=0.4426，但未转化为 boundary IoU |

对应 receipt：

- `artifacts.local/evidence/ustrf-sc/sanpo-boundary-distance-ablation-20260720/report.json`
- `artifacts.local/evidence/ustrf-sc/sanpo-boundary-distance-ablation-100step-20260720/report.json`

这组 paired GPU 结果排除了“仅增加训练步数”或“直接添加 signed-distance loss”就能得到 boundary 能力的假设。下一项研究变量应改变：补充 boundary/step 的独立连续来源或以实例/线段/近场几何监督替代当前像素稀疏的四类分类目标；再以至少两个覆盖相近的 held-out session 复验。

## real-only 覆盖匹配复验

为隔离 v3 中 procedural 零边界 session 的影响，在 `sanpo-v4-real-canonical-r3-20260713` 的 8 个真实 session 中选择 `9m1... + SRHp...` 作为 holdout。其 boundary 覆盖比为 **0.906**（满足 0.8–1.25 区间），其余 6 个真实 session 用于优化；其余协议保持为 GPU、100 step、256 输入、同一模型与采样策略。

| seed pair | baseline boundary IoU | `+ distance` boundary IoU | Δ boundary | Δ global mIoU |
| --- | ---: | ---: | ---: | ---: |
| 2026072001:2026072002 | 0.01011 | 0.01155 | +0.00145 | −0.00079 |
| 2026072003:2026072004 | 0.00995 | 0.01368 | +0.00373 | −0.00593 |
| 2026072005:2026072006 | 0.00266 | 0.00312 | +0.00047 | −0.00311 |
| 平均（n=3） | — | — | **+0.00188**（sd 0.00168） | **−0.00328** |

三次初始化均出现同方向的 boundary 改善，说明 auxiliary 在覆盖匹配的 real-only 数据上不再是 v3 那样的零效应；但收益很小、global mIoU 略降，故它目前应保留为 Pareto trade-off 候选，而非“胜出模型”。后续应增加 seed 数、延长训练并加入 boundary IoU + global mIoU 的双目标选择，而不是只追单一指标。

对应 receipt：

- `artifacts.local/evidence/ustrf-sc/sanpo-v4-real-boundary-distance-ablation-100step-20260720/report.json`
- `artifacts.local/evidence/ustrf-sc/sanpo-v4-real-boundary-distance-ablation-100step-seed2-20260720/report.json`
- `artifacts.local/evidence/ustrf-sc/sanpo-v4-real-boundary-distance-ablation-100step-seed3-20260720/report.json`

## 结构与训练长度对照

在相同 real-only 覆盖匹配切分上，额外比较了高分辨率 `detail OS4 / semantic OS16` 与默认 `OS8 / OS32`：

| 路径 | step | baseline boundary IoU | auxiliary boundary IoU | Δ boundary | Δ global mIoU | 当前判断 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| OS4 / OS16 | 100 | 0.01305 | 0.01178 | −0.00127 | −0.00345 | 首次高分辨率尝试未胜出 |
| 默认 OS8 / OS32，seed 1 | 300 | 0.00377 | 0.00482 | +0.00105 | +0.00476 | 边界/全局同向，但单 seed 不足 |
| 默认 OS8 / OS32，seed 2 | 300 | 0.00147 | 0.00148 | +0.000002 | +0.00153 | 全局正向，边界近零 |
| 默认 OS8 / OS32，300 step 平均 | — | — | — | **+0.00053** | **+0.00315** | 主要是整体收敛收益，边界效应不稳定 |

对应 receipt：

- `artifacts.local/evidence/ustrf-sc/sanpo-v4-real-os4-os16-distance-ablation-100step-20260720/report.json`
- `artifacts.local/evidence/ustrf-sc/sanpo-v4-real-default-distance-ablation-300step-20260720/report.json`
- `artifacts.local/evidence/ustrf-sc/sanpo-v4-real-default-distance-ablation-300step-seed2-20260720/report.json`

当前可量化结论是：增加步数能在该真实切分上改善 global mIoU；仅靠提高语义支路分辨率或加入 signed-distance 辅助，尚未稳定解决稀疏 boundary/step。下一个模型实验应改成显式 boundary proposal / line-instance 或近场风险区域的重采样监督，而不是继续做同一损失的微调。

## 稀疏边界调参的反证

为避免把“尚未试够常见超参数”当作解释，继续在同一 real-only 切分上运行了三种直接干预：

| 干预 | 关键设置 | baseline boundary IoU | auxiliary boundary IoU | Δ boundary | Δ global mIoU | 结论 |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| 强边界重采样 | guided crop=0.90，boundary target=0.90，300 step | 0.00355 | 0.00346 | −0.00010 | +0.00165 | 更多边界 crop 未带来边界收益 |
| 高输入密度 | 384px，100 step | 0.00590 | 0.00347 | −0.00243 | −0.00396 | 更高像素数反而变差 |
| 强类别权重 | max class weight=20，100 step | 0.00857 | 0.00879 | +0.00022 | −0.00852 | 增加 boundary 输出但效应极小且全局代价高 |

对应 receipt：

- `artifacts.local/evidence/ustrf-sc/sanpo-v4-real-boundary-resample-distance-300step-20260720/report.json`
- `artifacts.local/evidence/ustrf-sc/sanpo-v4-real-384-distance-ablation-100step-20260720/report.json`
- `artifacts.local/evidence/ustrf-sc/sanpo-v4-real-boundary-weight20-distance-100step-20260720/report.json`

这三项反证把下一步压缩为模型形式变化，而不是继续的参数扫描：需要独立的 boundary/line proposal 或基于相邻帧与几何的近场结构目标，并用 boundary precision/recall、跨 session IoU 和整体 mIoU 共同选择。

## 显式 boundary probability head 的 GPU 对照

为验证“独立头”本身是否足够，在同一 real-only 覆盖匹配切分上增加了一个全分辨率的 binary boundary probability head。它从 `lraspp_fuse` 接出，与 baseline 使用相同的主干、语义损失、session-balanced sampler、300 step 和三组初始化；只增加 positive-balanced BCE 辅助目标（正样本权重上限 32）。因此这是结构对照，不是继续调 distance 的权重。

| seed pair | baseline mIoU | `+ boundary head` mIoU | Δ mIoU | Δ semantic boundary IoU | head precision | head recall | head F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026072101:2026073101 | 0.389071 | 0.384618 | -0.004453 | -0.000208 | 0.004506 | 0.025390 | 0.007654 |
| 2026072102:2026073102 | 0.381068 | 0.377827 | -0.003241 | +0.000015 | 0.014650 | 0.073963 | 0.024456 |
| 2026072103:2026073103 | 0.378760 | 0.377729 | -0.001032 | -0.000308 | 0.004081 | 0.020304 | 0.006797 |
| 平均（n=3） | — | — | **-0.002909** | **-0.000167** | — | — | — |

这不是有效候选：三组 global mIoU 均下降，semantic boundary IoU 仅一组出现极小正值，独立头自身也呈现大量假阳性（每组预测约 26–29 万正像素，而 held-out 真值为 52,107 像素；F1 最高仅 0.0245）。因此，“从单帧语义融合特征直接接二值边界头”这条路线已经被此切分上的 GPU 配对数据否定，不应进入默认模型或再做同类超参数扫描。

下一项结构变量应当含有当前 head 缺失的信息：相邻帧可见性/运动一致性、metric depth 平面与边缘，或显式线段实例。它应先输出可审计的近场结构提案，再由 USTRF 风险场消化；不能把概率 head 的像素热图直接解释为台阶事件或安全走廊。

对应 receipt：`artifacts.local/evidence/ustrf-sc/sanpo-v4-real-boundary-probability-ablation-300step-20260720/report.json`。

## 解析 metric-depth 时序基准（GPU 审计）

为避免在缺少逐帧 pose/ground-plane 绑定的公开视频上伪造几何结论，新增了一个小型解析几何基准。它直接给出深度栅格、相机内参、camera-to-body 外参、body-local ground plane、两个相邻帧的精确前向位移及目标真值；使用 RTX 5060 的 Torch CUDA 审计其数组、哈希和重投影一致性。

| 指标 | 结果 |
| --- | ---: |
| 双帧序列 | 14 |
| depth 帧 | 28 |
| 可作静态时序匹配的 body/head 目标对 | 8 |
| 缺失深度但不得生成 `DROP` 的可见性缺口对 | 2 |
| 已知静态目标的 ego-motion 重投影 RMSE | 0.000 m |
| target body/head 几何分类样本与准确率 | 16 / 1.000 |
| 可见性缺口样本与错误 `DROP` 数 | 4 / 0 |
| 解析 `DROP` 样本与 GPU proposer recall | 4 / 1.000 |
| Kotlin CSV/TSV manifest batch replay 获准序列 | 14 / 14 |
| Kotlin static target temporal match | 8 / 8 |
| Kotlin visibility-gap sequence / false `DROP` | 2 / 0 |
| Kotlin expected/detected `DROP` sequence | 2 / 2 |
| 有限 depth 样本 | 86,016 |
| 有效正 depth 样本 | 14,686 |

审计 receipt：`artifacts.local/evidence/datasets/ustrf-synthetic-temporal-geometry-v6-20260720/qa/audit.json`。v6 同时提供带哈希 CSV depth 与无第三方依赖的 TSV pair manifest，并加入可见性缺口与已观测远距地面跳变的对照；`UstrfSyntheticTemporalGeometryReplay` 已在 JVM 中消费其 14 条真实 manifest 行。该数据集只证明解析几何、frame binding 和度量管线可以被一致地测试；它**不**是现实环境代表性、真实用户事件或生产安全效果的证明。它的直接用途是把 `UstrfMetricDepthGeometryAdapter`、`UstrfTemporalGeometryConsistency` 和 `UstrfGroundVisibilityDropProposer` 从单一 fixture 扩展到带哈希 manifest 的可重复时序输入。

## 合成动态轨迹与 TTC 真值回放

为满足 NAV-P0-07 对“合成横穿/对向场景中 TTC 误差可测”的最低要求，新增 `scripts/generate_ustrf_synthetic_dynamic_ttc_benchmark.py` 和 `UstrfSyntheticDynamicTtcReplay`。它生成同一用户身体坐标系中的相邻帧已关联目标轨迹、已验证或故障注入的 ego pose delta、相对速度/TTC/最近接近距离/人体碰撞半径真值；GPU 与 Kotlin 消费同一个 dependency-free TSV manifest。RTX 5060 审计结果为 **9** 条轨迹、**7/9** 可准入、**2/9**（未验证 pose、低置信 track）严格拒绝，**6** 条可计算 TTC，解析相对速度最大误差 **0 m/s**，GPU 的公式重算相对 manifest TTC 最大误差 **1ms**，碰撞标签准确率 **1.000**。Kotlin 回放对同一 bundle 得到 **6/6** TTC、最大误差 **1ms**、**4/4** 碰撞标签一致；receipt：`artifacts.local/evidence/datasets/ustrf-synthetic-dynamic-ttc-v1-20260720/qa/audit.json`。

这里的“track”是已关联的 metric 轨迹输入，并没有宣称像素级 tracker、光流、自运动估计或真实行人检测已经完成；该基准证明的是 `UstrfEgoCompensatedMotionPromoter` 与 `UstrfTtcEstimator` 在精确/故障可控条件下的数值和 frame-binding 行为。进入真实动态性能结论前，还需要带轨迹真值的连续 RGB-D/VIO 数据，以及按 TTC 分段的危险召回和误停止率。

## 人体胶囊走廊与安全监督器的 GPU/Kotlin 闭环回放

为将 NAV-P0-08（人体轨迹）、NAV-P0-09（走廊规划）、NAV-P0-10（故障降级）和 NAV-P0-14（场景回归）从 8 个手写 fixture 扩展为带真值的批量回放，新增 `scripts/generate_ustrf_synthetic_corridor_safety_benchmark.py` 与 `UstrfSyntheticCorridorSafetyReplay`。它固定使用 `synthetic-body-local-v1`、0.5m grid、五条候选人体胶囊走廊（offset −2…2，半宽 1 cell），并为 ground traversability、占用、下坠、头部障碍、动态 TTC、中央未知、pose lost 与 stale geometry 提供解析标签。GPU 在生成侧独立计算胶囊—危险 cell 相交；Kotlin 再经 `UstrfPerceptionAssembler → RiskField → CorridorPlanner → SafetySupervisor` 回放相同 TSV，而非只在 Python 中模拟规划器。

RTX 5060 生成/审计 **256** 个场景：**224** 个含危险或故障、**59** 个按真值应 STOP、**32** 个 clear 场景。Kotlin replay 得到 **256/256** action 一致、**59/59** 应 STOP 覆盖、clear STOP **0**、**240/240** 非故障场景的候选走廊选择一致、**16/16** pose-lost/stale 故障 STOP。第一版 CUDA 真值曾因 Python slice 的右端 exclusive 语义漏掉胶囊右边界；跨语言回放在 `O:1:3` 场景立即暴露该不一致，修复后才形成 v4 receipt。这是本轮最有价值的检查：它验证的是互相独立的 GPU 几何真值与 Kotlin 安全链的契约，而不是拿同一段判断代码自证。

receipts：`artifacts.local/evidence/datasets/ustrf-synthetic-corridor-safety-v4-20260720/qa/audit.json`、`.../qa/kotlin_replay.json` 与 `.../qa/preview.html`。这使“受控解析真值下的 action/走廊/故障一致性”首次可报告，但仍不测量视觉感知错误、用户步态、人体实际包络、真实台阶/下坠或真机 latency；因此只计为 `offline-theory-only`，不提高任何设备或用户端安全授权。

上述 receipt 由 `scripts/report_ustrf_sc_research_benchmark.py` 汇总为一条命令的 JSON + HTML 基准报告：`artifacts.local/evidence/ustrf-sc/research-benchmark-v1-20260720/research_benchmark_report.{json,html}`。本轮为 **CONDITIONAL_RESEARCH_GO**：解析几何、解析动态 TTC、公开源原生时序一致性三个研究 Gate 通过；设备 metric geometry 准入仍明确 **BLOCKED**。脚本把每个 Gate 的授权边界写入输出，避免“GPU 上通过的合成/公开数据检查”被误读为手机、眼镜或用户端安全授权。

### VKITTI2 连续动态轨迹真值

为避免只靠解析轨迹验证 TTC，新增对 [Virtual KITTI 2 官方文本真值](https://europe.naverlabs.com/proxy-virtual-worlds-vkitti-2/) `Scene01/clone` 的 GPU source-native 审计。其 `pose.txt` 提供逐帧、逐相机的 track ID、物体 world/camera 3D 位置，`extrinsic.txt` 提供 world-to-camera 变换，`bbox.txt` 提供 `isMoving` 标记。审计先将前帧物体坐标通过 `E_current × inverse(E_previous)` 变换到当前相机，再比较当前坐标，从而只测对象自身运动而非自车运动。

RTX 5060 对 **6,767** 个连续前向 track 对的结果为：源标注动态对 **363** 个，0.05m/frame 阈值的 self-motion-compensated 动态判别 **362 TP / 0 FP / 1 FN / 6,404 TN**，精度 **1.000**、召回 **0.9972**；其中 **179** 条动态对的源原生最近接近在 30 帧 horizon 内。receipt：`artifacts.local/evidence/datasets/vkitti2-textgt-v1-20260720/extracted/Scene01/clone/qa/vkitti_dynamic_track_audit.json`。

该数据包不携带逐帧 timestamp receipt，且相机安装与车辆坐标并非用户身体坐标；因此报告明确 `physical_ttc_seconds_admitted=false`、`ustrf_motion_input_admitted=false`。它证明公开高保真动态轨迹可用于验证自运动补偿的数值链，但不能输出“秒级助盲 TTC”或直接进入 USTRF safety input。VKITTI2 许可证为 **CC BY-NC-SA 3.0 / 非商用研究用途**，应继续与可商用训练数据隔离。更新后的汇总报告为 `artifacts.local/evidence/ustrf-sc/research-benchmark-v2-20260720/research_benchmark_report.{json,html}`，其中动态源原生 Gate 已通过、设备 Gate 仍为 BLOCKED。

### Argoverse AV1 带时间戳源原生 TTC

VKITTI2 缺少时间 receipt 后，进一步下载 [Argoverse AV1 官方 Motion Forecasting sample](https://www.argoverse.org/av1.html)：5 条真实道路场景的 `TIMESTAMP`、AV 轨迹与其他对象轨迹。`scripts/audit_argoverse_av1_timestamped_ttc.py` 对每个相邻 timestamp：以当前 AV 位移确定 forward/left 轴，把前后目标位置都表达在当前 AV 坐标，随后以实际 `dt` 计算相对速度、接近时间和最近距离。RTX 5060 结果为 **1,282** 条前向 track 对，`TIMESTAMP` 中位间隔 **0.10017s**（范围 **0.08393–0.13395s**），**384** 条为接近对；在 3s horizon、2m 车辆级最近距离下出现 **5** 条源原生 TTC 候选。receipt：`artifacts.local/evidence/datasets/argoverse-av1-forecasting-sample-v1.1-20260720/extracted/forecasting_sample/data/qa/argoverse_timestamped_ttc_audit.json`。

这完成了“真实时间轴下 TTC 的数值回放”，但它没有 RGB-D、相机标定或人体坐标，故仍显式 `ustrf_motion_input_admitted=false`；不能评价像素 tracker、视觉深度、用户身体包络或助盲危险召回。下载包内条款为 **CC BY-NC-SA 4.0**，仅保留为非商用研究基准。汇总报告升级为 `artifacts.local/evidence/ustrf-sc/research-benchmark-v3-20260720/research_benchmark_report.{json,html}`：6 个 Gate 中 5 个研究/数据 Gate 通过，唯一设备准入 Gate 仍 BLOCKED。

### CARLA 行人同步 RGB-D 与相机位姿

为补上 Argoverse 没有 RGB-D、TartanAir 没有稳定局部地面这一侧的**源原生输入完整性**，下载了公开 [CARLA Pedestrian RGB-D 数据卡](https://huggingface.co/datasets/mkxdxd/carla-dataset-ped) 中 `Town01/pedestrian` 的一个 200 帧 shard，只抽取同一行人场景连续前 20 帧。每帧包含 `rgb.png`、float32 `depth.npy`、`camera.json`（内参及可逆 `c2w/w2c`）和 `metadata.json`。`scripts/audit_carla_ped_rgbd_slice.py` 在 RTX 5060 CUDA 上检查模态/frame-id 绑定、全深度有限且正、内参/配置恒定、位姿互逆以及时间单调性。

结果为 **20/20** 帧 source-native 准入，分辨率 **704×1280**，深度范围 **1.410–100.000m**、中位 **19.973m**，相邻 timestamp 固定 **20ms**（metadata：50fps），相邻 camera translation 中位 **0.1540m**。该导出矩阵为正交且可逆，但包含图像轴反射（行列式 −1）；receipt 因而将其明确标记为 `left_handed_or_image_reflected`，而不是暗中当作 USTRF 人体坐标。receipt：`artifacts.local/evidence/datasets/carla-stage2-ped-town01-shard000000-20260720/slice20/qa/carla_rgbd_slice_audit.json`。

这只证明该公开仿真 slice 的 RGB、metric depth、camera pose 与时间轴可一起用于算法压力测试；它仍显式 `ustrf_metric_geometry_input_admitted=false`。缺少的是设备/人体安装外参、body-local ground truth、可关联目标轨迹和 drop/head 事件标签，故不能据此量化助盲风险召回，更不能放开 device safety input。数据卡声明 CC-BY-4.0，但作为第三方发布资产，其来源链应在任何训练扩展前再复核。汇总报告升级为 `artifacts.local/evidence/ustrf-sc/research-benchmark-v5-20260720/research_benchmark_report.{json,html}`：**8** 个 Gate 中 **7** 个研究/数据 Gate 通过，设备准入 Gate 保持 BLOCKED。

### Bonn 真实动态 RGB-D 与 OptiTrack 位姿

为给解析/仿真结果增加独立的真实动态 RGB-D 压力来源，下载 [Bonn RGB-D Dynamic Dataset](https://www.ipb.uni-bonn.de/data/rgbd-dynamic-dataset/) 官方 `rgbd_bonn_moving_obstructing_box` 单序列。该来源提供已注册 RGB/depth、TUM-format 的 OptiTrack 相机轨迹和官方标定；序列名称明确包含移动遮挡物，但不提供逐帧物体轨迹或助盲事件标签。`scripts/audit_bonn_rgbd_dynamic_source.py` 在 RTX 5060 上审计 archive hash、590 帧 RGB、592 帧 depth、593 个 pose、内参与 depth registration，并显式统计而非掩盖异步对齐尾部。

结果为 RGB—depth 配对 **100%** 位于 20ms 内，RGB—pose **99.83%** 位于 20ms 内（唯一尾部为 **33.26ms**，仍在一个 30Hz source-frame 的 40ms hard cap 内），RGB/depth/pose 中位速率为 **29.83/29.97/29.99Hz**，序列时长 **19.74s**。48 帧 GPU depth 抽样的正深度覆盖 **86.41%**，正深度中位 **2.156m**；OptiTrack 四元数最大单位范数误差 **5.96e-7**。`scripts/audit_bonn_rgbd_dynamic_reprojection.py` 用相邻 OptiTrack pose 在原生相机坐标中重投影 24 对 RGB-D，得到中位有效投影 **83.86%**、逐对中位绝对深度残差的中位 **13.33mm**、逐对 P95 残差的中位 **84.07mm**。动态箱体、遮挡、无效深度和 pose/depth 噪声均保留在残差中，未被改写为 object track 或危险标签。

receipts：`artifacts.local/evidence/datasets/bonn-rgbd-moving-obstructing-box-v1-20260720/extracted/rgbd_bonn_moving_obstructing_box/qa/bonn_rgbd_dynamic_source_audit.json` 与 `.../bonn_rgbd_dynamic_reprojection_audit.json`。它通过的是“真实 source-native RGB-D/pose 的时间与几何自洽” Gate；缺少 sensor-to-body、用户局部地面、物体轨迹与 assistive event truth，故两个报告都保持 `ustrf_metric_geometry_input_admitted=false`。汇总报告升级为 `research-benchmark-v6-20260720`：**9** 个 Gate 中 **8** 个研究/数据 Gate 通过，设备准入仍是唯一 BLOCKED Gate。

### REveL Dynamic 逐帧人体 2D 标注

为补齐“外部逐帧人类目标真值”而非继续只用车辆/解析轨迹，下载 [REveL Dynamic](https://uts-ri.github.io/revel/) 的公开 `images.zip` 与 `labels.zip`；官方项目页说明该采集含动态人、RGB、IMU、LiDAR 和 Vicon 传感器/人员位姿，而本轮刻意只消费较小的 RGB 与 2D 标签包。`scripts/audit_revel_dynamic_rgb_labels.py` 在 RTX 5060 上用 CUDA 对全部 YOLO box 作几何范围检查，并对同一 helmet-colour class 的相邻单框连续片段并行计算中心位移与 IoU。

结果为 **8,580/8,580** RGB—label 完整配对，**8,364** 个非空标签帧、**13,018** 个标注框、两个 helmet-colour class（**6,531 / 6,487** boxes），所有 box 的归一化边界均有效；时间戳中位帧间隔 **43.341ms**（**23.073Hz**）。class 0/1 分别有 **6,455 / 6,421** 对连续单框片段，中心位移中位 **0.01295 / 0.01301**（归一化图像尺度），连续 IoU 中位 **0.842 / 0.861**。receipt：`artifacts.local/evidence/datasets/revel-dynamic-images-labels-v1-20260720/qa/revel_dynamic_rgb_labels_audit.json`。

这通过的是 `external_2d_dynamic_object_truth_admitted=true`：可用于离线 2D 标注完整性和类别条件时序连续性度量。当时未消费的 `dynamic.bag` 含 Vicon/LiDAR/IMU 的原生时序和标定资料，故 v7 仍明确拒绝 metric depth、物理 TTC、人体坐标安全走廊、assistive event truth 和设备安全准入。汇总报告 `research-benchmark-v7-20260720` 为 **10** 个 Gate 中 **9** 个研究/数据 Gate 通过。

### REveL Dynamic Vicon 人体—传感器米制轨迹

随后完整下载并校验 `dynamic.bag`（**7,287,305,421 bytes**，SHA-256 `6b10752b0d4cb401751e57f3ac55ebe45fcbb785f89d8a43fe1cbfd30dc0b08a`）。`scripts/audit_revel_dynamic_bag_inventory.py` 读 ROS 索引得到 **371.805s**、**566,483** 条消息；其中 `/vicon/world` 下有 event/LiDAR sensor suite **34,223** poses、green/yellow helmet **26,404 / 25,406** poses。`scripts/audit_revel_dynamic_vicon_trajectories.py` 在 RTX 5060 上解码这些 transform，使用同一 Vicon world frame 将两类 helmet 与传感器套件逐帧最近邻对齐，再在 CUDA 上计算 range 和相对运动。

所有三条 Vicon 轨迹均为有限值且四元数单位范数通过；源页面说明 Vicon 丢失时会回退原点，本段的原点回退计数为 **0**。但原始 ROS 时间序列含微秒级相邻记录及位置尖峰，直接求导会产生虚假高速度。因此 audit 同时保留 raw receipt，并将 **5–50ms** 间隔与 **≤5m/s** world-step 明确为连续性筛选：sensor/green/yellow 分别保留 **31,277 / 24,111 / 23,176** 个连续 pose 对，筛掉 **9 / 45 / 52** 个速度尖峰候选。

在 20ms 对齐上限下，green 与 yellow 的同步有效 pose 比例为 **94.61% / 97.31%**，时间差中位仅 **0.0057 / 0.0097ms**；得到 **22,644 / 22,465** 个连续相对运动对。对应的 sensor-relative range 中位为 **3.551 / 2.883m**、P95 为 **5.998 / 5.288m**；world-frame 相对速度中位 **0.769 / 0.739m/s**、P95 **1.519 / 1.544m/s**。receipts：`artifacts.local/evidence/datasets/revel-dynamic-bag-v1-20260720/qa/revel_dynamic_bag_inventory.json` 与 `.../qa/revel_dynamic_vicon_trajectory_audit.json`。

这首次通过 `external_metric_person_sensor_trajectory_truth_admitted=true`：可做离线米制行人—传感器相对轨迹与 range/range-rate 分段研究。它仍然不是可直接灌入 USTRF 的 physical assistive TTC：标注的是 helmet marker 与 source sensor-suite，不是用户身体包络、手机/眼镜安装外参或已标注 assistive event。汇总报告现为 `research-benchmark-v8-20260720`：**11** 个 Gate 中 **10** 个研究/数据 Gate 通过，唯一 device metric-geometry admission 继续 BLOCKED。

### REveL RGB 标签—Vicon 3D 跨模态重投影

为验证 2D 标注、动态 bag 和 calibration 不是三份互不相干的材料，新增 `scripts/audit_revel_rgb_vicon_reprojection.py`。它以 `dynamic.bag` 的 `/dvs/image_raw` 顺序绑定 **8,580** 张 archive RGB 与 bag image timestamps；archive 文件名相对 bag timestamp 的中位偏移 **30.04ms**、P95 **35.13ms**，故只把文件名作顺序键，而将实际 Vicon 同步锚定到 bag timestamp。使用官方 `T_v_c`、`K` 和 4 参数畸变，将同一类别的 green/yellow helmet Vicon marker 投到 **346×260** RGB 像素平面，并与同类 YOLO box 比较。

在 20ms sensor/person Vicon 同步约束下，green/yellow 分别有 **4,348 / 4,199** 个可用标注帧；其同步时间差中位约 **1.76ms**，P95 约 **5.71 / 6.11ms**。green 的投影点在对应 box 中的比例为 **89.61%**，yellow 为 **97.04%**；最近同类 box 外距离 P95 为 **2.62px / 0px**。数据中各有一帧出现同一 helmet 类多 box（共 2 帧），审计按“任一同类候选框命中”统计，并保留 `ambiguous_same_class_frame_count=1`，没有擅自挑选一个框。receipt：`artifacts.local/evidence/datasets/revel-dynamic-bag-v1-20260720/qa/revel_rgb_vicon_reprojection_audit.json`。

这通过 `source_cross_modal_2d_3d_alignment_admitted=true`：它证明公开 RGB 标签、官方标定与 Vicon helmet 轨迹之间存在量化的同源几何一致性，而不是将它误称作 detector AP、米制深度、physical assistive TTC 或人体走廊真值。汇总报告升级为 `research-benchmark-v9-20260720`：**12** 个 Gate 中 **11** 个研究/数据 Gate 通过，唯一 device metric-geometry admission 仍 BLOCKED。

## 公开高保真仿真输入：TartanAir JapaneseAlley

已下载公开 TartanAir 预处理镜像的 `JapaneseAlley/Hard/P002`，只抽取连续前 20 帧的 RGB、float32 depth 与 camera parameter 包。GPU 结构审计通过：depth 为 **20×480×640**，全部有限且为正，正样本占比 1.000，最小深度 1.260m、中位数 4.613m；每帧 parameter 包均含 `camera_pose` 和 `camera_intrinsics`，相邻 translation 模长为 0.0057–0.0830m。审计 receipt：`artifacts.local/evidence/datasets/tartanair-japanesealley-p002-slice20-v2-20260720/qa/structural_audit.json`。这已经满足“同步 metric depth + 连续 pose + RGB”的输入前提，但审计明确将 `ustrf_body_mapping_admitted=false`：TartanAir camera/body 轴和 pose direction 尚未由该 slice 独立绑定。仍须补齐这一约定，以及局部 ground-plane、障碍/下坠事件标签后才能进入 metric Adapter 的准入链。

对下半图 depth 做 GPU RANSAC 的候选地面审计进一步支持拒绝：20 帧的中位内点率仅 **0.2115**，估计相机到候选平面的中位距离 **4.282m**，且逐帧结果在约 0.39–7.49m 之间剧烈波动。该片段不能提供稳定的用户局部地面平面，故不得被提升为 USTRF geometry input。receipt：`artifacts.local/evidence/datasets/tartanair-japanesealley-p002-slice20-v2-20260720/qa/ground_plane_candidate_audit.json`。

同归档的 `P000` 独立对照排除了“只是 RANSAC 不稳定”的解释：其 20 帧候选平面中位内点率达到 **1.000**，但相机到平面的中位距离仍为 **3.813m**。这是稳定但不符合用户脚下局部地面的几何，不能因内点率高而放行。receipt：`artifacts.local/evidence/datasets/tartanair-japanesealley-p000-slice20-v1-20260720/qa/ground_plane_candidate_audit.json`。

为把“候选地面被拒绝”与“源时序是否自洽”分开，新增 `scripts/audit_tartanair_temporal_reprojection.py`：它在 TartanAir 的原生 `cam2world`、`x-right/y-down/z-forward` 坐标中，使用 RTX 5060 将前帧的每个 metric-depth 样本经相邻 pose 投影到后帧，再与后帧 depth 比较；不作 body 轴转换、地面推断或安全事件标注。P000 的 19 对相邻帧中，中位“逐对中位绝对深度残差”为 **5.56mm**、逐对 P95 残差的 P95 为 **110.47mm**、中位有效投影覆盖为 **97.49%**；P002 对应为 **3.27mm**、**21.36mm**、**96.88%**。receipts：`artifacts.local/evidence/datasets/tartanair-japanesealley-p000-slice20-v1-20260720/qa/temporal_reprojection_audit.json` 与 `artifacts.local/evidence/datasets/tartanair-japanesealley-p002-slice20-v2-20260720/qa/temporal_reprojection_audit.json`。这支持“该公开片段的 depth/pose 对可用于时序算法压力测试”，但两个 report 都显式保留 `ustrf_geometry_input_admitted=false`；它不能替代身体坐标、局部地面或助盲事件真值。

据此固定公开/合成片段的候选地面预筛：中位内点率至少 0.60、相机—候选平面距离位于 0.80–2.20m、20 帧短窗的距离跨度不超过 0.50m，并且仍需单独的 world-to-body/up-axis receipt。该门槛下 P002 因内点率与 7.105m 距离跨度失败，P000 因 3.813m 中位距离失败；二者均不进入 USTRF geometry input。

运行命令：

```powershell
.\.venv-export312\Scripts\python.exe scripts\sanpo_deterministic_linear_probe.py `
  --dataset-root artifacts.local\evidence\datasets\sanpo-v3-canonical-evidence-v4-20260713 `
  --training-gate-report qa\training_gate_report.json `
  --feature-weights artifacts.local\evidence\segmentation-candidate\p1-sigmoid-no-pooled-bn-20260713\candidate.weights.h5 `
  --backend tensorflow --feature-layer lraspp_fuse --input-size 256 `
  --pixels-per-class-per-record 4 --feature-batch-size 8 `
  --maximum-samples-per-class 128 --repeats 2 `
  --report artifacts.local\evidence\ustrf-sc\sanpo-linear-probe-fuse-20260720\probe_report.json
```

该结果是离线研究基线，不是助盲事件真值、设备几何验证或产品效果证明。

## REveL YOLO11n 有界 2D detector 基线

原全量 `batch=64/imgsz=320/FP16` 尝试两次触发 Windows `0x133 DPC_WATCHDOG_VIOLATION`，且均未生成结果 JSON；因此旧运行不提供 detector 指标。为避免继续用高负载碰撞系统边界，benchmark 改为默认 `batch=1/imgsz=256/FP32`、显式 `--max-frames`、确定性 uniform 取样、独立输出和 CUDA allocator 限制；外部 PowerShell 守护器逐秒采样 GPU 并在温度、监控或时间越界时终止精确子进程。

8/32/128 帧阶梯通过后，完成 512/8,580 帧 uniform bounded run：

| 项目 | 结果 |
| --- | ---: |
| 标注框数 | 770 |
| AP50 | 0.92747 |
| precision | 0.83313 |
| recall | 0.88831 |
| F1 | 0.85984 |
| small recall（37） | 0.24324 |
| medium recall（354） | 0.87571 |
| large recall（379） | 0.96306 |
| elapsed / FPS（含 250ms/batch 节流） | 145.27s / 3.524 |
| PyTorch peak allocated / reserved | 180.19 / 346 MB |
| 守护最高温度 / 功耗 / 整卡显存 | 49°C / 35.74W / 1,453MB |
| 相关 System events | 0 |

主要结论不是平均 AP50 较高，而是 small-box recall 仅 24.3%，显著低于 medium/large。普通 COCO person detector 对远处小目标的提前发现不足，不能作为单独安全权威；下一配对变量是相同 index set 上 320px 输入能否以可接受负载换取 small recall，而不是直接恢复全量高 batch。

receipts：

- `artifacts.local/evidence/ustrf-sc/revel-yolo11n-guarded-bounded-512-20260720-r1/benchmark.json`
- `artifacts.local/evidence/ustrf-sc/revel-yolo11n-guarded-bounded-512-20260720-r1/guard_report.json`
- `artifacts.local/evidence/ustrf-sc/research-benchmark-v10-20260720/research_benchmark_report.json`

V10 为 **13** 个 gate 中 **12** 个研究/数据 gate 通过；唯一 `device_metric_geometry_admission` 继续 BLOCKED，`production_authority=false`。

### 可重复性、分辨率反证与距离分层

随后在相同 512 个 uniform index 上完成 r2。aggregate detector 指标与 r1 精确一致，并保存 512 行逐帧 `details.jsonl`；SHA-256 为 `47cfb30d7cf1862dd85628332f3b9526708c1de76deaa1e24691beeb4396f530`。r2 守护最高 46°C、41% utilization、整卡显存 1,302MB、22.38W，0 个相关 System events。逐帧审计得到 TP/FP/FN 为 684/137/86，共 155 帧含 FP 或 FN、13 帧无高于 score floor 的预测；small misses 为 28，并形成 13 个可复现连续片段。

预先声明的 128 帧 256px/320px 配对没有支持提升分辨率：small recall 两者都为 0.4，320px 的 F1 从 0.90176 降到 0.88614，差值 -0.01562。因此 comparison receipt 给出 `do_not_scale_candidate_to_512`，没有继续运行 320px/512 帧。

逐帧 detector receipt 与 REveL bag 的 RGB timestamps、green/yellow helmet 和 sensor-suite Vicon 轨迹对齐后，770 个 GT boxes 中 502 个获得 source sensor-local range。0–5m 为 420/448，recall 0.9375，Wilson 95% CI [0.9112,0.9564]；5m 以上为 39/54，recall 0.7222，CI [0.5911,0.8238]。Vicon-aligned small boxes 为 3/14，且 14 个的 source range 全部大于 5.3m；small miss 的 range 中位 8.97m。这个结果将 small-box 漏检解释为 far-range early-detection 问题提供了来源内证据，但仍不等于用户身体距离、physical assistive TTC、人体走廊或安全事件召回。

主要 receipts：

- `artifacts.local/evidence/ustrf-sc/revel-yolo11n-guarded-bounded-512-20260720-r2/benchmark.json`
- `artifacts.local/evidence/ustrf-sc/revel-yolo11n-guarded-bounded-512-20260720-r2/details.jsonl`
- `artifacts.local/evidence/ustrf-sc/revel-yolo11n-guarded-bounded-512-20260720-r2/failure_analysis.json`
- `artifacts.local/evidence/ustrf-sc/revel-yolo11n-guarded-bounded-512-20260720-r2/vicon_failure_alignment.json`
- `artifacts.local/evidence/ustrf-sc/revel-yolo11n-resolution-sensitivity-128-20260720-r1/comparison.json`
- `artifacts.local/evidence/ustrf-sc/research-benchmark-v12-20260720/research_benchmark_report.json`

V12 为 **14** 个 gate 中 **13** 个通过；新增 `public_detector_source_vicon_range_stratification` 仅获 `source-range-stratification-only`，唯一 `device_metric_geometry_admission` 继续 BLOCKED，`production_authority=false`。

### Source radial range-rate、approach/recede 与 TTC-proxy 分层

下一轮没有启动 GPU，而是在相同 512-frame / 770-box detector receipt 上执行 CPU/NumPy 对齐。协议在看 detector outcome 前冻结：使用 **rosbag record time**，为每个 bag image timestamp 选择严格包围它的相邻原生 person Vicon pose；两端分别绑定 sensor pose，并复用 `5–50ms` 连续间隔、单轨迹 `≤5m/s` world-speed、person/sensor `≤20ms` 同步门。禁止用约 0.7 秒间隔的 512 个稀疏 detector sample 直接求导。

令 `r0/r1` 为两端 helmet marker—event/LiDAR sensor marker 的 source range，`r_mid=(r0+r1)/2`，`v_r=(r1-r0)/dt`。冻结 deadband 为 `0.10m/s`：`v_r≤-0.10` 为 approaching，`v_r≥0.10` 为 receding，其余为 quasi-static；只对 approaching 计算 `TTC-proxy=r_mid/(-v_r)`。因严格包围会使用 image timestamp 之后的 Vicon pose，该指标显式标记为 `offline_noncausal=true`。

| source radial 状态 | GT boxes | matched | missed | recall | Wilson 95% CI |
| --- | ---: | ---: | ---: | ---: | --- |
| approaching | 204 | 190 | 14 | 0.93137 | [0.88812, 0.95868] |
| quasi-static | 103 | 93 | 10 | 0.90291 | [0.83045, 0.94641] |
| receding | 181 | 164 | 17 | 0.90608 | [0.85474, 0.94053] |

770 个框中，旧 range 对齐仍精确保持 `502`；连续 motion 对齐为 `488`（覆盖 `63.38%`）。其余 282 个框按失败原因保留：259 个当前 source Vicon 不可用、9 个无严格包围 pair、14 个被连续性门拒绝，没有从分母中静默删除。

| TTC-proxy 分层 | GT boxes | matched | recall | Wilson 95% CI |
| --- | ---: | ---: | ---: | --- |
| 0–1s | 0 | 0 | `null` | `null` |
| 1–2s | 1 | 1 | 1.00000 | [0.20655, 1.00000] |
| 2–3s | 9 | 9 | 1.00000 | [0.70085, 1.00000] |
| ≥3s | 194 | 180 | 0.92784 | [0.88253, 0.95653] |

小于 3 秒的 proxy 只有 10 个框，虽然本样本中 10/10 检出，但分母太小，不能解释为近碰撞安全能力。距离×运动交叉表进一步显示 5m 以上的 approaching / quasi-static / receding recall 分别为 `17/21=.80952`、`2/6=.33333`、`20/26=.76923`；主要失败仍与远距离、小目标混杂，不能从全局 approach/recede 均值推断方向性优势。完整交叉分层及每格 Wilson 区间保存在机器收据中。

精确复跑两次后，aggregate 与逐框内容完全相同：

- `artifacts.local/evidence/ustrf-sc/revel-yolo11n-vicon-radial-stratification-20260720-r1/alignment.json`，SHA-256 `4f2750d38869aecf3576f5635a3c1db36af186e074dea689c6113485de0cc012`；
- 同目录 `details.jsonl`，770 行，SHA-256 `155863e2725ccac5a237b98153fd275fb4f64faf764fe4ab6f828e219059d3ef`；
- 同目录 `run_receipt.json` 固化输入哈希、CPU/Python/NumPy/rosbags 版本、执行命令与重复性；
- `artifacts.local/evidence/ustrf-sc/research-benchmark-v13-20260720/research_benchmark_report.json`，SHA-256 `885220d3010dd6b692490557de89ac37a9be9ecd3a2924446a7cbc0599059141`。

V13 新增 `public_detector_source_vicon_radial_motion_stratification`，为 **15 gate / 14 pass**，授权仅为 `source-motion-stratification-only`。TTC-proxy 是“恒定径向速度到零 marker range”的离线代理，不含 closest approach、人体半径、轨迹曲率或加速度；helmet marker 不是用户身体，event/LiDAR marker 不是手机或眼镜 body frame。它不是 physical assistive TTC、assistive event truth、设备准入或生产授权，唯一失败门仍为 `device_metric_geometry_admission`。
