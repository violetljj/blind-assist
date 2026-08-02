# HFTF Stage C D5 TartanGround Development pilot

## 结论

TartanGround 已从“目录看起来足够大”推进到九个可执行结果：

1. 官方 Hugging Face revision
   `388faf9c800568cfc6828fa47e063f8369397eb3` 完整覆盖锁定 catalog 的
   198 个 differential-drive parents 和 7,722 个 archive paths；
2. 三个 outcome-open 探索窗口证明 RGB/depth/seg/pose 共同时间轴和 metric
   geometry 可用，并观察到少量但明确的 future-label 非冗余。
3. 六个 train environments、两个 dev environments 的 student Development
   证明 RGB 标签可学习；直接从随机初始化联合训练 history 失败，而从 single
   checkpoint 分阶段微调 history 出现小幅、随机性可重复但环境不稳健的增量。
4. 扩展到 15 个 environments 的三折 environment-held-out Development 后，
   保留水平方向轴的 directional head 在 3 seeds × 3 folds 的九个 paired 单元中
   8 胜 1 负；但 joint、零初始化逐点 residual 和 3×3 spatial residual 三类
   未对齐 history fusion 都没有建立跨折增量。
5. predicted-known 的完全与平方根逆频率重加权能提高部分 body/event recall，
   但不能同时守住 negative false-active；该标量损失修补路线停止。
6. 将 observability 与 alert permission 解耦，并加入高度分层的因果时间确认后，
   事件召回与误激活在多数 paired Development 单元同时改善，但 clearance 和
   false-alert fragmentation 仍未解决。
7. 要求 body lane 具有跨 distance cells 的空间支持后，v2 同时修复 v1 的
   clearance/fragmentation，并让 directional 相对 pooled 的表示增量首次在
   selective synthetic event proxy 上形成多数 paired 改善。
8. 固定 v2 后，在 6 个此前完全未使用的 outcome-unseen environments 上复用
   3 seeds × 3 folds checkpoints；directional 相对 pooled 的 event recall
   9/9 提高，false-active 7/9 降低，建立了跨新环境的 selective-event 迁移正结果，
   但 head false-alert fragmentation 与逐环境 false-active guardrail 尚未解决。
9. 不调模型或 v2，直接进入 30-session SANPO 真实连续事件集。9 个 directional
   checkpoints 全部命中 16/16 正事件，证明真实 RGB recall signal；但各自对
   13–14/14 负事件误报、只清除 0–2/16，固定 kernel 的真实事件效用不成立。
10. 保持相同 source-session-held-out folds 与监督不变，把 relation head 从
    30-dimensional output field 前移到固定 encoder 的 `128×3×6` spatial feature。
    相对 output-field head，false alerts 9/9 减少、clearance mean 增加 1.78，
    recall mean 仅减 0.22；空间表示增量成立，但 0/9 超过当前 YOLO 的三指标 Pareto。
11. 事件配对显示 HFTF 平均补回 YOLO `2.56/3` 个漏检，建立互补性正信号；但
    rank-2 spatial canary、9-checkpoint 静态 YOLO-HFTF fusion 和 1 秒 causal
    transition canary 都未形成 Pareto。当前瓶颈转为真实关系监督规模，不再堆 head。

当前终态为：

`DIRECTIONAL_SPATIAL_STRUCTURE_MULTI_SEED_CROSS_ENVIRONMENT_INCREMENT_SUPPORTED_IN_DEVELOPMENT /
UNALIGNED_HISTORY_FUSION_INCREMENT_NOT_SUPPORTED /
UNCALIBRATED_SYNTHETIC_EVENT_TRANSFER_NOT_SUPPORTED /
KNOWN_LOSS_REWEIGHTING_EVENT_INCREMENT_NOT_SUPPORTED /
HEIGHT_SPATIOTEMPORAL_SELECTIVE_DECISION_KERNEL_SIGNAL_SUPPORTED_IN_DEVELOPMENT /
DIRECTIONAL_SPATIAL_STRUCTURE_SELECTIVE_EVENT_TRANSFER_SIGNAL_SUPPORTED_IN_DEVELOPMENT /
DIRECTIONAL_SPATIAL_STRUCTURE_SELECTIVE_EVENT_TRANSFER_REPLICATED_ON_OUTCOME_UNSEEN_TARTANGROUND_ENVIRONMENTS_IN_DEVELOPMENT /
REAL_EVENT_RECALL_SIGNAL_SUPPORTED_ACROSS_NINE_CHECKPOINTS_IN_DEVELOPMENT /
FIXED_KERNEL_REAL_EVENT_SPECIFICITY_AND_CLEARANCE_NOT_SUPPORTED /
DIRECTIONAL_REAL_EVENT_PARETO_INCREMENT_NOT_SUPPORTED /
CENTRAL_VS_LATERAL_ACTIONABILITY_PROFILE_NOT_SUPPORTED /
WEAK_RELATION_HEAD_SPECIFICITY_CLEARANCE_SIGNAL_SUPPORTED_IN_DEVELOPMENT /
WEAK_RELATION_HEAD_REAL_EVENT_PARETO_INCREMENT_NOT_SUPPORTED /
FIXED_ENCODER_SPATIAL_RELATION_HEAD_OVER_OUTPUT_FIELD_GUARDRAIL_INCREMENT_SUPPORTED_IN_DEVELOPMENT /
FIXED_ENCODER_SPATIAL_RELATION_HEAD_REAL_EVENT_PARETO_INCREMENT_NOT_SUPPORTED /
YOLO_HFTF_EVENT_COMPLEMENTARITY_SIGNAL_SUPPORTED_IN_DEVELOPMENT /
LOW_RANK_SPATIAL_RELATION_HEAD_CANARY_INCREMENT_NOT_SUPPORTED /
STATIC_YOLO_HFTF_FUSION_PARETO_INCREMENT_NOT_SUPPORTED /
CAUSAL_TRANSITION_FUSION_CANARY_INCREMENT_NOT_SUPPORTED`

这足以把 directional single 提升为 HFTF 当前 Development reference，并停止
pooled/grid 与无对齐 history fusion；不需要先完成 197-parent 产品级 census。
它不证明 HFTF 超过主线或系统具有安全效用。

## Provider 与映射

- official dataset：`theairlabcmu/TartanGround`
- pinned revision：`388faf9c800568cfc6828fa47e063f8369397eb3`
- provider files：34,673
- catalog parents：198
- catalog archive paths：7,722
- revision 中缺失路径：0
- 全量 URL map SHA-256：
  `c3961c4c32f16af040745681e0a8ced4b9dca37bf96bfb11f1cb71a6fa2ee957`

稳定地址由
`https://huggingface.co/datasets/theairlabcmu/TartanGround/resolve/{revision}/{archive_path}`
机械生成。解析工具允许在网络或文件故障后修复重跑，不是 one-shot，也不烧毁 source。

## 三个探索 sentinel

- `AbandonedCable/Data_diff/P1000`
- `MiddleEast/Data_diff/P1002`
- `WaterMillNight/Data_diff/P1002`

每个 `metadata.zip` 都包含：

- finite positive `robot_height` 与 `time_step=0.1 s`；
- 12 个 camera pose streams，行数分别为 609、1,194、681，并与 `num_poses`
  完全一致；
- RGB/depth/seg 的 zero-based frame ID 集合分别与 pose 行
  `0..num_poses-1` 完全一致；
- 动态 `lcam_front` 6-DoF pose；
- 约 0.25 m 的固定左右 stereo baseline。

官方文档给出的 640×640 pinhole、`fx=fy=320`、10 Hz、同步采样和 NED pose
约定，与实际 payload 一致。

## Future-label pilot

每个 parent 取一个 outcome-open 25-raw-frame span，在 9 个 5 Hz anchors 上比较
`.4/.8 s`：

- field：`6 direction × 6 distance × foot/body/head`
- 标准探索速度：1.0 m/s
- baseline：current depth 对未来 candidate-envelope 位置的几何标签
- oracle：对应 future depth/pose 对同一位置的几何标签
- 三个窗口只读取各 13 张所需 depth frame；读取失败可修复重跑

汇总结果：

| 指标 | 结果 |
|---|---:|
| 双方 known 的 future cell-observations | 2,555 |
| 风险状态变化 | 54 |
| risk onset | 30 |
| risk clearance | 24 |
| future-only newly-known | 43 |
| 状态变化占 common-known | 2.11% |

按 parent 的风险状态变化为 `15 / 30 / 9`，三条均非零。future oracle 的
height-specific risk cell-observations 合计为 foot 284、body 522、head 221；这只说明
三个高度层在探索窗口中都有非退化输出，不是风险 prevalence。

坐标链通过 source payload 自身交叉核验：把 current depth 点用 pose 投到 future
camera 后，`.4/.8 s` 的 pair-median relative depth error 在三个 parent 上为
`.00068–.00144`，落在 5% 内的点比例中位数为 `.871–.988`。因此观测到的 future
差异不能用明显的 NED/OpenCV 坐标接错解释。

## Environment-clustered student Development

### Corpus

六个 train environments：

- `AbandonedCable`
- `CoalMine`
- `Gascola`
- `OldScandinavia`
- `Rome`
- `SeasonalForestWinterNight`

两个 dev environments：

- `MiddleEast`
- `WaterMillNight`

每个 source 取一个 centered 81-raw-frame span，形成 33 个 5 Hz anchors；每个
anchor 使用 `[-.8,-.6,-.4,-.2,0] s` 五帧 RGB，并具有
`current/.4/.8 s × foot/body/head × 6×6` teacher field。总计：

| 项目 | 结果 |
|---|---:|
| train samples | 198 |
| dev samples | 66 |
| RGB/depth PNG | 592 |
| manifest SHA-256 | `f9832d5b91d530fe70bfb65d55d639d5f040a7efa02ddb8abf40bf6d61981b76` |
| samples JSONL SHA-256 | `649d8ffc1e550b209ed64fcc87de20858da707089a5c31b7c00fabc14591ec75` |

592 个 PNG 均通过解码检查，264 条样本的 RGB 路径全部存在。网络中断、局部文件和
输出写入错误均属于可修复工程故障，不关闭 source。

### 比较设计

- backbone：ImageNet MobileNetV3-Small feature encoder；
- input：`5×RGB @ 128×224`；
- temporal fusion：同一个 depthwise `5×1×1` Conv3D；
- 输出：risk + known，三个 horizon、三个 height、`6×6`；
- 两臂参数量完全相同：1,087,464；
- checkpoint selection：dev 上 near/far × body/head 四组 F1 的宏平均；
- foot 仍报告，但不用于 selection，避免类别比例主导选择；
- 固定空间先验由 train 中每个 field cell 的均值构造。

### 绝对学习性与直接 history 结果

| 模型 | future body/head macro F1 | future body/head micro F1 | 相对 single macro |
|---|---:|---:|---:|
| train cell-prior | 0.2874 | 0.3261 | -0.2562 |
| single，seed 17 | 0.5435 | 0.6057 | 0 |
| history from scratch，seed 17 | 0.4996 | 0.5521 | -0.0439 |

single 明显超过固定空间先验，支持 RGB future-field learnability。history 从随机
初始化直接联合训练则降低整体 F1 和 recall，不能作为正结果。

交叉输入诊断显示，single checkpoint 在不重训时把 repeated-current 输入换成真实
history，macro F1 从 `0.5435` 升至 `0.5509`；history checkpoint 换回 single 输入
仍只有 `0.4980`。这把失败定位为训练/优化问题，而不是证明历史帧没有信息。

### 分阶段 history fine-tune

固定同一个 seed-17 single checkpoint，使用较小学习率对真实 history 微调 5
epochs；微调前的真实-history evaluation 作为 epoch 0。三个微调随机种子结果：

| fine-tune seed | selected epoch | macro F1 | macro delta vs single | micro F1 |
|---:|---:|---:|---:|---:|
| 17 | 5 | 0.5549 | +0.0114 | 0.6270 |
| 29 | 3 | 0.5565 | +0.0130 | 0.6304 |
| 43 | 4 | 0.5512 | +0.0077 | 0.6259 |

macro delta 平均 `+0.0107`，范围 `+0.0077..+0.0130`；micro F1 平均增量
`+0.0221`。三次 aggregate delta 同号，说明“先学静态场景、再学 history”的短程
优化信号对微调随机性可重复。

但 environment-level 结果不支持稳健增量：

| environment | single macro F1 | fine-tune macro F1 range | delta range |
|---|---:|---:|---:|
| MiddleEast | 0.5903 | 0.6130–0.6249 | +0.0227..+0.0346 |
| WaterMillNight | 0.4111 | 0.4035–0.4082 | -0.0076..-0.0028 |

全部 aggregate 提升都来自 `MiddleEast`；`WaterMillNight` 三次均轻微下降。相对
single，三次 fine-tune 的整体 body/head FPR 从 `0.4232` 降至
`0.4098–0.4123`，recall 从 `0.7535` 升至 `0.7810–0.7877`，但这些 aggregate
改善不能覆盖最差环境的负 delta。

## Outcome-open environment expansion

为区分 `WaterMillNight` 特例与一般环境迁移，增加其白天 counterpart
`WaterMillDay/Data_diff/P1002`，并按固定哈希顺序加入六个此前未使用的 P1000
environments：`Downtown / JapaneseAlley / NordicHarbor / Supermarket /
OldTownNight / GreatMarsh`。共新增 231 samples、518 PNG；samples SHA-256 为
`fad64102b9c1bcbeb5a93662f0f8c5acb30ea615668daf22f4d851ac3f958049`。

原 pooled single 在七环境的 aggregate macro F1 为 `0.3444`。三个原 staged
history checkpoint 均低于 pooled single，environment wins/losses 最好也只有
`2/5`；它们虽在 AUROC/AP 上有小幅改善，但没有转化为环境稳健的 thresholded
field effect。

诊断发现原 pooled head 丢弃全部空间结构后一次生成 6×6 field，并在新环境中严重
过预测；expansion 的 near/far head positives 仅约 `7.6%/8.3%`，原 pooled head
对应 AUROC 约 `0.491/0.472`。因此下一模型改动不是调阈值，而是保留输出方向与
feature-map 方向的对应关系。

directional head 的参数量为 `1,017,804`，少于 pooled 的 `1,087,464`。它在原
两环境 dev 上 macro F1 从 `0.5435` 降到 `0.5172`，但在七个新增环境上从
`0.3444` 升到 `0.3905`，AUROC `+0.0494`、AP `+0.0421`，且 6/7 environments
胜出。这个结果只用于提出跨环境复核，不单独作为成功终态。

## 15-environment cross-environment Development

原 8 个与新增 7 个 environments 合并为 495 samples，并构造三折：

- 每折 10 train / 5 dev environments；
- `WaterMillDay` 与 `WaterMillNight` 固定在同一折，避免 family leakage；
- checkpoint selection 使用五个 dev environments 各自 future body/head macro
  F1 的等权平均，防止 cell 数较多环境主导选择；
- 每个结构 seed 17、20 epochs；这是 outcome-open Development，不是 held-out
  promotion evidence。

三折结果：

| fold | pooled env-macro F1 | directional env-macro F1 | delta |
|---:|---:|---:|---:|
| 0 | 0.4738 | 0.4796 | +0.0058 |
| 1 | 0.3860 | 0.3972 | +0.0112 |
| 2 | 0.3584 | 0.4391 | +0.0806 |
| mean | 0.4061 | 0.4386 | +0.0326 |

directional 在 15 个 dev environments 中 11 胜、4 负。折均 aggregate
body/head macro F1 `+0.0327`、micro F1 `+0.0411`、AUROC `+0.0459`、AP
`+0.0587`，FPR `-0.0098`。最差环境是 `GreatMarsh`，macro delta
`-0.1788`；最大改善是 `MiddleEast` 的 `+0.2629`。因此支持的是跨折平均增量和
方向一致性，不是每环境支配。

更完整的 3×6 spatial grid 在 fold 0 的 environment-macro F1 只有 `0.4581`，
低于 pooled 和 directional，因此没有进入其余两折。

### Paired multi-seed replication

在相同三折上增加 seed 29/43，并对每个 seed 同时重训 pooled 和 directional，
避免把初始化差异误作结构增量。九个 paired fold×seed 单元结果为：

- environment-macro F1：8 胜 1 负，mean `+0.0351`，median `+0.0385`，
  range `-0.0046..+0.0806`；
- 三个 seed 的三折 mean delta：`+0.0326 / +0.0424 / +0.0304`；
- 三个 fold 的三 seed mean delta：`+0.0260 / +0.0150 / +0.0643`；
- 45 个 environment×seed 比较中 30 胜、15 负。

九单元折均 aggregate macro F1、micro F1、AUROC、AP delta 分别为
`+0.0357 / +0.0375 / +0.0395 / +0.0448`，各自均为 8/9 单元改善。
唯一 environment-macro 反向单元是 seed43/fold1 的 `-0.0046`；该单元
aggregate macro/micro 仍为 `+0.0153/+0.0200`。

但 threshold calibration 还不稳定：recall mean delta `+0.0797`，FPR mean delta
却为 `+0.0229`，且 FPR 在 6/9 单元变差。特别是 seed43/fold0 的 FPR
`+0.1957`。因此 multi-seed 复核强化的是表示/排序与 F1 候选，不允许直接推提醒层
false-alert 改善。

`GreatMarsh` 的最差 delta 不是一般亮度问题，而是 extreme height-mixture shift：
fold0 train 的 future body/head positive rate 为 `48.9%/15.3%`，GreatMarsh 为
`93.1%/0.97%`。seed17 directional 把整体 FPR 从 `0.459` 降到 `0.146`，但
body recall 降到约 `0.27`，导致 macro F1 `-0.1788`。后续 calibration/decision
kernel 必须分别守住 body critical recall 与 head false alerts，不能只优化 pooled
macro。

### Calibration 与 synthetic event transfer

两个不读取 dev outcome 的 train-side calibration 都未建立稳健修复：

- 按加权 BCE 的解析逆变换使用 `w/(1+w)` 阈值，显著降低 head FPR，却几乎清空
  head recall，seed17 三折 macro F1 全部下降；
- 每个 horizon×height 在 10 个 train environments 上按 environment-macro F1
  选阈值，seed17 只在 fold1 从约 `0.397` 升到 `0.432`，fold0/2 分别降到约
  `0.438/0.391`。

因此停止后处理阈值搜索。为直接检查 representation 增量能否穿过连续决策，增加一个
synthetic teacher-derived event proxy：

- lane unit：`environment × near/far × body/head × direction`；
- truth positive：任一 teacher-known distance cell risk ≥0.5；
- truth negative：六个 distance cells 全 known 且都非风险；其余为 unknown；
- candidate active：任一 distance cell 同时 predicted-known 与 predicted-risk；
- 连续 positive lane-frames 组成事件，并报告 hit/miss、negative false-active 与
  clearance。

它不是人类事件 truth，不是用户路线，也不是 App decision kernel，只是
Development 表示到连续行为的最小压力测试。3 seeds × 3 folds 结果：

| 指标 delta（directional - pooled） | mean | median | 正/负/零 |
|---|---:|---:|---:|
| event recall | +0.0102 | -0.0069 | 4 / 5 / 0 |
| false-active lane-frame rate | +0.0207 | -0.0182 | 3 / 6 / 0 |
| clearance rate | +0.0841 | 0 | 4 / 2 / 3 |
| body event recall | -0.0482 | -0.0805 | 4 / 5 / 0 |
| body false-active rate | -0.0565 | -0.0370 | 1 / 6 / 2 |
| head event recall | +0.0820 | 0 | 4 / 4 / 1 |
| head false-active rate | +0.1544 | +0.1351 | 6 / 3 / 0 |

三个 folds 的完整负 lane-frame exposures 只有 `55/114/187`，且 seed 重复不增加
truth exposure，所以 clearance/false-active 只作诊断。结果显示 directional 主要把
行为从 body alerts 重分配到 head alerts：body false-active 降低但 recall 也下降，
head recall 上升但 false-active 更明显上升。cell F1/排序正结果没有稳定转化成事件
行为改善，当前终态为：

`UNCALIBRATED_SYNTHETIC_EVENT_TRANSFER_NOT_SUPPORTED`

## Known-loss objective intervention

事件代理审计显示，risk-only 的 body 激活仍较高，主要损失发生在
`predicted-known AND predicted-risk` 的 known gate。为检验这是否只是 known
正类稀疏造成的训练偏置，在不改变数据、架构、threshold 和事件定义的情况下增加
两种训练目标：

- `balanced`：每个 horizon×height 使用 train negatives/positives 作为 known
  positive weight；
- `sqrt_balanced`：使用上述比值的平方根，作为 plain 与完全补偿之间的对数空间
  中点。

完全 balanced 已运行 3 seeds × 3 folds。相对同 seed/fold 的 directional
reference：

| 指标 delta（balanced - directional） | mean | 方向 |
|---|---:|---|
| environment-macro F1 | +0.0010 | 8/9 正，但最差单元 -0.0468 |
| event recall | +0.0941 | 7/9 正 |
| false-active lane-frame rate | +0.0435 | 5/9 恶化 |
| body event recall | +0.1492 | 8/9 正 |
| body false-active rate | +0.0578 | 5 恶化、3 改善、1 零 |
| clearance rate | -0.0101 | 无稳定改善 |

完全补偿确认 known gate 是可干预瓶颈，但以更高误激活换取召回，不能作为事件级
reference。为避免无界调权，只追加一个 seed17 三折的 sqrt-balanced 判别：

| fold | event recall delta | false-active delta | clearance delta |
|---:|---:|---:|---:|
| 0 | -0.0688 | +0.0182 | 0 |
| 1 | +0.0928 | +0.0263 | -0.0714 |
| 2 | +0.0759 | +0.0214 | -0.0571 |

其 cell-level future body/head macro F1 mean delta 只有 `+0.0009`；事件召回
mean `+0.0333`，但三折 false-active 全部恶化，mean `+0.0220`。因此不扩展
seed29/43，也不继续搜索标量权重：

`KNOWN_LOSS_REWEIGHTING_EVENT_INCREMENT_NOT_SUPPORTED`

这是有效的算法权衡负结果，不是 protocol INVALID。它只关闭当前 known 正类
reweighting；directional 的 representation 正结果仍成立。

## Height-temporal selective decision kernel

known-loss 结果表明问题不是“known 正类权重不够大”，而是把可观测性估计直接当成
提醒许可。先在同一 outcome-open Development 数据上比较最小 kernel family：

- `risk-only`、概率和、概率积与 high-risk override 等静态放宽规则；
- 2/3 个连续 anchor 确认和 causal 2-of-3；
- body/head 分别使用不同的门控和确认长度。

静态放宽规则均提高召回，但同时增加误激活。最初 v0 用 body risk-only 连续 3 帧、
head 0.9 override 连续 2 帧；总体 recall/false-active 分别
`+0.1302/-0.0537`，但 head recall mean `-0.0708`。因此 v1 将高置信 head
override 改为立即响应：

| height | base activation | 因果确认 | 高置信 override |
|---|---|---:|---|
| body | risk ≥0.5，不使用 known 硬门 | 3 anchors（0.6 s） | 无 |
| head | known ≥0.5 且 risk ≥0.5 | 2 anchors（0.4 s） | risk ≥0.8 立即响应 |

v1 相对同一 directional checkpoint 的 hard-known-and-risk kernel：

| 指标 delta（v1 - hard） | mean | median | paired 方向 |
|---|---:|---:|---:|
| event recall | +0.1705 | +0.1852 | 8 正 / 1 负 |
| false-active lane-frame rate | -0.0245 | -0.0351 | 7 改善 / 2 恶化 |
| response-delay median | 0 | 0 | 9 不变 |
| body event recall | +0.3569 | +0.3158 | 9 正 |
| head event recall | +0.0038 | +0.0133 | 5 正 / 4 负 |
| head false-active rate | -0.2286 | -0.0182 | 6 改善 / 2 恶化 / 1 零 |
| clearance rate | -0.0503 | -0.0238 | 1 改善 / 5 恶化 / 3 零 |

因此正结果按实际层级保留为：

`HEIGHT_TEMPORAL_SELECTIVE_DECISION_KERNEL_SIGNAL_SUPPORTED_IN_DEVELOPMENT`

它表明已训练 risk field 中存在能被因果选择性 kernel 利用的连续信号；不是人类事件
效用或系统安全结论。false-alert event count mean 仍增加 `+0.78`，说明更少的
false-active frames 可能碎成更多短 alert，clearance 也尚未守住。

为区分 kernel 与 representation 的贡献，又把 v1 同时应用于 pooled 与 directional。
directional - pooled 的 event recall mean 只有 `+0.0144`（5 正/4 负），
false-active mean `-0.0006`（4 改善/3 恶化/2 零）；body recall 甚至 mean
`-0.0438`。因此 v1 的正信号主要来自 decision kernel，本轮仍不建立 directional
representation 穿过事件层的稳健增量。原
`UNCALIBRATED_SYNTHETIC_EVENT_TRANSFER_NOT_SUPPORTED` 不被偷偷改写。

### v2 spatial-support repair

逐 lane 诊断显示，v1 的 body 只需任一 distance cell risk≥0.5 就能触发整条 lane。
这允许孤立高分 cell 形成短 false-alert runs，并在 teacher truth 清除后继续激活。
不能用最大提醒时长强制静音，因为那会在持续危险中游戏化 event recall；v2 改用
风险场自身的空间支持：

- body base：至少 `3/6` 个 distance cells risk≥0.5；
- body override：任一 distance cell risk≥0.8；
- base/override 合并后仍需连续 3 个 anchor；
- head 完全继承 v1，不改变其 recall/false-active。

v2 相对 directional hard reference：

| 指标 delta（v2 - hard） | mean | median | paired 方向 |
|---|---:|---:|---:|
| event recall | +0.1352 | +0.1217 | 8 正 / 1 负 |
| false-active lane-frame rate | -0.1091 | -0.0526 | 7 改善 / 2 恶化 |
| clearance rate | +0.0566 | 0 | 3 改善 / 4 恶化 / 2 零 |
| false-alert event count | -0.78 | -2 | 6 改善 / 2 恶化 / 1 零 |
| body event recall | +0.2886 | +0.2368 | 9 正 |
| body false-active rate | -0.0871 | -0.1296 | 5 改善 / 3 恶化 / 1 零 |
| response-delay median | 0 | 0 | 9 不变 |

相对 v1，v2 主动接受 mean `-0.0352` 的 event-recall 回退，换来
false-active `-0.0847`、clearance `+0.1069`、false-alert events `-1.56`；
false-active 8/9 非劣、clearance 9/9 非劣、fragmentation 9/9 非劣。因此 v2
取代 v1：

`HEIGHT_SPATIOTEMPORAL_SELECTIVE_DECISION_KERNEL_SIGNAL_SUPPORTED_IN_DEVELOPMENT`

同一个 v2 同时应用于 pooled 与 directional 后，directional - pooled 为：

| 指标 delta | mean | median | paired 方向 |
|---|---:|---:|---:|
| event recall | +0.0810 | +0.0635 | 8 正 / 1 负 |
| false-active rate | -0.0739 | -0.0088 | 5 改善 / 3 恶化 / 1 零 |
| clearance rate | +0.1958 | +0.0286 | 5 改善 / 2 恶化 / 2 零 |
| body event recall | +0.0919 | +0.0526 | 9 正 |
| body false-active rate | -0.2154 | -0.1750 | 8 改善 / 1 零 |
| head event recall | +0.0696 | +0.0800 | 6 正 / 3 负 |
| head false-active rate | +0.0341 | 0 | 4 恶化 / 1 改善 / 4 零 |

这建立了第一个 representation→decision 的 Development 正信号：

`DIRECTIONAL_SPATIAL_STRUCTURE_SELECTIVE_EVENT_TRANSFER_SIGNAL_SUPPORTED_IN_DEVELOPMENT`

它仍不是 fresh transfer：v2 的 3-cell support 和 0.8 override 均在这些
outcome-open folds 上选择，negative exposure 仍只有每折 `55/114/187`，head
false-active 也未形成 guardrail。旧 hard-kernel 负终态继续有效，新结论只描述
selective v2。

## History mechanism repair

原 single 训练把当前帧重复五次。对 5-tap、1×1 temporal convolution 来说，这只
约束五个时间权重之和；各时间位置权重本身不确定。直接换成真实 history 会先产生
任意扰动，然后微调再尝试修复。为排除这个结构性伪差异，增加：

1. `current + zero-initialized 1×1 temporal residual`；
2. 冻结 directional single，只训练 2,304 个 residual 参数；
3. 允许邻域运动的 zero-initialized 3×3 temporal residual，只训练 20,736 个参数。

三者的 epoch 0 都与 directional single 精确相同。结果仍不支持 history：

- 原 joint history 相对 directional single 三折 delta 为
  `-0.0140 / -0.0123 / +0.0017`；
- zero-init residual 全模型微调三折都选择 epoch 0；
- 1×1 residual-only 三折都选择 epoch 0；
- 3×3 residual-only 仅 fold 2 为 `+0.0029`，fold 0/1 都选择 epoch 0。

因此当前负终态精确限制为：

`UNALIGNED_HISTORY_FUSION_INCREMENT_NOT_SUPPORTED`

它不证明历史 RGB 没有信息。只有引入显式 feature alignment、flow/ego-motion
compensation 或新的时序表征后，才值得重开 history；不再继续无对齐结构或学习率
搜索。

## Outcome-unseen TartanGround transfer

### 固定选择与 corpus

在 v2 参数和 evaluation code 已固定后，从 base + expansion 未使用的
TartanGround environments 中机械选择 transfer cohort：

- selection seed：`HFTF_D5_OUTCOME_UNSEEN_TRANSFER_V0`；
- 排除前 15 个已使用 environments；
- 只保留同时具有 metadata、front RGB、front depth 的 P1000 parents；
- 按 `sha256(seed:environment)` 升序取前 6 个；
- engineering failure 可修复重跑，不因下载、路径或解码错误烧毁 environment。

选择结果为：

- `ModularNeighborhoodIntExt`
- `Fantasy`
- `GothicIsland`
- `OldIndustrialCity`
- `Hospital`
- `OldTownFall`

每个 environment 生成 33 个 5 Hz anchors，共 198 samples、444 个 RGB/depth
PNG 和 222 张 unique history RGB images。全部 PNG 解码成功，全部样本路径存在，
teacher label 重解码一致，且 selected 与此前 used environment sets 无交集。
manifest SHA-256 为
`292315ce65c14e53fc94981b28326390a71566581b60b87b824ef34719858e38`，
samples JSONL SHA-256 为
`f2669c336c36b24b98ee24bfa9c27db42d2a70869b7e85a0528416e0d4b9b237`。

这个 transfer corpus 包含 266 个 positive lane events、1,608 个 positive
lane frames、130 个 complete negative lane frames 和 20 个 clearance-eligible
events；六个 environments 都同时具有正、负机会。它比原三折每折
`55/114/187` 个 negative frames 增加了独立环境覆盖，但 130 个 negative frames
和 20 个 clearance events 仍然偏少。

### 表示层迁移

先不经过 event kernel，直接在 9 个原训练 checkpoint pairs 上比较 directional
与 pooled 的 future body/head field：

| 指标 delta（directional - pooled） | mean | median | paired 方向 |
|---|---:|---:|---:|
| environment-macro F1 | +0.0473 | +0.0822 | 7 正 / 2 负 |
| aggregate macro F1 | +0.0298 | +0.0657 | 6 正 / 3 负 |
| aggregate micro F1 | +0.0284 | +0.0397 | 5 正 / 4 负 |
| recall | +0.1538 | +0.2462 | 7 正 / 2 负 |
| false-positive rate | +0.1420 | +0.1412 | 9 恶化 |
| precision | -0.0120 | -0.0283 | 3 正 / 6 负 |

54 个 environment×seed×fold F1 比较中 37 胜、17 负。也就是说，directional
的 field F1/recall 增量在新环境中仍有多数支持，但裸 0.5 threshold 的 FPR
9/9 变差；不能把表示层结果直接写成提醒行为改善。

### 固定 v2 下的表示到事件迁移

复用完全相同的 `height_spatiotemporal_selective_v2`，不在 transfer outcomes 上
修改 support count、threshold、confirmation length 或 override：

| 指标 delta（directional - pooled，同一 v2） | mean | median | paired 方向 |
|---|---:|---:|---:|
| event recall | +0.1809 | +0.1692 | 9 正 |
| false-active lane-frame rate | -0.0727 | -0.0923 | 7 改善 / 2 恶化 |
| clearance rate | +0.0444 | +0.1000 | 6 改善 / 3 恶化 |
| response-delay median | 0 | 0 | 9 不变 |
| false-alert event count | +3.56 | +3 | 6 恶化 / 2 改善 / 1 零 |
| body event recall | +0.1794 | +0.1803 | 9 正 |
| head event recall | +0.1821 | +0.1528 | 6 正 / 2 负 / 1 零 |

因此当前最强、但仍严格限层的正终态是：

`DIRECTIONAL_SPATIAL_STRUCTURE_SELECTIVE_EVENT_TRANSFER_REPLICATED_ON_OUTCOME_UNSEEN_TARTANGROUND_ENVIRONMENTS_IN_DEVELOPMENT`

这里的 “replicated” 指 v2 在固定后迁移到未参与数据、模型或 kernel 开发的六个
TartanGround environments；不是把这些 environments 宣称为真实人类 held-out
event truth。

### 决策层贡献与反例

在 directional checkpoint 上单独比较 decision kernels，v2 相对 hard
known-and-risk 的 event recall `+0.1412`（8/9 正）、clearance `+0.1444`
（8 改善/1 零）、false-alert event count `-4.89`（8 改善/1 零），但
false-active lane-frame rate mean `+0.0359`，5/9 恶化。v2 相对 v1 则在
false-active、clearance、false-alert count 三项上都是 9/9 非劣，代价是 recall
7/9 回落。空间支持对 v1 fragmentation 的修复可以迁移，但 v2 尚未相对 hard
建立全部 guardrail 支配。

还必须保留两个聚合反例：

- directional 相对 pooled 虽然按全部 negative frames 加权的 false-active
  7/9 改善，但 54 个 environment cells 的等权 mean delta 为 `+0.0129`，
  其中 31 恶化、16 改善、7 不变；
- false-alert event count mean 增加 `+3.56`；height 分解显示 head count
  8/9 恶化、1/9 改善，说明更少的 active frames 仍可能碎成更多 head alerts。

所以当前证据支持“方向结构 + 固定选择性 kernel 在合成新环境上提高事件召回，并在
多数 checkpoint pairs 降低加权误激活”，不支持“每环境都更稳”“假警更少”或
“HFTF 已超过主线”。

## Consumed SANPO real parent-event transfer

### 最短真实事件接缝

不新建协议 ceremony，直接复用已经存在的 RISKSEG-R0 event view：

- 30 个 source-session-disjoint parent events、1,920 张真实连续 RGB；
- 8 个 blocking-obstacle positives、8 个 boundary-level-change positives；
- 7 个 parallel-curb negatives、7 个 normal-walkable negatives；
- 正事件具有 alertable/passed intervals，来自两路隔离 RGB review 与
  output-blind adjudication；
- 同一 cohort 的当前 YOLO reference 为 `13/16` hits、`6/14` false-alert
  events、`5/16` cleared。

这个 cohort 已被主线实验消费，因此这里只是 Development，不重新包装成 fresh 或
held-out。HFTF checkpoints 从未使用 SANPO 训练；real-event adapter 在任何 HFTF
outcome 前固定为：

- current RGB 单帧，重复输入的 single arm 用数学等价的 encoder fast path；
- 保留 v2 的 body/head、near/far 和 5 Hz causal confirmation；
- 只合并对应直行路线 `-15°..+15°` 的中央 direction indices 2/3；
- 任一中央 near/far × body/head lane active 时，当前帧提醒；
- 以 causal zero-order hold 把 5 Hz 状态映射回原 10 Hz event timeline；
- event hit、negative false alert 与 passed clearance 完全复用 RISKSEG 口径。

### 真实 recall 正信号与效用负结果

9 个 directional checkpoints 的结果高度一致：

| 指标 | 9-checkpoint 结果 | 当前 YOLO |
|---|---:|---:|
| positive-event hits | 16/16，9/9 单元相同 | 13/16 |
| critical misses | 0/16，9/9 单元相同 | 3/16 |
| false-alert events | 13–14/14，median 14 | 6/14 |
| cleared events | 0–2/16，median 0 | 5/16 |
| response delay from alertable start | median 2–4 个 10 Hz frames | 不同链路，不直接比较 |
| all-frame active fraction | mean 87.08% | 未以同一 raw state 定义 |

因此两个层级必须同时保留：

`REAL_EVENT_RECALL_SIGNAL_SUPPORTED_ACROSS_NINE_CHECKPOINTS_IN_DEVELOPMENT`

`FIXED_KERNEL_REAL_EVENT_SPECIFICITY_AND_CLEARANCE_NOT_SUPPORTED`

前者不是被后者抹掉：从纯合成 TartanGround 训练得到的 9 个模型全部命中
pedestrian/cone blocking 与 curb/stair/boundary 两类真实正事件，说明 risk field
不是在真实 RGB 上完全失效。后者说明当前绝对阈值 + 中央 lane union 几乎常开，
无法形成可用提醒。

### directional 相对 pooled

同一 real-event adapter 下再比较 9 个 paired pooled checkpoints：

- directional hit count 4/9 更高、5/9 相同、0/9 更低，mean delta `+2.22 events`；
- false-alert count 2/9 改善、3/9 恶化、4/9 相同，mean delta `+0.33`；
- cleared count 2/9 提高、3/9 降低、4/9 相同，mean delta `-1.89`；
- directional active-frame count mean 为 1,672/1,920，pooled 为
  1,464.4/1,920。

directional 的真实域贡献仍主要表现为 recall，而没有同时改善 specificity 与
clearance；18 个模型中没有一个在 hits、false alerts、cleared 三项上
Pareto-dominate 当前 YOLO。因此精确终态是：

`DIRECTIONAL_REAL_EVENT_PARETO_INCREMENT_NOT_SUPPORTED`

这关闭的是当前未校准 real-domain adapter，不关闭 directional 的合成表示正结果，
也不证明真实帧没有可用于区分正负事件的相对方向结构。下一步先检查中央/侧向风险
profile 是否能区分“前方侵入”与“平行路沿”，再决定是否值得做 session-held-out
weak event calibration；不在同一结果上直接搜索更多绝对阈值。

### Central-vs-lateral profile diagnostic

诊断保持 alert output 完全不变，也不搜索 threshold。每个 checkpoint 只把
alertable positive interval 与完整 negative event 各压成一个 event-level median，
比较五类 field evidence 的中央绝对值、中央减侧向均值和中央峰值减侧向峰值：

| profile | all-negative AUC mean/median | parallel-curb AUC mean/median |
|---|---:|---:|
| `risk_mean / central_mean` | 0.5893 / 0.6071 | 0.5327 / 0.5625 |
| `risk_mean / central_peak-lateral_peak` | 0.5655 / 0.5938 | 0.5208 / 0.5089 |
| `risk_mean / central_mean-lateral_mean` | 0.5501 / 0.5312 | 0.4990 / 0.4732 |
| `body_k3 / central_mean-lateral_mean` | 0.5060 / 0.4777 | 0.4772 / 0.4464 |
| `head_known_risk / central_mean-lateral_mean` | 0.5025 / 0.5179 | 0.4633 / 0.4286 |

绝对中央 risk 对 normal-walkable 有弱排序信号，但没有稳定分离 parallel curb；
相对方向 profile 在最关键的 parallel-curb 对比上接近或低于随机。因此：

`CENTRAL_VS_LATERAL_ACTIONABILITY_PROFILE_NOT_SUPPORTED`

这说明真实 recall 的主要来源是整体高风险响应，而不是已经学会“前方侵入 vs
平行但不阻塞”的关系。简单换 absolute threshold、手工 central-minus-lateral
公式或继续 v2 搜索都缺少机制依据；后继必须显式加入真实 actionability relation
监督，并以 source-session-held-out 输出评价。

### Source-session-held-out weak relation head

后继固定 HFTF backbone，不访问 test session 做标准化、拟合或 threshold。30 个
events 在每个 bucket 内按固定 hash 排为 5 folds；train labels 只来自：

- positive event 的 alertable 5 Hz frames；
- positive event 的 passed 5 Hz frames；
- 完整 negative event frames；
- alertable 与 passed 之间的 transition gap 不进入训练。

输入是五类 HFTF field profile × 6 directions 共 30 个特征；每个 event 总权重相等，
正负类权重各 0.5。每折只在 24 个 train sessions 上做加权标准化和 L2 logistic fit，
test 使用固定 probability 0.5 与 5 Hz 两步因果确认。

9 个 backbone 的 out-of-fold 结果为：

| 指标 | relation-head 范围 / mean | 相对 fixed v2 |
|---|---:|---:|
| hits | 11–16/16，mean 13.22 | mean -2.78；8 降 / 1 同 |
| false-alert events | 8–13/14，mean 11.22 | mean -2.56；9/9 改善 |
| cleared | 4–11/16，mean 7.22 | mean +6.67；9/9 改善 |
| response delay | median 3–11 frames | 描述性 |

因此关系监督确实能稳定压低常开和恢复清除：

`WEAK_RELATION_HEAD_SPECIFICITY_CLEARANCE_SIGNAL_SUPPORTED_IN_DEVELOPMENT`

但没有一个 checkpoint 同时非劣于当前 YOLO 的 `13 hits / 6 false alerts /
5 cleared`；0/9 Pareto，精确负终态为：

`WEAK_RELATION_HEAD_REAL_EVENT_PARETO_INCREMENT_NOT_SUPPORTED`

这不是“关系监督无效”：它在 9/9 上把两个 guardrails 向正确方向移动。失败点是当前
output field 中的 30 个低容量特征无法在保住 recall 的同时完成 actionability
分离。下一次模型改动把同一 held-out 监督前移到 HFTF encoder 的 spatial feature
map；不搜索 logistic L2、threshold、confirmation 或 fold 分配。

### Fixed-encoder spatial relation head

空间头严格复用上述 5-fold source-session split、frame labels、event/class
weights、0.5 probability threshold 与 5 Hz 两步确认。唯一表示变化是从五类
output profiles × 6 directions 前移到固定 HFTF pointwise fused feature：
adaptive pool 为 `128 channels × 3 rows × 6 directions = 2,304 features`。
encoder/backbone 不解冻；每折 test sessions 仍不参与标准化或拟合。为适配维度，
L2 strength 固定为 `1.0`，没有做结果后搜索。

9 个 directional backbones 的 out-of-fold 结果为：

| 指标 | spatial-head 范围 / mean | 相对 output-field head | 相对当前 YOLO |
|---|---:|---:|---:|
| hits | 12–14/16，mean 13.00 | mean -0.22；3 胜 / 2 同 / 4 负 | mean 相同；3 胜 / 3 同 / 3 负 |
| false-alert events | 7–11/14，mean 9.00 | mean -2.22；9/9 改善 | mean +3.00；9/9 更差 |
| cleared | 6–12/16，mean 9.00 | mean +1.78；6 胜 / 1 同 / 2 负 | mean +4.00；9/9 改善 |
| response delay | median 2–6 frames | 描述性 | 尚未建立 lead-time 增量 |

空间特征同时稳定改善 specificity，并在多数 checkpoints 改善 clearance，而平均
recall 基本持平。因此保留模型表示层正终态：

`FIXED_ENCODER_SPATIAL_RELATION_HEAD_OVER_OUTPUT_FIELD_GUARDRAIL_INCREMENT_SUPPORTED_IN_DEVELOPMENT`

但 9 个 checkpoints 没有一个把 false alerts 压到 YOLO 的 6/14 或更低，0/9
同时非劣于 YOLO 的 hits/false-alert/cleared，故系统比较负终态仍为：

`FIXED_ENCODER_SPATIAL_RELATION_HEAD_REAL_EVENT_PARETO_INCREMENT_NOT_SUPPORTED`

后一个终态只限制“已超过当前系统”的主张，不撤销前一个空间表示正结果。本 cohort
已经回答 representation placement，不再用来搜索 grid、L2、threshold 或
confirmation。是否解冻 backbone 取决于后续互补性与样本容量诊断；其结果无论如何
仍是 consumed Development，不能升格为 fresh validation。

### Complementarity 与 fusion 边界

空间头和当前 YOLO 的错误并不相同。9 个 checkpoints 的事件级配对为：

| 诊断 | 9-checkpoint mean |
|---|---:|
| HFTF 补回 YOLO miss | 2.56/3 |
| HFTF 丢失 YOLO hit | 2.56 |
| 两者同时 miss | 0.44 |
| event-level OR hits | 15.56/16 |
| event-level OR false alerts | 11.00/14 |
| event-level AND hits | 10.44/16 |
| event-level AND false alerts | 4.00/14 |

因此保留：

`YOLO_HFTF_EVENT_COMPLEMENTARITY_SIGNAL_SUPPORTED_IN_DEVELOPMENT`

这不是一个可部署 OR/AND policy。为检验能否学习选择，固定 YOLO device trace 的
7 个 causal 200 ms features（detection count、actual alert、raw/stable risk、
left/center/right）并与 30 个 HFTF profiles 拼接；仍用原 5 folds、event/class
weights、L2、0.5 threshold 与两步确认。9 个 OOF 结果为：

| 指标 | static fusion 范围 / mean | 相对 HFTF weak head |
|---|---:|---:|
| hits | 11–14/16，mean 12.89 | mean -0.33；1 胜 / 5 同 / 3 负 |
| false-alert events | 8–11/14，mean 9.78 | mean -1.44；7 胜 / 1 同 / 1 负 |
| cleared | 3–11/16，mean 6.89 | mean -0.33；4 胜 / 2 同 / 3 负 |

0/9 超过当前 YOLO，且整体弱于 fixed-encoder spatial head：

`STATIC_YOLO_HFTF_FUSION_PARETO_INCREMENT_NOT_SUPPORTED`

完整 spatial head 每折约 497–619 training frames，却有 2,305 个自由参数且
regularized train loss 已约 `0.002`。固定 rank-2 把参数降到 293 的 canary 没有
改善：seed17/fold0 从 `13 hits / 8 false / 7 cleared` 变为 `13/10/7`：

`LOW_RANK_SPATIAL_RELATION_HEAD_CANARY_INCREMENT_NOT_SUPPORTED`

把静态融合扩为 current + 1-second delta + 1-second causal prefix mean 的 111-feature
transition canary 得到 `11/9/9`；clearance 上升但 recall 降到 68.75%：

`CAUSAL_TRANSITION_FUSION_CANARY_INCREMENT_NOT_SUPPORTED`

这三个负结果不是“互补性不存在”，而是当前 30 events 对更多结构自由度已经
information-limited。为了避免把 Development 变成同 cohort 的无限 head search，
rank/history/fusion-feature/L2/threshold 路线在此停止；下一步先扩充与这 30 sessions
隔离的真实 actionability relation supervision。

## 边界与下一实验

当前正结果只支持：

`teacher feasible + RGB learnable + multi-seed directional spatial-structure increment
+ height-spatiotemporal selective decision-kernel signal
+ directional selective-event transfer signal
+ outcome-unseen TartanGround selective-event replication
+ real SANPO positive-event recall signal
+ fixed-encoder spatial relation representation increment
+ YOLO-HFTF event complementarity signal`

尚未支持：

- history 对独立环境具有稳定增量；
- directional 的 threshold calibration 能跨 seed 稳定；
- known 正类标量重加权能同时改善召回与 false-active；
- selective v2 能在每个 outcome-unseen environment 同时守住 false-active；
- head false-alert fragmentation 在 selective v2 下形成稳定 guardrail；
- 固定 v2 在真实事件上能守住 specificity 与 clearance；
- directional 在真实事件上相对 pooled 或当前 YOLO 形成 Pareto 增量；
- central-vs-lateral profile 能稳定区分 parallel curb；
- output-field weak relation head 能形成真实事件 Pareto 增量；
- fixed-encoder spatial relation head 能形成真实事件 Pareto 增量；
- 静态或 1 秒 causal YOLO-HFTF fusion 能形成真实事件 Pareto 增量；
- 真实事件 warning lead time 相对当前 YOLO 改善；
- HFTF 超过当前主线或进入 App。

下一步保留 directional + v2 为合成 Development reference，也保留真实 recall
正信号，不再在 21 个 TartanGround environments 或 SANPO outcomes 上盲搜绝对阈值。
output-field relation head 已证明 guardrail 可学；fixed-encoder spatial head 又证明
信息放置位置具有稳定增量，事件配对也证明 HFTF/Yolo 互补，但当前融合仍未形成
YOLO Pareto。30-event cohort 的 head search 到此停止；下一步优先发现并物化与现有
30 sessions 隔离的 SANPO relation-supervision pool，先扩大真实正负关系覆盖，再
固定比较 spatial head 与 fusion head。没有新增监督前不解冻 backbone、不修改
v2、calibrator、空间头超参数或 history。

## 复现

```powershell
E:\codex-tools\tools\venvs\blindassist-venv-export312\Scripts\python.exe `
  scripts/research/hftf/resolve_stage_c_d5_s0b_p0c_tartanground_provider.py `
  --revision 388faf9c800568cfc6828fa47e063f8369397eb3

E:\codex-tools\tools\venvs\blindassist-venv-export312\Scripts\python.exe `
  scripts/research/hftf/run_stage_c_d5_tartanground_development_pilot.py

E:\codex-tools\tools\venvs\blindassist-venv-export312\Scripts\python.exe `
  scripts/research/hftf/materialize_stage_c_d5_tartanground_development_corpus.py

E:\codex-tools\tools\venvs\blindassist-venv-export312\Scripts\python.exe `
  scripts/research/hftf/train_stage_c_d5_tartanground_development_student.py `
  --samples artifacts.local/evidence/hftf/stage-c-d5-tartanground-development-corpus-v0/samples.jsonl `
  --pretrained artifacts.local/models/hftf/torch/hub/checkpoints/mobilenet_v3_small-047dcff4.pth `
  --output-root artifacts.local/evidence/hftf/stage-c-d5-tartanground-development-student-v0/single-seed17 `
  --arm single --seed 17 --epochs 20

E:\codex-tools\tools\venvs\blindassist-venv-export312\Scripts\python.exe `
  scripts/research/hftf/materialize_stage_c_d5_tartanground_development_expansion.py

E:\codex-tools\tools\venvs\blindassist-venv-export312\Scripts\python.exe `
  scripts/research/hftf/build_stage_c_d5_tartanground_cross_environment_folds.py

E:\codex-tools\tools\venvs\blindassist-venv-export312\Scripts\python.exe `
  scripts/research/hftf/evaluate_stage_c_d5_tartanground_event_proxy.py `
  --samples artifacts.local/evidence/hftf/stage-c-d5-tartanground-cross-environment-v1/fold-0/samples.jsonl `
  --pretrained artifacts.local/models/hftf/torch/hub/checkpoints/mobilenet_v3_small-047dcff4.pth `
  --output artifacts.local/evidence/hftf/stage-c-d5-tartanground-cross-environment-v1/event-proxy/seed-17/fold-0.json `
  --model pooled artifacts.local/evidence/hftf/stage-c-d5-tartanground-cross-environment-v1/training/fold-0/pooled-single-seed17/checkpoint.pt `
  --model directional artifacts.local/evidence/hftf/stage-c-d5-tartanground-cross-environment-v1/training/fold-0/directional-single-seed17/checkpoint.pt `
  --reference pooled

E:\codex-tools\tools\venvs\blindassist-venv-export312\Scripts\python.exe `
  scripts/research/hftf/evaluate_stage_c_d5_tartanground_decision_kernels.py `
  --samples artifacts.local/evidence/hftf/stage-c-d5-tartanground-cross-environment-v1/fold-0/samples.jsonl `
  --pretrained artifacts.local/models/hftf/torch/hub/checkpoints/mobilenet_v3_small-047dcff4.pth `
  --checkpoint artifacts.local/evidence/hftf/stage-c-d5-tartanground-cross-environment-v1/training/fold-0/directional-single-seed17/checkpoint.pt `
  --output artifacts.local/evidence/hftf/stage-c-d5-tartanground-cross-environment-v1/decision-kernel/height-temporal-selective-v1/seed-17/fold-0.json

E:\codex-tools\tools\venvs\blindassist-venv-export312\Scripts\python.exe `
  scripts/research/hftf/materialize_stage_c_d5_tartanground_outcome_unseen_transfer.py

E:\codex-tools\tools\venvs\blindassist-venv-export312\Scripts\python.exe `
  scripts/research/hftf/evaluate_stage_c_d5_tartanground_development_checkpoints.py `
  --samples artifacts.local/evidence/hftf/stage-c-d5-tartanground-outcome-unseen-transfer-v0/samples.jsonl `
  --pretrained artifacts.local/models/hftf/torch/hub/checkpoints/mobilenet_v3_small-047dcff4.pth `
  --output artifacts.local/evidence/hftf/stage-c-d5-tartanground-outcome-unseen-transfer-v0/evaluation/cell-level/seed-17/fold-0.json `
  --model pooled single artifacts.local/evidence/hftf/stage-c-d5-tartanground-cross-environment-v1/training/fold-0/pooled-single-seed17/checkpoint.pt `
  --model directional single artifacts.local/evidence/hftf/stage-c-d5-tartanground-cross-environment-v1/training/fold-0/directional-single-seed17/checkpoint.pt `
  --reference pooled --role transfer

E:\codex-tools\tools\venvs\blindassist-venv-export312\Scripts\python.exe `
  scripts/research/hftf/evaluate_stage_c_d5_tartanground_event_proxy.py `
  --samples artifacts.local/evidence/hftf/stage-c-d5-tartanground-outcome-unseen-transfer-v0/samples.jsonl `
  --pretrained artifacts.local/models/hftf/torch/hub/checkpoints/mobilenet_v3_small-047dcff4.pth `
  --output artifacts.local/evidence/hftf/stage-c-d5-tartanground-outcome-unseen-transfer-v0/evaluation/height-spatiotemporal-selective-v2/seed-17/fold-0.json `
  --model pooled artifacts.local/evidence/hftf/stage-c-d5-tartanground-cross-environment-v1/training/fold-0/pooled-single-seed17/checkpoint.pt `
  --model directional artifacts.local/evidence/hftf/stage-c-d5-tartanground-cross-environment-v1/training/fold-0/directional-single-seed17/checkpoint.pt `
  --reference pooled --role transfer `
  --decision-policy height_spatiotemporal_selective_v2

E:\codex-tools\tools\venvs\blindassist-venv-export312\Scripts\python.exe `
  scripts/research/hftf/evaluate_stage_c_d6_sanpo_real_event_transfer.py `
  --checkpoint artifacts.local/evidence/hftf/stage-c-d5-tartanground-cross-environment-v1/training/fold-0/directional-single-seed17/checkpoint.pt `
  --name directional-seed17-fold0 `
  --output artifacts.local/evidence/hftf/stage-c-d6-sanpo-real-event-transfer-v0/directional/seed-17/fold-0.json

E:\codex-tools\tools\venvs\blindassist-venv-export312\Scripts\python.exe `
  scripts/research/hftf/diagnose_stage_c_d6_sanpo_direction_profile.py `
  --checkpoint artifacts.local/evidence/hftf/stage-c-d5-tartanground-cross-environment-v1/training/fold-0/directional-single-seed17/checkpoint.pt `
  --name directional-seed17-fold0 `
  --output artifacts.local/evidence/hftf/stage-c-d6-sanpo-direction-profile-v0/seed-17/fold-0.json

E:\codex-tools\tools\venvs\blindassist-venv-export312\Scripts\python.exe `
  scripts/research/hftf/run_stage_c_d6_sanpo_weak_relation_head.py `
  --checkpoint artifacts.local/evidence/hftf/stage-c-d5-tartanground-cross-environment-v1/training/fold-0/directional-single-seed17/checkpoint.pt `
  --name directional-seed17-fold0 `
  --output artifacts.local/evidence/hftf/stage-c-d6-sanpo-weak-relation-head-v0/seed-17/fold-0.json

E:\codex-tools\tools\venvs\blindassist-venv-export312\Scripts\python.exe `
  scripts/research/hftf/run_stage_c_d6_sanpo_spatial_relation_head.py `
  --checkpoint artifacts.local/evidence/hftf/stage-c-d5-tartanground-cross-environment-v1/training/fold-0/directional-single-seed17/checkpoint.pt `
  --name directional-seed17-fold0 `
  --output artifacts.local/evidence/hftf/stage-c-d6-sanpo-spatial-relation-head-v0/seed-17/fold-0.json

E:\codex-tools\tools\venvs\blindassist-venv-export312\Scripts\python.exe `
  scripts/research/hftf/run_stage_c_d6_sanpo_low_rank_spatial_relation_head.py `
  --checkpoint artifacts.local/evidence/hftf/stage-c-d5-tartanground-cross-environment-v1/training/fold-0/directional-single-seed17/checkpoint.pt `
  --name directional-seed17-fold0 `
  --output artifacts.local/evidence/hftf/stage-c-d6-sanpo-low-rank-spatial-relation-head-v0/seed-17/fold-0.json

E:\codex-tools\tools\venvs\blindassist-venv-export312\Scripts\python.exe `
  scripts/research/hftf/run_stage_c_d6_sanpo_yolo_hftf_fusion_head.py `
  --checkpoint artifacts.local/evidence/hftf/stage-c-d5-tartanground-cross-environment-v1/training/fold-0/directional-single-seed17/checkpoint.pt `
  --name directional-seed17-fold0 `
  --output artifacts.local/evidence/hftf/stage-c-d6-sanpo-yolo-hftf-fusion-head-v0/seed-17/fold-0.json

E:\codex-tools\tools\venvs\blindassist-venv-export312\Scripts\python.exe `
  scripts/research/hftf/run_stage_c_d6_sanpo_causal_transition_fusion_head.py `
  --checkpoint artifacts.local/evidence/hftf/stage-c-d5-tartanground-cross-environment-v1/training/fold-0/directional-single-seed17/checkpoint.pt `
  --name directional-seed17-fold0 `
  --output artifacts.local/evidence/hftf/stage-c-d6-sanpo-causal-transition-fusion-head-v0/seed-17/fold-0.json
```

网络读取完成后可用 `--skip-fetch` 重算 geometry result。生成数据位于 ignored
`artifacts.local/evidence/hftf/stage-c-d5-s0b-p0c-provider-resolution-20260802/`。
