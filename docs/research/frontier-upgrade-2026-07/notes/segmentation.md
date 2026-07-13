# 边界感知轻量分割精读笔记：PIDNet、Mobile-Seed 与 Mobile-PID 最小路线

## 1. 范围、证据等级与结论

本笔记只回答一个工程问题：在不替换 MobileNetV3 骨干、不绕过既有训练和晋级门的前提下，如何用边界分支和受控融合改善 `boundary_step_curb`、最差 seed 与跨 session 稳定性。

证据标签如下：

- **[全文证据]**：来自本地 PDF 正文、公式、图表或作者明确写出的限制，均标注 PDF 页码。
- **[本地证据]**：来自 BlindAssist 仓库文档或当前代码，只描述当前仓库状态。
- **[工程推断]**：由论文机制和本地问题共同推出的待验证假设，不当作论文结论。

结论先行：

1. **最值得迁移的不是完整 PIDNet 或 Mobile-Seed，而是三项可拆分机制**：高分辨率 detail 路径、显式几何边界分支、由边界决定 detail/context 权重的融合。PIDNet 为三者提供较强的正式会议证据；Mobile-Seed 为“小模型中直接增加边界监督可能反而伤害语义任务”提供关键反例。[全文证据：PIDNet PDF p.4–7；Mobile-Seed PDF p.3–8]
2. `boundary_step_curb` 是 BlindAssist 的**语义风险类**，论文中的 boundary 是由相邻类别变化得到的**几何对象边界**。两者有关但不等价。把论文的 boundary 指标直接当作 `boundary_step_curb` IoU 是概念错误。[本地证据：`docs/SANPO_SEGMENTATION_CANDIDATE.md`；全文证据：Mobile-Seed PDF p.1–2]
3. 当前图已实现 `sigmoid` context gate 且移除了 pooled BatchNorm，但默认仍是 OS8 detail / OS32 semantic、直接 concat、无显式边界分支。OS4/OS16 接口已存在于代码，尚无本任务包内的重跑结果，因此必须先完成修正基线复跑，不能把代码存在当成质量改善。[本地证据：`scripts/sanpo_segmentation_model.py`；`docs/SANPO_P0_SEED_FACTOR_AUDIT_2026-07-13.md`]
4. 推荐的最小路线是 **Mobile-PID-lite**：保留 MobileNetV3 和当前 LR-ASPP context gate，先单独验证 OS4/OS16，再增加一个 GroupNorm 的多尺度 D 分支和 Light-Bag 式空间门；边界辅助损失和双任务一致性按条件逐级打开。不要一次替换骨干、分辨率、损失、融合和训练预算。[工程推断]

---

## 2. 当前本地问题与论文机制的对接边界

### 2.1 已确认的本地事实

- 固定 sampler、改变 model seed 时，head-only selection score 为 `0.1739–0.4424`，范围 `0.2685`；固定 model seed、改变 sampler 时范围只有 `0.0112`。前者约为后者的 `24.1×`。五组最差场景均为 `step_curb`。[本地证据：`docs/SANPO_P0_SEED_FACTOR_AUDIT_2026-07-13.md`]
- 当前训练合同以 optimizer step 比较，使用预注册多 seed、session-balanced sampler、rare-class crop，并以 dev mIoU 与 `boundary_step_curb` IoU 的调和评分选模；blind 不得被 trainer 读取。[本地证据：`docs/SANPO_TRAINING_PROTOCOL.md`]
- 当前权威模型函数已将 LR-ASPP pooled 分支修正为 `global pool -> 1×1 conv -> sigmoid`，无 pooled BN；默认 detail 为 OS8、semantic 为 OS32，融合是 concat 后单个 `1×1` semantic logits。代码也支持 OS4 detail 和 OS16 semantic，但它们不是默认值。[本地证据：`scripts/sanpo_segmentation_model.py`]

### 2.2 不能混淆的两个“边界”

| 概念 | BlindAssist `boundary_step_curb` | PIDNet / Mobile-Seed boundary |
|---|---|---|
| 含义 | 台阶、路沿、可通行区边界等语义风险类 | 相邻语义类别发生变化的像素带 |
| 当前主指标 | class IoU；并进入事件层作为边界证据 | boundary F-score、Boundary IoU 或边界 BCE |
| 典型错误 | 把平行路沿直接升级为障碍提醒 | 纹理边缘或标注轮廓误差影响几何边界 |
| 可迁移关系 | 更清晰的空间轮廓可能帮助细窄风险类 | 只能作为辅助表征与诊断，不能替代风险类标签 |

**[工程推断]** 因此新增 D 分支应输出独立的 `geometric_boundary`，而主输出仍维持四类 taxonomy。报告必须同时给出：

1. `boundary_step_curb` class IoU；
2. geometric Boundary IoU / F-score；
3. 二者在 `step_curb`、平行路沿负例和阴影/纹理强边缘上的混淆。

只有几何边界变好而 `boundary_step_curb` 或事件误提醒变差，不能视为成功。

---

## 3. 证据卡 PID：PIDNet

### 3.1 元数据和全文状态

- 题名：*PIDNet: A Real-Time Semantic Segmentation Network Inspired by PID Controllers*
- 作者：Jiacong Xu, Zixiang Xiong, Shankar P. Bhattacharyya
- 会议：CVPR 2023，CVF Open Access 正式会议全文
- 本地文件：`.downloads/papers/2026-07-frontier-upgrade/01_PIDNet_CVPR_2023.pdf`
- 页数：11；SHA256：`21e13de72f1420a6472e72d0d69e0ae1c697adba2d597b11c602c606509aeaf4`
- 阅读范围：PDF p.1–11；方法与实验重点为 p.3–8，p.9–11 为参考文献。

### 3.2 论文解决的问题

**[全文证据，PDF p.1–3，Fig.2–3]** 作者把两分支网络解释为 PI 控制器：detail/P 路径保留局部和高频信息，context/I 路径累积大范围低频语义。直接融合时，低频上下文可能淹没边界和小物体，形成作者所称的 spatial overshoot。D 分支用边界作为高频响应，为 detail/context 融合提供阻尼。

这不是“PID 数学证明了三分支一定更好”。论文给出的是类比、简化一维卷积分析、特征可视化和消融证据。对 BlindAssist 可用的核心是**高频边界信号参与空间融合**，不是控制理论名词本身。

### 3.3 结构机制

| 机制 | 全文定位 | 可核验内容 | 对当前项目的含义 |
|---|---|---|---|
| 三分支 P/I/D | PDF p.4，Fig.4 | P 保持高分辨率 detail；I 聚合局部和全局 context；D 提取高频并预测边界。图中 P、D 主体保持约 1/8，I 最深到 1/64 | 不应直接复制尺度；可借用“浅层 detail + 深层 semantic + 边界 gate”职责划分 |
| Pag | PDF p.5，Eq.7–8，Fig.5 | 通过 P/I 对应像素特征相似度的 sigmoid，选择性把 I 语义注入 P，避免 P 被上下文淹没 | 比当前无条件 concat 更有针对性，但增加了中途交互，不是第一轮最小改动 |
| PAPPM | PDF p.5，Fig.6；p.7，Table 3 | 把 DAPPM 串行聚合改为并行并减少每尺度通道；在作者平台上同为 78.8 mIoU，PAPPM+Bag 为 93.2 FPS，DAPPM+Bag 为 83.7 FPS | 当前已有轻量 global context gate；不建议首轮再引入 PPM，避免同时改变 context 和 boundary 两个因素 |
| Bag / Light-Bag | PDF p.5，Eq.9–11，Fig.7 | `σ=sigmoid(D)`；边界处更信 P/detail，非边界处更信 I/context。Light-Bag 用两个 1×1 路径近似重融合 | 是 Mobile-PID 首选迁移点，因为它让 D 直接影响语义输出且可做很小的头部改动 |
| 边界监督与 BAS loss | PDF p.4，Eq.5–6；p.7，Table 4 | weighted BCE 训练 D；BAS loss 只在高边界置信区域强调语义 CE；作者固定 `λ0=0.4, λ1=20, λ2=1, λ3=1, t=0.8` | 这些权重依赖 Cityscapes 标签密度与损失尺度，不可直接移植；先验证 D/fusion，再决定辅助损失 |

### 3.4 关键实验证据

1. **通用两分支网络加 ADB+Bag 能提升 mIoU，但速度代价不小。** BiSeNet(Res18) 从 75.4 到 76.7 mIoU，同时作者平台 FPS 从 63.2 降到 52.1；DDRNet-23 从 79.5 到 80.0，FPS 从 51.4 降到 39.2。[全文证据：PDF p.6，Table 1]
2. **Pag+Bag 协作优于简单组合。** PIDNet-L 的相关消融最高为 80.9 mIoU；作者强调 detail 保护应贯穿 lateral connection 与 final fusion，而不是只在最后补一次。[全文证据：PDF p.6，Table 2；p.7，Fig.8–9]
3. **边界监督在该架构和数据上有贡献。** PIDNet-L 从仅 `l0` 的 78.8 mIoU，加入 boundary loss `l1` 后为 79.9，再加入 BAS loss `l3` 为 80.5，OHEM 后为 80.9。[全文证据：PDF p.7，Table 4]
4. **论文速度不是手机速度。** PIDNet-S 在 2048×1024、RTX 3090、batch 1、融合 BN 后报告 93.2 FPS、47.6 GFLOPs、7.6M 参数；实验环境是 PyTorch 1.8/CUDA 11.2/cuDNN 8.0。[全文证据：PDF p.6，Implementation/Inference；p.8，Table 6] 这些数字只能说明论文平台上的相对权衡，不能推断 Android TFLite 延迟。

### 3.5 局限、负迁移风险与停止点

- **精确边界标签依赖。** 作者在结论中明确说，由于边界预测用于平衡 detail/context，精确的边界标注更有利，但代价高。[全文证据：PDF p.8，Conclusion]
- **数据和视角差异。** 主要证据来自车载 Cityscapes/CamVid；BlindAssist 是手机/可穿戴第一视角且可能有倾斜、模糊、遮挡。论文不能证明对 `step_curb` 有效。[全文证据 + 工程推断]
- **完整 PIDNet 不是最小模型。** PIDNet-S 的 7.6M 参数和 47.6 GFLOPs高于当前小型 head 的量级；直接替换骨干会破坏因果归因并扩大端侧风险。[全文证据：PDF p.8，Table 6；工程推断]
- **停止条件。** 如果 Light-Bag 只提高 geometric boundary 指标，却不提高最差 seed 的 `boundary_step_curb`/selection，或同机 P95 明显恶化，应停止扩展 Pag/PAPPM，而不是继续堆完整 PIDNet。[工程推断]

---

## 4. 证据卡 MSEED：Mobile-Seed

### 4.1 元数据和发表状态边界

- 题名：*Mobile-Seed: Joint Semantic Segmentation and Boundary Detection for Mobile Robots*
- 作者：Youqi Liao 等
- 本地来源：arXiv:2311.12651v3
- 本地文件：`.downloads/papers/2026-07-frontier-upgrade/02_Mobile_Seed_arXiv_2023.pdf`
- 页数：8；SHA256：`83c24ec89b7ad52608a6cdd9b0cab5d5c8ddc09f039ae72760feb09812cb4c31`

**发表状态处理：** PDF p.1 带有 *IEEE Robotics and Automation Letters* 页眉，并写有 2024-02-20 accepted/recommended for publication，但同页 DOI 仍是占位描述，本地可追溯来源和 inventory 均记录为 arXiv 预印本。本笔记因此按**预印本证据**使用，不把它表述为已核验的正式期刊发表。

### 4.2 论文解决的问题

**[全文证据，PDF p.1–3]** Mobile-Seed 认为轻量语义分割通常牺牲局部细节或采用过简 decoder，而语义边界对 SLAM、标定、操作等机器人任务有用。它采用共享 stem、语义流、从多阶段语义特征抽取的边界流，以及 AFD 动态融合。其 boundary 是“语义不连续区域”，论文明确区别于亮度、纹理或光照变化产生的普通 edge。[PDF p.2，Fig.2 与正文]

### 4.3 结构与损失机制

| 机制 | 全文定位 | 可核验内容 | 对当前项目的含义 |
|---|---|---|---|
| 共享 stem + 双流 encoder | PDF p.3，Fig.3，Eq.1 | 共享 stem 为两个 MobileNetV2 block；语义流使用 AFFormer-T；边界流读取早期 stem 和语义流各阶段特征，经 3×3 conv + GroupNorm + ReLU，再上采样 concat | “多尺度 D 分支 + GN”比复制第二个 backbone 更适合当前小 batch；但 AFFormer 不应替换当前骨干 |
| AFD | PDF p.4，Fig.5，Eq.2–6 | 分别 GAP 语义/边界特征，经 MLP 和分组 affinity 得到通道权重，融合为 `(1+w_s)F_s+(1+w_b)F_b` | AFD 是**输入相关的通道融合**，不是 PIDNet 那种逐像素 boundary gate；对细路沿是否更好需实测 |
| 分类损失 | PDF p.4，Eq.7 | 同时监督辅助语义、融合语义和 binary boundary | 多头监督会把不同梯度推向共享低层，可能导致小数据冲突 |
| 双向一致性 | PDF p.4–5，Eq.8–13 | semantic-to-boundary 用局部模板从语义预测构造伪语义边界；boundary-to-semantic 在 GT 或高置信边界处强调语义 CE，阈值为 0.8；总损失 `Lcls+Lreg` | 可作为“直接边界监督伤害语义任务”时的条件修复，不应在第一步无证据地全量加入 |

### 4.4 关键实验证据与反例

1. **相对 AFFormer-T 的增益。** Cityscapes val 上 mIoU 从 76.2 提高到 78.4；3 px 语义边界 mean F-score 从 68.0 提高到 72.2。论文报告 Mobile-Seed 为 2.4M 参数、31.6G FLOPs、23.9 FPS，而 AFFormer-T 为 2.2M、23.6G、27.8 FPS。[全文证据：PDF p.6，Table I；p.7，Table II–III]
2. **跨数据集仍是同类公共数据，不是 SANPO 域泛化证明。** CamVid mIoU/BIoU 从 71.6/41.2 到 73.4/45.2；PASCAL Context 59 类从 45.7/20.7 到 47.2/22.1，60 类从 41.4/14.9 到 43.0/16.2。[全文证据：PDF p.7，Table IV]
3. **最关键反例：直接边界监督会伤害语义分割。** 仅语义流为 76.2/41.3（mIoU/BIoU）；加入边界流但不加 boundary loss 为 77.7/42.1；再直接加入 boundary loss 后下降到 76.9/41.6；加入双任务一致性后恢复并提高到 78.4/43.3。[全文证据：PDF p.6–7，Table V] 这证明“有 boundary 标签就直接加 BCE”不是安全默认值。
4. **AFD 的增益有限但成本可控。** ADD 为 77.7 mIoU、0.96G、24.3 FPS；CAT 为 78.0、1.91G、23.6 FPS；AFD 为 78.4、0.96G、23.9 FPS。[全文证据：PDF p.8，Table VI] 该 0.96G 是论文 decoder 消融口径，不能等同当前 Android 图上的 MACs 或延迟。

### 4.5 局限与负迁移风险

- **证据等级。** 正式期刊状态未在当前可追溯元数据中闭合，按预印本降级；不能写“RA-L 已正式发表”。
- **AFD 不是空间边界门。** 它通过 GAP 得到通道权重，可能无法针对同一帧中局部台阶/路沿位置选择 detail。把它称为 per-pixel boundary fusion 会误读论文。[全文证据：PDF p.4，Eq.2–6]
- **训练规模不匹配。** Cityscapes 使用 160K iterations、batch 8、1024×1024，并以 ImageNet 预训练 AFFormer-T 为语义流；当前 real-only 小数据和 384×384、batch 6 的方差问题更严重。[全文证据：PDF p.5，Implementation]
- **速度证据不适用于手机。** 论文 FPS 在 RTX 2080 Ti、原图分辨率、batch 1 测得；训练在 RTX 4090。[全文证据：PDF p.5] 只能在最终 TFLite、目标手机、同一 runner 上判断端侧成本。
- **边界标签噪声可能放大冲突。** 当前 `boundary_step_curb` taxonomy 与几何 boundary 不同，且小数据中机器传播标签可能不够精确；双任务冲突风险可能高于 Cityscapes。[工程推断]

---

## 5. 三种结构的机制对照

| 维度 | 当前 MobileNetV3 + LR-ASPP | PIDNet | Mobile-Seed | 建议 Mobile-PID-lite |
|---|---|---|---|---|
| 骨干 | MobileNetV3Small，默认 OS32 | 自建残差三分支 | AFFormer-T 语义流 + 轻边界流 | 保留 MobileNetV3Small |
| detail | 默认 OS8，1×1 投影 32ch | P 分支持续高分辨率 | 共享 stem/多阶段浅层进入 B | 先验证 OS4；固定轻量投影 |
| semantic/context | 默认 OS32，高层 96ch，global sigmoid gate | 深 I 分支 + PAPPM/DAPPM | 语义流高层特征 | 先验证 OS16；保留现有 context gate |
| boundary | 无独立几何边界输出 | D 分支，边界 BCE | 多尺度 B 分支，binary/semantic boundary | 多尺度 D-lite，GroupNorm；独立 `geometric_boundary` |
| fusion | high/low concat，1×1 logits | Pag 中途交互 + Bag/Light-Bag 空间融合 | AFD 输入相关通道融合 | 第一版只用 Light-Bag 空间 gate；AFD 暂不叠加 |
| 冲突处理 | 无双任务冲突 | 边界 BCE + BAS loss | 双向一致性，且有直接监督变差的消融 | 先测无边界监督，再测小权重监督；观察到冲突才加 consistency |
| 当前证据缺口 | 修正图尚未在本任务包中完成同矩阵重跑 | 无手机和 SANPO 证据 | 预印本、无 SANPO/手机证据 | 需三 seed、INT8、真机事件门逐层证伪 |

**选择理由 [工程推断]：** PIDNet 的 Light-Bag 是逐像素空间 gate，更贴近细局部风险；Mobile-Seed 的 GN 和“直接边界监督冲突”消融更贴近当前小 batch/高 seed 方差。第一版组合这两点即可，不应再同时引入 Pag、PAPPM、AFD 或新骨干。

---

## 6. Mobile-PID-lite 最小可证伪实验

### 6.1 实验前置门 E1-0：先闭合修正基线

在任何新分支之前，使用当前 `lraspp_sigmoid_no_pooled_bn_v1`：

- 保持 384×384、alpha 1.0、decoder 96、batch 6、head-only 100 optimizer steps、每 25 step 评估；
- 原样复跑 P0 的五组 model/sampler seed OFAT；
- 核对新报告中的 architecture revision、detail/semantic stride、blind access 字段；
- 若固定 sampler 的 model-seed selection 范围仍接近 `0.2685`，才把“修正 gate 仍不足”作为进入结构实验的证据。

这一步只回答 pooled gate/数值稳定性，不引入 OS4、OS16 或边界分支。[本地协议约束]

### 6.2 尺度门 E1-A：只改高低分辨率端点

| 变体 | Detail | Semantic | 其他变化 |
|---|---:|---:|---|
| A0 | OS8 | OS32 | 修正后的权威基线 |
| A1 | OS4 | OS32 | 只验证更浅 detail 是否保留细窄结构 |
| A2 | 由 A0/A1 中胜者固定 | OS16 | 只验证上下文分辨率变化；保持 dilation 合同与同权重迁移 |

使用同数据、step、batch、loss、seed、评估频率。禁止同时改 512 分辨率、decoder 160 或 backbone。只有 A1/A2 在**三 seed 最差值和 `step_curb`**上有一致改善，才进入 D 分支。单个幸运 seed 不算通过。

### 6.3 架构门 E1-B：最小 D-lite + Light-Bag

建议的第一版头部合同如下：

```text
MobileNetV3 shared backbone
  P/detail:  selected OS4 or OS8 -> retain current low projector/BN/ReLU, C=32
  I/semantic:selected OS16 or OS32 -> retain current high projector/BN/ReLU
              -> current context sigmoid gate -> upsample -> 1x1 Conv, C=32
  D-lite:     P feature + one intermediate feature + I feature
              each 3x3 Conv -> GroupNorm -> ReLU, C=16
              upsample + concat -> 1x1 Conv -> geometric_boundary_logit
  sigma:      sigmoid(geometric_boundary_logit)
  Light-Bag:  Conv1x1((1-sigma)*I + P) + Conv1x1(sigma*P + I)
  main:       1x1 Conv -> 4-class logits -> bilinear upsample
```

这里的 GN 只用于新增 D-lite，P/I 继续使用当前 head 的 BN；原因是 Mobile-Seed 的小边界分支使用 GN，而当前 batch 较小。**不建议第一轮全局替换现有 head BN**，否则又增加一个变量。[全文证据：Mobile-Seed PDF p.3；工程推断]

### 6.4 条件消融：把“分支、监督、冲突修复”分开

| 变体 | D-lite/Light-Bag | 几何边界直接损失 | 一致性 | 要回答的问题 |
|---|---|---|---|---|
| B0 | 无 | 无 | 无 | 尺度胜出基线 |
| B1 | 有 | 无 | 无 | D 是否仅通过主语义损失就学到有用 gate |
| B2 | 有 | 有 | 无 | 直接边界监督是帮助还是产生任务冲突 |
| B3（条件触发） | 有 | 有 | 有 | 仅当 B2 几何边界变好但主 mIoU/类 1 IoU变差时，能否像 Mobile-Seed 一样修复冲突 |

建议 B2 的辅助项先固定为：

```text
L_total = 0.90 * L_semantic_current
        + 0.10 * (weighted_BCE_geometric + soft_Dice_geometric) / 2
```

正类权重继续使用 cap `4.0`，不移植 PIDNet 的 `λ1=20`。0.10 是**预注册的工程起点，不是论文最优值**；第一轮不做大范围 loss sweep。B3 的 consistency 只在 B2 符合触发条件时增加，并保持其余配置不变。[工程推断]

### 6.5 几何边界标签与度量合同

1. 从四类 GT 的相邻类别变化生成 `geometric_boundary`，而不是把 class 1 直接当几何边界；生成脚本、半径和哈希进入 manifest。
2. 对 HUMAN 与 MACHINE 标签分别报告边界宽度、连通度和空 mask 比例；若 MACHINE 边界明显更碎，B2 首轮只在 HUMAN 像素上计算几何辅助损失，MACHINE 只保留主语义损失。[工程推断]
3. 同时报告 1 px 与 3 px 容忍度的 boundary F-score/BIoU，并保留 class 1 IoU。Mobile-Seed 的 3/5/9/12 px 阈值对应 1024×2048 Cityscapes，不能原样搬到 384 输入。[全文证据：Mobile-Seed PDF p.5；工程推断]
4. 增加三个定性错误桶：真实 `step_curb` 漏边界、平行路沿负例被过度增强、阴影/纹理 edge 被误当 semantic boundary。

### 6.6 评价与晋级顺序

结构实验的首要排序：

1. worst model seed selection；
2. worst model seed `boundary_step_curb` IoU；
3. macro-session 与 worst-session/scene；
4. unknown precision、recall、IoU 与 covered accuracy；
5. geometric boundary F-score/BIoU；
6. 参数、峰值内存、TFLite/INT8 保真与目标手机 P95；
7. 连续场景 event recall、false alerts/min、post-event clearance 和 repeated alerts。

前五项是离线候选证据；第六、七项仍按既有 `offline_training_quality -> int8_fidelity -> device_event` 链路执行。任何离线改善都不授权替换 App 默认模型。

---

## 7. 成功、失败与停止条件

### 7.1 进入下一阶段的必要条件

- 使用相同预注册 seeds；不挑最好 checkpoint 代替整体报告。
- 相比同尺度 B0，B1/B2/B3 至少同时提高 worst-seed selection 与 worst-seed `boundary_step_curb` IoU；macro-session 和 unknown 门不得通过牺牲覆盖率伪改善。
- geometric boundary 改善必须能在 `step_curb` 定性桶中看到，而不是只在大物体轮廓上提高。
- 模型进入端侧前必须重新导出并通过跨后端、INT8 fidelity 和同机连续事件门。

### 7.2 立即停止或回退

1. **概念失配停止：** geometric boundary 上升但 class 1 IoU、step_curb 最差场景或事件误提醒变差，回退 D/fusion；不再以“边界更清楚”辩护。
2. **监督冲突停止：** B2 相比 B1 主 mIoU/类 1 IoU下降；只允许触发一次 B3 consistency 对照。B3 仍不能恢复则删除直接边界监督，保留 B1 或完全回退。
3. **seed 停止：** 平均值提高但 worst seed 未提高，或 model-seed 范围没有实质收窄，不晋级；优先查初始化、归一化和梯度尺度，不扩展 backbone。
4. **域停止：** 只在 machine-heavy session 改善、HUMAN session 不改善，或反之，先做标签质量分层；不把结果解释为通用泛化。
5. **端侧停止：** 论文 GPU FPS 不参与判断。若 TFLite/INT8 同机 P95、内存或 fidelity 不过既有门，回退到 B0/B1 的更小分支。
6. **事件停止：** class 1 IoU 提高但平行路沿负例、登阶后 repeated alerts 或 false alerts/min 恶化，不进入生产路径。
7. **证据治理停止：** trainer 读取 blind、配置/权重/manifest 哈希不匹配、只报告单 seed，整轮作废。

---

## 8. 证据—论点映射

| Source ID | 全文事实 | 可支持论点 | 报告引用位置 | 风险 |
|---|---|---|---|---|
| PID-p4 | Fig.4 的 P/I/D 三分支与训练损失；D 预测边界 | 显式边界分支可作为 detail/context 融合控制信号 | 4.1 Mobile-PID 架构 | Cityscapes 边界与风险类不等价 |
| PID-p5 | Eq.9–11 的 Bag/Light-Bag 用逐像素 `sigmoid(D)` 融合 P/I | 空间 gate 比无条件 concat 更贴近细局部风险 | 推荐架构融合层 | 未证明手机收益 |
| PID-p6-T1 | 给 BiSeNet/DDRNet 加 ADB+Bag 提升 mIoU同时降低作者平台 FPS | 边界分支有效但不是免费午餐 | 成本与停止条件 | GPU FPS不可外推 |
| PID-p7-T4 | boundary loss 与 BAS loss 逐步提高 PIDNet-L mIoU | 边界监督可能改善语义分割 | 条件辅助损失 | 权重和标注质量不可移植 |
| PID-p8 | 作者明确依赖较精确边界标注 | 标签质量必须先审计，机器标签不能默认可信 | HUMAN/MACHINE 分层 | 只是一句局限，无噪声实验 |
| MSEED-p3 | 多尺度边界流使用 3×3 conv + GN + ReLU | 新增 D-lite 可用 GN，避免小 batch 新 BN | 最小实现 | 预印本、架构不同 |
| MSEED-p4 | AFD 是 GAP 后的通道 affinity 和 residual fusion | AFD 不等于逐像素边界 gate | 机制对照 | 不能由作者结果判断局部台阶 |
| MSEED-p7-T5 | 直接 boundary loss 使 77.7 mIoU降到76.9，consistency恢复到78.4 | 双任务监督冲突必须做分支/监督/一致性拆分 | B1–B3 消融 | 单数据集/单训练设置 |
| MSEED-p8-T6 | AFD较 ADD 提高0.7 mIoU，decoder FLOPs相同，FPS略低 | 动态融合值得作为后续候选，不宜和 Light-Bag 同时首测 | 后续路线 | 论文 GPU 口径不适配手机 |
| LOCAL-P0 | model-seed selection 范围0.2685，sampler范围0.0112；最差均为step_curb | 首要目标是 worst seed 和最弱场景，不是最佳 seed mIoU | 执行摘要、实验门 | OFAT未估计交互 |
| LOCAL-CODE | 当前已修正 pooled gate；默认OS8/OS32，无显式D | 先复跑修正基线，再做尺度和D分支 | E1-0/E1-A | 代码状态不等于结果证据 |

---

## 9. 阅读与自检记录

### 9.1 已读页数

- PIDNet：11/11 页；正文与方法/实验表格 p.1–8 全读，参考文献 p.9–11 核对。
- Mobile-Seed：8/8 页；全文、公式、Table I–VI 与结论全读。
- 合计：19/19 PDF 页。
- 关键页面已通过 Poppler 渲染并视觉核对：PIDNet PDF p.4、5、7；Mobile-Seed PDF p.3、4、7、8。

### 9.2 自检清单

- [x] PIDNet 页数、标题、正式 CVPR 来源和 SHA256 与 inventory 对齐。
- [x] Mobile-Seed 页数与 SHA256 已由 `pdfinfo`、`Get-FileHash` 对照 inventory 验证。
- [x] 所有数值均来自明确的 PDF 表格或本地审计文档；没有用摘要外推未报告指标。
- [x] 已区分 `boundary_step_curb` 语义类与 geometric boundary。
- [x] 未用 RTX 3090/2080 Ti FPS 推断手机延迟。
- [x] Mobile-Seed 按预印本降级，没有表述为已核验正式期刊发表。
- [x] 未建议一次替换整个骨干；实验按 gate修正、尺度、分支、监督、一致性逐步拆分。
- [x] 已给出失败/停止条件，并保持 blind、INT8 和 device event 晋级边界。
