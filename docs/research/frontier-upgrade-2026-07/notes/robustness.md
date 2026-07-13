# 鲁棒性、标签噪声与 abstention 精读笔记

> 范围：MRFP、UPC、SWSEG、ValUES、Kandinsky。本文只回答“这些论文的证据能支持什么、在 BlindAssist 中怎样做最小可证伪迁移”。
>
> 证据标签：**[论文报告]** 表示论文在其数据和协议下的结果；**[仓库事实]** 表示当前仓库、canonical 数据或报告中已核对的事实；**[迁移推断]** 表示尚未在 BlindAssist 上验证的设计判断。三者不得互换。

## 1. 结论先行

这五篇论文不能组成一个“全部叠加”的大模型。对当前项目最有建设性的组合是分成三条互不混淆的链：

1. **先恢复标签质量可观测性，再讨论半监督去噪。** 当前 P0 seed baseline 使用的 real-only r3 canonical，其 `label_authority` 全部是 `source_ground_truth`，但 SANPO 原始序列仍区分 `HUMAN_ANNOTATED` 与 `MACHINE_ANNOTATED`。本地逐帧 raw join 得到 train 为 38 HUMAN / 362 MACHINE，dev 为 39 HUMAN / 161 MACHINE；canonical 目前没有透传该字段。没有这个字段，UPC/SWSEG 的“可靠标注/弱标注”角色无法被审计，直接使用会把来源 GT 与人工 GT 错当成同一概念。[仓库事实]
2. **MRFP 是域泛化训练增强，不是当前 seed 初始化方差的直接修复。** 它在训练期扰动高/低分辨率特征、推理期移除；但论文验证的是 GTA/Synthia 到多个真实道路域，当前主 baseline 是 real-only SANPO。只有在 P1 head 已稳定后，重新引入 synthetic/procedural 预训练或把 session 视作未见域时，MRFP 才值得进入受控消融。[论文报告 + 迁移推断]
3. **ValUES 最应该立即吸收的是评价框架。** 把“预测模型 C1、uncertainty measure C2、聚合 C3、下游任务”拆开；以风险-覆盖、失败检测、OoD、校准分别评价。当前 `unknown_nonwalkable` 是语义类兼显式 abstain 合同，不能被 softmax entropy 或 conformal set 偷换。[论文报告 + 仓库事实]
4. **Kandinsky 只能在独立 calibration split 上做后处理。** 它改善的是预测集合的覆盖误差，不改善 mIoU，也不自动改善事件安全。连续帧不满足朴素 i.i.d./exchangeability 假设；若没有独立、session 隔离的 calibration 数据，只能称“经验覆盖审计”，不能声称 conformal guarantee。[论文报告 + 迁移推断]
5. **推荐顺序：P0 质量字段 → P1 head 稳定 → P2 MRFP-HRFP 消融 → P3 UPC-lite → P4 SWSEG → P5 ValUES/Kandinsky 后处理。** 任一步只要最差 seed、最差 session、`step_curb`、unknown 门或事件门变差，就停止该分支，不用平均 mIoU 掩盖失败。

## 2. 当前仓库事实与论文迁移边界

### 2.1 数据和来源门

- **[仓库事实]** 仓库中有两个容易混淆但行数不同的闭环产物：
  - `sanpo-v3-canonical-evidence-v4-20260713`：`training_manifest.jsonl` 300 行（train 200、dev 100），`reviewed-source-manifest.jsonl` 420 行（另含 blind 120）；训练视图中 200 行是 `source_ground_truth`、100 行是 `procedural_ground_truth`。
  - P0 seed 审计与最新训练协议使用的 `sanpo-v4-real-canonical-r3-20260713`：`training_manifest.jsonl` 600 行（train 400、session-held-out dev 200），`reviewed-source-manifest.jsonl` 720 行（另含 2 个 official-test blind session、120 帧）；本笔记后续的当前 baseline 均指这个 real-only r3，不把 evidence-v4 的 300 行写成 600 行。
- **[仓库事实]** 训练/dev 共 12 个互斥 SANPO-Real session，每个 50 帧；四个 scene bucket 在 train 各 2 个 session、dev 各 1 个 session。blind 不得用于训练、checkpoint、阈值或 calibration。
- **[仓库事实]** canonical 中 600 帧均记为 `source_ground_truth`；这表示标签来自来源数据且有哈希闭环，不等于每帧都是人工标注。
- **[仓库事实]** 对 real-only r3 的 600 行，通过 `training_manifest.jsonl` 的 `image_sha256` 回连 `test-artifacts.local/datasets/**/manifest.draft.jsonl` 中的 `source.sha256`，再统计原始 `source_annotation_quality`，600/600 命中，得到 train `38 HUMAN + 362 MACHINE`、dev `39 HUMAN + 161 MACHINE`。该统计不适用于上面的 300 行 evidence-v4；现有 real-only r3 canonical schema 未保存该字段，因此 trainer/quality gate 目前无法直接按质量分层。
- **[迁移推断]** 在 schema、总门和报告中透传 `source_annotation_quality` 之前，不启动 UPC/SWSEG 正式比较；否则无法证明可靠 donor 只来自 HUMAN，也无法输出 HUMAN/MACHINE 分项。

### 2.2 当前模型与门禁

- **[仓库事实]** 384×384 MobileNetV3Small + LR-ASPP 的五组 head-only OFAT 中，固定 sampler 改 model seed 的 selection 范围是 `0.1739–0.4424`；固定 model seed 改 sampler 的范围是 `0.4312–0.4424`。五组最弱场景均为 `step_curb`。
- **[仓库事实]** 当前协议按 optimizer step 比较、三 seed 报告，checkpoint 使用 mIoU 与 boundary IoU 的调和评分；采样按 session 平衡并有 rare-class crop。
- **[仓库事实]** 候选必须依次通过 `offline_training_quality → int8_fidelity → device_event`。`unknown_nonwalkable` 报告 precision/recall/IoU、abstain rate、known coverage 和 covered accuracy；任何单门变绿都不授权替换 App。
- **[迁移推断]** 论文中的平均 mIoU 或单模型覆盖率只能作为机制先验；BlindAssist 的主判据仍是 worst-seed、worst-session/scene、boundary/unknown、独立 blind 事件门和端侧 P95。

## 3. 逐篇证据卡

## 3.1 MRFP：Multi-Resolution Feature Perturbation

**来源**：Udupa et al., CVPR 2024，全文 11 页，本地 ID `MRFP`。

### 论文报告了什么

- **[论文报告，p.1–2]** MRFP 针对 sim-to-real single-domain generalization，假设低分辨率/低频特征偏向风格，高分辨率/高频特征偏向域特定细纹理；仅扰动风格不足以阻止模型过拟合源域细节。
- **[论文报告，Fig.2 / p.4–5]** 方法由两类训练期模块构成：
  - HRFP：把 backbone stage-0 浅层特征送入随机初始化的 overcomplete autoencoder，逐层放大空间分辨率（约 1.2 倍、最高 2 倍）以收缩 receptive field，再把随机高频扰动加回浅层；
  - NP+：按 batch channel statistics 扰动低分辨率特征的均值/方差；
  - MRFP+ 还把 HRFP+ 输出加到 decoder 倒数第二层。三个模块以独立 `p=0.5` 随机启用，推理时全部移除。
- **[论文报告，Table 1 / p.6]** ResNet-50 DeepLabv3+，GTA→BDD/Cityscapes/Mapillary/Synthia：baseline 平均 31.21，MRFP 37.09，MRFP+ 39.27。这里的 `+7.56` 是相对 baseline 的绝对 mIoU 点数，不是 BlindAssist 预期增益。
- **[论文报告，Table 2 / p.6]** adverse weather 平均：MRFP+ 25.87，对比 IBN 17.61、ISW 20.37；说明组合扰动在论文的雾/雨域上有效。
- **[论文报告，Table 3 / p.6]** MobileNetV2 backbone：baseline 26.20，MRFP 31.10，说明机制并非只在 ResNet-50 有效，但没有验证 MobileNetV3Small/LR-ASPP。
- **[论文报告，Table 5 / p.7–8]** 单项 HRFP+ 38.26、NP+ 35.58、MRFP+ 39.27；HRFP+ 比 NP+ 更强，二者组合最好。SCFP 37.48 低于 HRFP+，作者将其解释为 overcomplete/decreasing-RF 有贡献。
- **[论文报告，p.6]** 所有主要结果是三次独立运行均值，训练为 40k iteration、batch 16、SGD、ImageNet 预训练；论文没有报告最差 seed、session-held-out、unknown、boundary 细类或事件安全。

### 可迁移部分

- **[迁移推断]** 先试 `HRFP-only`，后试 `HRFP + NP+`。理由不是“论文 HRFP 更强”这么简单，而是当前 batch 6，NP+ 的 batch statistic 方差可能把现有小 batch 不稳定放大；HRFP 更容易作为纯训练期扰动隔离。
- **[迁移推断]** 为避免把 augmentation randomness 混进 model seed，新增独立 `perturbation_seed`，三 model seed 共用同一扰动序列；MRFP 随机权重固定且不参与学习。
- **[迁移推断]** 作用点限制在修正后的 P1 head/backbone 浅层；不得同时改输入尺寸、decoder channels、loss 权重或 sampler。
- **[迁移推断]** 若重新引入 SANPO-Synthetic/procedural 数据，采用“synthetic pretrain + real train/dev 评价”的单向协议；real dev 从不参与 MRFP 参数或随机策略选择。

### 不适用与风险

- **[论文局限]** 证据来自城市道路 19 类与较大 backbone，不含薄路沿/台阶的专门边界指标。
- **[迁移风险]** 当前最大问题是 head 初始化高方差。再加入随机 convolution/BN 可能改善泛化，也可能增加方差；因此它不是 P1 前置修复。
- **[迁移风险]** `step_curb` 依赖细粒度几何；过强地扰动高频可能把真正需要的细边界也当域特征抹去。

### 最小实验与失败条件

1. 固定当前 384/alpha1.0/decoder96、sampler seed、step budget；完成 P1 head 后用相同三个 model seed 比较 baseline 与 HRFP-only。
2. 记录 selection、mIoU、boundary IoU/recall、unknown IoU、macro/worst-session、worst-scene、每 seed checkpoint 曲线；新增 perturbation activation count。
3. 只有 HRFP-only 在 **三 seed 均值与 worst-seed 同向改善** 且 `step_curb` 不下降，才追加 NP+。
4. **失败条件**：worst-seed selection 或 boundary recall 下降；seed range 扩大；`step_curb` 仍最弱且无改善；出现全 boundary/全 unknown 捷径；训练数值异常。任一触发即停止，不进入 INT8。

## 3.2 UPC：Uncertainty-aware Patch CutMix

**来源**：Fang et al., ICCV 2023，全文 11 页，本地 ID `UPC`。

### 论文报告了什么

- **[论文报告，p.1–2]** 伪标签噪声往往成片出现并集中在对象边缘；逐像素阈值/erase 会破坏形状上下文。UPC 先定位高不确定 patch，再用有标签图像的可靠 patch 替换。
- **[论文报告，Eq.1 / p.4]** 由 teacher softmax entropy 生成 pixel uncertainty map。
- **[论文报告，Eq.3–6 / p.4–5]** 将图像划为 `N×N` patch，按 patch 内 entropy 求和，选 top-k；对 image 与 mask 同步 CutMix，donor 必须来自 labeled pair。
- **[论文报告，Eq.7 / p.5]** RAS 为同一 unlabeled 样本生成多个 donor/不同 k 的版本，缓解 teacher uncertainty 不准和固定 k 不适合所有样本的问题。
- **[论文报告，Table 1–3 / p.6–7]** UPC 在 Pascal/Cityscapes 多种 labeled fraction 上改善 CPS/ST++/U2PL；例如 Cityscapes U2PL 的 1/16 从 70.30 到 75.31，但数据、backbone 与当前项目不同。
- **[论文报告，Table 4 / p.7]** 以 MSCOCO 为 OoD unlabeled 时也有增益；这说明方法可利用分布不同的未标数据，但不能证明可安全处理 SANPO MACHINE mask。
- **[论文报告，Table 6 / p.7]** 1/4 VOC blender + ST++：baseline 76.60，CutMix 77.82，Patch CutMix 78.73，加入 RAS 79.47。
- **[论文报告，Table 7–8 / p.8–9]** uncertainty selection 一贯优于 random，低于 oracle；默认 `N=4,k=5`，RAS 用 `k=2,3,5`。最优 k 随 N 变化，不能机械复制到 384×384 的薄边界任务。

### 对 HUMAN/MACHINE 的正确迁移

- **[仓库事实]** 当前 MACHINE 帧不是“无标签图片”：它们带有 SANPO 来源机器 mask；HUMAN/MACHINE 质量字段却未进入 canonical。
- **[迁移推断]** UPC 不能直接把 MACHINE mask 当论文中的可靠 labeled mask，也不能拿 blind 当 unlabeled pool。最接近论文语义的映射是：HUMAN=train reliable donor；MACHINE=train weak/unlabeled target；teacher 为仅用 train 构建的冻结/EMA 模型；dev、calibration、blind 均不做 donor。
- **[迁移推断]** 第一版 UPC-lite 不启用 RAS，只用单一中等替换率；先确认 donor 质量与边界没有被 CutMix 几何破坏，再引入多 k。
- **[迁移推断]** patch 不能只按固定 4×4。应以输入尺度和目标几何定义，例如先比较 6×6/8×8，同时设置“不得把 HUMAN donor 的 `unknown_nonwalkable` 大面积贴入目标”与 class-aware donor quota。

### 不适用与风险

- **[论文局限]** UPC 研究 teacher 自生成 pseudo-label 的噪声，不是外部机器标注噪声；两种噪声分布不等价。
- **[迁移风险]** 当前 train 只有 38 HUMAN 帧且来自连续 session，donor 多样性严重不足；重复贴块会让模型记住少量人工帧。
- **[迁移风险]** entropy 低不代表正确。高置信错误会保留，尤其是当前 `step_curb` 薄边界。
- **[迁移风险]** CutMix 的硬矩形边缘可能产生伪边界，恰好污染 `boundary_step_curb`。

### 最小实验与失败条件

1. **硬前置**：canonical、训练报告和 gate 中加入 `source_annotation_quality`，并验证 split/session 统计；HUMAN donor SHA256 可追踪。
2. 在 P1 稳定模型上，固定单 seed probe：baseline all-source-GT vs `HUMAN supervised + MACHINE pseudo + UPC-lite`；MACHINE 原 mask只用于离线质量分项，不进入 pseudo 分支损失。
3. 记录 HUMAN-dev 与 MACHINE-dev 的四类指标、边界带指标、pseudo-label coverage、被替换 patch 比例、donor 使用分布和高置信错误抽检。
4. probe 通过后才做三 seed；RAS 作为第二个单因素实验。
5. **失败条件**：HUMAN-dev boundary/unknown 任一下降；MACHINE 提升但 HUMAN 下降；donor 由少数帧垄断；伪边界增加；worst-seed 或 seed range 变差；pseudo coverage 通过阈值塌缩到极低或接近 100%。

## 3.3 SWSEG：Sliced-Wasserstein Alignment and Uniformity

**来源**：Lu et al., CVPR 2025，全文 11 页，本地 ID `SWSEG`；方法名在正文排版为 `SWSEG/SWSEG`，本笔记统一写 SWSEG。

### 论文报告了什么

- **[论文报告，p.1–3]** 现有半监督分割主要在 decoder 输出做 weak/strong consistency，忽略 encoder 表示。SWSEG 把 self-supervised 的 alignment 与 uniformity 引入特征层，意图既对齐同一图像的不同增强，又避免表示坍缩。
- **[论文报告，Fig.3 / p.4]** FixMatch 式双流：weak image 生成 pseudo-label，strong image 计算 unsupervised CE；共享 encoder 的 weak/strong feature 经训练期 MLP projection，再计算 SWD。
- **[论文报告，Eq.5–7 / p.5–6]** 用 Gaussian-projected 2-Wasserstein 的 closed form 近似 Monte-Carlo SWD，并增加 off-diagonal covariance decorrelation `Lreg`；总损失是 `Ls + λ1Lu + λ2Lswd + λ3Lreg`。MLP 推理时移除。
- **[论文报告，Algorithm 1 / p.6]** Gaussian-SWD 计算 weak/strong embedding 的均值差与中心化二阶矩差；论文声称较 Monte-Carlo SWD 更确定、更省内存。
- **[论文报告，Table 1–3 / p.6–7]** 在 ADE20K、Cityscapes、VOC 多个 labeled fraction 上超过 supervised-only；但对 contemporaneous SOTA 并非所有 split 都第一，例如 Cityscapes 1/16 76.8 低于 CorrMatch 77.3，1/4 79.5 低于 RankMatch 80.0。
- **[论文报告，Table 4 / p.7]** VOC ablation：MSE 70.2、CE 77.1、Monte-Carlo SWD 77.6、Gaussian-SWD 77.4、GSWD+Lreg 78.1、再加 MLP 79.0；完整组合最好，但单一 GSWD 不高于 MC-SWD。
- **[论文报告，Table 5 / p.7；Fig.4–6 / p.8]** 加 SWD 后 uniformity 指标和可视化更均匀，边界定性结果更干净；论文没有报告跨 seed 初始化方差、unknown abstain、session-held-out 或 event gate。

### 可迁移部分

- **[迁移推断]** SWSEG 更像“P4 表示正则”，不是 P1 初始化修复。只有 UPC/半监督角色已经明确、head 本身不再高方差，才评估它是否降低 weak/strong 表示漂移。
- **[迁移推断]** 对当前 batch 6，先实现不含 BatchNorm 的轻量 projection head（论文原实现含 Conv-BN-ReLU，但小 batch 可能不稳），并将 projection seed 与 model seed 一致记录；这属于迁移改造，必须与论文版区别标注。
- **[迁移推断]** strong augmentation 只允许语义安全变换；任何改变路沿几何、模糊薄边界或破坏 unknown 语义的增强均拒绝。
- **[迁移推断]** 在报告中增加 feature alignment、uniformity、off-diagonal covariance 和 per-class prototype separation，只作为诊断，不替代 segmentation/event 指标。

### 不适用与风险

- **[论文局限]** 理论近似依赖高维投影与弱相关条件；Lreg 正是为减相关加入。BlindAssist 小 batch、连续帧、高相关 session 未满足同样条件。
- **[迁移风险]** uniformity 变好不等价于 seed 稳定，更不等价于 `step_curb` 或事件安全改善。
- **[迁移风险]** MACHINE pseudo-label 高置信错误与 weak/strong consistency 可能形成 confirmation bias；SWD 会让错误表示更稳定。
- **[迁移风险]** 同时引入 Lu、Lswd、Lreg、MLP 有多个自由度，若不按单因素顺序会无法归因。

### 最小实验与失败条件

1. 固定已通过 UPC-lite 的数据角色与 teacher；先加 `Lswd`，再单独加 `Lreg`，最后加 MLP，每步只改一个因素。
2. 单 seed probe 监控显存、step time、梯度范数和表示诊断；通过后才按三个预注册 seed 跑完整预算。
3. 主判据：worst-seed selection、HUMAN-dev boundary/unknown、macro/worst-session；uniformity 只作解释变量。
4. **失败条件**：uniformity 改善但主门下降；seed range 扩大；small-batch NaN/梯度爆炸；额外训练成本过高；MACHINE 分项提升但 HUMAN 分项下降。

## 3.4 ValUES：系统化不确定性验证

**来源**：Kahl et al., ICLR 2024，全文 35 页，本地 ID `VALUES`。

### 论文报告了什么

- **[论文报告，Fig.1 / p.2–3]** 不确定性方法要拆成：C0 segmentation backbone、C1 prediction model、C2 uncertainty measure、C3 aggregation。很多论文只改 C2，却用很差的 C3 比 baseline，导致归因错误。
- **[论文报告，p.3–5]** 三个评价要求：R1 用显式 ambiguity/shift reference 验证 AU/EU；R2 消融 C0–C3；R3 在 OoD、active learning、failure detection、calibration、ambiguity modeling 多个下游任务评价。
- **[论文报告，Eq.1 / p.3]** Bayesian 分解中 predictive entropy = mutual information（EU）+ expected entropy（AU），但作者强调真实数据上未必能干净分离。
- **[论文报告，p.5；Appendix A / p.13–15]** 指标包括 OoD AUROC、failure detection 的 AURC/E-AURC、active learning 相对 random 的增益、Platt scaling 后 ACE，以及多标注 ambiguity 的 NCC/GED。
- **[论文报告，p.6；Appendix C / p.21]** 对比 deterministic softmax、MC-dropout、5-model ensemble、TTA、SSN；ensemble 用 5 个 seed，TTA 使用多种 label-preserving augmentation。
- **[论文报告，p.6；Appendix F / p.23–24]** 三种聚合：image sum、最大不确定 patch、阈值后均值。作者证明 image-sum 会与前景大小相关；聚合策略对结果影响很大，必须按数据/task benchmark。
- **[论文报告，Fig.2–3 / p.7–9]** AU/EU 分离在 toy 上有效，但真实数据常不完全成立；ensemble 跨任务最稳健，TTA 常是较轻替代；SSN 擅长 ambiguity modeling 但 failure detection 不稳。
- **[论文报告，Table 4 / p.26]** GTA→Cityscapes 中 ensemble/TTA 的 MI + image aggregation 对 OoD AUROC 分别为 0.90/0.94，说明 TTA 在该设置可接近 ensemble；这不是端侧实时开销结论。
- **[论文报告，Table 6 / p.35]** GTA/Cityscapes 的 aggregation 差异很大，例如 deterministic MSR 的 OoD AUROC：patch 0.3343、image 0.6999；证明不能把 pixel entropy 简单求和后直接触发提醒。

### 对 BlindAssist 的直接吸收

- **[迁移推断]** 立刻把不确定性报告拆成四层：
  1. C1：single seed、三-seed ensemble、受限 TTA；
  2. C2：1-MSR、predictive entropy、ensemble/TTA MI；
  3. C3：固定中心走廊 patch、scene-aware patch、threshold mean；
  4. downstream：像素失败、session失败、event failure、OoD session、risk-coverage。
- **[迁移推断]** 三 seed ensemble 首先是离线诊断器：若 MI 在 `step_curb` 和坏 seed/session 上升，说明初始化方差可被显式检测；它不等于修复，也不自动进入手机推理。
- **[迁移推断]** TTA 只允许与四类 mask 语义一致的变换，并单独报告乘数延迟；若用于端侧，只能在低频复核路径而非每帧主路径。
- **[迁移推断]** `unknown_nonwalkable`（语义 unknown）与 entropy/MI（模型不确定）必须分别报告。语义 unknown 的 truth 来自标注 taxonomy；模型不确定只能决定是否额外 abstain，不能改写 truth。
- **[迁移推断]** 将当前 unknown gate 扩展为 risk-coverage curve：按 uncertainty 从高到低拒绝像素/patch/session，观察 known covered accuracy、boundary recall、critical event miss 随 coverage 的变化；不得只报一个“最佳阈值”。

### 不适用与风险

- **[论文局限]** 主要真实数据是医疗 3D 分割，GTA/Cityscapes ambiguity 是人工类翻转；与自然路沿歧义、SANPO 标签质量不同。
- **[迁移风险]** ensemble/TTA 提高推理成本；其离线优势不能直接外推端侧 P95。
- **[迁移风险]** entropy 高可能来自真实多义、域外、边界或模型失败；在没有显式 reference 时给它贴 AU/EU 标签是不诚实的。
- **[迁移风险]** C3 若按 dev 搜索太多形状/阈值，会把 evaluation set 变成训练集。

### 最小实验与失败条件

1. 不改模型，先在当前三 seed dev 输出 per-pixel softmax 与 ensemble samples；实现 MSR/PE/MI、三种预注册聚合和 session risk-coverage。
2. 只用 dev 标签做**诊断**，不选择生产阈值；对四 scene/session、HUMAN/MACHINE 分层报告 AUROC/AURC/coverage 曲线。
3. 若模型冻结后需要固定 abstention 阈值，必须转入独立 calibration split，见第 5 节。
4. **失败条件**：aggregation 只在一个 session 有效；risk-coverage 中拒绝更多样本却不降低风险；boundary/critical event 被优先拒绝导致 recall 下降；TTA 延迟超过预算；阈值依赖 blind。

## 3.5 Kandinsky Conformal Prediction

**来源**：Brunekreef et al., CVPR 2024，全文 9 页，本地 ID `KAND`。

### 论文报告了什么

- **[论文报告，p.1–3]** inductive conformal prediction 用独立 calibration data 的 non-conformity score 分布构造预测集合。论文采用 `s(x)=1-p(y_true|x)`，目标是 prediction set 覆盖真类的概率至少 `1-α`。
- **[论文报告，p.1–2]** imagewise/marginal calibration 把所有像素当样本，数据多但可能掩盖局部严重失校；pixelwise/conditional calibration 每个像素坐标只能得到 N 个 calibration image 样本，低数据时噪声很大。
- **[论文报告，p.4]** Kandinsky 处在两者之间：先估计每个像素的 non-conformity curve，再按曲线相似性和空间先验聚类，汇总 cluster 内 score，最后每个像素使用其 cluster 的 quantile。
- **[论文报告，p.4–5]** 三种 cluster：k-means、带环形参数先验的遗传算法 GenAnn、Fourier Concentric Clustering。作者明确承认没有普适空间先验，聚类需要任务知识。
- **[论文报告，p.5–8]** 四组实验：COCO 20,000/100 calibration images、Medical Decathlon 77/27 patients。低数据下 Kandinsky 均优于 pixel/image baseline；20,000 COCO 高数据时 pixelwise 最好。
- **[论文报告，Table 1 / p.8]** COCO-100 mean coverage error：pixel 0.131、image 0.091、k-means 0.086、GenAnn 0.068、FCC-ann 0.060。Decathlon-27：pixel 0.249、image 0.175、k-means 0.168、GenAnn 0.152、FCC 0.165。
- **[论文报告，p.6]** COCO segmentation model只用 678 training images、2,869 test images；calibration 与 test 独立。论文评价 coverage error，不声称提高 segmentation accuracy。

### 对 BlindAssist 的正确迁移

- **[迁移推断]** 第一版不用 GenAnn/FCC 搜索复杂形状，而用预注册的固定几何区：底部中心近场、底部左右、地平线/远场、上部背景；原因是胸前相机存在空间先验，但当前独立 calibration session 太少，数据驱动 cluster 容易过拟合。
- **[迁移推断]** 对四类输出构造 class prediction set。建议的 abstain 映射仅为候选策略：`|Cα(pixel)|=1` 且唯一类为 known 时保留；集合为空、多类或包含互斥风险解释时转为额外 abstain。这个映射必须重新跑 unknown precision/recall/IoU 与 covered accuracy。
- **[迁移推断]** prediction-set coverage 与事件安全是两件事。最终需把 conformal abstain 后的 mask 送入现有事件层，重新测 event recall、critical miss、false alerts/min、clearance、repeat 和 P95。

### calibration 假设与风险

- **[论文条件]** inductive CP 的覆盖依赖 calibration/test 的交换性；论文按独立 images/patients 评估。
- **[仓库事实]** 当前每个 session 有 50/60 个连续帧，高度相关；把每帧或每像素当独立 calibration sample 会夸大有效样本量。
- **[迁移推断]** 最终形式化版本应按 session/block 做 split-conformal，或至少以 session 为抽样单位做 block bootstrap/coverage interval。若独立 session 数不足，只报告 empirical coverage，不写“保证 95%”。
- **[迁移风险]** 相机俯仰、身体运动和设备安装变化会破坏固定空间 cluster；必须按 session/OoD 姿态报告覆盖误差。
- **[迁移风险]** 选择 cluster、α、non-conformity score 与 abstain 规则都可能消耗 calibration 数据。任何一项看过 blind 后再改，覆盖和晋级证据失效。
- **[迁移风险]** conformal 覆盖保证真类在集合中，不保证集合小、不保证 unknown precision，也不保证零 critical miss。

### 最小实验与失败条件

1. **诊断版**：模型/聚合全部冻结后，对 4 个 dev session 做 leave-one-session-out；3 个 session 估 quantile，1 个 session 测 CE20、set size、abstain、四类/场景指标，轮换四次。明确标记为 empirical audit，不声称有限样本覆盖保证。
2. **正式版硬前置**：新建独立 calibration split，session 与 train/dev/blind 全隔离；四 scene 至少各有独立 session。若需要正式 session-level 95% quantile，4 个 session 远远不够，应扩展独立 session 数后再谈 guarantee。
3. 比较 imagewise、pixelwise、固定几何 Kandinsky；只在 calibration 内定 α/cluster，dev 仅做开发评价，blind 只在所有选择冻结后运行一次。
4. **失败条件**：coverage 在某 scene/session 明显失配；平均 set size 过大；unknown precision/recall 任一不过门；关键边界被过度 abstain；event recall/critical miss 变差；需要用 blind 调 α；P95 超预算。

## 4. 证据—论点映射

| Source | 可支持论点 | BlindAssist 使用位置 | 不能支持的外推 |
|---|---|---|---|
| MRFP p.4–8, Table 1–5 | 训练期高/低分辨率特征扰动能改善论文中的 sim-to-real DG，且推理零新增模块 | synthetic/procedural→real 或跨 session DG 单因素消融 | 不能保证修复当前 head seed 方差、薄边界或事件安全 |
| UPC p.4–9, Table 6–8 | patch-wise uncertainty-guided replacement 优于 random/CutMix，可靠 donor 可降低 pseudo noise | HUMAN donor + MACHINE weak target 的 UPC-lite | 不能把 MACHINE mask 自动当论文 pseudo-label，也不能证明高置信错误已被清除 |
| SWSEG p.4–8, Table 4–5 | 特征 alignment/uniformity 正则可改善论文半监督分割 | 已建立可信 weak/strong 流后的表示正则 | uniformity 不等于 seed 稳定、unknown 或事件安全 |
| ValUES p.2–9, App. A/F/H | C1/C2/C3 和 downstream task 必须分开评价；聚合可主导结果 | 不确定性审计、risk-coverage、session/event failure detection | 不能在无 reference 时把 entropy 直接命名为 AU/EU |
| Kandinsky p.1–8, Table 1 | 低 calibration data 下，空间聚类可降低 pixelwise coverage error | 冻结模型后的独立 calibration/abstain 后处理 | 不提高 mIoU，不保证连续帧交换性，不保证事件零漏报 |

## 5. calibration split 的硬合同

### 5.1 角色隔离

| Split | 允许 | 禁止 |
|---|---|---|
| train | 更新模型、teacher、MRFP/UPC/SWSEG 训练 | 生成最终校准 quantile 后再回训同一模型 |
| dev | checkpoint、结构/损失单因素比较、诊断曲线 | 固定最终 α/threshold 后仍反复改模型并复用同一 dev 声称无偏 |
| calibration（新增） | 在模型、聚合和 score 全冻结后，拟合 Platt/quantile/α/固定 cluster 参数 | 梯度训练、checkpoint、选择 backbone/head/loss、加入 donor pool |
| blind | 所有选择冻结后一次性离线/设备晋级 | 训练、校准、阈值、cluster、α、失败后回看再调 |

### 5.2 最小数据要求

- **[迁移推断]** 不从现有 4 个 dev session 切 calibration：每个 scene 只有 1 个 dev session，拆分会立刻破坏场景覆盖。
- **[迁移推断]** 最低工程方案是新增至少 4 个 official-train SANPO session，四个 scene 各 1 个，只做 calibration；但这只能支持经验校准，无法支撑强 session-level conformal 结论。
- **[迁移推断]** 正式 conformal 结论需要更多独立 session。连续 50 帧不能按 50 个独立样本计算置信度；报告必须同时给出 frame-level 数字与 session/block-level 区间。
- **[迁移推断]** calibration manifest 必须透传 `source_annotation_quality`。HUMAN/MACHINE 可分别检查覆盖误差，但最终 quantile 是否混合两类必须预注册；不能看到 blind 后再决定。

### 5.3 防止语义偷换

- semantic `unknown_nonwalkable`：数据 taxonomy 的第 4 类；仍按现有 precision/recall/IoU 门评价。
- model uncertainty：MSR/entropy/MI/conformal set size；是后处理信号，不是真值类。
- extra abstain：由冻结规则将不确定 known prediction 转成拒绝；必须单独报告“原生 unknown”与“额外 abstain”，并重新计算 known coverage、covered accuracy 与事件指标。

## 6. 组合顺序与最小实验矩阵

| 阶段 | 唯一改变量 | 预算 | 进入条件 | 退出/停止条件 |
|---|---|---|---|---|
| R0 质量透传 | canonical 保存 HUMAN/MACHINE + 分项报告 | 无训练 | 来源总门仍绿、600 行可回连 | 任一哈希/split/session 门变红 |
| R1 head baseline | P1 sigmoid gate/去 pooled-BN 后的稳定基线 | 既有 5-run OFAT + 3 seed | worst-seed 和 range 明显优于旧 head | 仍由 model seed 主导且无改善，则先修 head，不上论文模块 |
| R2 MRFP | HRFP-only；随后才 NP+ | 1 seed probe → 3 seed | head 稳定；perturbation seed 独立 | worst-seed、step_curb、boundary 任一下降 |
| R3 UPC-lite | HUMAN donor + MACHINE pseudo target | 1 seed → 3 seed | 质量字段可审计；teacher 冻结 | HUMAN-dev 下降、donor 垄断、伪边界增加 |
| R4 SWSEG | Lswd → Lreg → MLP 单因素 | 每步 1 seed probe，最终 3 seed | UPC/weak-strong 流通过 | uniformity 与主门背离、小 batch 数值不稳 |
| R5 ValUES | C1/C2/C3 不确定性审计 | 无再训练 | 模型冻结 | risk-coverage 不单调或聚合不跨 session |
| R6 Kandinsky | 固定几何 cluster + conformal abstain | calibration-only | 独立 calibration split | 覆盖失配、unknown/event/P95 任一不过门 |

最终只允许两类组合进入完整三段 gate：

- **轻量训练组合**：稳定 head +（HRFP 或 UPC，先不叠 SWSEG）；
- **后处理组合**：通过 offline/int8 的单模型 + 冻结 calibration 规则。

不要第一轮就 `MRFP + UPC + SWSEG + Kandinsky`。那会同时改变特征分布、伪标签、loss landscape 与输出决策，既无法解释提升，也无法定位退化。

## 7. 建议新增的报告字段

1. `source_annotation_quality_counts.{split,session,scene,class}`；
2. `metrics_by_annotation_quality.HUMAN_ANNOTATED/MACHINE_ANNOTATED`；
3. `model_uncertainty.{msr,pe,mi}` 与 `aggregation.{patch,image,threshold}`；
4. `risk_coverage.{pixel,session,event}`，同时保存 boundary/unknown/critical miss；
5. `conformal.{score,alpha,cluster_spec,calibration_manifest_sha256,coverage_error,set_size}`；
6. `abstain_breakdown.{semantic_unknown,extra_model_abstain}`；
7. `randomness.{model_seed,sampler_seed,perturbation_seed,teacher_sha256}`；
8. 所有 calibration/quality 报告固定写入 `blind_holdout_access=not_accessed`。

## 8. 页数、核验与自检

### 已读页数

| ID | 页数 | 阅读范围 | 核心定位 |
|---|---:|---|---|
| MRFP | 11 | 全文逐页（正文 1–8，参考文献 9–11） | Fig.2；Table 1–5 |
| UPC | 11 | 全文逐页（正文 1–9，参考文献 9–11） | Eq.1/3–7；Table 1–8 |
| SWSEG | 11 | 全文逐页（正文 1–8，参考文献 9–11） | Fig.3；Eq.5–7；Table 1–5 |
| ValUES | 35 | 全文逐页（正文 1–9，参考文献 10–12，附录 13–35） | Fig.1–3；Table 4–6；Appendix A/F/H |
| KAND | 9 | 全文逐页（正文 1–8，参考文献 8–9） | Eq.1–6；Table 1 |
| **合计** | **77** | 5 篇本地全文 | — |

### 自检

- [x] 所有核心论文 claim 都有页码、公式、图或表定位；未用摘要替代全文。
- [x] 明确区分论文报告、仓库事实与迁移推断。
- [x] 没有把 MRFP benchmark 增益外推成 BlindAssist 预期增益。
- [x] 没有把 pseudo-label 方法用于 blind；blind 不进入训练、donor、校准或阈值选择。
- [x] calibration split 不参与梯度、checkpoint 或架构选择；连续帧未冒充 i.i.d. 样本。
- [x] semantic unknown、model uncertainty 与 extra abstain 三者分开。
- [x] 所有实验都回到 worst-seed、worst-session/scene、boundary/unknown、event gate 和 P95，而非只看平均 mIoU。
- [x] MRFP、UPC、SWSEG 均给出最小实验与明确失败条件。
