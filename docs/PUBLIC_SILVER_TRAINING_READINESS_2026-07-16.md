# 公开视频银标训练就绪性诊断（2026-07-16）

## 结论

当前 r3 campaign 的 11 条非弃权 episode 已通过数据门禁、对象轨迹冻结特征门禁和 prototype/bootstrap 五组短跑稳定性门禁，可以继续做**暂定的风险轮廓/生命周期 head 原型**。这不是生产晋级：许可、来源哈希、逐帧文件哈希和暂定训练授权均已通过，但标签仍是 GPT/VLM provisional supervision，禁止用于阈值校准、blind 结论或默认模型替换。

### 2026-07-17 更新：r3 的总体门已不足以代表机制覆盖

本节更新优先于下方 r3 历史结论。r4/r5 证明：把不同风险机制合并计数会高估训练就绪性。当前应判定为“**总体线性信号存在，但机制级数据与 head 稳定性仍未闭合**”，不进入 SAM/ASAM，不保存或部署 head。

- r4 在 `gie8` 中增加一组静态家具 `passable → narrowing` 同源 pair，得到 12 条非弃权 episode、正负各 6、3 组 pair。冻结对象轨迹 balanced accuracy 为 `.75`，但五组短跑优化中位数仅 `.6667`、只有 `2/5` 达标；风险轮廓/潜生命周期 MIL 中位数 `.5833`，pair 排序中位数 `.6667`。
- r4 的冻结 DINO balanced accuracy 为 `.6667`；新增的显式相对深度走廊轮廓 probe 只有 `.50`。因此简单的全局语义表征或深度分位数/占用统计均不能解决静态通道净空。
- r5 独立加入 `Chcne` 第二组静态通道收窄同源 pair，得到 14 条非弃权 episode、正负各 7、4 组 pair。冻结对象轨迹 balanced accuracy 回升到 `.7857`；风险轮廓/潜生命周期 MIL 中位数回升到 `.7857`，pair 排序中位数 `.75`，表明补同机制来源有效。
- r5 的五组 prototype/bootstrap 短跑 balanced accuracy 为 `.6429/.7857/.6429/.7857/.9286`，只有 `3/5` 达标，稳定性门仍失败。source 留出时，`gie8` 与 `Chcne` 的静态 no-alert 均被判成 alert，说明 COCO 对象轨迹能识别“场景危险感”，却不能可靠判断静态夹道是否仍有足够净空。
- 新增机制级数据门：`dynamic_agent_approach` 与 `static_corridor_narrowing` 必须各有至少 3 个独立 matched source。r5 当前两类都只有 2 个，因此 `mechanism_coverage_gate=false`。下一批应各补至少 1 个高质量同源 pair，实际建议补到 4–5 个以避免单来源影响。
- r4/r5 与另一个独立模型方向完全隔离；不使用后者的数据、指标、权重、目录或晋级结论，只保留方法层面的风险提醒。

### 2026-07-17 r6 更新：低置信度 pair 可用于诊断，但不能计入覆盖

主线 r6 从 `JtMY` 同一官方 SANPO train session 的既有公开 RGB 中隔离出 226–300 的开放接近段，以及 339–429 的马车近距离横穿段。它没有修改 r5，也没有读取独立模型方向；新时间线和子实验分别写入 `artifacts.local/evidence/datasets/sanpo-weak-jtmy-counterfactual-20260717` 与 `artifacts.local/evidence/public-video-provisional-training-r6-20260717`。

- 原正例置信度 `.63` 被原样保留。机制覆盖门现要求 matched pair 中每条 episode 的置信度都至少 `.65`；所以动态机制 `all_matched_pair_count=3`，但合格 `matched_pair_count=2`，`JtMY` 明确列入 `excluded_low_confidence_pair_ids`。静态机制仍为 2 对/2 来源。
- 旧总体数据门仍会因 5 个总体 pair、各类独立来源和冻结 probe 通过而给出 `head_short_runs_authorized=true`，但它已被机制+置信度门覆盖；后者保持红色，禁止把“总体门绿”解释成训练就绪。
- r6 冻结对象轨迹 balanced accuracy `.7321`，混淆矩阵 `[[6,2],[2,5]]`；`JtMY` pair 在 source 留出中为 `0/1 -> 0/1`，但五个 pair 的平均增量余弦降到 `-.0065`，不存在统一 prototype 方向。
- 五组 prototype/bootstrap 为 `.6696/.75/.9375/.6696/.7946`，中位数 `.75`、只有 `3/5` 过线，head 稳定性仍失败。
- 风险轮廓/潜生命周期 MIL 为 `.8661/.8036/.7946/.8661/.8661`，中位数 `.8661`；pair 排序率为 `.8/1/1/.8/1`，中位数 `1.0`，五组都正确排序 `JtMY`。这支持继续把生命周期/时序风险轮廓作为主原型，但不能把模型对低置信度银标的拟合当作标签已证实。
- label sensitivity 显示移除 `JtMY` 正例会使 balanced accuracy 降低 `.0863`，而移除其早段 no-alert 会提高 `.0536`；说明马车横穿信号有用，但“同一路线开放负例”仍与现有负类分布存在差异，后续应补更稳定的路口意图或相同行进方向证据。
- 置信度加权 OFAT 只改变 loss：每个类仍占总损失 `.5`，类内按 hash-bound silver confidence 线性归一化；其余特征、LOSO、bootstrap、seed、步数和超参数不变。加权与等权的五组 balanced accuracy、最差 seed、pair 排序全部相同，最大单 episode 概率变化约 `.01`。因此不再继续搜索平方/温度化等权重函数；置信度应继续用于数据门和复核路由，而不是期待它修复缺失机制。

r6 证据：

- `artifacts.local/evidence/public-video-provisional-training-r6-20260717/mechanism_coverage_confidence_15ep.json`，SHA256 `ead765a49f94ea63d9b439ecdf42366bcf56ea4593a7759ba54a923af8149490`
- `artifacts.local/evidence/public-video-provisional-training-r6-20260717/frozen_object_trajectory_source_group_15ep.json`，SHA256 `c035f3889635151e7033e0e4a70c3da83dbfa0ae120392d315bf51c8a2afb519`
- `artifacts.local/evidence/public-video-provisional-training-r6-20260717/prototype_bootstrap_five_short_runs_15ep.json`，SHA256 `30d9b1b6698abd94ea2407df30b03f7812d0a579f8d13dc3b1eaf48ebe2c49e0`
- `artifacts.local/evidence/public-video-provisional-training-r6-20260717/risk_lifecycle_mil_five_runs_15ep.json`，SHA256 `1fa5d796d0c90c1d2532821877a0aea244705e3f5b1b3a5f8df7d0294b63a7f9`
- `artifacts.local/evidence/public-video-provisional-training-r6-20260717/label_sensitivity_15ep.json`，SHA256 `660318742d894cbf96696f374a4e79d380fab10a3ae1662f0e5479cddfc5e2af`
- `artifacts.local/evidence/public-video-provisional-training-r6-20260717/risk_lifecycle_mil_equal_loss_rerun_15ep.json`，SHA256 `e724d0493191d2aa8a37ddd30dc21b90d16312c54e0839d20dd8a556359ca33d`
- `artifacts.local/evidence/public-video-provisional-training-r6-20260717/risk_lifecycle_mil_confidence_weighted_15ep.json`，SHA256 `bddc0ed8e269a96e9d5d963096cbf4832063cad5f1c6d15ed2d42507b4c4b01b`

### 2026-07-17 r7 更新：机制覆盖首次通过，但新来源证明表示仍不足

r7 从 r6 不可变复制，并只增加一个独立的 Wikimedia Commons 公开视频包。素材为 POPtravel 的曼谷 Sukhumvit Road 第一视角步行视频；来源合同绑定 [Wikimedia Commons 文件页](https://commons.wikimedia.org/wiki/File:Walking_in_BANGKOK_-_Thailand_-_Sukhumvit_Road_-_4K_60fps_(UHD).webm)、作者、CC BY 3.0、原始 YouTube 链接和 YouTubeReviewBot 的许可确认记录。下载的 240p 转码大小为 `241402495` 字节，SHA256 为 `8f0efe24eddd939e8396abc60cfa35789003e9a3b9f115b9538182d0060e6a17`。验证器只在上述 review 字段完整且作者一致时接受 CC BY 3.0；普通未复核 CC BY 3.0 仍拒绝。

- 新增两组高置信 matched pair：车道口清空 → 面包车近场横穿（动态）；砂堆占道 → 绕过后清空（静态）。四条 episode 置信度分别为 `.72/.88/.86/.78`。
- r7 共 19 条非弃权 episode、7 组 pair。机制审计中，动态 `all=4`、合格 `3` 对/3 来源；静态合格 `3` 对/3 来源。低置信 JtMY 仍排除，机制覆盖门首次通过。
- 冻结对象轨迹 balanced accuracy `.7389`，混淆矩阵 `[[7,3],[2,7]]`，总体线性门通过；但五组 prototype/bootstrap 为 `.6833/.7444/.7889/.5833/.6778`，中位数 `.6833`、仅 `2/5` 达标，稳定性门失败。
- 风险轮廓/潜生命周期 MIL 为 `.7222/.6722/.7389/.5833/.7833`，中位数 `.7222`；pair 排序率 `.5714/.5714/1/.7143/.8571`，中位数 `.7143`。与 r6 的 `.8661` / `1.0` 相比明显下降，说明第三机制来源增加了有效分布压力，而不是简单增加分母。
- 冻结 DINO/Depth pooled feature 只有 `.3611`；显式相对深度走廊轮廓只有 `.5222`，两者均未通过。故不能把修复简化成“换强 backbone”或“加入手工深度占用统计”。
- 新来源 source 留出下，轨迹 probe 预测为 `1/1/1/0`，期望为 `0/1/1/0`；主要错误是把车道口清空负例判为 alert。该 episode 被 quarantine 后 balanced accuracy 提升约 `+.15` 至 `.8889`，但这是 post-hoc 影响诊断，不授权删标或翻标。
- 目标轨迹统计解释了失败：近场面包车只有 1 条持续轨迹；砂堆正例与绕过后负例具有完全相同的 `14 detections / 6 tracks / 5 persistent tracks`。当前 COCO 对象轨迹既漏掉局部遮挡车辆的稳定占用，也无法表达非 COCO 的砂堆/地表障碍。
- 独立实现的冻结 free-space topology OFAT 也失败：它只从当前主线 SANPO 分割 logits 计算自适应路径、宽度、瓶颈、偏移和非可走类概率，不读取独立方向。拓扑单独 balanced accuracy `.5222`（负/正召回 `.60/.4444`）；与对象轨迹拼接后 `.6278`（`.70/.5556`），低于轨迹基线 `.7389`。拓扑/fusion 的 pair 增量平均余弦分别为 `.0349/-.0272`。四条新来源 episode 在 walkable `.50` 阈值下最小路径宽度均为 `0`，连清空段也无法识别，因此不继续调阈值或搜索拓扑参数。
- 结论从“数据集不足”更新为“机制覆盖已达到最低门槛，但表示合同不足”。下一实验应是独立实现的近场占用、静态地表障碍和时间遮挡表征；不继续搜索 head 优化器、confidence weighting 或 SAM/ASAM。
- 隔离仍是硬约束：本线没有读取或修改 `secondary-corridor-causal` 的数据、代码、权重、指标和产物；只允许在文字层参考其思路。

r7 证据：

- `artifacts.local/evidence/public-video-provisional-training-r7-20260717/r7_build_receipt.json`，SHA256 `dc0cbe6de714e3ac67bef5fdc3bf4ae8e2c9b85cbbd39c42da43e866e1ac933a`
- `artifacts.local/evidence/public-video-provisional-training-r7-20260717/mechanism_coverage_confidence_19ep.json`，SHA256 `59dfa67b7bab5a07c6ae7a6464e26081004d6030e5c90b93d869888356dc8e52`
- `artifacts.local/evidence/public-video-provisional-training-r7-20260717/frozen_object_trajectory_source_group_19ep.json`，SHA256 `0a460982ea9b703b33905613fb0abe523dba6b8afcf942cff3cd0dbb01bf9162`
- `artifacts.local/evidence/public-video-provisional-training-r7-20260717/frozen_depth_dino_source_group_19ep.json`，SHA256 `16a940f949ee04512c08f090ed004d7151d27e26e7c8b7b3e3fa25a5f2861120`
- `artifacts.local/evidence/public-video-provisional-training-r7-20260717/frozen_depth_corridor_profile_source_group_19ep.json`，SHA256 `5034cc56309155d0c1a3e94ab60e6b28456d71d7c9b9fef85ef08add971e7f9c`
- `artifacts.local/evidence/public-video-provisional-training-r7-20260717/prototype_bootstrap_five_short_runs_19ep.json`，SHA256 `2ab461a3537843ff8a85d1c4877be85dddcc18fca2811146635e1f47e5893cbc`
- `artifacts.local/evidence/public-video-provisional-training-r7-20260717/risk_lifecycle_mil_five_runs_19ep.json`，SHA256 `d047c04ab1cb0094b3add9e8e45cf4613de67d3ed2f6ce8174cb4f805ddd4515`
- `artifacts.local/evidence/public-video-provisional-training-r7-20260717/label_sensitivity_source_group_19ep.json`，SHA256 `f35378391948da77a0df2109c383684f06f8e1f3a1265baab548b273911c8f02`
- `artifacts.local/evidence/public-video-provisional-training-r7-20260717/frozen_free_space_topology_fusion_source_group_19ep.json`，SHA256 `be519fc2034e72933bcb7140e80ca773bd2bcbd07f273199f0d8e0d5f39fd4a9`

### 2026-07-17 r7.1 更新：受控 train-only 反事实稳定了 head，但 post-event clear 仍失败

本节继续使用 r7 的 19 条真实 provisional episode 作为唯一评测分母，没有把合成图、公开视频候选拒绝样本或独立模型方向的任何产物计入指标。`secondary-corridor-causal` 仍保持代码、数据、权重、指标、目录和实验编号完全隔离；当前主线脚本对该路径 fail closed。

- 运动补偿占用 probe 使用 homography 注册后的下方走廊灰度残差和可靠性。motion-only balanced accuracy `.5167`，trajectory+motion `.6833`；压缩 motion 与 trajectory 融合恢复到 `.7389`，但预测与 trajectory-only 完全相同。因此残差提供了局部机制线索，却没有增加稳定的 source-isolated 判别能力。
- 机制 temporal-range 审计中，动态、静态 pair 的 source 内 alert/no-alert 顺序均为 `3/3`；动态留出端点正确 `6/6`，静态只有 `4/6`。这证明静态障碍信号存在，但不能共用绝对阈值，需要 source-relative 基线和生命周期组织。
- MIL 加入首个可靠帧的 registered-residual/object-overlap 变化，以及完整非留出 pair 的 logistic ranking loss 后，五组 balanced accuracy 为 `.8389/.7333/.7389/.6833/.8389`，中位数 `.7389`、最差 `.6833`、`4/5` 达到 `.70`；pair 排序中位数 `.8571`。与原始 MIL 相比，head 组织得到改善，但最差 seed 仍不稳定。
- 继续扫描的 [Novi Sad POPtravel](https://commons.wikimedia.org/wiki/File:Walking_in_NOVI_SAD_-_Serbia_-_4K_60fps_(UHD).webm) 柱体候选被行人、卡车和转向/路线变化混杂；[Trubarjeva street roadworks](https://commons.wikimedia.org/wiki/File:Walking_down_the_Trubarjeva_street.webm) 主要为固定机位，且有白色面包车横穿。二者均被拒绝，没有强行构造真实 r8。
- 随后建立独立的受控合成 train-only 包：从 `Chcne`、`vcz` 和 Wikimedia Bangkok 的真实 no-alert 父 episode 复制完全一致的 clear 帧，再确定性叠加逐帧放大的路障或砂堆。共 3 对、6 条合成 episode、18 张图；9 张正例带 alpha-derived 精确 mask/bbox，9 张负例是未改动 clear 图。YOLO、COCO、manifest 和 18 张 contact-sheet 视觉 QA 均通过。
- 每条合成 pair 都绑定 `parent_source_id`。LOSO 中只允许进入训练折，父真实 source 被留出时自动排除对应 2 条合成 episode；合成样本永不进入真实混淆矩阵、balanced accuracy、pair 指标或任何 calibration/blind/production 结论。
- 最佳候选 `temporal baseline + pairwise ranking + train-only synthetic static pairs` 的五组真实 balanced accuracy 为 `.7944/.7944/.8944/.7944/.7444`，中位数 `.7944`、最差 `.7444`、`5/5` 达到 `.70`，pair 排序中位数 `.8571`。相对无合成候选，中位数提高约 `.0556`，最差提高约 `.0611`。
- lower-corridor 外观统计 OFAT 为 `.7389/.6889/.8389/.7889/.8389`，中位数 `.7889`、最差 `.6889`，没有超过最佳候选且重新出现低于 `.70` 的 seed，因此停止外观通道和阈值搜索。
- 残留错误仍集中在事件结束：砂堆 alert 在 `5/5` seed 中被正确判为 alert，但绕过后的 clear 在 `4/5` seed 中仍被判为 alert。当前只可判定“head 稳定性初步闭合”，不能判定真实生命周期语义闭合，更不能晋级生产。

r7.1 新证据：

- `artifacts.local/evidence/public-video-provisional-training-r7-20260717/frozen_motion_compensated_occupancy_fusion_source_group_19ep.json`，SHA256 `b26e18ea87e31bb1d2b5494c453098b3357a80c01f99740b23192331d7f754d6`
- `artifacts.local/evidence/public-video-provisional-training-r7-20260717/mechanism_temporal_range_leave_one_pair_out_12ep.json`，SHA256 `5b5f59422bc30f3d2d249cd39998b42c7b6777ba32d4e2fb99a07ac7b9f96d88`
- `artifacts.local/evidence/public-video-provisional-training-r7-20260717/risk_lifecycle_mil_baseline_channels_pairwise_five_runs_19ep.json`，SHA256 `c8b7c28993913a6f7f8f84126a3505208287e71dd3c92808387571795cbc04e7`
- `artifacts.local/synthetic/mainline-static-counterfactual-r8/controlled-pairs-v1/build_receipt.json`，SHA256 `79273244d32694ab9bd566eb48969b5d40862ba89c367cc75f0fc63978cabae6`
- `artifacts.local/synthetic/mainline-static-counterfactual-r8/controlled-pairs-v1/qa/manual_review.json`，SHA256 `f30494a89c949e31d10984cef83c1747d7b2ea0d651506119ef9c2db1098f02e`
- `artifacts.local/evidence/public-video-provisional-training-r7-20260717/risk_lifecycle_mil_synthetic_static_aug_pairwise_five_runs_19real_6synthetic.json`，SHA256 `4f8e68c04d5c40787fb5df262318c1a4bbd414c974660f2ab0f9203601f5eb3a`
- `artifacts.local/evidence/public-video-provisional-training-r7-20260717/risk_lifecycle_mil_synthetic_static_aug_appearance_pairwise_five_runs_19real_6synthetic.json`，SHA256 `d851a1b99e501a29373c39dbcd61d70a65a9ae6784445ffe9e351d26092f6040`

下一步只测试能显式表示“近期安全基线恢复、障碍退出、事件终止”的生命周期通道或静态障碍 teacher；继续保持真实 source 留出评测与 train-only 合成隔离。SAM/ASAM 仍为第三顺位。

### 2026-07-17 r7.2 更新：从错误的末帧聚合转向因果静态事件退出

逐帧风险曲线证明，砂堆 post-event clear 的错误不是“末帧始终危险”：五组中首帧都低风险，中间帧被通用对象特征拉高，末帧在部分 seed 已恢复，但 smooth-max 仍被中间峰值锁住。以下实验均保持 19 条真实 episode 为唯一评测分母，且继续拒绝 `secondary-corridor-causal` 路径。

- 严格 terminal pooling 只用最后因果帧做 episode BCE 和 pair ranking，五组 balanced accuracy 为 `.6278/.6333/.7833/.6833/.6333`，中位数 `.6333`、最差 `.6278`；pair 排序中位数 `.7143`。该方法损失动态接近证据，明确停止。
- train-only synthetic mask teacher 使用冻结 DINO 稠密 patch token，以 composite mask 内 patch 为正、同位置 exact-clear patch 为负；每折仍排除父真实 source。它的真实 LOSO balanced accuracy `.5778`，负/正召回 `.60/.5556`，并把 Bangkok 砂堆与清空段方向判反。合成 mask 可用于结构训练，但当前视觉域不能充当真实静态障碍 teacher。
- 冻结 YOLOE-11s prompt-free 模型使用内置 4585 类，不接收文本 prompt、银标、source mask 或合成样本。预注册 surface-material/barrier 词表后，它在砂堆最接近帧产生一个 `sand box` 检测（confidence `.5845`），三张 post-event clear 均为零检测。
- prompt-free 静态语义单独 balanced accuracy `.6111`；与 COCO 轨迹直接拼接为 `.5778`，均不能作为全局线性表示。但在完全 held-out 的 Bangkok source 上，融合预测 `0/1/1/0`，四条均正确，说明静态语义应作为稀疏专家而不是全局特征。
- 新增无学习参数的因果 event-exit router。它只在以下四项同时满足时关闭上一静态事件：同 source 紧邻前段有 surface-material 检测；当前段无 surface-material；时间间隔不超过 `5000 ms`（无时间戳时 manifest gap 不超过 3）；当前 source-isolated COCO trajectory 预测 no-hazard。路由决策不读取当前银标，也没有 episode ID 白名单。
- 路由只找到一个候选：砂堆 alert → `2000 ms` 后 clear。五组 balanced accuracy 从 `.7944/.7944/.8944/.7944/.7444` 提升到 `.8444/.8444/.8944/.8444/.7944`，中位数 `.8444`、最差 `.7944`；全部 non-degrading。post-event clear 达到 `5/5` 正确，alert recall 五组均保持 `.8889`。
- `1000 ms` gap 负控不产生 exit candidate，五组结果完全回到未路由基线，证明修复来自可解释的连续事件退出，而不是无条件把某类高风险样本改成 no-alert。
- 当前只关闭“已知失败样本 + 一个 source”的原型问题，尚未通过跨来源 event-exit gate。至少还需 2–3 个独立 surface-material/静态障碍退出来源，才能评估退出 precision、动态新风险保护和事件级假提醒。

r7.2 新证据：

- `artifacts.local/evidence/public-video-provisional-training-r7-20260717/risk_lifecycle_mil_terminal_pooling_synthetic_static_aug_pairwise_five_runs_19real_6synthetic.json`，SHA256 `ef4f54bef65173ff6d7c1198eca1522d06c633e3ea45ad877d78062be28ae81a`
- `artifacts.local/evidence/public-video-provisional-training-r7-20260717/frozen_dino_synthetic_mask_static_teacher_source_group_19ep.json`，SHA256 `6aa21bc02b29907d9dcde632f070550f495c5bb7f3275fffd559efa3309da319`
- `artifacts.local/evidence/public-video-provisional-training-r7-20260717/frozen_prompt_free_static_semantic_fusion_source_group_19ep.json`，SHA256 `041a97745832e0457c8ffa1bf607e020d9ad56586ae59ef8544939a11a0301e2`
- `artifacts.local/evidence/public-video-provisional-training-r7-20260717/semantic_exit_router_five_runs_19real.json`，SHA256 `80b4f4badad700e5931d4f4aace88cc49f756f65d016ac4009b237d28a176697`
- `artifacts.local/evidence/public-video-provisional-training-r7-20260717/semantic_exit_router_gap_1s_negative_control_19real.json`，SHA256 `2d5d6c0bd18894e5c5da9a8b1240bd8d1869e0f49591a2ef42c23ba406e105f5`
- YOLOE prompt-free 权重 SHA256 `292bdf157a9ec7315f34b567cb93467c5043cd1889a1cc18abbfdeb88d7a948d`

下一步保持 router 合同不变，优先补独立连续退出来源，而不是继续搜索 gap、semantic confidence 或类别词表。当前不写入 Android `RiskEventTracker`，也不授权生产晋级。

最新证据：

- `artifacts.local/evidence/public-video-provisional-training-r5-20260717/frozen_object_trajectory_source_group_14ep.json`，SHA256 `36b80d545c09af9fe2594bd580da8d18cda3a4b9f691c6229e736f3f12c6f5be`
- `artifacts.local/evidence/public-video-provisional-training-r5-20260717/prototype_bootstrap_five_short_runs_14ep.json`，SHA256 `453d605012402532e21b8499d5a9e4b5e5f79c0cda363b349b2d89ddf377cf72`
- `artifacts.local/evidence/public-video-provisional-training-r5-20260717/risk_lifecycle_mil_five_runs_14ep.json`，SHA256 `b5171d4b649aede351fb23156c678d3971c6ca5cbdf766e1eedf39a91556429e`
- `artifacts.local/evidence/public-video-provisional-training-r5-20260717/mechanism_coverage_14ep.json`，SHA256 `bf5c267194ff5c8ef9f25d57ec125a90444a06db0ce06867059e775f7608a032`

- 正例为 6 条、来自 6 个独立 source；负例为 5 条、来自 5 个独立 source，达到每类至少 5 个独立 source。
- vcz 的 `clear → stroller intrusion` 与 SK1 的 `clear → pedestrian approach` 形成 2 组显式 `counterfactual_pair_id`，达到最低门槛。
- MobileNetV3、Depth Anything V2/DINO、静态分割走廊池化和固定残差光流均未过线；改用冻结 YOLO proposals 形成的对象轨迹、相对尺度、下方走廊重叠和持续性特征后，source-isolated balanced accuracy 达到 `.8167`，两类召回 `.80/.8333`，通过门槛。
- prototype 初始化加 source bootstrap 的五组 80-step 短跑中，balanced accuracy 为 `.65/.7333/.7333/.8167/.7333`，中位数 `.7333`，5 组中 4 组通过线性门槛，稳定性门禁通过。
- 11 条 episode 之间不存在跨 source 的重复帧哈希，弃权样本没有进入 probe，同一 source 的正负 episode 在评估中被整体留出。

补入两组匹配反事实后，MobileNetV3 从 r2 的 `.40` 提升到 `.6167`，DINO 从 `.50` 提升到 `.6333`，证明数据结构确实是先前失败的重要原因。对象轨迹特征进一步过线，说明根因不只是 head 优化：通用静态 pooled representation 缺少“对象相对走廊如何随时间变化”的结构，而简单线性 head 在获得该结构后已经可学。SAM/ASAM 仍保持第三顺位，不应替代数据扩充和生命周期结构诊断。

## 确定性冻结特征结果

两个 probe 都使用 `source_id` 分组留一：同一公开视频以后即使切出多个 episode，也必须整体进入同一留出折，不能通过 episode 切分泄漏。

| 冻结特征 | 混淆矩阵（真值行、预测列） | 正例召回 | 负例召回 | balanced accuracy | 结论 |
|---|---:|---:|---:|---:|---|
| MobileNetV3 OS8+OS32 | `[[2,3],[1,5]]` | .8333 | .40 | .6167 | 未通过 |
| Depth Anything V2 / DINO-S | `[[3,2],[2,4]]` | .6667 | .60 | .6333 | 未通过 |
| YOLO12n proposals + deterministic object trajectories | `[[4,1],[1,5]]` | .8333 | .80 | .8167 | 通过 |

证据：

- `artifacts.local/evidence/public-video-provisional-training-r3-20260716/frozen_feature_probe_os8_os32_pair_alignment_11ep.json`，SHA256 `f32b0faddea14f4a94554efb9124937f6d5c4e729a04ea59000b7544cf383b42`
- `artifacts.local/evidence/public-video-provisional-training-r3-20260716/frozen_depth_dino_source_group_11ep.json`，SHA256 `080d9ea14d54405875705a0994e840f70cb0bd0566551af1bb2621d3141e9f6b`
- `artifacts.local/evidence/public-video-provisional-training-r3-20260716/frozen_object_trajectory_source_group_11ep.json`，SHA256 `ab54c01002f6a18830750a8185855e299916e1a24d18ddb191e930d2835f6596`

### 走廊相对池化与 prototype 方向检查

在不训练 backbone 的前提下，另行预注册了一种固定空间池化：用现有四类分割 logits 的 walkable channel，只在 lower-center 区域分别计算 walkable/non-walkable 加权特征和两者差值。结果为混淆矩阵 `[[1,4],[3,3]]`、负例召回 `.20`、正例召回 `.50`、balanced accuracy `.35`，明显差于普通冻结特征。报告为 `artifacts.local/evidence/public-video-provisional-training-r3-20260716/frozen_feature_probe_corridor_relative_source_group_11ep.json`，SHA256 `ad853e7b6e5b514e5967a4544b8ed285984dc1539882f741feca3f8c8f611309`。

因此现有像素分割输出不能稳定充当风险走廊坐标系，仍只适合作为辅助监督。它在 SK1 上能区分 clear/approach，却把 vcz 的 clear/intrusion 方向判反，说明静态 walkable 概率缺少一致的相对运动语义。

同一 MobileNet 冻结特征还计算了两组显式反事实的 `no-alert → alert` 单位增量方向。SK1 与 vcz 的余弦为 `-.0522`，未达到预注册的 `.20`；说明静态 MobileNet 表征中不存在可复用的 prototype 方向。对象轨迹向量的两组原始增量余弦也只有 `.1395`，仍未达到 `.20`，但它的 source-isolated 线性分类已经过线。因此 prototype 初始化只作为 fold 内起点，最终结论依赖 source 留出和五组 bootstrap，而不依赖两组 pair 共享单一原始方向。

随后只做了一次固定的显式运动单因素诊断：在 192×192 上计算 Farneback flow，减去全局中值平移，再把 lower-center 残差幅度、正散度、径向扩张及其 episode mean/max/last-minus-first 追加到冻结 MobileNet 向量。结果混淆矩阵仍为 `[[2,3],[1,5]]`、balanced accuracy `.6167`，与静态基线完全相同；两组反事实增量余弦仍为 `-.0513`。报告为 `artifacts.local/evidence/public-video-provisional-training-r3-20260716/frozen_feature_probe_residual_motion_source_group_11ep.json`，SHA256 `4ed69ed8e4c65c5e2a14005b1cbd730032c2f179710a6662f8b69e3758949878`。

所以稀疏 1 FPS 的通用光流残差也不足以形成一致 lifecycle 方向。后续不继续搜索手工光流参数；应让风险轮廓/生命周期原型直接接收对象轨迹、相对尺度变化和走廊关系的结构化时序输入。

## prototype/bootstrap 五组短跑

新增 `scripts/run_public_silver_prototype_bootstrap_short_runs.py`。每个 source 留出折只用训练 source 计算标准化参数，以两类 prototype 的单位差向量和中点 bias 初始化线性 softmax head，再在每类内部按 source 有放回抽样，执行 80 step 确定性 full-batch Adam。冻结 detector/轨迹提取器，不保存权重。

| seed | 负例召回 | 正例召回 | balanced accuracy |
|---:|---:|---:|---:|
| 2026071601 | .80 | .50 | .65 |
| 2026071602 | .80 | .6667 | .7333 |
| 2026071603 | .80 | .6667 | .7333 |
| 2026071604 | .80 | .8333 | .8167 |
| 2026071605 | .80 | .6667 | .7333 |

稳定性门禁要求至少 4/5 组满足 balanced accuracy `.70` 且两类召回至少 `.50`，中位数至少 `.70`，并且任何一组任一类召回不低于 `.40`；本轮通过。证据为 `artifacts.local/evidence/public-video-provisional-training-r3-20260716/prototype_bootstrap_five_short_runs_11ep.json`，SHA256 `84ae652907ca774990968177d12e86bc2230f04ddca74def81b3699c9d0f7b39`。

误差并非随机消失：`1sft` 行人走廊正例五次均漏报，`We` 开放人行道负例五次均误报；`gie8` 静态家具侵入和 vcz 婴儿车侵入随 bootstrap 波动。这说明下一个主线应增加“风险轮廓 + 生命周期”的显式时序组织，并补同类型 matched negative/positive，而不是继续调线性优化器。

## 风险轮廓 + 潜生命周期 MIL 原型

新增 `scripts/run_public_silver_risk_lifecycle_mil_head.py`。它从冻结对象 proposals 构造每帧对象数、相对面积、底边、固定走廊重叠、威胁值、类别组和一阶时间差；一个小型线性逐帧 head 产生风险曲线，再用 smooth-max multiple-instance pooling 接受唯一可用的 episode 级 alert/no-alert 监督。像素分割不进入主 head，继续保持 `auxiliary_only`。

当前银标没有可信的逐帧生命周期边界，因此 `approach/alertable/post_event` 只从学到的风险曲线解码为**潜变量诊断**，不计算或声称生命周期准确率。五组 source-bootstrap LOSO 的 balanced accuracy 为 `.7333/.6333/.9167/.9167/.7333`，中位数 `.7333`；两组 matched counterfactual 在全部五组运行中都保持 alert 概率高于对应 no-alert，排序 10/10 正确。证据为 `artifacts.local/evidence/public-video-provisional-training-r3-20260716/risk_lifecycle_mil_five_runs_11ep.json`，SHA256 `af329f062b60a94c8650f6bf5232c20ad348f63cc79853067bbd4f6b6d4719d6`。

该原型证明逐帧风险轮廓是可行方向，但最差运行只有 `.6333`，且 `1sft` 仍五次均漏报，所以当前不能保存或部署 head。下一批数据优先覆盖“持续存在但尺度变化不强的走廊内行人”与“开放路线中存在大量 lateral objects 的 no-alert”两类硬样本，并为正例补明确 alertable 起点/cleared 终点后，才可把潜生命周期升级为受监督生命周期头。

为区分“结构性硬样本”和“可能的银标歧义”，又增加了只用于审计的逐 episode quarantine 敏感性分析。冻结轨迹基线仍为 `.8167`；隔离 `1sft` 后为 `.90`，但隔离明确的 SK1 matched-positive 后也同样为 `.90`，说明 11 条样本下单条正例对分母和训练折的影响都很大，不能据此 post-hoc 删除或翻转标签。视觉复核显示 `1sft` 是宽路上持续存在、尺度增长不强的行人群，其“是否应提醒”确有策略歧义；因此它被提升为独立语义复核和同类数据补充优先级，而不是被排除。报告为 `artifacts.local/evidence/public-video-provisional-training-r3-20260716/trajectory_label_quarantine_sensitivity_11ep.json`，SHA256 `b707bda5ebbc1073d43dea73e3b35e7dabb7af2626294b06ce03d7bbbe825841`；该报告显式 `post_hoc_analysis_only=true`、`training_gate_authorized=false`。

## 可执行训练前门禁

新增 `scripts/audit_public_silver_training_readiness.py`。它会重新验证全部 v2 银标包和每个绑定图像文件，并执行以下检查：

1. 非弃权正负类各至少 5 个独立 `source_id`；
2. 至少 2 个显式 `counterfactual_pair_id`，且每对同时包含 alert/no-alert；
3. 不允许同一帧哈希伪装成多个独立 source；
4. 评估必须按 `source_id` 分组留出；
5. 只有数据门禁和冻结特征线性可分门禁同时通过，才会将 `head_short_runs_authorized` 置为 `true`。

采用对象轨迹 probe 后的最新 11-episode 审计报告为 `artifacts.local/evidence/public-video-provisional-training-r3-20260716/training_readiness_object_trajectory_11ep.json`，SHA256 `ec26c48cfa69829513deb369072a3bfcf64074de2c132707dfaf4bae432c028d`。独立 source、匹配反事实、跨 source 帧隔离和冻结特征门禁均通过，`head_short_runs_authorized=true`，无 failure reason。

该脚本只控制实验顺序，不授权校准、blind 评测或默认/生产模型替换。

## 新数据的优先顺序

下一批继续优先补充同对象、近似视角和近似路线条件下的 alert/no-alert 对。由于静态分割走廊池化与通用残差光流都已失败，下一种表征必须使用对象级轨迹、相对尺度变化和时序走廊关系，而不是再调整 walkable mask/光流阈值或重复静态池化。只把同一长视频拆成更多窗口不能增加独立来源数，且同一 source 的全部窗口必须在同一评估折。

已检查的 qtty、wBP 与 4P1 本地序列不强行标成负例：qtty 的中心/近场被大范围整人和车辆脱敏遮挡；wBP 存在明显横向转向、路口和斑马线；4P1 全程运动模糊且近距离多人穿行。它们应继续弃权，避免用错误负例“修好”指标。HEm 与 C-g5 各只有 3 个可用 RGB 帧，也不满足时序判断要求。Chcne 原 252–387 窗口同样因大幅转向而不采用；2026-07-17 另行向前补取 117–252 帧，只截取 177–207 与 222–252 的稳定同路线片段，作为独立 r5 静态机制 pair，未回写原弃权判断或 r4。

当前已完成确定性 probe、五组 prototype/bootstrap 短跑、距离场独立负向 OFAT、两版风险轮廓/潜生命周期 MIL 原型和机制级数据门。下一步不再调线性优化器：先为动态接近与静态收窄各补至少 1 个、优先 2–3 个独立同源 pair，并增加逐帧 lifecycle interval；像素/距离场继续保持辅助监督，SAM/ASAM 仍为第三顺位。

## r7.3 跨来源公开视频退出发现审计（2026-07-17）

本轮没有扩大 r7 的真实评测分母。四条新增 Commons 连续步行视频共约 112 分钟；5 秒采样 1,346 帧，320 输入下得到 92 条 prompt-free 退出 proposal。全部 6 条 surface proposal 和最高置信 barrier proposal 经 GPT 多帧视觉复核均不构成干净风险退出，不能进入训练或评测。

连续缺失门能减少闪烁：10 秒持久化将 92 条降至 77，20 秒降至 62，但原 6 条 surface 错检全部保留。640 输入、`.05` 置信度、10 秒持久化将候选进一步降至 51（3 surface、48 barrier），剩余抽样仍全部为普通纹理、建筑、围栏、固定设施、车辆或人群。近场 box 几何门又误删真实施工边界，因此也被否决。结论是当前开放词汇检测器适合 proposal，不适合独立决定事件退出。

Hof 公交后窗施工片段被确定性时间反转为 discovery-only 退出反事实。10 个连续一秒缺失把 3 条闪烁候选压到一条 `18s → 19s`，与 GPT 的约 19–21 秒视觉清除窗口一致。它只作为退出持久化回归，不是真实行人 episode，不计入 balanced accuracy，不授权训练、校准、blind 或生产。

后续采集策略改为：先由大模型审阅许可视频 overview 并提出少量粗时间窗，再用冻结语义/轨迹和持久化门做验证；停止全视频低阈值 YOLOE 广撒网。r7.2 指标保持原样，Android `RiskEventTracker` 暂不修改，`secondary-corridor-causal` 继续完全隔离。

## r7.4 真实 Hof 退出与风险轮廓并集（2026-07-17）

- 新增 Commons 作者自发布 CC BY 4.0 的 Hof 公交车窗施工短片。原时序 `0–6s` 持续可见施工区，`7s` 离开，`8–10s` 持续清空；视频 SHA256 `1d9dff54e8ff89c4f40b66f29818eddca5948c564ef64c2f92a02d09efcc9e4c`。
- 冻结 prompt-free 扫描在 `6s` 检出 `construction site` `.5028877`，`7–10s` 连续缺失；GPT 多帧审阅接受 `6s → 7s` 为 discovery-only 退出边界，置信度 `.92`。
- 外部挑战先冻结候选，后使用 GPT 边界评分。三样本持久化下，`surface_only` 在 `1s → 2s` 过早退出；`barrier_only` 和 `risk_profile_union(surface + barrier)` 都在 `6s → 7s` 精确命中。报告 SHA256 `7c4cdc7b724ee6a3a1ef326c8e1f81d5231b724e6601bb4f4eda3a8ab83195d8`。
- 隔离的主线风险轮廓回放仍只识别曼谷砂堆后 clear 一条候选。五组 balanced accuracy 为 `.8444/.8444/.8944/.8444/.7944`，中位数 `.8444`、最差 `.7944`，与 r7.2 一致且全部不退化；报告 SHA256 `d18b89e6a70d392fa842874eb2adf6e61ec9990bca844a336cb94be1b1d9eb3f`。
- 1 秒 gap 负控无候选，五组精确回到 `.7944/.7944/.8944/.7944/.7444`；报告 SHA256 `ffc79d68561e3a8d30e6abf7594b463bd28cf0b63aea5652ef06e3f653390668`。

当前判定：风险轮廓并集解决了新 Hof 挑战中单 surface 过早解除的问题，同时不伤害现有 r7 结果。但 Hof 为侧向公交视角，不能作为行人 should-alert 真值。保持非训练、非校准、非 blind、非生产，不修改 Android `RiskEventTracker`。

## r7.5 公开来源许可与连续性采集门（2026-07-17）

- VLM overview 审阅拒绝 3 条许可候选：两条 Ljubljana Commons 工地视频是固定/摇摄视角；Vimeo Gympie 是带航拍和多次硬剪辑的巡场蒙太奇。拒绝报告 SHA256 `03270c1796393fa22da46c34f0fff0e73c52efe11da555186b26f127eed7c764`。
- 新增平台无关 discovery registry v2 和 Vimeo CC-BY 单页台账。在线包装器每次只做一个官方搜索页请求；条目始终先写为不可训练候选，搜索页许可过滤不替代条目级许可证明。
- Vimeo 唯一新结果的条目许可为 CC BY，但场景描述只是泛谈出行延误，`walking/roadwork` 属于文本碰撞，下载前拒绝。
- 登录态 YouTube 页面确认 Addis Ababa 候选在内容上非常接近所需的连续施工走廊步行，但视频许可未闭合，因此不下载。另一条 90 秒工地旁步行视频明确声明视频版权，Creative Commons 链接只覆盖音乐，按许可作用域错误拒绝。
- 统一采集审计 SHA256 `8a6c22482e590a3f030d769b5da00ce2aa3372e863ce2cafbf8d8cef9a3541c4`。新增合格行人第一视角退出来源为 `0`，而门槛仍要求至少 `2` 个独立来源；训练入口、校准、blind、Android runtime 和默认模型替换继续关闭。

这轮说明公开数据问题不是“网上没有相关画面”，而是相关画面常被默认 YouTube 许可、音乐 CC、静态机位或剪辑破坏因果连续性。下一轮必须按 `视频条目级许可 -> 连续性/VLM -> 冻结语义与轨迹挑战` 的顺序采集，不能先下载再用关键词或大模型判断补齐许可。

## r7.6 Hof 冻结 DINO 方向跨来源检索负实验（2026-07-19）

- 用 Hof 视频 `0/3/6s` 风险帧与 `8/9/10s` 清空帧建立冻结 Depth Anything V2 DINO-S 方向，零可训练参数；在 Greenwich、上海、哈尔滨、Worms 四条许可明确的连续视频中每 5 秒采样一次，共覆盖 1,346 帧。
- 原始投影在四个来源中均为负且量纲不具备跨来源校准含义，因此只允许使用来源内 robust-z 和持续下降排序 proposal，不能解释为“施工概率”。
- 对 12 个最高分窗口做 2 秒密集多帧复核，`0/12` 通过。候选实际是室内外切换、近景物体/商店、历史砖石或岩壁纹理、固定门洞/花园入口和镜头转向，没有连续的施工/障碍风险退出。
- 合格独立来源仍为 `0/2`，所以后续风险轮廓、持续性和轨迹挑战未运行；训练、校准、blind、Android runtime 和默认模型替换继续关闭。审阅报告 SHA256 `05b5fe9637598144fce86a7fb5fb839baccf8714e4f8520ff1d33f6908d05112`。

该负实验排除了“只需用一个正负原型方向扫描更多无标签视频”的捷径。当前数据池并不含可由该方向可靠找回的真实行人施工退出；事后把 DINO 与 YOLOE proposal 做并集或权重搜索只会在同一批非因果候选上过拟合，不能替代许可优先的新来源采集。

## r7.7 Fremantle 长施工区生命周期负挑战（2026-07-19）

- 在不计入行人门禁的前提下，使用 CC BY 2.5 AU 的 Fremantle 原时序车载记录测试长施工区。GPT 密集时间线复核给出视觉风险退出 `178s -> 179s`，随后 `179–195s` 稳定净空；这只是车辆视角的外部机制参考。
- 冻结 prompt-free 语义对 262 个一秒样本产生 10 个 barrier 退出：8 个早于参考、2 个晚于参考、0 个命中。risk-profile union 与 barrier-only 完全相同，surface-only 无候选。
- 在 `150–178s` 风险仍存在的 29 个样本中，barrier 仅激活 2 次，覆盖率 `.06897`，最长连续空窗 20 秒；在 `179–195s` 的 17 个稳定净空样本中又出现 1 次 `construction site` 假激活。报告 SHA256 `8e37b59d75ca2c2985c06e4cd826491bebb59eaa0d4fc05415744953b277a7a4`。

因此 r7.4 在短 Hof 视频通过的是“检测连续且退出清楚”的有限条件，不足以证明长事件生命周期可用。当前退出规则缺少独立 clear evidence：语义缺失既可能是模型漏检，也可能是真净空。后续生命周期原型应采用三态 `present/uncertain/clear`，把 detector absence 仅视为不确定性，把明确净空/可通行表征作为退出证据；在获得可分的 clear 表征前，不修改 Android `RiskEventTracker`。

## r7.8 施工标志词表与三态生命周期原型（2026-07-19）

- 发现冻结 prompt-free 内置词表已有 `barricade`、`cone`、`construction worker`、`traffic cone`，但既有 barrier 子集未纳入。新增默认关闭的探索开关把这四类映射到 barrier；旧基线保持原样，无文本 prompt、无学习参数。
- Fremantle 风险窗口语义覆盖由 `2/29` 提升到 `23/29`，最长空窗由 20 秒降至 2 秒；稳定净空仍有 1 个单帧假激活。相邻退出候选由 10 个降至 3 个，但仍是 1 个过早、2 个过晚、0 个精确命中，说明扩词表修复了风险存在表征，尚未修复事件生命周期。
- 固定三态合同为 `3` 帧入口窗内至少 `2` 帧激活、缺失先进入 uncertain、连续 `3` 帧缺失才确认 clear，clear 后单帧激活不能重开。Fremantle 得到唯一事件区间 `last_active=177s / first_absent=178s / confirmed_clear=180s`，包含 GPT 的 `178→179s` 视觉边界，并抑制 `182s/239s` 单帧假激活。报告 SHA256 `302152ec2cb3ab2287bec387fdbc064a447f0fe5987a3d14f3694b2d56ce5744`。
- 同一参数不调参回放 Hof 原 baseline scan，得到唯一事件 `last_active=6s / first_absent=7s / confirmed_clear=9s`，包含原 `6→7s` 参考。报告 SHA256 `3f73e28d97278d68533ed434ec077939b3b8dc2a405c21c234e66e3a2a9de37e`。

这是“风险轮廓 + 生命周期头”方向的首个双来源机制通过，但不是训练就绪：施工 marker 集合由 Fremantle 失败诊断后提出，属于探索性特征；Fremantle 与 Hof 都是车辆视角。新增合格行人来源仍是 `0/2`，Android `RiskEventTracker`、训练、校准和 blind 保持关闭。下一条许可明确的连续来源必须在审阅前冻结 marker 集合和 `2/3 + 3-clear` 参数，才能提供前瞻证据。

## r7.9–r7.11 前瞻负控与机制专家收敛（2026-07-19）

- r7.9 在画面复核前冻结完整 work-zone marker 与 `2-of-3 / 3-clear` 合同，但普通 Greenwich 门栏和 Shanghai 街景开启了假事件，仅 `2/4` nuisance 控制通过。单类删除和近场几何均不能同时保住 Fremantle 正例与四个控制，合同/失败报告 SHA256 分别为 `48a8319f61bb58f9e319460e5dbb655340f8729de06871718cb258550cce5fa1` / `b4ce3f45ec2791924492df35c0402dd5325d888a08ed733cd816fc4e8b97a038`。
- r7.10 把 dense multi-cone 独立成机制专家。计数阈值 `1/2/3/4` 中只有每帧 `>=2 traffic cone` 同时通过 Fremantle 与旧控制；冻结后又通过 `4/4` 新 nuisance 控制。但在 Hof 稀疏施工桩段只覆盖 `2/18` 帧，无法建立事件。因此 r7.10 是有效窄专家，不是通用 barrier 表征。前瞻负控/稀疏风险报告 SHA256 为 `d81541eeefc43997ce26def42761364e66455f01e3387fec9182db19cd0ec348` / `3335a4e466c3e9b679c9406f64dc54aad9cc20a7408eacd09fd7926dea5a9c79`。
- r7.11 使用 detection box 内 `high_saturation_fraction > dark_fraction` 的相对颜色规则，同时接收 `traffic cone` 与 `barricade`，不使用绝对颜色阈值或几何门。`clear_absent=5` 是测试的 `3–6` 中最小充分值：Fremantle 唯一退出区间覆盖视觉边界；Greenwich/Shanghai 固定设施不开户；Hof 稀疏风险在 `22s` 开启并保持 open uncertain。审计 SHA256 `1ace0742e75c63df14e34c60aef66e253c4c38d07c6a53d807b763f3d61f6065`。
- r7.11 合同在新窗口打开前以 SHA256 `3e6e6f410ce03053f4e8d6e38475156f69cf6a41352d8619541a7c8ae526bca7` 冻结。第四批 4 个新 nuisance 窗口共 190 帧、35 个原始 marker 检测，最终 `4/4` 零事件且终态 clear，报告 SHA256 `c0c7dde0cdf0545a45d9a36fce2a971afe79ad02c0b7a931b87371e133bb746f`。

结论：当前证据支持“机制特定正证据专家 + present/uncertain/clear 生命周期”，反对把任意 `barrier/construction site` 检测缺失直接解释为净空。r7.11 只有前瞻负控，没有冻结合同后的独立真实正退出来源；车辆视角也不能替代行人门禁。因此训练、校准、blind、Android runtime 和 `RiskEventTracker` 修改继续关闭。

## r7.12 本地来源全集与独立正样本门（2026-07-19）

- `public_video_local_source_inventory_r712.json` 将 7 份本地来源注册表中的视频按文件 SHA256 去重，审计严格要求注册表全集覆盖、来源 ID 唯一、视频路径/哈希一致，并拒绝任何影响 r7.8、r7.10 或 r7.11 冻结参数的来源冒充 prospective positive。库存合同 SHA256 `2110df3b8973a68efa7f466cb2ffe2d7f2f8a4c60c281339b11054fa62af7090`。
- 14 个唯一视频中，Moira/Ljubljana/Anasskoko 是固定或平移机位；Duisburg/Vimeo 有硬剪辑；Greenwich/Shanghai/Harbin/Worms 的完整时间概览与候选复核未发现施工离场；Fremantle 与两条 Hof 视频已参与生命周期或特征派生，不能作为冻结后独立来源。
- 自动门禁输出：独立真实正退出来源 `0/1`，行人第一视角独立正退出来源 `0/1`，总门禁失败。报告 SHA256 `9070787e41d701129ec56597a09cd78c0dce834e48ce9efeefaaab7fc99bb370`。

这把缺口从“可能还有未整理视频”收敛为可执行条件：只有新增 item-level 许可、原始连续时序、未影响冻结参数且大模型复核为真实风险到清空的来源，才会自动打开下一阶段；换文件名、重复下载、反向视频、固定机位消失或剪辑跳转均不能通过。

## r7.13 冻结正退出验收器（2026-07-19）

- `evaluate_public_video_chromatic_marker_prospective_positive.py` 要求四份带 sidecar 的输入同时一致：r7.11 冻结合同、合同绑定的 chromatic feature report、r7.12 来源谱系审计和打开画面后形成的大模型多帧复核。复核必须反向引用已冻结 feature report 与来源审计 SHA，防止看完结果后替换输入。
- 每条正来源必须同时通过来源资格、唯一事件覆盖视觉边界、风险窗口覆盖率 `>=.4`、稳定净空激活率 `<=.1`、终态 clear 且无未关闭事件；硬剪辑、非原时序、派生污染和视频 SHA 不一致直接拒绝。
- 复核模板 SHA256 `07d872826ff68231e327c231dd52de2002b34b16256fc6bf45255673a0e355bd`。8 个离线回归分别证明合格路径能通过，并能拒绝来源污染、事件错位、稳定净空假激活、合同/报告漂移与硬剪辑。

该验收器只消除了未来执行歧义，没有补造证据。当前仍无可运行的真实 held-out positive，训练、校准、blind、Android runtime 和生产门保持关闭。

## r7.14 pair-relative 生命周期变化 probe（2026-07-19）

- `run_public_silver_pair_relative_lifecycle_probe.py` 对 r7 的六组合格同源 matched pair 做零训练参数的变化方向审计。输入必须通过 sidecar、package root、mechanism report SHA 和独立方向隔离校验；时间顺序只读取 SHA 绑定 source manifest 的 `frame_index`。
- 合同不使用绝对场景阈值：后段机制分数高于前段时预测 `open_event`，低于前段时预测 `close_event`，相等时弃权。结果为 `6/6`：动态接近 `3/3`、静态收窄/恢复 `3/3`，共 `5` 次开事件和 `1` 次关事件。
- Bangkok 砂堆后的已知 post-event clear 被正确路由为关事件，但其归一化 margin 只有 `.0412`。无门槛 sign 诊断通过；在不作为验收阈值的 `>5%` margin 压力审计下，该唯一关事件会弃权，而其余 `5/5` 保持正确。这说明方向信号存在，但关事件稳定性证据仍薄弱。
- 报告 SHA256 `421e2d5be2c7ca81991ebe36ccadc0033c4503515acb828345fbb0c63a51ec68`。当前结论只支持下一版 head 采用“机制通道 + 相对基线 + 三态生命周期”，并要求近期可信参考状态；任意帧冷启动、连续窗口确认和跨来源正退出仍未解决。训练校准、blind、Android runtime、`RiskEventTracker` 与生产替换继续关闭。

## r7.15 SK1 retrospective 动态关事件压力样本（2026-07-19）

- source-manifest 全集检查发现，六组 qualified pair 中只有 SK1 在既有风险 episode 后还有未使用的连续 frame `8–9`。大模型先按原时序冻结 `5–7 risk → 8–9 clear` 判断，再打开冻结 detector 分数；review SHA256 `d620eb14494751ca61e9e97f591857e36e21a01c9cf6e33b5adbe3372dc15452`。
- 同一 YOLO12n/320/`.15` 合同精确复现已发布风险分数 `.0349100`，后段净空分数为 `.0123757`；相对差 `-.0225343`、归一化 margin `.6455`，动态 `close_event` 正确。这样现有离线诊断拥有静态和动态各一个关事件，但二者都不是新来源前瞻证据。
- 结果 SHA256 `8e1d212893a3e77729b911b4d93d0d8588cef1586cbe68478ecea6c43a0bf6bb`，6 个纯回归通过。review 的手填 UTC 时间有几分钟笔误，未改写原证据；勘误 SHA256 `4458000e09c595e1a7b66cc456da55ab826330873ea5b3c204ea92023f099497` 记录 review/sidecar 先于推理输出的文件顺序，并明确文件时间不是密码学时间戳。

结论：相对基线的关事件方向不再只依赖 Bangkok 静态样本，但 r7.15 来自同一已审阅派生 source，不能改变 r7.12 独立正来源 `0`，也不能授权训练校准、blind、Android runtime、`RiskEventTracker` 或生产替换。

## r7.16 双证据生命周期融合原型（2026-07-19）

- `run_public_silver_dual_evidence_lifecycle_fusion.py` 绑定 r7.14、r7.15、5 秒 risk-profile 退出路由及其 1 秒 gap 负控。所有输入必须有匹配 sidecar、相同 package root，并保持独立实验方向隔离。
- 决策不依赖全局绝对 scene threshold：可信 clear 基线 + 归一化上升 `>=.05` 才开事件；可信 risk 基线 + 归一化下降 `>=.05` 可直接关事件；`0–.05` 的弱下降必须同时命中同 source、同前后 episode 的因果语义退出边界；冷启动、零变化、缺少互证或证据冲突统一输出 `uncertain`。
- 现有 7 个转移 `7/7`：5 个开事件、1 个强下降关事件、1 个弱下降互证关事件。4 个 fail-closed 控制 `4/4`，证明 1 秒 gap 下的弱下降、语义缺失单独出现、上升/退出冲突和无可信参考都不会错误清空。
- 报告 SHA256 `b0f0a4c46bd2a0de6df6ba9a859811280c594898017e47aa98f2d21172acddc7`。`.05` 来自 r7.14 的事后 stress grid，不能称为前瞻校准；报告因此同时记录 `retrospective_acceptance=true` 与 `prospective_acceptance=false`。

结论：当前最合理的中期 head 已从抽象方向收敛为“机制风险通道 + 可信相对基线 + 双证据关事件 + uncertain 冲突态”。但它还没有冻结后新来源挑战，仍不修改 Android `RiskEventTracker`，不授权训练校准、blind 或生产。

## r7.17 冻结双证据生命周期前瞻合同（2026-07-19）

- 合同 `public_video_dual_evidence_lifecycle_contract_r717.json` SHA256 `e7439ab3beac677ac913a0bb51155378ce2b2898c61dc4c38399c31235cd6175`。它在新来源视觉复核前冻结 r7.16 的 `.05` strong margin、动态 YOLO12n、静态注册残差、prompt-free 语义模型/输入尺寸/语义组、5 秒互证 gap、`clear/risk/uncertain` 状态和所有授权 false；并明确 r7.16 属于 post-hoc derivation，不能作为前瞻通过证据。post-clear 动态 guard 直接比较冻结 occupancy peak，不接受 feature report 注入 hazard verdict。
- 前瞻流程要求完整视频先按 1 秒采样生成不可变 feature report，且生成时 review window 必须未知。之后大模型只能在原时序中选择至少 3 个 pre-clear 样本、3 个 risk 样本和 6 个 post-clear 样本，不能修改任何特征值。review 模板 SHA256 `03923a78b3dd48bee49581d36f2fbe2ac8d2110a1e8f49c2fd3ee82742b48ffa`。
- `evaluate_public_video_dual_evidence_prospective_lifecycle.py` 从冻结特征计算 open、close 和 post-clear stability；同时验证来源库存、视频 SHA、视觉边界、完整无缺帧采样、合同/feature/inventory sidecar 和三项 fail-closed 控制。
- `extract_public_video_dual_evidence_features.py` 对完整视频按 `[0,duration)` 的 1 秒时间表做有界 batch 采样，只输出冻结 occupancy、注册 residual 与 semantic count；不接收 review window、label 或 hazard verdict，并复验视频及两份模型 SHA。
- 5 个提取器测试、7 个合同测试与 13 个验收器测试覆盖采样边界、冻结语义类、margin/语义组/冷启动漂移、授权抬升、feature report 替换、生成顺序倒置、硬剪辑、来源污染、弱下降缺少互证、视觉边界错位、缺帧、非法哈希、post-clear 重开，以及真实 sidecar `run()`/输出哈希路径。

当前没有生成真实 r7.17 验收结果，因为 r7.12 的 14 个本地视频中没有合同冻结后的独立正来源。r7.17 的意义是把下一条来源的成功标准提前固定，消除继续事后调规则的空间；它本身不打开训练、校准、blind、Android runtime、`RiskEventTracker` 或生产门。

工程 smoke 已在明确不合格的 40.117 秒 Hof 公交视频上真实执行：41 个一秒样本跨多个 batch 完成 YOLO12n occupancy、YOLOE semantic count 和 motion residual，feature report SHA256 `95f2bb6f8bc60f68622b4ecee57669b2850cde128bb39d49da8bc320b7b2a031`。随后使用已在合同前被审阅、且污染 r7.10/r7.11 的 smoke review 挑战验收器，程序在生命周期 scoring 前返回结构化 chronology rejection，目标 result 和 sidecar 均未创建。审计 SHA256 `9e0897599d2057411f1c606982213422b118878b9dc9f748bfc5190db9a9f59e`。该 smoke 只证明工程可运行且旧来源不能换壳，正来源门仍为 `0`。

## r7.18 首个真实独立行人前瞻挑战（2026-07-19）

- 网络恢复后新检查三条许可绑定候选。Commons Trubarjeva 在 CC BY 3.0 下可用，但原时序近似固定且没有 clear-risk-clear；Pexels 3874684 许可允许使用且为连续 POV，但施工背景从头持续到尾，车辆只在道路中经过，不能冒充行人风险离场。两条都在正式验收前 fail closed。
- YouTube `TVCX9tpaty8` 的 item metadata 明确给出 `Creative Commons Attribution license (reuse allowed)`。下载的 360p MP4 为 `31,679,092` bytes、SHA256 `551f8483e112b38afd0a91840fb56c8539cfcbe3a6475465677a342c695e63ec`；完整视频先冻结 457 个一秒特征样本，报告 SHA256 `9ff5ecbe0abba930463cc24c4679a2419a53364f0df472008df706c7743504e4`，之后才生成 5 秒全片概览和 300–405 秒逐秒复核。
- 大模型原时序复核确认无明显硬切，并选择 `340–345s` 开阔参考、`365–371s` 巨型混凝土管/开挖近场风险、`378–389s` 完整稳定人行道。来源库存审计首次通过独立正例 `1/1` 与行人正例 `1/1`，SHA256 `7847f19e6494dda6af56c4a4ee4f8b101664009d16044437b151090d57d3bb50`。
- 冻结 r7.17 验收仍失败。静态 residual range 从视觉 clear 的 `.3058160` 降到 risk 的 `.1055915`，open change `-.6547222`，`strong_open_passed=false`；close change `-.6958017` 正确关事件，稳定净空不重开，所有 fail-closed 控制通过。结果 SHA256 `633591eafd019e2e5d6ae91c63268bde666de4564465798d7364884dcd3aae96`。

结论：来源缺口已经真实闭合一次，但模型门仍未闭合。当前最强证据指向静态特征鲁棒性，而不是 head 优化：相机转动和 360p 压缩让视觉 clear 窗的注册残差范围高于真实管道侵入窗，head 对错误排序正确弃权。不得用改窗口、改 `.05`、SAM/ASAM 或事后校准挽救本次结果。下一轮必须先冻结新的运动不变静态特征，再用另一条独立来源前瞻验证；训练、校准、blind、Android runtime、`RiskEventTracker` 和生产授权保持关闭。汇总审计 SHA256 `3cd003c196de87b94d4b5afe209f247a54b3bf64f12a572ba8e27f31816520fe`。

## r7.19 冻结静态表征诊断（2026-07-19）

- background-normalized residual 只在 Rice 三段排序上有效，三组旧静态 matched pair 没有共同通过项；报告 SHA256 `7a15d0e0443b359c6db0b23a4c6dd1c2dda0e0b1c3a5f63af8436e848bc695c9`。
- 冻结 ADE20K SegFormer 的固定中心软自由空间在 `Chcne`/砂堆方向正确、`gie8` 反向，且把 Rice 工地前净空判得比管道风险更不可走；报告 SHA256 `26ef0ea50707d94a7be5e2ec6faeb6a35f571d0103c4bea32c8362b51fda4f8f`。
- 无阈值自适应路径修复了 Rice 的风险绕行偏移，但仍不能统一旧静态 pair；报告 SHA256 `642ad6b6ce6e266f76d0367d68896092b2bf463e584820b0a1f8a120abc24be7`。停止 mask/类别/阈值搜索，下一批数据按静态子机制建 matched episode；这些结果不授权训练、校准、blind、Android 或生产替换。
- 新来源恢复采集后，18 条 YouTube 候选因 item-level CC 字段为空未下载；两条许可明确的 Commons Ljubljana 工地视频在登记后只下载 360p，并于 5 秒概览中确认无行人前进和 clear-risk-clear，特征提取前即拒绝。triage SHA256 `d6ab3842d651ed3576fa55ae341c0754319e93edef08d4aaa03433377aee90e1`。
- 固定 16 维多通道风险轮廓（局部变化、绝对净空、路径占用、绕行偏移）后，与 163 维对象轨迹在同一 19 episode、同一 `source_id` 留一闭式 ridge 上比较。trajectory-only balanced accuracy `.7389`，profile-only `.4222`，fusion `.7389`；融合没有严格改善，pair-delta 方向仍不一致，Rice 轮廓投影的 open/close 也未通过。报告 SHA256 `d370192ca5ca81d1693eb26274e43daa859210251afd2983d1ead561be005db1`。线性特征门失败，故依合同跳过五组 prototype/bootstrap；这把当前根因进一步收敛为机制异质与监督合同不统一，而非 head 优化。
- 网络恢复后的 30 条 YouTube 逐项许可核验只得到一个明确 CC 候选 `SI7uinNg7jk`。它在预登记后下载 360p，但 10 秒概览显示多机位、多活动连续切镜，不满足单一 ego-pedestrian 生命周期，在特征提取前拒绝。该失败不降低许可或连续性门。

## r7.20 机制路由与 train-only 静态反事实（2026-07-19）

- 嵌套 leave-one-source-out 机制专家没有闭合根因。trajectory 的 unified/router/oracle/routed balanced accuracy 为 `.5833/.5833/.5000/.6667`，risk-profile 为 `.5833/.3333/.5833/.5000`，fusion 为 `.4167/.4167/.5000/.5833`；没有 oracle 或 observable routed gate 通过。报告 SHA256 `8c29ea04130b2af9a65a407c58d9a318586166b13a8d0ff8320eaf5c531f5f7a`。因此不能用简单 mixture-of-experts 代替表征修复。
- 新建 6 个真实父来源、3 个障碍族、6 对/12 图的照片级静态反事实集。它严格 `train_only`，只提供分类/pair-ranking 暂定监督；无 bbox、mask 或像素真值，父 source 留出时所有合成后代一并排除。GPT/VLM 多图复核 SHA256 `522c6a98b90370b42f21081f95740b2d68cac193d7ef0332bf4b3182eba953de`。
- provenance/response audit v2 通过来源、SHA、无泄漏、无像素标注与每族双来源门，SHA256 `2e7dcbcdb30278098aee2d33669a11b7b0d3b38a912d243d846560b168cd4b8d`。视觉合格但冻结 teacher 反向响应的 `vcz_temporary_barrier` 被保留为 hard counterexample。
- 冻结四通道 pair-delta 增广把三组真实静态排序从 `1/3` 改善到 `2/3`，Rice open 仍失败；SHA256 `eb2e574bc4c86d2d7aaa454c990c240dbc9346dc9062511f2610bd159d9578bd`。完整 450 维 ADE 语义 adapter 仅过真实 `1/3`，但 Rice open/close 通过；SHA256 `1d5dc9f2700c8cdf179e4a5bdd11574d6ffc02f215acbdab8f0490ea93a0d42b`。

结论：合成反事实已有跨父来源的训练信号，但现有 frozen representation/head 没有统一真实旧 pair 与 Rice。两条互补方向属于 post-hoc，不能融合后宣称通过。下一步是扩充 hard counterexamples 并训练受限表示 adapter；真实 source-isolated pair 与新前瞻来源闭合前，不启动生命周期 head、SAM/ASAM、Android 或生产替换。

固定单种子非线性 adapter 随后也未通过：4 个等权 positive-evidence unit、300 step、无超参搜索时，真实 source-isolated 静态排序仍为 `1/3`，Rice open 失败而 close 通过；SHA256 `b8630d4c08823b3bce026f9db77203e4a9d61c17c1e06b564e32f41ef1ffeb7c`。依预先门禁不运行五组 bootstrap。这排除了“只把线性 adapter 换成小 MLP”作为当前解法。

## r7.21–r7.23 DINO 区域表征、独立前瞻与多专家生命周期（2026-07-19）

- 逆向反事实 2 对通过 train-only、许可、哈希、无 bbox/mask 和父来源隔离审计，SHA256 `784adac3c5b3d707c2159ec29e13ef174307235ab56dbd495aec2f494f4fc943`。
- 固定 DINOv2-S 最后一层、五个预定义区域的 1920 维 pair probe 在三组旧真实静态 pair 与 Rice open/close 上全部通过，SHA256 `37f90be695d16cf3937a7c019cc8e2e8c04a5cb482489ee89aecb3b3fbcdfd5f`；五个固定 source-bootstrap 短跑也为 `5/5` 全过，SHA256 `69223aae8d1ff4d6273cf0acf4f37d7ab01b49dc6182054d72265f1e38fa7852`。这是看过 Rice 失败后的 retrospective gate，只授权冻结下一份前瞻合同。
- r7.22 DINO 合同 SHA256 `6b5924ab35b9866a4ec23994518e5316de1b7ccc1e127e2e79a5535392389845`。冻结后的 Japan 连续行人视频含右侧小交通锥，DINO open `-.0048568` 失败而 close `.0432368` 通过；结果 SHA256 `cde840bb604766d5dcd0c3f471e32d9db64c5e4b171c271113622d16c97a4995`，不得事后改窗或改方向。
- 更早冻结的 r7.11 彩色施工标志专家在同一视频上按原合同前瞻通过：风险窗激活率 `.8`、稳定净空 `.0`、终态 clear，结果 SHA256 `ad71ad76a2859611401e13fad7ebc1de80c1927d1dc466d60593fd5bfb190191`。这证明小型机制专家能覆盖通用静态表征的盲区，但不是 DINO 门通过。
- r7.23 原型采用“独立正证据 OR 开事件；所有打开通道分别确认 close；冲突保持 present/uncertain；缺失不能单独清空”，原型 SHA256 `ed147893f8ee49f4b9cc3648aa7ec0d6227d8671e94ad6e711ba3327c0218f63`。在任何新来源登记前纠正误标的 UTC 字段后，冻结合同 SHA256 `4918fe3e6053a3dd7b13d200ea219dfbdfee2a6e20dbc7be67fefc5b18317071`，Japan 不得计入下一次融合前瞻验收。
- Igriska/Erjavceva 为固定街对面施工，Norrköping 为河岸平移/转动且无前向障碍，两者均在完整特征冻结后经原时序复核拒绝，没有伪造正例。

当前结论：线性/短跑阶段证明“合适区域特征可分”，但单一通用通道仍漏掉小而靠边的施工标志；主架构应是风险轮廓 + 多专家生命周期，像素分割和距离场仅辅助。r7.23 仍需一条合同冻结后的新许可连续来源；在其通过前，训练、校准、blind、Android runtime、`RiskEventTracker`、生产替换和 SAM/ASAM 均不授权。

## r7.24 Matoaka 合同后负控（2026-07-19）

- Commons 连续行人视频在下载前绑定 r7.23 合同；视频 SHA256 `017f860e002c75d093206772800bd68cb1c19f226b74e4d5a933798916347821`。视觉审阅前分别冻结 DINO 全片特征 SHA256 `1fb898825505deedb8f611e6de62406a628e2b055a7e14a797646f90c6318a9a` 和彩色专家全片特征 SHA256 `1e526a7a2e155856559766008fab3ad8c4f75315cbb441db15ed2db5fa1e008f`。
- 178–188 秒的橙色交通锥在马路对面，步行者一侧始终净空；原时序 review SHA256 `0b0484c810c530076119f523c235e8342f63b5013767d36b2a3d03ab2a969aa2`。720–840 秒的机械/堆放物也因隔着排水沟而拒绝。
- 冻结合同仍会假开事件：DINO open `+.0375678`、close `-.0551055`；彩色事件区间 `174000–192000ms` 与负控重叠。多专家负控结果 SHA256 `938c03fa3ac03a341ecb3b0885946496ad5ebe67268586f2c7b55ccf7f0ac0ef`，`negative_control_passed=false`。

因此 r7.23 不是可晋级架构。无条件 OR 把“通用场景变化”和“远处施工标志存在”误当成行人路径风险；下一版必须要求机制证据同时具有路径相关性和接近/侵入时序。固定中心 nearfield flag 虽能过滤 Matoaka，但也过滤 Japan 真正小锥桶，不能作为事后补丁。继续保持像素/距离场辅助、SAM/ASAM 第三顺位，所有训练与端侧授权关闭。

## r7.25–r7.26 径向接近合同与合同后负控（2026-07-19）

- 回顾性径向接近诊断在 Japan 正例通过，在 Matoaka 5 个假事件和历史车辆事件上全部拒绝；报告 SHA256 `8881048ddd5f3fab4f10fc56eba435d0bc572bf3b039d65b6f988cebae70efcc`。该规则要求至少 5 个接受样本、首末各 3 个中位数、纵向前进至少 `.05` 且大于水平扫动，并要求正面积增长。
- 新来源登记前冻结 r7.25 合同 SHA256 `6746957ce89d4f133c21802632ba4c5c972b01d0d61e34895b11b12392e9a8ce`。DINO 只可辅助；开事件必须同时满足原 r7.11 颜色证据和径向接近，缺失不能开事件或直接证明净空。
- Bramwell 与 Stegna 两条 CC BY 3.0 连续行人视频在下载/审阅前登记。2163 个全片逐秒特征先冻结，SHA256 `efd3127df05aca4a280dacbdb92d2e47b083989f1845918bc941cb63b827e160`；冻结候选报告为 0 个事件，SHA256 `576fe7da8cdd213e2d4ee71ef6c4dff7fb56f9cdd8907bba563aed7bdc222c9a`。
- 后验视觉复核将 Bramwell `375–385s` 固定为路肩锥桶近距离旁路负例：通行路面保持开放，短暂识别由转身/横扫主导。负控结果通过，SHA256 `9c3910fc5467a0890831c721eeba3eda9371164ca121c2d11658f9ca58ec2752`。Stegna 只作拥挤动态场景 context，未形成可评分的施工标志风险生命周期，不计门禁信用。

当前训练就绪结论仍为 **false**：新增证据只支持一个合同后负例，没有合同后独立正例，召回门未闭合。大模型视觉判断仍为 provisional silver、非人工真值；不得授权训练、校准、blind、Android runtime、`RiskEventTracker` 或生产替换。

## r7.27 Tai Wo 来源拒绝（2026-07-19）

- CC0 短片 `Roadwork with no dust control 02` 在下载和视觉复核前绑定 r7.25；视频 SHA256 `4e6b47fc218a7e96cfa601309315dd73462957516fb1a07d15343d0baf0ba252`。10 个逐秒特征先冻结，SHA256 `da2202521c783a37d9a3babc6ca70709e3a37050651b8aa7196b016c5bafc833`；0 候选报告 SHA256 `1d87766d92bc69b30bb6098f5ac7b5aaf6198e42b359ff160839eb34a00fc0f1`。

## r7.28–r7.29 独立来源与首次生命周期假清除（2026-07-19）

- r7.28 Rice Street CC BY 连续 POV 在视觉复核前冻结视频 `551f8483...`、颜色特征 `1baa692d...` 和 0 候选 `4acf7972...`。施工物主要邻接通道，后段存在路线选择歧义，故只作 context，不强制赋正负事件标签。
- r7.29 Edmonton/Kampala 在 r7.25 后登记。组合特征 SHA256 `ea351ffe25201fb0e2ef6fa99d6e5cb6261784d5e429cc0fde673f519bd86ade`；唯一冻结候选报告 SHA256 `0c350a4659485c9df490e3937fbb586c303d6a6da5aee46209aa7d123394c78c`。
- Edmonton 的入口 671 秒正确，但 r7.25 在 697 秒清除；逐秒大模型银标显示同一窄施工走廊风险持续到约 735 秒，形成 38 秒假清除。失败报告 SHA256 `d6230f066e0684c1f65a88b2c6564071e08376011263d776fc71ed498bc7ca42`。这是生命周期 fragmentation fail，不能用后验合并回写为 pass。

## r7.30–r7.32 非对称生命周期修复状态（2026-07-19）

- r7.30 回顾性 `5–15` 秒缺失扫描的最小通过值为 9，诊断 SHA256 `a1070be1326b455c2ae83789c2254ada97c2465eedc4523c4633888d6e1528ca`。随后在新来源前冻结合同 SHA256 `b692f72758d7f34021a4dd02dd65371fa24a9ddc7faa48b821fb6003dd158169`：径向只负责 entry，开后颜色证据恢复续同一事件且不重复提醒，9 秒连续缺失才 clear。
- r7.31 Dallas 独立连续步行负控先冻 321 个特征 `e6777a75...` 和 0 候选 `6c530633...`，再复核为“牌在草地、锥桶在道路边缘、步道清晰”；负控通过，结果 SHA256 `0186c526aef0b728d08f1aedaa6e46f9c5e4e73cbba52a9109688383d0a5fa5a`。
- r7.32 `More cones, barriers and lights.` 在特征/0 候选冻结后确认是车库对象演示，不是行人走廊；拒绝报告 SHA256 `264d0d6f6f9628a78d11a96a2eb446046281a568ac8b96600193f4f519c9f08f`。
- r7.33 Cape Town CC BY 连续窗口先冻结特征 `968082ec...` 与 0 候选 `38c550e9...`，再把固定混凝土柱和宽广场旁路锥桶复核为无走廊风险；负控通过，结果 SHA256 `eafef19730cf4e62774e35063f1677300f5a6913ebd5405f241085c11aee8556`。
- 训练就绪仍为 **否**：r7.30 缺少“真实径向 entry 后迅速视觉净空”的独立负压力，也尚无第二条冻结后正例。大模型复核是 provisional silver，不是人工真值；训练、校准、blind、Android runtime 和生产替换全关闭。

## r7.34 Jakarta 高密度负控与局部短窗诊断（2026-07-19）

- CC BY Jakarta `08:00–09:20` 在下载/画面复核前登记。80 秒片段 SHA256 `f2a0d3ec...84fdb`；冻结特征包含 221 个锥桶检测，SHA256 `ab5b9cba...34b4b`，冻结径向候选仍为 0，SHA256 `988b87ea...4735`。
- 逐秒大模型银标确认锥桶/水马沿开放弯道边界排列而不侵入行进路径；负控通过，结果 SHA256 `3ef05cb89c6df9cc939fa6370f673b96c23c330b484bdfc5518d214f8ff2ee3b`。
- 5/7/9/12 接受样本的局部滑窗诊断会在 Jakarta 全部误开，并在其他既有负控上误开；报告 SHA256 `a410ef8e...9fb9`。它证明缺失的是路径占用表示，不是简单的事件窗长度。该诊断不授权新合同、训练或端侧改动。

## r7.35–r7.36 路径关系表示与新来源状态（2026-07-19）

- r7.35 以透明锥桶资产和确定性合成构造 3 组等数量 Jakarta path-relation pair：clear/risk 各 4 个相同锥桶，仅横向路线关系变化，干预掩码外像素变化均为 0。生成报告 `7a1297a6...70671`，manifest `74443a9e...f6e4`，只准 train-only 表示诊断。
- 冻结 DINO 区域方向在合成留一、镜像、Japan、Edmonton 和 Cape Town 上方向正确，但把 Jakarta 开放高密度边界投成 `+.00882` 风险；严格门失败，报告 SHA256 `a4494f81b9cccca082bb4c65ac5b34ff3d8a45d50453f51058848745e85b7d33`。缺口从“是否侵入”推进为“侵入后剩余净宽/绕行余量”。
- r7.36 Commons Trubarjeva 13 秒来源先冻特征 `aad2aec4...160d8` 与 0 候选 `cd67d219...c2d9`，后验画面没有文字描述暗示的 risk-to-clear 生命周期；来源拒绝 SHA256 `70730ac7...edfc`，不计门禁信用。
- 训练就绪仍为 **否**。没有新的独立快速净空正压力通过；Android、`RiskEventTracker`、校准、blind 和生产模型均不变。

## r7.37 Tampere 元数据/视角错配（2026-07-19）

- CC BY 3.0 条目文字明确写有行人通道被施工压窄，但先登记、下载并冻结全片后，138 个一秒采样与 192 个检测仍产生 0 个 r7.25 径向候选。视频 SHA256 `7f930a13...f47f850`，特征 SHA256 `18d85399...30b2b`，候选 SHA256 `e6918d21...36d99`。
- 后验概览显示固定路边机位横扫公交、汽车和围栏，不是连续行人第一视角；拒绝 SHA256 `6ce7687f...fe74a`，不计正负门信用。训练就绪仍为 **否**；下一轮来源检索必须同时满足步行/POV 与施工/人行道受阻元数据。

## r7.38–r7.41 障碍感知净宽与距离场诊断（2026-07-19）

- r7.38 的 SegFormer argmax 连通路线在 clear 场景也频繁断裂，四组真实 delta 均为 0（`7e5a72b1...d15d5`）；r7.39 的软可走概率×净距恢复连通，却把 Japan/Edmonton 风险判得更宽并误伤 Cape Town（`c4e6ca6d...e5861`）。
- r7.40 自适应中心线距离场保留 Japan/Edmonton 正方向，却同时误伤 Jakarta/Cape Town，合成 pair 仍并列（`cba812f7...ce73d`）。只做一次物理尺度纠正的 r7.41 仍失败：Japan 反向、两个负例假收窄，SHA256 `7c906401...ba03`。
- 因此距离场明确降为 auxiliary，不能升级为主风险分数，也不再搜索膨胀或阈值。训练就绪仍为 **否**；下一诊断是带正负 prototype 的 source-isolated DINO 线性 probe。

## r7.42 DINO 正负 prototype 根因门（2026-07-19）

- 零参数 source-isolated prototype 在合成后代/真实 Jakarta 父 source 联动留出的前提下，正例 `5/5`，负例仅 `2/5`，balanced accuracy `.70`；报告 SHA256 `4fd55dfc...d249f`。三个假正是 Jakarta 边界、Bramwell 转弯路肩锥桶和 Dallas 路缘锥桶。
- 这说明当前瓶颈仍是 feature 对相机转向/场景变化与 ego-path 侵入的混淆，不是 head 初始化；五组 bootstrap 被代码门禁跳过。训练就绪仍为 **否**，SAM/ASAM、Android 和生产模型保持不变。

## r7.43 clear-drift nuisance 投影（2026-07-19）

- 唯一 OFAT 是把 real marker-clear DINO delta 中与同一 clear 窗前后漂移平行的分量投影掉；其余样本、fold、prototype 和阈值完全不变。负例召回从 `2/5` 升到 `3/5`，balanced accuracy `.80`，SHA256 `738d0ce6...35ca0`。
- Bramwell 转弯假正被修复，但 Jakarta 边界和 Dallas 路缘锥桶仍失败。因此相机漂移只是部分根因，五组 bootstrap 仍关闭；下一步必须增加跨父来源的等数量路径关系反事实，而不是继续改投影。

## r7.44–r7.46 多来源路径反事实与位置敏感 DINO（2026-07-19）

- r7.44 在 Bramwell/Dallas 两个独立父来源上建立 2 对、4 张 train-only 等数量反事实；同锥桶只横移，父像素掩码外不变。接受版 generation/review SHA256 分别为 `ac294ee6...f59e96`、`b9963c76...bd08a`。来源后代现在按各自 `parent_source_id` 与真实样本联动留出。
- r7.45 沿用 r7.43 clear-drift、正负 prototype 与 0 阈值，加入两组后 balanced accuracy 为 `.3714`，正召回 `1/7`；SHA256 `9be97362...bb422`。这是方向冲突，不授权五组 bootstrap。
- r7.46 只把特征改成固定 `4×4` DINO patch 网格以保留粗空间位置，结果 balanced accuracy `.3429`，SHA256 `7864be42...aac0b`。因此失败不能归因于区域均值；跨场景 DINO 表观特征本身没有统一路径侵入方向。
- 训练就绪仍为 **否**。停止 DINO pooling/投影/阈值搜索；下一候选必须显式输入障碍位置与预测 ego 路线关系，生命周期头、SAM/ASAM、Android 和生产替换继续等待该表征门闭合。

## r7.47–r7.48 显式 ego-route 关系与事件生命周期组合（2026-07-19）

- r7.47 在 marker 掩码内恢复潜在可走支持后再冻结 ego route，并测 route-to-obstacle q10 距离；正例 `2/2`，负例 `2/5`，balanced accuracy `.70`，合成原图 `3/5`、镜像 `5/5`。SHA256 `6a72165c...0f87e`。单帧静态几何仍不能识别转向、宽前场和边界路线意图，不能成为主 head。
- r7.48 采用事件级组合：径向接近和相对路线关系共同开事件，颜色证据/9 秒缺失管理 present→uncertain→clear。现有 2 正/5 负得到 balanced accuracy `1.0`，Edmonton 生命周期桥接通过，SHA256 `11f75c26...a8ecb1`；分割和距离场明确降为 auxiliary。
- 完整门仍未通过，因为五个负控均没有真实冻结径向入口，不能压力测试路线关系 veto；Japan 完整 r7.30 生命周期也未验证。恢复网络后的许可/连续性检索没有新增合格来源，审计 `e8e24451...a3382d`。
- 训练就绪仍为 **否**。下一硬证据是至少一个独立真实“径向接近但安全侧向”负事件，以及 Japan 的因果全生命周期重放；在此之前不运行五组 bootstrap、SAM/ASAM 或 Android 修改。

## r7.49 Japan 因果生命周期重放（2026-07-19）

- 每个时刻只用 prefix 样本重算 r7.25；路线基线来自冻结 `2–7s` pre-risk clear，r7.30 的 9 秒缺失与单次提醒保持不变。滚动径向和路线关系最早在 `8s` 联合通过。
- 冻结风险窗口从 `10s` 开始，因此提醒提前 `2s`，entry timing 门失败；事件仍覆盖完整风险并在 `22s` 正确 clear，且只提醒一次。SHA256 `95ce8201...01d56`。
- r7.48 的完美事件身份指标不能证明生命周期完成。训练就绪仍为 **否**；下一合同必须在新来源前定义可接受提前量或直接监督 time-to-contact/剩余宽度，禁止事后改风险窗或 relation 阈值。

## r7.50 新来源专用的前瞻提醒时序合同（2026-07-19）

- 冻结合同 SHA256 `9b3c9fb4...41067`，固定验收区间为 `material_risk_onset - 3000ms <= reminder <= latest_useful_reminder`。3 秒提前量是在 r7.49 Japan 失败后提出，故 Japan 继续保留失败且不得计入 r7.50。
- 前瞻正来源必须在下载/视觉复核前登记并冻结全片特征与候选；视觉复核需独立给出 material onset、最晚有效提醒和稳定 post-clear 窗。GPT/VLM 结论仅是 provisional silver。
- 还必须有不同真实来源的 true-radial safe-lateral 负控：冻结 r7.25 确实开出径向入口，但冻结 r7.47 路线关系必须把它否决。合成或 GPT-only 例不能补足此门。
- 训练就绪仍为 **否**。合同不授权训练、bootstrap、SAM/ASAM、校准、blind、Android runtime 或生产替换；下一步只允许按合同寻找和冻结新来源。

## r7.51 来源去重、径向筛查与 London 正例（2026-07-19）

- Pexels 3874684 被 item-ID 谱系门识别为 r7.17 已用来源，不允许不同转码冒充新样本。Pexels 5234995 是新条目，但全片 25 个冻结样本产生 0 个 r7.25 候选，故在视觉复核前失去 true-radial 负控资格。
- 新注册的 Commons London POPtravel 为 CC BY 3.0、55 分钟连续第一视角。240p SHA256 `ee68ca32...82836`；3301 个逐秒特征 SHA256 `81e5317e...83680`，唯一冻结径向候选为 `2678–2687s`（`088f37bb...5102b`）。
- 候选后验大模型银标复核把它判为正角色而非安全侧向负控：红色锥/隔离桩进入下方中心路线。固定 onset `2681s`、最晚提醒 `2684s`、风险结束 `2687s`、稳定净空 `2688–2699s`。
- 冻结路线增量 `+0.953596`；`2678s` 的唯一提醒正好处于 onset 前 3 秒边界，`2696s` 在九次缺失后 clear，所有正例时序检查通过。报告 SHA256 `7043eeaf...1eb8c`。
- 训练就绪仍为 **否**：独立正例已补齐，但不同真实来源的 true-radial safe-lateral 负控及 route veto 仍缺失。禁止用 London 同源窗口、Pexels 零候选或合成/GPT-only 例替代。

## r7.52 Ulm true-radial safe-lateral 压力测试（2026-07-19）

- Maribor 完整冻结 2100 个逐秒样本和 215 个目标检测后仍为 0 个 r7.25 候选，故预视觉拒绝。Ulm 则在注册后的 2177 个逐秒样本中冻结出 3 个径向事件；视频、特征、候选 SHA256 分别为 `67efb35b...caa48`、`7053981f...18cf`、`bf1c1fc6...a3b9`。
- 原时序大模型复核把 `1504–1510s` 判为 provisional true-radial safe-lateral：道路两侧红白路桩持续放大，但中央缺口开放、相机直接穿过。另两段因施工围挡和实际转向重叠被拒绝。复核计划在看图前冻结，SHA256 `81ca66c6...e6c1`。
- 冻结 r7.47 路线关系给出 clear median `0`、marker median `.928303`、delta `+.928303`，没有执行所需 veto；失败报告 SHA256 `cb02a1c0...d9ea1`。这说明现有 SegFormer adaptive-route + marker-distance 表征仍会把两侧边界当作 ego-route 侵入。
- 训练就绪仍为 **否**。不允许事后调阈值或几何规则回救 Ulm；prototype/bootstrap、SAM/ASAM、Android、校准、blind 和生产模型保持关闭。下一表征实验必须把可通行中央缺口/未来路径连续性作为显式输入，并把 Ulm 固定为外部回归压力样本。

## r7.53 未来帧 ego-trace 离线教师（2026-07-19）

- 该合同在 Ulm 失败后冻结，只能做 retrospective representation diagnostic。固定未来 `1/2/3s` 的 ORB+RANSAC 单应性，把未来底部中心锚点反投影到当前帧，再检查冻结 marker 是否占用实际未来轨迹；不允许参数搜索。
- London/Ulm 有效帧率均为 `1.0`。Ulm mean/median intrusion 均为 `0`；London mean 为 `.266667`，但预设 median 仍为 `0`，所以方向门失败，报告 SHA256 `fc9f4b0a...362f3`。
- 该信号最多作为无人工标注的离线路线辅助监督来源，不能作为主风险分数或 prospective gate。训练就绪仍为 **否**；不把聚合器从 median 事后改成 mean，不运行 bootstrap、SAM/ASAM 或 Android 修改。

## r7.54–r7.60 未来路线教师与蒸馏根因（2026-07-19）

- r7.54 稀疏 ORB 教师在有效帧上已完全分离 3 正/6 负，但 Japan 覆盖仅 `.25`；r7.55 固定换成无匹配阈值的 DIS dense future flow 后，9 个事件覆盖率均为 `1.0`，最弱正例 Japan `.0833` 仍高于最强负例 Ulm `.0476`，完整诊断门通过（`5b097322...3eea6`）。未来实际路线是目前首个能处理 Ulm 双侧路桩的跨来源变量。
- r7.56 仅用过去光流的因果恒速外推失败（`42d3f616...e91a0`）；r7.57 手工当前/过去特征和 r7.58 冻结 DINO 全局特征虽分别取得 teacher-active AUROC `.8427/.8116`，但都无法事件级分离（`923f4583...50af`、`744e27ba...78f84`）。
- r7.59 合并空间路线距离场在整来源留一下达到 pixel AUROC `.9311`；r7.60 三 horizon heatmap 仍有 `.9161`，说明 auxiliary target 可学。但 marker-overlap/argmax 事件 readout 均未完全分离，尤其 Japan/London 与 Ulm；报告分别为 `547f087c...30471`、`c4571611...f401c`。
- 因此不再继续换 ridge/head/readout。当前可执行方向是用全部许可连续视频自动生成更多 `causal clip -> future 1/2/3s route fields`，再训练专门的时序路线辅助 head，并以完整来源留出验证。该工作不需要人工事件标注，但仍不产生事件真值、Android 或生产授权。

## r7.61 自动时序路线辅助 manifest（2026-07-19）

- 冻结合同只使用许可/特征哈希已绑定的连续来源，要求当前有 marker 检测、前后各 3 秒可用，每来源最多 128 条确定性均匀样本；风险/事件标签、人工和 GPT/VLM 均不参与 target 生成。
- DIS future teacher 生成 1/2/3 秒路线锚点。manifest 共 `753` 条、`10` 个 source，SHA256 `05424d63...8428`；全部 item 唯一、event label 为 null、三锚点完整。审计报告 SHA256 `86a186eb...d6e6`。
- 这使下一轮可以合法运行 source-isolated temporal route auxiliary train-only prototype，但仍不授权风险事件训练或端侧替换。应先验证完整空间 past-flow/短 clip 表征能否在 held-out source 上恢复未来 route field，再讨论五组 seed 或端侧轻量化。

## r7.62–r7.64 专用 temporal route head（2026-07-19）

- r7.62 的固定 43 通道时序输入和 61,955 参数卷积 head 在 753 条 marker-only 样本上取得 route-field AUROC `.91058`，但 localization `.1222`，事件分离失败（`98ec998a...902cc`）。
- r7.63 将自动数据扩展为 2,102 条全连续因果帧、10 个完整来源留出；manifest `59923fbc...e0e`，事件标签仍全部为 null。r7.64 仅替换数据后 AUROC `.91974`、MAE `.0420`，但 localization `.11787`，Japan `0`、London `.10`、Ulm `.14`（`eaa3c8ef...aa0c`）。
- 结论是更多无需人工的未来路线辅助数据能改善像素级 teacher 拟合，却没有闭合跨来源事件路线定位；五组 seed、风险训练和端侧替换继续关闭。

## r7.65–r7.66 冻结相对峰值风险轮廓（2026-07-19）

- r7.65 只读审计发现，只有 marker 区域相对全局峰值比在旧 3 正/6 负事件上完全分离；它是事后 readout，报告 `f4f535db...a9cf`，不能作为 blind 或历史晋级证据。
- r7.66 在新来源前冻结固定阈值 `.68`、半物高 marker 扩张、r7.25 入口与 r7.30 生命周期；合同 `73076ff9...ecf3`，冻结权重 `18690367...7901`。验收必须由不同新视频哈希提供正事件和 true-radial safe-lateral 负例，且旧来源 ID/哈希全部代码拒绝。
- 当前只允许 strictly-offline diagnostic。正例除分数外还必须通过固定提醒窗口、风险持续、九帧清除和同事件一次提醒；未闭合前训练、校准、blind、Android、生产和 SAM/ASAM 均不授权。

## r7.66 第一轮新来源筛选（2026-07-19）

- Spiegelgasse（CC0）121 个逐秒样本得到 0 个冻结径向候选，按预登记在看帧前拒绝。
- Alicante（CC BY 3.0、全视频新 SHA）共冻结 3,781 个逐秒样本。前 20 分钟 0 候选；顺序搜索未看过的剩余区间得到 `2408–2424s` 唯一候选。候选后复核发现是餐厅露台/楼梯/栏杆的 barricade 误检，并无真实施工 marker，因此在 r7.66 路线分数前拒绝，不能记为负控。
- Pexels 2980886 按 item ID 和 Pexels License 先登记后下载，但时长只有 `3.47s`，无法满足 r7.25 至少 5 个一秒接受样本，在特征/视觉前拒绝。
- 这一轮没有合格的新正事件或 true-radial safe-lateral 负例；r7.66 仍待外部来源验证，所有训练和端侧权限继续关闭。

## r7.66 第二轮 Bristol 全片预视觉拒绝（2026-07-19）

- 在下载或看帧前登记 Commons 上已许可复核的 CC BY 3.0 POPtravel Bristol 步行视频；固定 240p 视频 SHA256 为 `1d2d74adab3023ea8f9abbb48f31d683f3600afc8d1641f28ac44df60399dc76`。
- 预登记前 20 分钟得到 1,200 个一秒样本、66 个 traffic-cone 和 188 个总目标检测；合同绑定特征 SHA256 `cfd6dbf0...78ae`，冻结 r7.25 候选为 0（`f8aac721...6cfb`）。
- 仍未看过的后半段被单独登记为自适应 proposal acquisition，明确不给 blind/calibration 信用。2,161 个样本含 87 个 cone、344 个总目标检测（`91b2f6cb...40a4`），径向候选仍为 0（`828b8186...7e0e`）。ffprobe 与 OpenCV 的 7 ms 时长口径差在采样前机械收紧并单独记录，没有改模型或阈值。
- 全片 3,361 个样本、532 个目标检测没有冻结径向事件，因此在任何画面复核和 r7.66 评分前拒绝（`92a0ccf7...ff88`）。Bristol 不获得正例或 true-radial safe-lateral 负例信用；五组、训练、SAM/ASAM、Android 与生产门继续关闭。

## r7.66–r7.69 Bangkok 压力对与 head 根因闭环（2026-07-19）

- 新 Bangkok Modern Center 视频在候选前完成许可/哈希登记。冻结 `300–311s` 为真实径向 safe-lateral provisional 负例、`328–339s` 为路线侵入 provisional 正例。r7.66 分别为 `.80168/.87608`，安全负例越过 `.68`，相对峰值合同被证伪；正例单事件生命周期通过不改变总失败。
- r7.55 future teacher 在同一对上为 `.11111/.33333`，表明离线路线 target 保留正确方向。r7.67a 直接对 marker-conditioned 43 通道因果特征做 132 维确定性 ridge，10-source LOSO pooled AUROC `.883995`，Bangkok 旁路 margin `.15003`（`e4ae6777...bc12`）。这只证明条件化关系信号存在，不是独立前瞻成绩。
- 更严格的 r7.68a source-class bootstrap/source-macro 五组为 `0/5` 通过；prototype-only balanced 中位数 `.8534`，优化后 `.7245`，主要失败在正类 recall `.39–.56`。连续命中比例 soft target 与 fixed binary active 决策不一致，禁止通过搜索阈值回救（`3e17d71a...f246`）。
- r7.69 配对距离场辅助 A/B 虽把 distance MAE 从常数约 `.40` 降到 `.20–.27`，主分类 balanced 中位增量仅 `+.00019`，门失败（`8edd27a4...d59a`）。因此训练就绪仍为 **否**；不保留距离辅助，不启动 SAM/ASAM，不修改 Android/默认模型。
- r7.70 已把 Bangkok 两段物化为各 12 帧、共 24 个唯一哈希的真实同源 matched contrast，manifest `90287f20...2612`。它只增加 representation-training candidate 覆盖，必须 parent-source 联动隔离，不能让训练就绪转为“是”。

## r7.71–r7.77 成对排序、target 审计与因果生命周期（2026-07-19）

- r7.71 最近时刻同来源 pair-ranking 通过：七个混合来源中位/最弱 AUROC `.85/.67385`，pair ordering `.792857`（`caa37275...c6d`）。r7.72 五个优化头单跑可用，但优化中位 `.84889` 比 bootstrap prototype `.90833` 低 `.05944`，稳定性门失败（`be949071...cf7`）。head 优化不是当前优先项。
- r7.73 的零训练 prototype 生命周期在 Bangkok safe-lateral `304s` 假开，positive 到 `337s` 才开，晚于 `336s` 最晚有效提醒（`1615d161...b25`）。r7.74 几何匹配和 r7.75 几何残差化分别降至中位 AUROC `.7733/.675`，均停止（`48914904...6f7`、`7c03f19d...a2f`）。
- 审计确认旧 target 将任一未来 horizon 命中视作 active。r7.76 冻结改为至少 `2/3` horizon 命中后，中位/最弱 AUROC `.96774/.71959`、pair ordering `.94231`（`7b2f5d89...9d2`），离线特征可分性明显成立。
- r7.77 仅替换 target 后，Bangkok safe/positive 仍在 `304s/337s` 开启（`c18cff4e...db1`）。因此训练就绪仍为 **否**：离线多数时域 target 可分，但当前因果输入没有在转向发生前表达路线选择。继续收集独立真实 matched-radial episode；不调 lifecycle 阈值，不启动 SAM/ASAM，不修改 Android 或默认模型。

## r7.78–r7.78a Düsseldorf 独立 safe-lateral 压力（2026-07-19）

- 预登记并冻结 Commons Düsseldorf 视频前 20 分钟：1,200 帧、85 cone、223 总目标，两个径向候选（`529f29b5...fbd4`）。大模型原时序 provisional review 拒绝 `117–127s` 的街角转向混杂，把 `900–910s` 固定为真实径向但始终位于开放路线左侧的 safe-lateral 负例。
- r7.66 事件分数 `.771165 > .68`，未能 veto（`19ba400d...19d4`）。r7.78 首跑在任何 prototype 分数前因事件内两个 detector gap fail closed；r7.78a 预先明确只取 9 个冻结接受样本并核对计数。
- r7.78a 多数 target 零训练 prototype 仍在 `906s` 打开，峰值相对分数 `2.6933`（`ab880466...54af`）。这是一条独立来源外部诊断，不是 prospective 晋级证据；它再次证明离线可分不等于因果生命周期可用。
- 训练就绪保持 **否**。需要新的独立 matched-radial episode 与显式 causal route-intent/turning supervision；禁止用 threshold、gap、baseline、optimizer 或 SAM/ASAM 搜索回救。

## r7.79–r7.81c 因果根因与事件角色 probe（2026-07-19）

- r7.79 的固定低维 waypoint ridge 仅把 localization 从 `.11787` 改到 `.11583`，提升 `.00204`，且事件分离失败（`cc54e30...81e6`）。这否定“只是 heatmap readout 太扩散”的解释。
- r7.80 的两级语义能让 positive 在最晚窗口 `336s` 升级，并避免 Düsseldorf safe 升级，但 Bangkok safe 在 `307s` 仍假升级。r7.80a 修正 `confirmed_clear_timestamp_ms` 读取后确认三事件都能清除；总门仍为 false，因为安全事件误升级不是报告字段问题（`6e466af8...95e9`）。
- r7.81c 的事件级 route-role LOSO 只保留 marker-present 帧后有 106 帧、8 个事件/来源，事件 AUROC `0`、balanced accuracy `.2`、positive recall `0`（`3f1a69ff...beb3`）。数据量与机制匹配不足，当前 head 形成来源捷径；该系列只作失败诊断，不计 prospective 晋级。
- 因此训练就绪仍为 **否**。可以继续扩展 provisional event-role 候选，但必须按 parent source 隔离；不得翻转分数、搜索阈值、启动 SAM/ASAM 或修改端侧默认模型。

## r7.82–r7.84 Cologne/Cardiff 高召回扩源（2026-07-19）

- Cologne 前 40 分钟 2,400 个逐秒样本、190 个目标检测没有固定 r7.25 事件；Cardiff 全片 3,241 个样本、875 个目标检测也为 0。两者表明 detector proposal 充足不等于持续径向路径事件充足。
- 只作为 acquisition audit 的五样本局部 gate 在 Cardiff 找到 3 个窗口，而宽度 `7/9/12` 全为 0。先冻结复核窗口后再看原时序：两个黄色地面警示牌保留为 provisional path-intrusion/right-side-pass 正候选；路口多锥桶因拍摄者计划右转、非单目标轨迹而排除。
- 复核 JSON SHA256 `088bad8f...17a9`，明确记录 `canonical_r725_events_added=0`、`new_independent_sources=0`。Cardiff 两个保留项只能一起进入 train 或 holdout；不计人工真值、校准、blind 或生产证据。
- 数据层面的实际缺口已经更具体：缺的是多个独立来源、同一路径关系机制、原时序可判的 matched-radial episode，不是普通公开视频总帧数。训练、五组优化、SAM/ASAM、Android runtime 和默认模型继续关闭。

## r7.85 因果可行动性合同修正（2026-07-19）

- 冻结的在线状态只使用当前/过去 trace：`context_attention -> intervention_needed -> route_clear`。Bangkok 原 safe-lateral 事件在 `307s` 进入干预、`311s` 清除；正例在 `336s` 干预且持续；Düsseldorf 只需 context。三种冻结预期全部通过（`0120009d...07e5`）。
- 因此原先把 Bangkok 最终成功侧向通过当作 `no alert` 的做法存在因果标签泄漏：未来用户动作可能正是提醒应该促成的结果。训练 target 应改为在线 actionability，最终 safe pass 只能作为 response/lifecycle 属性。
- `blindassist_public_video_silver_labels_v3` 已把该边界变成验证规则：`candidate_no_alert` 只允许 `no_attention`；`context_only / intervention_then_route_clear / persistent_intervention` 必须保持 alert；`causal_evidence_basis` 必须是 `past_or_current_only`。
- 该修正消除了一个假负标签来源，但没有证明模型已经可部署。新独立来源、方向侧可通行证据、校准、blind 和同设备事件门仍未闭合；训练就绪保持 **否**。
## 2026-07-19 更新：可行动性重标与路线意图输入缺口

- r7.86 统一审计 12 个既有事件，发现旧 route-role 与 current/past-only actionability 不一致 `3/12=25%`；r7.89 扩展为 16 个事件、11 个来源，其中 4 个干预事件来自 3 个独立来源，数据覆盖门首次满足确定性 probe 的最低条件。
- r7.90 事件均值 readout 失败（AUROC `.0795`、干预召回 `0`）；r7.91 风险轮廓 + 冻结生命周期虽改善到 balanced `.5227`，仍未跨来源通过。当前 RGB/DINO/past-flow 不足以在实际转向前恢复路线选择。
- r7.93 完成 ADVIO-15 官方包的哈希和同步审计；r7.94/r7.95 在 `201 turn / 199 straight` 平衡样本上分别得到 AUROC `.4770/.4746`。原始轴与旋转不变 IMU 均不能预测尚未发生的自主转向。
- r7.96 的 current-only 转向确认负控也失败（AUROC `.3465`、balanced `.3878`）；未经 device-to-world/route 姿态对齐的手机 IMU 不能直接作为路线确认输入，Android sensor benchmark 授权保持关闭。
- readiness 仍为 **false**。下一次可授权的 OFAT 不是 head/optimizer，而是“显式 route-intent token + r7.91 risk profile”，且必须有独立来源事件验证。ADVIO 为 CC BY-NC 4.0，只允许隔离的非商用研究，不计生产训练、校准或 blind 信用。
- r7.97a 已完成该接口的 oracle 上限：16 个事件/11 个来源上 intervention recall `1.0`、context recall `.8333`、balanced `.9167`；12 份 feature report 与 13 个本地视频 SHA 全部重算匹配。这授权继续构建隔离的真实 route provider 接口，但不授权把 future-video teacher 用于 eval/runtime，也不改变总体 training readiness=false。

详细证据见 [PUBLIC_VISUAL_INERTIAL_ROUTE_INTENT_2026-07-19.md](PUBLIC_VISUAL_INERTIAL_ROUTE_INTENT_2026-07-19.md)。

## r7.99–r8.07 三态路线选择与转向候选复核（2026-07-19）

- r7.99 固定三态模板在既有事件上 balanced `1.0`，但 r7.99a 揭示全部 4 个 intervention 都是 `STRAIGHT`，LEFT/RIGHT 只有 context，完整 provider 门关闭。
- r8.00 的 5 个 mean-anchor-x 候选经 r8.02 全部拒绝。该 x 是 future lower-center 光流对应位置，不是方向标签；移动卡车、相机运动和 detector 假框会制造伪方向。
- r8.03 背景 yaw 搜索在 753 行上得到 13 LEFT/17 RIGHT；r8.05 确定性交叠缩到 3 段；r8.07 复核确认 2 段为平行施工边界、1 段为建筑误检，新增 LEFT/RIGHT intervention 为 `0/0`。
- readiness 保持 **false**。这不是继续训练 head、调扩张率或 SAM/ASAM 能补的缺口；需要独立于光流 x 的真实/外部 route choice，以及匹配障碍确实阻断该分支的连续事件。现有候选可作为困难 context 反例，不能作为事件真值或训练正标。

## r8.08–r8.09 许可优先来源审计（2026-07-19）

- 冻结的 Commons/Vimeo 定向查询没有形成新事件。Commons 唯一命中为警方执法/dashcam 语义误项，未下载；Vimeo Burwell 虽有条目级 CC BY 3.0 许可，但全片是剪辑工程新闻材料而非连续行人 POV。
- LEFT/RIGHT intervention 仍为 `0/0`，readiness 保持 **false**。许可和大模型复核都不能替代缺失的因果事件覆盖；五组训练、SAM/ASAM、Android、校准、blind 与生产门不变。

r8.10–r8.11 的 Internet Archive 许可过滤查询又返回 3 项历史档案误命中，全部在下载前拒绝；没有改变上述判断。

## r8.12–r8.18 路线条件合成诊断与距离场 OFAT（2026-07-19）

- 三个父来源的 train-only 路线反事实通过精确 mask/bbox 与全图 QA；通用 `static_obstacle` 类名版本因潜在漏标语义被拒，正式 v2 只标 `inserted_temporary_obstacle`。
- exact-field 线性 head balanced `1.0`，冻结 DINO 二值风险图路线 readout `.6249`；双障碍族 factorial 后仍 `.6332`，证明问题不是 head 或单一外观缺失。
- bbox 距离场辅助监督是唯一变化时，路线 readout 升至 `.9156`，固定 open 生命周期 `.9429`。距离场作为风险轮廓辅助监督获得继续进入真实 provisional 诊断的授权。
- 五组 prototype/bootstrap mean `.8774`、std `.0090`、worst seed `.8682`，因 mean 未达 `.90` 且 worst clear recall `.7971 < .80` 而失败；禁止事后改门或启动 SAM/ASAM。
- readiness 仍为 **false**：上述全部指标来自合成 train-only LOSO，不能替代真实事件、真实 provider、校准、blind、INT8 或同设备事件门。

## r8.19–r8.23 真实迁移、合成语义扩展与生命周期诊断（2026-07-19）

- r8.19 在 16 事件/11 来源上完成首个严格 source-LOSO 真实迁移：合成距离场的路线 readout balanced `.7083`、干预/上下文召回 `.75/.6667`，优于全局 `.5833` 与 r7.90 `.3182`，但冻结 context `.70` 门未过（`331a2f5d...fad60`）。
- r8.20 拼接全场背景后 balanced `.4583`，没有减少 context 误报；停止该特征路线（`ffe88d21...bf570`）。
- r8.21 的 traffic-cone 扩展含 108 图/324 路线样本、精确二值 alpha、81 个 composite 全图视觉 QA 与 243 个 composite 路线标签几何审计。它只增加 train-only 合成覆盖，不增加真实事件或 provider 信用。
- r8.22 只增加锥桶 asset family 后，真实 balanced `.5833`、干预 recall `.5`，比 r8.19 更差（`3f16e70b...691f1`）。禁止继续用未验证的合成类别堆叠回救。
- r8.23 的真实逐帧风险轮廓 + 固定 lifecycle 使用 218 帧（20 intervention、198 context/clear）和完整来源留一，事件 balanced `.5417`、context recall `.3333`；clear lifecycle `1/2`，总门失败（`6c098d1f...815b0`）。
- readiness 保持 **false**。已排除 head 优化、简单背景对比、单一合成语义缺口和事件均值聚合为主因。下一步只允许预注册的真实训练折局部风险监督 OFAT；不得调阈值、启动 SAM/ASAM、接 Android、做 calibration/blind 或替换默认模型。

## r8.24 训练折内真实 marker 距离场（2026-07-19）

- 每个完整 parent-source LOSO 折只使用训练来源 provisional bbox，测试来源 detection 从未进入 teacher 或 head。218 帧中 28 帧无检测，作为全零距离场；patch 权重按 near/far、来源、帧分层平衡。
- 全局 balanced `.6250`，路线 balanced `.5833`；路线干预/上下文召回 `.75/.4167`，比 r8.19 路线 `.7083` 退化 `.125`。报告 SHA256 `7750e172...d0013`。
- readiness 仍为 **false**。当前缺口不是 object localization，而是 route field 与局部视觉风险的可学习交互。下一 OFAT 只能是 source-heldout route-conditioned patch interaction；仍禁止阈值搜索、SAM/ASAM、Android、calibration、blind 和默认模型替换。

## r8.25 route-field × frozen patch interaction（2026-07-19）

- 冻结 DINO patch 经固定 seed-0 `384→32` 投影，连续 route polyline 场直接参与路线内/路线外/contrast pooling；同一函数的均匀场为负控，不使用 bbox、marker、obstacle-hit 或 lifecycle label。
- 均匀/路线 balanced `.4583/.5000`，路线 context/intervention recall `.75/.25`，相对 r8.19 退化 `.2083`。报告 SHA256 `ec1b3684...93b60`，全部预注册提升门失败。
- readiness 仍为 **false**，且不授权非线性 interaction head。当前最强证据仍是 r7.97a 的显式 route + detector geometry `.9167`；后续资源转向真实 route provider 与几何/lifecycle 验证，分割或风险场只作辅助监督。

## r8.26–r8.26a 显式路线端侧几何闭环（2026-07-20）

- r8.26 的 benchmark-only `route waypoints × detector bbox × lifecycle` 编译和 9 项真机测试通过，但全量逐帧 conformance 发现 59/654 anchor hit 与 30/218 frame score 不一致，首轮实现按门禁拒绝。
- 问题是将 normalized object height 同时作为 x/y 扩张；这不等价于 r7.97a 的像素高度扩张。r8.26a 改为先算 `margin_px`，再分别除以 frame width/height，并加入非正方形画面回归。
- 修复后 4 个离线单元测试、218 帧/654 锚点零不一致、SM-S9280/API 36 的 10 项 instrumentation 全通过；APK SHA256 `b543dd9d...62fe`。
- readiness 仍为 **false**。这只关闭了确定性几何实现风险，没有验证真实非未来 route provider、LEFT/RIGHT intervention、校准、blind、App/runtime 或生产门。模型训练不再是下一主项。

## r8.27–r8.27a Android 外部路线边界（2026-07-20）

- benchmark-only provider 已能解析带 provider/projection receipt、时效、置信度和 1/2/3 秒 waypoint 的 Android payload；risk-model/future-video/未来签发/过期/低置信/缺回执/畸形输入均 fail-closed。
- r8.27 因测试夹具返回类型错误在执行前编译失败；r8.27a 只修夹具后 build 成功，SM-S9280/API 36 上 provider 6 项加几何/生命周期 10 项为 `OK (16 tests)`。
- readiness 仍为 **false**。只授权继续寻找或实现隔离的真实 navigation/user-choice provider；不授权 App 接线、LEFT/RIGHT 覆盖声明、训练、校准、blind 或生产替换。

## r8.28–r8.28a 世界路线投影（2026-07-20）

- benchmark-only 投影器已固定 ENU→camera 旋转、针孔内参、回执时效/置信度、可见深度与画面范围门。r8.28 的 8 元素 identity 测试夹具被 `invalid_pose` 正确拒绝；r8.28a 只补齐第九个矩阵元素。
- SM-S9280/API 36 上投影 6 项加 provider/geometry/lifecycle 16 项为 `OK (22 tests)`。
- readiness 仍为 **false**：投影数学通过不等于真实 pose、camera calibration、navigation route 或 LEFT/RIGHT intervention 已验证，训练与 App/生产门不变。

## r8.29–r8.29d 真机投影输入能力（2026-07-20）

- SM-S9280/API 36 的两个后置 camera 均可推导针孔内参；r8.29d 无权限 context 中的精确 intrinsic/distortion null 后由 r8.30 证明是权限过滤，不能作为设备缺失结论。rotation vector 为 119 样本/2.454s、约 48Hz，四元数范数门通过。
- r8.29 编译错误与 r8.29a–c 收据导出错误均发生在能力结论之外并保留；r8.29d 通过 instrumentation status 得到 `OK (1 test)` 和完整收据。
- r8.30 使用已获 `CAMERA` 权限的 target context 通过 `OK (1 test)`：两颗后摄均公开精确 intrinsic、distortion 和 sensor-to-camera lens pose rotation，timestamp source 均为 `REALTIME`。候选 world-to-camera 组合不再需要先做独立旋转外参标定。
- r8.31 将该组合冻结为 `R_lens_pose * transpose(R_device_to_world)`，SM-S9280/API 36 上 `OK (5 tests)`，并与现有 world projector 接通；范围只到 raw camera-aligned sensor 坐标。
- r8.32–r8.34 已真机确认实际分析流无隐藏 crop、取得 CameraX 权威 sensor-to-buffer 矩阵，并完成精确 intrinsic + 90°像素/相机轴 + world projector 的确定性组合；分别 `OK (1 test) / OK (1 test) / OK (4 tests)`。
- r8.35 证明 30/30 相机帧均有 rotation-vector 前后样本，最大最近样本差 `9.67ms`、最大 bracket `19.756ms`，`OK (1 test)`；逐帧 SLERP 配对的数据条件成立。
- r8.36 原始低照线检测为 non-informative；r8.36a 同帧自适应复筛为 informative fail（10 条近重力消失点线，但 aligned length `8.78% < 10%`）。该负结果保留，不授权重投影链。
- readiness 仍为 **false**。下一独立门是短振动下 IMU `K R K^-1` 与 LK 光流的真实旋转重投影误差，随后仍需真实导航/LEFT-RIGHT 事件。
