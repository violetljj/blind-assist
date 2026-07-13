# Evidence-Claim Map

| Source ID | Citation / source type | Full-text finding | Usable fact | Supported claim | Citation slot | Risk |
|---|---|---|---|---|---|---|
| LOCAL-01 | real-only r3 manifests + training gate `4c68e434...` | 600 train/dev + 120 blind，14 session，10/10 green | 训练阻塞已从来源闭环转移到模型质量 | 当前主问题是质量稳定性，不是继续补形式化授权 | 报告 §2 当前事实 | 本地小规模数据，不代表泛化；旧 evidence-v4 不适用 |
| LOCAL-03 | P0 seed factor audit | model-seed selection range 0.2685，sampler range 0.0112；worst scene 均为 step_curb | model seed 相关随机状态是当前高方差主因 | 不能继续扫学习率/分辨率或挑幸运 seed | 报告 §2 当前事实、§5 P2/I0 | OFAT 非因果检验，未估交互 |
| LOCAL-04 | training/promotion contracts | optimizer-step、多 seed、offline→INT8→device event 三门 | 论文机制必须进入现有门禁 | 任何 mIoU 增益都不自动授权生产替换 | §2、§7 | 阈值仍需随数据扩展复审 |
| LOCAL-05 | traversability/event evidence | 90 帧 oracle v2 event recall 88.9%，错误提醒 25.9%；登阶后重复/再生是主失败 | 逐帧正确不等于事件安全 | 系统必须维护事件身份、阶段、清除和一次提醒 | 报告 §2 当前事实、§3.4、§5 T0/T1 | 90 帧 oracle，非训练模型/blind 证据 |
| LOCAL-06 | P1 LR-ASPP alignment audit | P1-A 最佳值升高但 range 扩至 0.2951；OS4/OS16 被拒绝 | 结构正确性不等于稳定性 | 保留 OS8/OS32，先 P2；Mobile-PID 需重新通过结构重入门 | 报告 §2、§3.1、§5 P1/P2/I0/E2 | head-only 短跑，未 INT8/设备门 |
| LOCAL-07 | one-off raw join + canonical schema | train/dev MACHINE 523/600，quality 未透传 trainer | source GT 不等于人工 GT | UPC/SWSEG 前必须恢复 HUMAN/MACHINE 可观测性 | 报告 §2、§3.2、§5 D0 | 一次性派生审计，尚非正式 gate |
| PID | Xu et al., CVPR 2023 | P/I/D 三分支、Pag/Bag；boundary loss 与融合有消融增益 | detail/context 直接融合会淹没细节；D 可作空间 gate | 只有重入门通过后，才测试 OS8/OS32 D-lite | 报告 §3.1、§5 E2 | Cityscapes/GPU，不是手机/SANPO；完整 PIDNet 较重 |
| MSEED | Liao et al., arXiv 2023 | 双流、GN、AFD、一致性；直接 boundary loss 曾使主任务退化 | 边界监督存在多任务冲突 | D 分支、边界损失和一致性必须分阶段消融 | 报告 §3.1、§5 E2 | 当前按预印本降级；boundary 与风险类不等价 |
| MRFP | Udupa et al., CVPR 2024 | HRFP+NP+ 改善论文 sim-to-real；推理不保留模块 | 训练期可同时扰动细纹理与风格 | 稳定性重入后可试 HRFP-only，且不增加端侧推理 | 报告 §3.2、§5 R1 | 可能破坏薄边界或扩大 seed 方差 |
| UPC | Fang et al., ICCV 2023 | patch entropy 定位噪声，以可靠 labeled patch 替换；优于 random/CutMix | 噪声在边界附近成片出现 | D0 通过后，HUMAN donor/MACHINE weak target 可受控验证 | 报告 §3.2、§5 D0/R2 | MACHINE 不是论文原生 unlabeled；高置信错误仍保留 |
| SWSEG | Lu et al., CVPR 2025 | Gaussian-SWD 优化 weak/strong alignment 和 uniformity；训练期 projection 可移除 | 表示层正则可能缓解 confirmation collapse | 只在 head 与伪标签角色稳定后作为 P4 正则 | 报告 §3.2、§5 R3 | uniformity 不等于 worst-seed 或安全改善 |
| VALUES | Kahl et al., ICLR 2024 | C1/C2/C3 和 downstream task 必须独立验证；aggregation 影响巨大 | ensemble 最稳，TTA 常是轻替代；AU/EU 真实数据不易分开 | 先升级不确定性评价与 risk-coverage，再选校准法 | 报告 §3.3、§5 U0 | 医疗+人工 GTA/CS ambiguity 与当前域不同 |
| KAND | Brunekreef et al., CVPR 2024 | 低 calibration data 下空间聚类降低 coverage error | 固定空间区域可共享 non-conformity 样本 | 冻结模型后用独立 session calibration 做额外 abstain | 报告 §3.3、§5 U1 | 连续帧非 i.i.d.；coverage 不保证事件安全 |
| STEPP | Ægidius et al., ICRA 2025 | 人类轨迹投影正样本、重建误差发现陌生地形；机器人实机 | 轨迹可生成弱 traversability 监督 | 可作为离线 anomaly teacher，不进入首轮端侧主链 | 报告 §3.5 | 机器人与人类/BLV 可通行条件不同；阈值校准弱 |
| DTERN | Xu et al., ICCV 2025 | local/global exemplar；VEC 同时约束一致性和有效性 | 旧 VC 可被全零稳定预测欺骗 | 先移植 VEC，再试 history-only 轻量 prototype | 报告 §3.4、§5 T1 | 完整模型吞吐显著下降，无手机证据 |
| BOFP | Baghbaderani et al., WACV 2024 | 双向 flow+occlusion attention 改善时序一致性 | 遮挡区域不应盲目传播或清除事件 | 双向只作离线上界，线上只试 causal forward hold | 报告 §3.4、§5 T2/T3 | 未来帧不可用于实时告警；mTC 可能奖励稳定错误 |
| AIGD | Jadhav et al., AAAI-SS 2025 | 未来 1 秒 FRONT/LEFT/RIGHT；iPhone 2 Hz | 路径方向可作低频意图先验 | 不让方向模型替代近场危险感知 | 报告 §3.6 | 8 名 sighted 模拟参与者，转向 recall 约 0.56–0.58 |
| ESCALATOR | Zhang, ICCVW 2025 position paper | 稀疏帧遗漏低信号连续运动 | 静态语义充分不等于动态安全充分 | 建立外观相同、运动相反的 hard-pair benchmark | 报告 §3.4 | 无新算法、无标准化量化证据 |
| VISASSIST | Gao et al., AAAI 2026 | 13,413 段真实视障用户视频；缺失信息、depth/direction 和幻觉失败 | VLM 无法可靠区分画面缺失与低质量 | quality/missing-information gate 必须独立于安全告警 | 报告 §3.6、§5 H1 | VideoQA、秒级延迟、非导航干预实验 |
| CLIPBLV | Massiceti et al., CVPR 2024 | 25 CLIP 在 BLV 数据平均低 15pp；质量/内容/文本均有差距 | 通用预训练规模不消除 BLV QoS 差距 | 所有 foundation teacher/VLM 必须做 BLV capture worst-group audit | 报告 §3.6、§5 H2 | 分类审计，不直接证明分割/导航失败 |

## 核心论点覆盖

| 核心论点 | 主支撑 | 交叉支撑 | 状态 |
|---|---|---|---|
| 下一次跨越不能靠单一更大模型 | LOCAL-03、LOCAL-05 | PID、VALUES、VISASSIST | 强 |
| P1 提高上限但没有关闭方差；近期先 P2，再决定结构重入 | LOCAL-03、LOCAL-06 | PID、MSEED 仅支持条件候选 | 强 |
| HUMAN/MACHINE 质量透传是半监督硬前置 | LOCAL-07 | UPC、SWSEG | 强 |
| unknown、model uncertainty、extra abstain 必须分开 | LOCAL-04 | VALUES、KAND、VISASSIST | 强 |
| 时序目标应是事件有效性，不是 mask 平滑 | LOCAL-05 | DTERN、BOFP、ESCALATOR | 强 |
| VLM 只能做低频解释/质量恢复，不能直接安全告警 | VISASSIST、CLIPBLV | ESCALATOR、AIGD | 强 |
| 正式晋级必须 worst-seed + blind event + INT8/设备预算共同通过 | LOCAL-03、LOCAL-04 | 全部迁移论文仅作机制先验 | 强 |
