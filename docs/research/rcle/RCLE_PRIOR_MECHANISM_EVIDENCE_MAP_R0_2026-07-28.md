# RCLE 前人机制—重叠—差异—可验证假设证据表 R0

日期：2026-07-28

状态：`EVIDENCE_MAP_COMPLETE / RELATED_WORK_AND_HYPOTHESIS_USE_ONLY`

## 口径

本表只回答五项前人工作实际支持什么、与 RCLE 在哪里重叠，以及能形成什么有边界
的研究假设。`原文机制` 与 `RCLE 推演` 分开记录；后者不是原论文结论。

证据层级：

- `FULL_TEXT`：本轮检查了论文正文；
- `OFFICIAL_ABSTRACT_PLUS_PAPER`：检查官方出版页摘要与论文正文入口；
- `ABSTRACT`：只把摘要可支持的内容写入 source-native finding；
- `DERIVED_INFERENCE`：针对 RCLE 的推演，不归因给原作者。

## 证据表

| ID / 文献 | 证据级别 | 前人机制与 source-native finding | RCLE 重叠 | 真实差异 | 可验证假设（RCLE 推演） | 风险与边界 |
| --- | --- | --- | --- | --- | --- | --- |
| E1 Stabinger, Rodríguez-Sánchez, Piater, *Monocular Obstacle Avoidance for Blind People using Probabilistic Focus of Expansion Estimation*, WACV 2016. [PDF](https://iis.uibk.ac.at/public/papers/Stabinger-2016-WACV.pdf) | `FULL_TEXT` | 头戴单目相机；Farnebäck dense flow；用流向量交点密度估计 FOE 分布；以 inlier proportion 表示 FOE 质量；在 FOE 附近拟合局部仿射 flow，以 divergence 近似接触时间；低质量帧弃用，并要求连续多帧质量合格。 | 同属视障辅助、单目 flow、局部仿射 divergence/expansion、质量门和时序持续性。 | E1 先估全局 heading/FOE，再在其附近算 divergence；RCLE 使用固定局部 cell、source pose 旋转模型、明确 support/condition/residual/abstention。E1 没有证明 RCLE 的姿态补偿机制，也不是 session 级独立性能证据。 | 若未来重启比较，应在同一自然 session 上比较 FOE-quality 路线与固定-cell RCLE 的 coverage、弃权和响应，而不是只比较单一触发率。 | E1 是直接前人，必须进 related work。RCLE 不能泛称“仿射 expansion、质量门或连续帧确认”为新颖点；创新主张需进一步检索。 |
| E2 Wulff, Sevilla-Lara, Black, *Optical Flow in Mostly Rigid Scenes*, CVPR 2017. [CVF](https://openaccess.thecvf.com/content_cvpr_2017/html/Wulff_Optical_Flow_in_CVPR_2017_paper.html) | `OFFICIAL_ABSTRACT_PLUS_PAPER` | 将自然场景分成 rigid 与 moving regions；静态区联合估计相机运动和 3D 结构，使用 Plane+Parallax 强约束；运动区用非约束 flow，并显式估计运动物体分割。 | 都试图把观察到的图像运动拆成自运动解释项与局部残差。 | MR-Flow 是多帧、显式分割、相机运动与三维结构联合估计的完整 optical-flow 方法；不是“减一个背景 flow 就得到动态目标”的轻量模块，也不直接验证 RCLE。 | 若强参考轨迹仍与 LK 一致地保留补偿后残差，应把下一问题指向 translation/depth/non-rigid model insufficiency，而不是继续调 LK。 | `DERIVED_INFERENCE`。不能把 MR-Flow benchmark 结果外推为助盲、端侧实时或风险有效性。 |
| E3 Ling, Sun, *ScaleFlow++: Robust and Accurate Estimation of 3D Motion from Video*, arXiv:2407.09797v2, 2024. [arXiv](https://arxiv.org/abs/2407.09797) | `ABSTRACT` | 从一对 RGB 同时估计 optical flow 与 motion-in-depth；核心是跨尺度特征匹配，并以端到端架构统一两项估计。摘要报告 KITTI monocular scene-flow 与 zero-shot 结果。 | Motion-in-depth 与 RCLE 的尺度变化/接近响应有概念联系；可观察 RCLE 失败是否仍存在强尺度方向信号。 | ScaleFlow++ 是学习式 dense 3D-motion 模型，无 RCLE 的 source-pose、局部仿射、弃权与端侧约束；输出不是 ground truth。 | 仅在参考轨迹诊断仍无法区分 tracker/model failure 时，才可另立版本检查 ScaleFlow++ MID 与 RCLE 符号、时间一致性；当前不执行。 | 预印本且本表只使用摘要级证据。不得称为 oracle、教师真值或安全证据；模型域偏移与训练数据影响必须保留。 |
| E4 Chalupka, Dickinson, Perona, *Generalized Regressive Motion: a Visual Cue to Collision*, 2015/2016. [arXiv](https://arxiv.org/abs/1510.07573), [DOI](https://doi.org/10.1088/1748-3190/11/4/046008) | `ABSTRACT` | 指出 looming 过去主要用于静态障碍接近；针对两个运动主体提出 GRM。摘要称几何分析支持其作为同类个体碰撞线索，agent-based modeling 中优于 looming。 | 直接提醒 RCLE：expansion/looming 对动态相遇关系可能不充分。 | GRM 研究移动主体几何与行为模型，不是头戴相机上的可部署局部 expansion estimator；不直接推出 bearing rate 是最佳扩展。 | 将来若基础局部运动机制重新获得支持，可预注册 expansion 与横向/方位变化的二维相图，检验 frontal approach 与 lateral pass 是否在 session 单位上可分。 | 当前机制审计为负，故 bearing 假设 `HOLD`。原文不能支持“RCLE 下一步必须加 bearing”的强结论。 |
| E5 Boretti, Bich, Zhang, Baillieul, *Visual Navigation Using Sparse Optical Flow and Time-to-Transit*, arXiv:2111.09669, 2021. [arXiv](https://arxiv.org/abs/2111.09669) | `FULL_TEXT` | 从 sparse optical flow 计算 time-to-transit；给出理想化 steering law，并用 ROS/Gazebo 与 Jackal robot 实验。实现使用 pyramidal LK、ORB 与多个 ROI；正文明确 perceived tau 在速度、航向或相机对齐不满足时会失真。 | 同样追求不用完整地图/米制深度的稀疏光流任务量，并依赖 ROI 与时间结构。 | E5 是闭环机器人导航/控制，常用走廊墙面特征并依赖运动与相机假设；RCLE 是局部接近响应诊断，不控制路线，也没有稳定速度/heading authority。 | 只有在独立运动机制成立后，才可测试 reciprocal expansion / tau 的时间一致性是否优于单 pair expansion；当前不执行。 | 不能把机器人走廊控制结果外推为视障用户碰撞预警、手机端性能或产品安全。 |

## 对当前 RCLE 的决策

1. **立即吸收为相关工作**：E1。它与 RCLE 的任务、信号和质量/时序门高度重叠，
   会约束论文的新颖性表述。
2. **支持当前唯一下一诊断的研究逻辑**：E2。先区分 tracker failure 与
   rotation-only model insufficiency，再谈增加新物理量。
3. **保留为后备参考、当前不运行**：E3。学习式 MID 只能做诊断参照，不能当真值。
4. **未来假设池**：E4、E5。它们说明 looming 可能不充分以及 tau 的条件，但当前
   rotation-compensation mechanism audit 已为负，bearing/tau 均保持 `HOLD`。
5. 路径走廊只保留为未来可视化层，不从这五篇文献获得当前算法权限。

## 论文写作边界

- 可以写：前人表明 FOE/divergence、mostly-rigid decomposition、cross-scale MID、
  GRM 与 tau 分别提供不同的运动解释框架。
- 不可以写：这些论文共同证明 RCLE 有效，或证明 bearing、深度、路径走廊是必然
  下一步。
- 当前 RCLE 最强可写结论仍是单 session Development Diagnostic 的负结果：
  坐标修正和去畸变后，rotation compensation 没有在冻结高角速度窗压低触发。
