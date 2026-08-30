# L10 / DTR 定向文献增量（2026-08-30）

状态：`LITERATURE_DELTA / NO_EXPERIMENT_EXECUTED / NO_ROUTE_AUTHORIZATION`

## 结论

知识库需要小幅更新，不需要继续大规模扩容。用户提供的本轮 Exa 审阅覆盖
4 个检索方向、46 个结果槽位；表格实际点名 **9 项**（L10 4 项、DTR 5 项），
不是正文概括中的 8 项。

后续默认节奏改为：

```text
新实验暴露瓶颈
-> 定向查 3--8 项
-> 选择一个新信息机制
-> 跑一个最小可证伪实验
-> 由结果决定是否继续检索
```

本文件只保存当前映射和停止边界。来源事实与路线使用已分别写入
`items/` 和 `uses/`，可由 `tools/knowledge.py` 检索。

## L10：先修参考侧可见门面

SceneNN 的当前失败是参考侧可见性没有权威：mesh 投影包络混入前景遮挡物。
下一实验应改变可见性信息源，不再换 DINO、SAM prompt 或调平面残差。

### 1. ScanNet++ 官方工具链 — `PLANNED`

- **工作：** [ScanNet++ toolkit](https://github.com/scannetpp/scannetpp)，ICCV 2023 数据集的官方工具链。
- **新机制：** mesh 到图像 rasterization、pixel-to-face、z-buffer depth、2D instance/semantic projection 和 object visibility。
- **当前映射：** 直接修复 SceneNN 参考门包络混入遮挡物的问题。
- **最小实验：** 固定现有候选与 query transfer，只把参考侧深度读取改为 `mesh rasterization -> z-buffer visible target pixels -> depth consistency -> visible plane fit`。
- **观察指标：** exact-door Top-1、wrong-door commit、centroid-inside、中位 IoU、世界坐标质心误差。
- **停止条件：** 若可见性资格化不能在不增加 wrong-door 的前提下改善门面定位，则停止该源；不转入 prompt、backbone、plane residual 或阈值 sweep。

### 2. VGGT-Segmentor — `DIRECT CHALLENGER`

- **工作：** [VGGT-Segmentor: Geometry-Enhanced Cross-View Segmentation](https://arxiv.org/abs/2604.13596)，2026 preprint。
- **新机制：** source mask、VGGT 跨视角特征、点引导预测与迭代 mask refinement 共同完成跨视角实例分割。
- **当前映射：** 在可见参考门面成立后，挑战现有门面跨视角 transfer，而不是替代身份权威。
- **最小实验：** 固定同一 source mask、query frames 和 evaluator，只替换 cross-view mask transfer。
- **观察指标：** exact-door Top-1、wrong-door commit、中位 IoU、centroid-inside、质心误差。
- **停止条件：** 若没有改善 exact-door/IoU，或以更多 wrong-door 换来覆盖，则关闭 challenger；不做 prompt 或 threshold rescue。

### 3. Fast Generative DeOcclusion — `MECHANISM RESERVE`

- **工作：** [Fast Generative DeOcclusion for Visual Geometry and Robotics](https://openaccess.thecvf.com/content/CVPR2026F/papers/Chen_Fast_Generative_DeOcclusion_for_Visual_Geometry_and_Robotics_CVPRF_2026_paper.pdf)，CVPR Findings 2026。
- **新机制：** 选择有信息量视角、2D amodal completion 与 depth-aware 3D integration 的快速生成式去遮挡。
- **当前映射：** 只借 visible/occluded 分层与遮挡诊断；不让生成内容填充参考门真值。
- **最小实验：** 先做只读遮挡状态分层，检查可见门像素与原包络误差是否集中在 occluded strata；不生成训练或评价标签。
- **观察指标：** visible/occluded 分层后的 wrong-door、IoU、质心误差与覆盖。
- **停止条件：** 若收益依赖生成的隐藏表面或像素，立即停止；生成几何永远不得作为几何真值、身份权威或确认依据。

### 4. Video2DoorTraversal — `LATER RESERVE`

- **工作：** [Video2DoorTraversal: Push Door Traversal via Simulated Door Twins](https://arxiv.org/abs/2608.20251)，2026 preprint。
- **新机制：** 从单段真实视频恢复实例对齐、可关节、可仿真的 door twin，并据此生成开门/穿越技能。
- **当前映射：** 仅在可见门面与 exact door 已成立后，作为门板、把手、铰链与 traversal 结构的后序参考。
- **最小实验：** 当前不执行；未来只先审查 door twin 的 metric structure error，不接机器人控制链。
- **观察指标：** 门板/把手/铰链结构误差、实例错绑和结构缺失率。
- **停止条件：** 只要当前瓶颈仍是可见门面或 exact-instance transfer，就不启动；不得用其 96.57%/80.95% 机器人成功率外推 BlindAssist。

## DTR：存在、可见性和行为分布分开

下面五项都不修改已冻结的 X31/C11。它们只为 C11 终结后的新协议或 CARLA
后序 source/evaluator 提供候选。核心表示假设是：

```text
existence probability
x current observability
x conditional instance distribution
x motion-branch distribution
```

### 5. OccSTeP — `EVALUATOR PRIORITY`

- **工作：** [OccSTeP: Benchmarking 4D Occupancy Spatio-Temporal Persistence](https://arxiv.org/abs/2512.15621)，2025 preprint。
- **新机制：** 把 dropped frames、错误语义、历史扰动和给定未来动作后的 proactive occupancy 纳入 persistence 评测。
- **当前映射：** 为 X31 后序协议补“遮挡期间保留了什么”的压力评测，不改变当前冻结 protocol。
- **最小实验：** 在新 protocol 上固定输入与动作，分别注入掉帧、视角缺失、语义错误和历史扰动；比较当前可见/不可见 strata。
- **观察指标：** occupancy IoU/calibration、CONTACT recall、false segments、dropout recovery、lead time，并按 observability 分层。
- **停止条件：** 若 persistence 指标不能解释任何 DTR event 差异，或只提高 occupancy IoU 而恶化事件表现，则不引入 world-model complexity。

### 6. BeyondSight — `DIRECT MECHANISM REFERENCE`

- **工作：** [BeyondSight: Object Permanence for End-to-End Autonomous Driving](https://arxiv.org/abs/2607.09138)，2026 preprint。
- **新机制：** 将 actor existence 与 instantaneous observability 解耦，维持持久 actor hypothesis，并按可见性条件评测。
- **当前映射：** 对准 X31 的遮挡后运动权限：不可见不等于不存在，但未观测也不能自动成为确定占据。
- **最小实验：** 新 protocol 中只增加 existence/observability 两个正交状态，冻结 ancestry、motion branches、route scorer 和 lifecycle。
- **观察指标：** visible/unobservable actor persistence、错误持久化、CONTACT recall、false segments、dropout recovery。
- **停止条件：** 若 object permanence 只把 false persistence 推高，或需要评价器身份才能在线更新，则停止，不延长保留时窗救结果。

### 7. Consistent Instance Field — `REPRESENTATION RESERVE`

- **工作：** [Consistent Instance Field for Dynamic Scene Understanding](https://openaccess.thecvf.com/content/CVPR2026/html/Wu_Consistent_Instance_Field_for_Dynamic_Scene_Understanding_CVPR_2026_paper.html)，CVPR 2026。
- **新机制：** 每个时空点分别表示 occupancy probability 与 conditional instance distribution，把存在与实例归属分开。
- **当前映射：** 为 X31 set-valued lineage 提供概率表示参考，不把 4D Gaussian reconstruction 直接搬进 DTR。
- **最小实验：** 先在冻结 ledger 上做 read-only factorization canary，检查 hard track/cell 是否掩盖 existence 与 identity 的不同不确定性。
- **观察指标：** occupancy calibration、instance assignment error、split/merge 后 lineage consistency、事件级 false persistence。
- **停止条件：** 若 factorization 不改变错误归因或事件决策，停止；不训练 4D field，也不把渲染质量当风险指标。

### 8. CarlaOcc — `TRUTH / DATA REFERENCE`

- **工作：** [CarlaOcc](https://github.com/fengyi233/carlaocc)，CVPR 2026 项目与数据集。
- **新机制：** 100K 帧实例级 panoptic occupancy、最高 0.05 m voxel、actor mesh 导出与物理一致 mesh voxelization。
- **当前映射：** 审查后续 CARLA occupancy truth 的实例、网格与 mesh provenance；不是替换当前 CARLA 0.9.16 冻结来源的命令。
- **最小实验：** 先比较一个相同场景的当前 truth 与 CarlaOcc-style mesh voxelization 的占据/实例差异，不下载全量数据。
- **观察指标：** voxel agreement、instance consistency、route-tube contact parity、truth generation cost。
- **停止条件：** 若差异只来自更高分辨率，或 CARLA UE5/0.10.0 环境差异主导结果，则保持为参考，不迁移工具链。

### 9. HABIT — `BEHAVIOR PRIORITY`

- **工作：** [HABIT: Human Action Benchmark for Interactive Traffic in CARLA](https://openaccess.thecvf.com/content/WACV2026/html/Ramesh_HABIT_Human_Action_Benchmark_for_Interactive_Traffic_in_CARLA_WACV_2026_paper.html)，WACV 2026。
- **新机制：** 把 mocap/video 人体动作重定向到 CARLA；从约 30K 动作中筛出 4,730 个交通兼容 SMPL 动作。
- **当前映射：** 补 CARLA 当前确定性脚本的人体行为分布：犹豫、加速、停走、返回、非线性横穿、姿态变化和遮挡后重现。
- **最小实验：** 固定地图、天气、路线、人数与 DTR 算法，只把一个 scripted-motion arm 换成 matched retargeted-motion arm。
- **观察指标：** 行为 strata 覆盖、CONTACT recall、false segments、lead time、事件 F1 与 deterministic-script gap。
- **停止条件：** 若新动作没有产生新的 route-risk/event strata，或环境集成成本远大于信息增量，则停止；不继续加天气或静态布景。

## 明确不再扩充的方向

- 通用 detector、SAM、开放词汇模型；
- 普通 scene-text OCR 排行榜；
- 通用 trajectory forecasting 或完整 planner；
- 泛化 VLM 助盲系统；
- 与当前故障无关的大规模安全治理框架。

本增量不授权实验执行、阈值调优、默认 App 集成、产品、用户收益或安全主张。
