# BlindAssist 待决方向

状态：current（只记录尚未决定的方向，不授予研究或产品权限）

本页是待决想法的短入口，不复制动态实验数字、执行计划或历史流水。旧记录完整保留在 [想法池历史归档](docs/history/idea/IDEA_ARCHIVE_THROUGH_2026-07-28.md)；当前产品事实以 [README.md](README.md) 为准，当前双环、SANPO 与暂停的 RCLE 状态以各自 current 文档为准。

## 维护规则

- 只有用户明确要求保留，或团队明确延后一个非平凡方向时，才新增条目。
- 每条只写问题、尚未验证的假设、进入实施前的门槛和权威入口。
- 一旦采用、拒绝或完成，就把决定写入对应 current/decision/snapshot 文档并从本页移除；不要在这里追加实验流水。
- 本页任何内容都不开放训练、formal claim、Android 接入、默认模型替换或产品安全表述。

## 当前待决

### 反事实风险事件与生命周期建模

- 问题：单帧/像素指标不能直接回答“同一真实风险事件是否及时提醒且只提醒一次”。
- 假设：具备来源隔离、matched negative、完整生命周期标注的事件级数据，可能比继续调单帧 head 更能改善误提醒与重复提醒。
- 实施门槛：先按 [SANPO 当前状态](docs/SANPO_CURRENT_STATUS.md) 和 [反事实采集协议](docs/SANPO_COUNTERFACTUAL_EPISODE_COLLECTION.md) 关闭许可、来源隔离、复核与数据覆盖门；未闭合前只允许协议/接口工作。

### 动态出行风险：30 项外部候选池（2026-08-24）

- 状态：`PENDING_CANDIDATE_POOL / NOT_AN_ACTIVE_ROUTE / NO_EXECUTION_AUTHORITY`。
- 逐项精读：[30 项论文、算法与项目精读笔记](docs/research/dynamic-travel-risk/DYNAMIC_TRAVEL_RISK_30_DEEP_READING_2026-08-24.md)，逐项记录核心机制、实验、核心价值、读后判断和不可迁移边界；精读不建立执行或生产 authority。
- 问题：如何从“检测到物体”转向“有证据表明目标正在进入用户的短期路径，或输入已不足以做决定”，并只在需要行动时提醒。
- 检索与筛选：用 Exa 在 24 个不同搜索面审阅了 `327` 个结果槽位（包含重叠，不等于 327 个独立来源）；只保留论文页、作者/机构项目页、官方仓库或官方数据页。同一论文与官方仓库合并为一项；排除了排行榜文章、纯厂商宣传和只提供通用检测 AP 而没有事件/路径/不确定接口的工作。
- 与已有资料的关系：已与 [USTRF 前沿论文指南](docs/research/ustrf-sc/USTRF_FRONTIER_PAPER_GUIDE_2026-07-22.md) 去重和交叉核对。本表只是待决候选池，不恢复已关闭的 USTRF 路线，也不把汽车/机器人指标写成助盲证据。
- 入选后的最小用法：先按失效层只取一个 baseline 和一个机制变化，并固定 `body/path corridor + event lifecycle + sensor health + abstention`；不把下表变成 30 臂 model zoo。

表中“证据域”只描述外部工作的证据来自哪里：`BLV 直接`表示有盲人/低视力参与者或真实助具使用，`助具技术`表示为助行设计但没有足够用户结果，`迁移`表示汽车/机器人/通用 ML 机制，`工具/数据`表示只提供资产或评测面。它们都不自动获得 BlindAssist 的功能、安全或默认 App 权限。

#### A. 直接碰撞、路径和低负担反馈

| ID | 候选（一手来源） | 主要痛点 | 可借用的最小部件 | 证据域 | 最强边界 |
|---|---|---|---|---|---|
| DR01 | [Collision-point × TTC 选择性提醒系列：2015 障碍课程](https://iovs.arvojournals.org/article.aspx?articleid=2281664) / [2021 居家随机临床试验](https://pmc.ncbi.nlm.nih.gov/articles/PMC8299358/) | 物体在附近但不进入身体路径；高频误报 | 不标类别，只在预测碰撞点进入身体带且 TTC 足够短时震动 | BLV 直接 | 两个研究都约降低 37% 接触/碰撞，但前者是外周视野受损障碍课程，后者记录接触而非伤害；设备始终是盲杖/导盲犬的补充 |
| DR02 | [BBeep](https://wotipati.github.io/projects/BBeep/paper/CHI%2719_BBeep_preprint.pdf) | 行人横穿；恒定告警造成社交和认知负担 | RGB-D 跟踪 + 相机旋转补偿 + 行人未来路径与用户 emergency line 交汇；只在预测碰撞时出声 | BLV 直接 | 机场研究只有 6 名盲人，手提箱形态、线性外推和“旁人主动让路”不能外推到手机独立防撞 |
| DR03 | [HEADS-UP](https://www.epfl.ch/labs/vita/research/prediction/heads-up/) | 头部转动与真实行人相对运动混杂 | 头戴 RGB-D/IMU 的旋转感知半局部轨迹表示；可做相对轨迹与 ego compensation 对照 | 助具技术 | 约 43k 帧/1k tracks，部分标签来自 detector/tracker/VIO 平滑；不是独立碰撞真值或 BLV 用户效果 |
| DR04 | [ARAwareness / ARAware](https://www.mdpi.com/1424-8220/24/13/4282) | 快速车辆需要比慢行人更早提醒；早告警过多 | 用目标类型、速度、距离、预测碰撞时间和严重度决定 alert deadline | 助具技术 | 实验由研究者模拟视障用户；检测精度与处理时间不等于避免碰撞 |
| DR05 | [MinD：动态环境中的实时最优局部避碰路径](https://openaccess.thecvf.com/content/ICCV2023W/ACVR/html/Surougi_Real-Time_Optimisation-Based_Path_Planning_for_Visually_Impaired_People_in_Dynamic_ICCVW_2023_paper.html) | 知道要避让，但不知道怎样最小移动才避开车/摩托 | 把目标类型、运动模式、TTC 和用户移动约束合成局部避碰优化 | 助具技术 | 实车原型参数驱动的模拟不等于开放人行道用户试验；它假设目标跟踪与碰撞时间已可用 |
| DR06 | [Corridor-Walker](https://www.masakikuribayashi.com/data/project/masaki_kuribayashi_mobilehci_2022/paper.pdf) | 路线相关性、上半身/盲杖盲区、无意义贴墙 | 手机 LiDAR occupancy grid → 可通行走廊/交叉口 → 空间音频、震动、TTS | BLV 直接 | 14 名盲人的室内静态走廊研究；没有横穿、TTC 或户外动态证据 |
| DR07 | [Wearable obstacle avoidance with cross-modal learning](https://doi.org/10.1038/s41467-025-58085-x) / [WOAD 官方仓库](https://github.com/MMCNJUPT/WOAD) | 单传感器退化、端到端时延、电池和音频/触觉负担 | video-depth 交叉学习、跨模态一致性/降级和分级音频+触觉调度 | BLV 直接 | 专用 RGB-ToF/FPGA/手机硬件约 400 g；小规模、非随机研究中的“100% 避障”不可作独立安全证据 |
| DR08 | [Project Guideline](https://github.com/google-research/project-guideline) | 从手机感知到可听行动的完整工程接口 | 可运行 Android/C++ 结构、路径相关障碍 aperture、追踪丢失时 STOP、开放式耳机反馈 | 工具/助具 | 只适用特制宽路和地面引导线，明确要求 sighted spotter；不是通用动态避障系统 |

#### B. 风险形成、未来占用与事件评测

| ID | 候选（一手来源） | 主要痛点 | 可借用的最小部件 | 证据域 | 最强边界 |
|---|---|---|---|---|---|
| DR09 | [Binary TTC: A Temporal Geofence](https://openaccess.thecvf.com/content/CVPR2021/html/Badki_Binary_TTC_A_Temporal_Geofence_for_Autonomous_Navigation_CVPR_2021_paper.html) | 精确 TTC 回归不稳，但仍需快速回答“会不会在某时限内进入” | 每像素多 horizon 二值 temporal geofence，作为事件触发的轻量对照 | 迁移 | TTC 相对相机平面，不是用户身体包络/真实路线的 physical TTC |
| DR10 | [Stochastic Occupancy Grid Map Prediction / SOGMP、SOGMP++](https://proceedings.mlr.press/v229/xie23a.html) | 行人可能有多个合理未来，单轨迹会漏掉分支 | 预测随机未来 occupancy map，以均值与样本方差 costmap 对照 body corridor，而不强制单 ID | 迁移 | 机器人传感和控制栈；VAE 样本方差未做风险校准，也没有头戴相机或 BLV 事件证据 |
| DR11 | [RiskProp](https://openaccess.thecvf.com/content/CVPR2026/html/Zou_RiskProp_Collision-Anchored_Self-Supervised_Risk_Propagation_For_Early_Accident_Anticipation_CVPR_2026_paper.html) | 只有碰撞帧，没有可信的主观“风险开始”帧 | 以 collision anchor 向碰撞前的帧反向传播风险信号；可借鉴事件 onset 弱监督和趋势约束 | 迁移 | 驾驶记录仪事故数据；横穿后离开/用户停步时风险不必单调，不得照搬 monotonic prior |
| DR12 | [Conformal Risk Tube Prediction](https://github.com/HCIS-Lab/CRTP) | 报警过早/过晚、在时空上闪烁、对不确定未来过度自信 | 经 conformal calibration 的时空 risk tube；评估 coverage、onset/release、Risk-IoU 与 nuisance duration | 迁移/工具 | ICRA 2026 新项目，车辆/CARLA 域、旧 CUDA 环境且外部复现少；概率覆盖不等于 BLV 安全 |
| DR13 | [PIE + scenario evaluation](https://github.com/aras62/PIE) | 哪个行人正在变得相关，以及多早提醒才有用 | 行人行动/看向/横穿/遮挡标注；事件分层、lead-time、轨迹/意图评估代码 | 工具/数据 | 车载视角、交叉口横穿为主；不包含穿戴式身体路线或触地事件真值 |
| DR14 | [RiskBench](https://github.com/HCIS-Lab/RiskBench) | 识别危险物是否真正改变下游行动 | 交互/碰撞/障碍/非交互场景分类，风险对象归因、预判与 planning-aware 指标 | 工具/数据 | 合成 CARLA 驾驶策略不是步行控制 truth；只可借用 taxonomy/评估接口 |
| DR15 | [OF-VO：Optical-flow Velocity Obstacle](https://arxiv.org/abs/2004.10976) | 侧方横穿者与用户路径将来交叉，但当前距离仍远 | 相对运动的 probabilistic velocity-obstacle collision cone | 迁移 | Turtlebot 与主动规划依赖 LiDAR；近距离局部可见和跟踪失败会破坏圆锥估计 |

#### C. 自运动、未知障碍、相机健康与选择性失效

| ID | 候选（一手来源） | 主要痛点 | 可借用的最小部件 | 证据域 | 最强边界 |
|---|---|---|---|---|---|
| DR16 | [Detection of Fast Incoming Objects with a Moving Camera](https://www.bmva-archive.org.uk/bmvc/2016/papers/paper146/paper146.pdf) | 用户转头/晃动产生全局 optical flow，被误当作物体接近 | 全局相机运动补偿后的 residual flow、looming 和质量门 | 迁移 | 无人机/快速物体视频；纹理不足、模糊、遮挡和身体几何仍未解决 |
| DR17 | [SelectiveNet](https://proceedings.mlr.press/v97/geifman19a.html) | 模型不可信时仍强制输出方向 | 联合学习 prediction + selection head，显式优化 risk–coverage 曲线 | 迁移 | 通用图像数据；目标 coverage 不提供分布外或安全保证，仍需 BlindAssist 事件损失 |
| DR18 | [Perceive With Confidence](https://proceedings.mlr.press/v270/dixit25a.html) | 不知道“没有障碍”还是“感知没看见” | 用 conformal calibration 把 occupancy perception 与 planner 的不确定联结，为退化/分布偏移保留风险上界 | 迁移 | 室内静态障碍和四足机器人；统计保证依赖校准/偏移假设，不覆盖动态 BLV 事件 |
| DR19 | [Monitoring and Adapting the Physical State of a Camera](https://elib.dlr.de/147340/) | 模糊、噪声或曝光问题导致下游检测失效，但系统仍沉默 | 任务相关的实时相机 self-health estimator；将输入质量作为独立 evidence channel | 迁移 | 已演示的主要是 motion blur/传感器噪声与车载检测；不含镜头完全遮挡、路线或助盲事件 |
| DR20 | [GSHI 退化感知相机可靠性监测](https://arxiv.org/abs/2605.05439) | 雨雪、低照、眩光、运动/离焦模糊、镜头遮挡和压缩等多种失效 | 单 RGB 同时估计退化类型、严重度、空间不确定图和全局健康指数；供“请减速并用盲杖确认”门控 | 迁移/观察 | 2026 预印本，主要用 KITTI 合成退化训练；健康分数不是任务 coverage 保证 |
| DR21 | [Evidence of Absence](https://arxiv.org/abs/2608.14952) | 相机被挡或路口遮挡时，引擎/胎噪已出现而视觉共证据缺失 | 声学方位/接近证据 + “应有视觉但缺失”的跨模态 advisory；在明确假阳预算下只提醒不控制 | 迁移/观察 | 2026-08 工作草稿；移动 ego 噪声是主阻塞，安静电动车/自行车反而是弱项，不能当通用解法 |
| DR22 | [Indoor Imminent Danger Detection with Ground Segmentation](https://ieeexplore.ieee.org/document/9211506) | 未知/临时障碍不在类别表内；地面涂色容易造成语义误判 | RGB-D+IMU 地面/自由路径 + 去地后点云 residual；建立不依赖物体名称的 traversability 通道 | 助具技术 | 室内 RGB-D 假设，没有 BLV 行走效果；室外 IR/阳光和细杆失效仍需另行评估 |
| DR23 | [Stereo + motion drop-off detection](https://homepages.inf.ed.ac.uk/msridhar/Papers/iros08_doffDetect.pdf) | 坑洞、路缘和下行台阶不是“前方物体” | 正障碍与 negative obstacle 分离，用 stereo + motion evidence 建独立 drop-off channel | 迁移 | 2008 机器人 stereo 管线；无穿戴式头动、用户试验或手机单目可行性 |

#### D. 可运行数据、压力源与反馈工程

| ID | 候选（一手来源） | 主要痛点 | 可借用的最小部件 | 证据域 | 最强边界 |
|---|---|---|---|---|---|
| DR24 | [JRDB family](https://jrdb.erc.monash.edu/) | 密集人群、遮挡、跟踪断裂与社交动作 | 360° RGB/RGB-D/LiDAR/里程计、2D/3D tracks、动作、pose 和 trajectory forecasting 评测；用于人群/遮挡分层 | 工具/数据 | 人体高度机器人仍不是手机/眼镜，且没有用户 intended route 或碰撞事件 truth；下载和许可需现场复核 |
| DR25 | [Aria Digital Twin + Project Aria Tools](https://www.projectaria.com/datasets/adt/) | 穿戴式时钟、标定、头动、对象位姿和像素/轨迹绑定 | 多相机+双 IMU+眼动、6DoF 设备/对象 pose、深度、mask 及 Apache-2.0 工具；适合时钟/位姿/输入健康 canary | 工具/数据 | 仅两个室内场景约 200 段，活动不是出行风险事件；数据条款需逐次核对 |
| DR26 | [REveL](https://uts-ri.github.io/revel/) | 安静/快速目标、运动模糊和相机自运动下的人员接近 | RGB + event + LiDAR + 1 kHz IMU + Vicon 人/传感器 pose；可做跨模态检测、ego compensation 和 source-motion 压力 | 工具/助具 | 约 14.1 分钟、2 个行人、室内 arbitrary Vicon frame；没有 body corridor、closest approach 或助盲 event truth |
| DR27 | [DynamicStereo / Dynamic Replica](https://github.com/facebookresearch/dynamic_stereo) | 区分 ego motion 与人/动物运动，并保持时序深度稳定 | 动态视频的 depth、camera pose、mask、flow 和长轨迹；作为动态深度/运动 residual 压力源 | 工具/数据 | 合成、stereo、CC BY-NC 4.0，解压约 2.2 TB；不是当前手机单目输入或真实风险事件 |
| DR28 | [Spring + RobustSpring](https://spring-benchmark.org/) | 模糊、天气、噪声、压缩和深度/双目不一致时，何时应退出 | 真实高分辨率 flow/disparity/scene-flow truth + 20 类时序一致退化；生成质量–coverage–abstention 曲线 | 工具/数据 | 汽车 stereo/4K 离线 benchmark；对 corruption 稳健不证明助盲事件或手机时延 |
| DR29 | [EgoTraj](https://github.com/yehiahmad/EgoTraj) | 穿戴者在真实城市中的多模态短/长期路径分支 | 75 段同步 RGB、6DoF 头部 pose、3D gaze 和场景标注；可检验 gaze/头动/历史对 wearer-path prediction 的增量 | 工具/数据 | 2026 预印本，预测的是穿戴者本身轨迹，不是其他目标进入身体路径或碰撞 truth |
| DR30 | [Open Source Soundscape](https://github.com/microsoft/soundscape) | 方向提示不应长时遮蔽环境声，风险提醒还要与导航/后台音频共存 | 开源空间音频、head-relative beacon、background coexistence 和 route authoring；可作“紧急事件抢占、非紧急叙述让行”的反馈基线 | 工具/交互 | MIT 仓库只是原产品的非 turnkey 子集，删除了第三方/部署组件；它不包含动态危险感知 |

筛选后仍没有一个候选同时提供“穿戴视角 + 用户身体/目标路线 + 临时障碍/动态目标 + 相机健康 + 事件 onset/clear + BLV 反馈结果”。因此最大缺口仍是一个可评分的组合接口，不是另一个通用 detector。

以后若明确授权这个问题，优先不是从 30 项中选一个“最强模型”，而是用下列最小四层区分失效：

1. `DR01/DR09` 作 object-agnostic `collision point × temporal geofence` 最小基线；
2. `DR03/DR10/DR15` 中只选一种相对轨迹、随机 occupancy 或 velocity-obstacle 机制，判断多目标未来是否提供增量；
3. `DR17–DR21` 建立独立的 sensor-health / selective-risk / cross-modal advisory，不允许“无提示”回退为“前方安全”；
4. 只在事件 truth 闭合后，用 `DR12–DR14/DR30` 评估 first-alert lead time、critical miss、false alerts/min、fragmentation/repeat、clearance 与环境声遮蔽。

任何一层缺失 route/body truth、时钟/位姿或输入健康证据时，输出只能是 `UNKNOWN / ABSTAIN / 请减速并用盲杖确认`，不是“前方安全”。

### TwinScene 与 AC4D 后续方向（含 TARO 迁移指针）

- 状态：`TwinScene 与 AC4D 待决 / GaugeFix+PARA 已迁出为 TARO 独立并行支线`。
- 共同问题：继续换 depth backbone 或直接学习 final clearance，无法区分尺度、support、boundary、
  可观测性和动态预测究竟哪一层提供信息增益；新方向必须先用 oracle/反事实把机制拆开。
- `GaugeFix` 想法：在有效 K/crop/rotation/resize receipt 和独立米制锚之后，只估计会改变
  body-swept clearance 的低维残余 gauge posterior，并以 TSVD/observable mask 保留不可观测方向；
  RGB-only、纯旋转或无 metric anchor 时必须保持 UNKNOWN。
- `PARA` 想法：先复用合法 ring-buffer 历史，再在身体保持静止时选择 yaw/pitch 或手机左右微基线，
  直接优化 task-query uncertainty/risk reduction；不得退化成通用 NBV、时序平滑或要求用户迈步。
- 采用决定：GaugeFix 与 PARA 已合并为 **TARO — Task-directed Active Risk Observability**，
  不再作为两个待决子路线；当前入口为 [TARO current](docs/research/taro/README.md)，详细命题、
  数学定义、数据合同、实验阶梯、强基线与 kill gates 见
  [TARO R0 路线指南](docs/research/taro/TARO_R0_RESEARCH_ROUTE_GUIDE_2026-08-10.md)。
- `TwinScene` 待决想法：构造 `真实基线 → 同 pose 数字基线 → 单变量 3D 干预` 三元组，
  以 exact depth/support/boundary treatment effect 做 factor finite-difference supervision；只有通过
  collider/render parity、effect-mask 外 artifact probe、跨 renderer/asset/site 和少量真实物理 pair
  审计后，才可另立离线数据/蒸馏路线。它不能把 synthetic effect 写成真实因果。
- `AC4D` 待决想法：先预测与候选 wearer path 无关的 stochastic future metric world belief，再由
  deterministic 4D body tube 查询任意 path/profile 的 first-contact survival；必须先在加速/转向、
  遮挡重现、多目标和 1.5–3 秒困难分层上以 oracle 超过 D44 + Kalman/IMM，普通 1 秒 ADE 改善不足以立项。
- 依赖关系：TwinScene 未来只能作为 TARO 的可选离线监督；TARO 未来只能在独立 oracle 通过后向
  AC4D 提供 current metric posterior；三者不得互相背书或一次性组成无法归因的大系统。
- 当前权限：TARO 只获得路线设计与唯一 P0 protocol-lock successor，execution=false；TwinScene、
  AC4D 没有 active route、数据、实现、训练或 outcome 权限；默认 App、产品与 safety 均不变。

### 真实眼镜终端

- 问题：当前正式能力仍以手机摄像头为主，模拟中心不等于真实硬件连接。
- 假设：先建立 `GlassesFrameSource`、设备状态与反馈适配边界，再以 USB 有线样机验证，能避免过早把 BLE、视频、音频和 UI 耦合。
- 实施门槛：以 [眼镜硬件路线](docs/GLASSES_HARDWARE_ROUTE.md) 为 current 入口；具备真实硬件、协议、断连降级、功耗/热与隐私验证后才进入实现。

### 答辩/课堂轻量展示包

- 问题：临场演示需要稳定材料，但不应重新污染日常主流程或把历史验证当成当前事实。
- 假设：文档、外部 APK 归档、debug-only 回放和少量授权截图已足以构成轻量展示包。
- 实施门槛：只有在明确的提交或答辩需求出现时，按 [演示指南](docs/DEMO_GUIDE.md) 生成一次性材料并重新验证版本、构建和设备状态。

### DA3 × Metric3D GeoMetric-Lite 双教师轻量几何

- 状态：`待评估 / 候选算法支线`
- 主张：DA3 的空间结构/跨帧几何与 Metric3D 的米制尺度可能互补，可蒸馏为移动端稠密几何学生。
- 保留理由：若教师互补性成立，可能改善 clearance、地面/障碍几何和时序稳定性；但必须先证明互补性，再购买双教师训练复杂度。
- 进入门槛：先做固定数据上的教师互补性、单帧/时序对照、disagreement 与 error 关系及 kill gate；通过后才能建立新版本研究路线。
- 禁止动作：未通过 R0 证据门前不得直接开始完整双教师训练、替代 DepthART/DA2、接入 Android 或改变默认 App。
- 原始方案：[BlindAssist_DA3_Metric3D_GeoMetric-Lite_R0.1.md](D:/edge/BlindAssist_DA3_Metric3D_GeoMetric-Lite_R0.1.md)

### DepthART R1 QAIRT/QNN/HTP 部署准入

- 状态：`待推进 / 已存在主线候选的部署工作包`
- 主张：DepthART-S 可沿 numerical parity、QAIRT graph compatibility、SelectiveScan lowering、HTP graph/runtime/performance 分层验证，争取成为可部署候选；R0 FAIL 必须保持不变。
- 保留理由：这是当前 DepthART 主线的部署与准入执行思路，能够把“算法收益”和“部署可行性”分开管理。
- 进入门槛：按 R1 顺序完成原始 ONNX parity、等价图改写 parity、Camera Embedder 处理、QAIRT 转换、SelectiveScan 实际判定、HTP smoke 和性能；任何阶段未闭合都不能写成 HTP PASS/FAIL 或生产替换。
- 禁止动作：不得修改 R0 FAIL、把转换阻塞提前写成 HTP FAIL、把单机性能当作准确率/安全证据、或直接接入默认 App。
- 原始纲领：[BlindAssist_DepthART_R1_QAIRT_QNN_HTP_部署与准入执行纲领_2026-08-07.md](D:/edge/BlindAssist_DepthART_R1_QAIRT_QNN_HTP_部署与准入执行纲领_2026-08-07.md)

### 异步慢快感知、流式记忆与潜在未来风险

- 状态：`待评估 / 系统与算法交叉想法`
- 主张：通过 slow/fast 异步感知、跨帧 streaming memory、轻量 latent future predictor、uncertainty gate 和 dynamic compute，可能在移动端降低平均计算成本并提升动态风险感知时效性。
- 保留理由：它不是单一模型替换，而是可拆成独立的时序、数据、延迟和性能研究，适合在 DepthART 周边并行验证。
- 去重边界：本条只保留 slow/fast scheduling、streaming memory 与 dynamic-compute 系统想法；
  task-query active observation 已迁入 TARO，4D first-contact/world-belief 算法已由上面的 AC4D 待决项承载。
- 进入门槛：先做单变量、可停止的最小实验：异步刷新收益、记忆/时序模型收益、未来风险预测收益、动态计算节省与最坏延迟；每个方向通过判别实验后再登记为独立路线。
- 禁止动作：不得一次性同时引入十二类技术、把概念综述当作结果、未测量就宣称降低端到端延迟，或绕过 current 入口直接接入默认 App。
- 原始调研：[2024–2026 新型轻量智能学习与实时视觉感知技术路线整理.md](D:/edge/2024–2026%20新型轻量智能学习与实时视觉感知技术路线整理.md)
