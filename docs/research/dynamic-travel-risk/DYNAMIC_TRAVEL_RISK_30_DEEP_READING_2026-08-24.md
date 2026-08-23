# 动态出行风险：30 项论文、算法与项目逐项精读

状态：`DATED_READING_NOTES / PENDING_CANDIDATES / NO_ROUTE_OR_PRODUCTION_AUTHORITY`

日期：2026-08-24

## 1. 为什么重读

[根目录候选池](../../../idea.md)解决的是“先找到 30 个可能有帮助的方向”，但候选卡片不足以支持后续取舍。本笔记回到每项的一手全文、作者项目页、官方数据页或官方仓库，逐项回答六个问题：

1. 它真正解决的是什么问题，而不是标题看起来像什么；
2. 输入、输出和部署假设是什么；
3. 核心机制怎样运作；
4. 实验实际证明了什么；
5. 对 BlindAssist 动态出行风险的核心价值是什么；
6. 读完以后，哪些应借、哪些不应借、还缺什么证据。

这里的“精读”指核对论文正文的方法、实验、消融和局限；项目型条目则核对官方设计文档、代码接口和数据说明。笔记不会把汽车、机器人、合成数据或研究者模拟结果改写成视障用户安全证据，也不会因项目开源就推定它可在当前 Android 产品中运行。

## 2. 统一判断尺度

- **核心机制**：作者系统中真正改变信息或决策的步骤，不以模型名代替解释。
- **核心证据**：数据规模、参与者、对照、指标和关键结果；没有用户试验就明确写没有。
- **核心价值**：只写它能补齐的一个失效层，例如身体路径、未来占用、输入健康或音频调度。
- **读后感**：我的技术判断，明确标为迁移推断，不冒充作者结论。
- **借用决定**：`借`、`只作评测/压力源`、`暂不借`；这只是未来研究优先级，不建立实施 authority。
- **安全语义**：任何条目都不能把“未告警”升级为“前方安全”。缺路线、时钟、位姿、目标未来或输入健康证据时，只能 `UNKNOWN / ABSTAIN`。

---

## DR01 — Collision-point × TTC：最强的人体接触结果锚点

**一手来源**：[2015 受控障碍场研究](https://iovs.arvojournals.org/article.aspx?articleid=2281664)；[2021 居家双盲随机临床试验](https://pmc.ncbi.nlm.nih.gov/articles/PMC8299358/)

**它真正问的问题。** 两项研究共同检验一个类别无关规则：只有预计撞击点靠近使用者且 time-to-collision 足够短才提醒。2015 年问它能否在受控路线减少周边视野损失者的身体接触；2021 年进一步问，在继续使用白杖或其他惯常助具的日常生活中，主动腕部振动是否优于同一设备静默运行。

**核心机制。** 胸前单目相机把相对接近拆成两个变量：空间上的 collision point 与时间上的 TTC；两者同时越过阈值才告警，不依赖 person/chair 等类别。论文没有公开 TTC/撞击点公式、数值阈值或完整跨帧状态机，因此这些不能被“复现式引用”。居家试验把相距不超过 2 秒的 warning instances 合成一个事件，保存前后各 2 秒视频；两名盲法审阅者依次判断视频是否有效、若轨迹保持是否为 true hazard、是否发生非故意身体/白杖接触，正常探杖不计作碰撞。

**核心证据。** 2015 年有 25 名周边视野损失者，在 41 m、46 个障碍的路线各走 8 圈；总碰撞中位数约 `6→3`，回归斜率 `0.63`，即约降低 37%，而低于 15 cm 的地面箱没有改善。2021 年 49 人入组、31 人有成对可用数据，4 周共记录 `368.17 h`；28,733 个设备事件中审阅 16,341 个，只有 4,067 个被判 true hazard。主动与静默模式的全部接触中位数约 `9.26 vs 13.79 / 100 true hazards·h`，调整后 rate ratio `0.63`（95% CI `.54–.73`）。

**最重要的证据边界。** 居家评估样本由设备报警触发；设备完全没看到、从未触发的危险不进入分母，因此全场景危险召回 `NOT_EVALUABLE`。研究证明的是“已触发 true hazard 中振动降低接触”，不是所有危险都可发现，也不是跌倒、受伤或完整安全得到改善。2015 年近地障碍的阴性结果同样说明胸前 TTC 通道不能覆盖白杖层。

**核心价值。** 这是 30 项中最强的真实行为结果锚点：风险应由身体路径相关性与时间紧迫性共同门控；评估必须把算法触发、真实危险和接触结果分层，并同时看行走速度副作用。

**读后感。** 两篇放在一起读，结论比“减少 37% 碰撞”谨慎得多，也更有用：选择性提醒确实可能改变行为，但感知覆盖仍是黑箱。它告诉我们先做事件分母和漏检审计，再讨论更复杂模型；不能用漂亮的条件内因果效应遮住设备没看到的世界。

**取舍。** `优先借` collision-point × TTC 双门、2 秒事件合并、主动/静默对照、盲法 hazard/contact adjudication；`不借`未公开阈值，不把胸前视场或 detected-hazard 分母当完整风险 authority。

---

## DR02 — BBeep：碰撞是双人协调事件，不只是给用户报警

**一手来源**：[CHI 2019 论文 PDF](https://wotipati.github.io/projects/BBeep/paper/CHI%2719_BBeep_preprint.pdf)

**它真正问的问题。** 在机场拥挤环境中，能否只对预测相撞的行人发声，并让盲人和附近明眼行人共同避让。它不试图识别所有物体，而把冲突限定为行人未来路径穿过使用者前方 emergency line。

**核心机制。** 行李箱上的 ZED RGB-D 以 YOLOv2 检测、深度跟踪和 ZED odometry 补偿箱体旋转。3D 检测间距小于 `1 m` 才关联，漏检 5 帧删除；保留 32 个位置，以前后半窗均值差估速度，再作短时匀速外推。若预测直线穿过用户前方宽 `0.7 m` 的 emergency line，则在 5 秒内发低紧迫声、2.5 秒内升级，任何障碍到 `≤70 cm` 时发停车铃。外放声既提醒用户也提醒对方，骨传导耳机只提醒用户。

**核心证据。** 现场观察先比较 7 种声音条件，共 399 条轨迹；声音相对静默增加行人的最近距离，5 秒提示优于 2.5 秒，声型紧迫度本身无显著差异。机场用户研究只有 6 名盲人、约 20 m 直线路线各 5 次，并有安全研究员。imminent events 的均值为外放 `0.41`、耳机 `2.00`、静默 `3.00`；外放显著优于耳机，但总 collision-risk 频次外放与耳机无差异。

**核心价值。** 5 s→2.5 s→立即停止的事件升级，以及“危险有时需要双方共同协调”这两个思想，比其 2019 年 detector/tracker 更重要。

**读后感。** 最有说服力的是外放与耳机的差异：系统改变了周围人的行为，而不只是增强用户感知。但这也引入隐私、羞耻感、公共噪声和对方戴耳机等新失效面；不能因为平均最近距离变大，就默认外放是普适正确的产品选择。

**取舍。** `借`分级持续确认、事件升级和“用户端/公共协同端”分轨；`不借`默认外放、单一 0.7 m 线或恒速直线作为安全 authority。复杂横穿、无避让空间与社会接受度仍需独立证据。

---

## DR03 — HEADS-UP：旋转补偿的数学动机强，效果证据弱

**一手来源**：[论文](https://arxiv.org/abs/2409.20324)；[EPFL VITA 项目页](https://www.epfl.ch/labs/vita/research/prediction/heads-up/)

**它真正问的问题。** 头戴相机的平移和转头会与行人真实运动混在一起；论文希望在“半局部”坐标中直接预测佩戴者与行人的相对轨迹，避免分别预测两个全局轨迹再求交。

**核心机制。** ZED Mini 的像素和深度先反投影 `p_camera=dK^-1[u,v,1]^T`，再由 `p_global=Rp_camera+t` 进入世界坐标。碰撞关系原本涉及行人预测 `f_i(Rp_camera+t)` 与佩戴者预测 `f_i(t)`；对线性 Kalman 情况，平移项可消去，近似把问题写成 `f_i(Rp_camera)≈0`，只要求可靠旋转。数据由 YOLOv8、ByteTrack、2.5 fps 下采样和 Kalman 平滑生成轨迹，短于 6 帧的轨迹删除。

**核心证据。** Easy/Hard/Uncontrolled 三组共 43,213 帧、959 条轨迹，含 RGB、深度、IMU、点云和 VIO pose。固定论文和项目页没有给 ADE/FDE、碰撞 precision/recall、报警时延或可核对的用户试验样本与结果；摘要式“在实时 user study 有效”没有正文表格支撑。轨迹标签又来自 detector/tracker/filter，而非独立物理碰撞真值。

**核心价值。** 它清楚地指出头部旋转是首要混杂，并提供 Easy/Hard/Uncontrolled 分层的穿戴数据，可用于测试 ego-motion compensation 是否在剧烈头动下保持相对运动一致。

**读后感。** 这更像一个值得保留的坐标系假设和数据起点，而不是已经成立的碰撞预测器。细读后，论文最重要的信息反而是“依赖准确旋转”；只要 IMU/VIO 退化，半局部表示也会失去 authority。

**取舍。** `借`旋转补偿、相对坐标和头动分层；`不借`自动平滑轨迹作碰撞真值，也不引用未披露的 user-study 成效。未来先用 source-native pose 验证表示，再谈非线性预测网络。

---

## DR04 — ARAware：严重度决定提前量，但常数几乎全是经验值

**一手来源**：[Sensors 2024 全文](https://www.mdpi.com/1424-8220/24/13/4282)

**它真正问的问题。** 对汽车、摩托车、自行车和行人，统一距离或 TTC 阈值会让慢目标过早提示、快目标来不及反应。系统因此先判断是否进入身体碰撞圆锥，再按类别严重度给不同 alert deadline。

**核心机制。** YOLOv8+DeepSORT 跟踪四类目标，ORB 静态区域匹配补偿相机平面运动；DisNet 用框相对尺寸和类别平均实体尺寸估深度，再反投影到 3D。用户和物体建模为圆，目标速度方向落进由两者半径和距离决定的切线夹角才是 critical moving object。系统估平均相对速度和碰撞时间，设基础反应时间 `5 s`，再按严重/较重/轻微/最低风险增加 `6/4/2/0 s`；达到 deadline 才发 TTS，并先按风险等级、再按 TTC 排序。

**核心证据。** 研究者模拟视障用户，最多 5 名志愿者按预定轨迹驾驶/骑行/步行。VMOT 96 段、28 分钟，VCRP 44 段、15 分钟。论文报告约 32 fps、CMO mAP `88.20%`、mAR `97.26%`、风险分类 mAP `91.69%`、速度平均绝对误差 `2.47 km/h`，碰撞预测平均提前 `2.83 s`。这些是编排视频的检测/分类结果，没有真实用户动作、接触或误报负担终点。

**核心价值。** `severity-conditioned deadline` 和风险优先队列解决了“安静快车应比慢行人更早提示”的产品问题，也提示 TTS 自身 1–2 秒延迟必须进入预算。

**读后感。** 系统链很完整，但安全半径、类别尺寸、反应时间和 severity margin 都是经验常数。mAP 回答的是作者标注规则能否被复现，不回答用户是否有足够空间执行动作。尤其“避开可见目标也很可能避开被遮挡目标”的推论没有证据，不能接受。

**取舍。** `借`风险条件化 deadline、同事件去重和队列排序；`不借`半径、`5+δ` 秒阈值、类别尺寸测距或遮挡安全推断。应改成可校准的事件 lead-time 与动作可执行性评测。

---

## DR05 — MinD：把“危险”转成最短局部动作，但前提几乎全是 oracle

**一手来源**：[ICCV Workshops 2023 全文](https://openaccess.thecvf.com/content/ICCV2023W/ACVR/html/Surougi_Real-Time_Optimisation-Based_Path_Planning_for_Visually_Impaired_People_in_Dynamic_ICCVW_2023_paper.html)

**它真正问的问题。** 当危险目标、其速度、使用者速度和最终目的地都已知时，怎样给出一个移动最少、仍能拉开安全时间、且不远离终点的地面局部目标点。

**核心机制。** 最近目标的距离除以双方速度和低于 safety limit 时启动优化。目标最小化用户移动距离；约束包括最大移动距离、在 `T_max` 内可达、按恒速外推目标、未来分离至少为 `sl(|v_O|+|v_B|)`、离开切线危险锥、最小位移以及不比当前位置更远离最终目的地。非凸约束以 sequential convex programming 在当前解附近线性化，位置变化 `<0.001` 时停止，不可行则重初始化。默认车/摩托、自行车、行人的远近安全时限分别设成不同经验值。

**核心证据。** 正式比较是 3,200 个二维 Monte Carlo 样本：单一目标恒速直冲、完美感知、用户固定 `5 km/h`、终点已知。MinD 平均躲避距离 `1.69 m`，NBRD `8.90 m`，调参 APF `17.27 m`；碰撞率分别 `0%/5.75%/86.94%`，求解约 `0.04 s`。没有 BVI 参与者、真实指令遵从误差、多威胁或感知不确定性。

**核心价值。** 它把风险层与动作层分开：不是多说一句“有车”，而是求一个最小可执行局部目标，并保留朝最终目的地推进的约束。

**读后感。** 优化问题写得清楚，`0%` 碰撞却主要说明约束与同一个理想生成世界自洽。真正困难的是用户能否按时理解/执行，以及目标轨迹、用户速度和定位误差怎样传播到约束；这些都被当成已知。它更适合做后端接口设计，不适合作现成安全算法。

**取舍。** `借`目标函数/约束分离、用户速度参数化和“不可行即停/重新规划”；`不借`固定安全时限、完美恒速预测或 0% 碰撞结论。只有上游事件和动作误差闭合后，才值得评估该层。

---

## DR06 — Corridor-Walker：身体走廊 relevance 有直接用户价值，也有速度代价

**一手来源**：[PACM HCI / MobileHCI 2022 论文 PDF](https://www.masakikuribayashi.com/data/project/masaki_kuribayashi_mobilehci_2022/paper.pdf)

**它真正问的问题。** 仅凭手持 iPhone LiDAR，能否让全盲用户在固定室内走廊中避开与身体走廊相关的障碍、少贴墙并识别路口，而不是播报整个场景。

**核心机制。** 10 fps 点云先按法线与重力、高度 `±0.1 m` 找地面，再以 `0.15 m` 网格累积可走性。障碍邻域成本在 3 个格内随距离衰减；系统在前方 100°、3.5 m 处寻找最长低成本连续空间中点，必要时每次缩短 0.5 m，以 A* 求局部路，再走到一半重规划。RANSAC 去墙后，前方 2 m/30° 区域障碍格超过 30% 才触发；`≤2 m` 用 TTS，`≤1 m` 连续振动，左右方向用 400 Hz 空间音。路口则把 occupancy grid 图像交给 YOLOv3 分类。

**核心证据。** 14 名全盲、日常独立出行且使用白杖者参加，训练 30 分钟；系统+杖与仅白杖在镜像路线中平衡。两条长路线的障碍杖接触约 `3.07→1.28`、`3.71→0.85`，墙接触也大幅下降，SUS `80.5`。代价是明显变慢，例如 37.4 m 路线约 `50.65→69.30 s`。11/14 不喜欢持续手持和稳定姿态，一次发生建图失败；研究主要是静态障碍，测的是杖接触而非身体碰撞。

**核心价值。** 它给“先求身体 clearance corridor，再决定是否提示”提供了比通用检测更直接的实现和真实用户证据，并显式覆盖高处空底家具等白杖较难发现的对象。

**读后感。** 这是一个很好的提醒：减少无关提醒并不一定提升整体出行效率。系统大幅减少墙/杖接触，却让路线慢约三成；所以动态风险也必须同时报告接触、到达时间、停顿和认知负担，不能只优化一项。

**取舍。** `借`身体高度窗、局部 clearance map、远距 TTS→近距振动升级和个体化距离；`不借`固定正交走廊、持续手持姿态或静态结果外推到户外横穿。

---

## DR07 — WOAD：跨模态工程完整，但“100% 避障”不是零碰撞

**一手来源**：[Nature Communications 2025 论文](https://doi.org/10.1038/s41467-025-58085-x)；[补充材料](https://static-content.springer.com/esm/art%3A10.1038%2Fs41467-025-58085-x/MediaObjects/41467_2025_58085_MOESM1_ESM.pdf)；[官方仓库](https://github.com/MMCNJUPT/WOAD)

**它真正问的问题。** 专用 IR+ToF 眼镜、FPGA/ARM、手机和触觉/音频能否在功耗、延迟与多模态退化之间做调度。它兼具检测、传输和反馈，是完整工程原型，但其结果术语需严格解读。

**核心机制。** 深度帧差先估距离、相对速度和方向；D-SAC 根据新/旧/无障碍状态选择丢帧和 320p/1080p 传输，以危险距离 `<3 m` 为奖励信号。IR 与 depth 分别经过 YOLOv5 前层，拼接后用 Transformer 自注意和跨模态残差调制，再进入检测后层。反馈把方向/距离主要交给额头与双颞振动，将 Attention/Warning/Danger 交给音频；近距动态时提高感知与传输强度。

**核心证据。** 12 名 BVI 志愿者据称连续使用 7 个月，代表场景包括突然出现的行人、楼梯和夜间车辆；8 人完成问卷。论文报告延迟 `<320 ms`、功耗 `≤4 W`、约 11 h 续航，以及所有选择场景“100% collision avoidance”。但补充定义把 collision avoidance rate 写成“成功检测障碍数/障碍总数”，本质更接近场景内检测召回，不是无身体碰撞 trial 的比例。假阳性来自阴影/雪花却没有 rate，随机化、盲法和是否同时持杖不清楚。

**仓库核对。** 组件、Android 和数据接口确实存在，但关键数据在外部 Drive，缺少端到端复现。论文描述的 DiscreteSAC 主块在训练脚本中被注释，活动代码使用其他 RL；Android 中 TTS 调用被注释并存在类别/拼写不一致。因此仓库证明“做过这些组件”，不证明当前提交精确复现论文整机。

**核心价值。** 近距危险提升帧率/带宽、IR+ToF 互补，以及把方向交给触觉、风险等级交给音频，都是降低单一音频通道负担的有用设计。

**读后感。** 工程工作量可观，但最醒目的 `100%` 与普通人理解的“零碰撞”不是同一指标。5 m ToF 面对 10–16 m/s 车辆的反应余量也值得怀疑。该项目适合拆部件学习，不适合整体作为安全证据。

**取舍。** `借`危险驱动采样/传输、跨模态一致性和感官分工；`不借`100% 安全表述、固定 3 m/振动常数或复杂硬件栈。误报、真实接触、车辆 lead time 和代码版本必须另审。

---

## DR08 — Project Guideline：最值得读的是失败路径

**一手来源**：[Google Research 官方仓库](https://github.com/google-research/project-guideline)

**它真正问的问题。** 腰部 Pixel 如何让盲/低视力跑者沿一条实体紫线前进，并在引导依据不足或前方占用时发 STOP。官方明确这是研究原型、不是受支持产品，且要求特制宽路和 sighted spotter。

**核心机制。** 分割出的引导线关键点反投影到世界并跨帧聚合；单目 ML depth 用 ARCore 3D 特征经 RANSAC 对齐尺度。occupancy clearance zone 默认前深 5 m、宽 4 m、相对相机高度 `-0.5～+1.2 m`，格内至少 5 点才占用；障碍 presence 经 30 帧窗口、至少 15 帧 latch。ARCore tracking lost、前方关键点不足、横向值 NaN，或 lateral offset `≥2 m` 持续 20 帧时，`ResetAndSendStopSignal` 会清环境、控制和历史队列并发送带原因 STOP。

**核心证据。** 仓库有 simulator、测试、模型和状态代码，但固定一手材料没有 BVI 样本数、障碍召回、碰撞率或对照结果。README 说 STOP 后回初始化，源码仍保留相关 TODO，说明文档状态机和恢复闭环没有完全一致。障碍逻辑也只有占用 presence，没有 TTC、目标轨迹或动态交叉判断。

**核心价值。** `reason-coded STOP + 清空陈旧状态 + 重新初始化` 是极强的 fail-closed 参考。系统把“我不知道还能否继续引导”当成一级输出，而不是继续给方向。

**读后感。** 演示容易让人关注沿线跑步，代码却表明真正成熟的思想是失败路径。与此同时，README 与 TODO 的缝隙证明：写出 STOP 状态不等于恢复流程真的闭合；BlindAssist 必须通过可执行状态测试验证 fail-closed，而不是只复制接口名。

**取舍。** `优先借`带原因 STOP、连续帧确认、当前身体走廊以及初始化/跟踪分离；`不借`紫线基础设施、1.5 m 常数或 occupancy presence 作动态风险判断。

---

## DR09 — Binary TTC：把难回归改成 temporal geofence

**一手来源**：[CVPR 2021 论文](https://openaccess.thecvf.com/content/CVPR2021/html/Badki_Binary_TTC_A_Temporal_Geofence_for_Autonomous_Navigation_CVPR_2021_paper.html)；[官方仓库](https://github.com/NVlabs/BiTTC)

**它真正问的问题。** 不先恢复精确深度、光流和连续 TTC，能否直接回答每个像素“会不会在某个时间阈值内撞到相机平面”，并按计算预算只运行少数关键 horizon。

**核心机制。** 对近似正视平面，TTC 可由两帧图像尺度比 `α` 推得：`τ=(t1-t0)/(1-α)`。方法把第二帧 feature 按每个候选 `τ_i` 对应的 `α_i` 缩放，预测二值概率 `B_τi(x,y)=P(τ(x,y)>τ_i)`，以 BCE 训练；多个阈值之差形成 TTC 区间概率，曲线面积近似连续 TTC。实际在 inverse-TTC/motion-in-depth 域均匀采样，并用光流辅助任务改善特征。

**核心证据。** 在 SceneFlow/KITTI 训练评测，最终 binary mIoU 约 `.9525`、percentage error `1.013`；`384×1152` 单一 geofence 在 V100 约 `6.4 ms`。仅 TTC 训练的 mIoU `.8808`，光流预训练并联合优化升到 `.9525`。失败包括 road bump 的突然垂直运动、剧烈旋转、罕见运动和物体显著自转；证据全部来自汽车数据，没有穿戴式或行人事件。

**核心价值。** 它是低延迟、类别无关的 looming/TTC 候选生成器，可以直接围绕“2 秒内”之类事件 horizon 训练，而不强迫输出噪声很大的精确秒数。

**读后感。** 这是一个非常漂亮的问题重写，但回答的是“撞相机平面”，不是“进入用户未来身体走廊”。纯横向横穿可能早期没有 looming，边缘逼近物也可能与路线无关。它必须位于 route/body gate 之下，不能单独发安全告警。

**取舍。** `借`多 horizon temporal geofence、动态计算预算和辅助 motion 训练；`不借`相机平面 TTC 作 physical route TTC。必须叠加 ego-motion health 与身体路线相交。

---

## DR10 — SOGMP/SOGMP++：未来占用的均值与方差双层

> **名称校正：** 原候选池把固定 PMLR 链接误写成 “SCOPE”。链接实际对应 *Stochastic Occupancy Grid Map Prediction in Dynamic Scenes*，方法名为 **SOGMP/SOGMP++**。正文不存在 SCOPE；本笔记和候选池均按一手全文修正。

**一手来源**：[CoRL 2023 / PMLR 全文](https://proceedings.mlr.press/v229/xie23a.html)

**它真正问的问题。** 资源受限机器人怎样在自身运动、静态几何和多种动态未来同时存在时，预测下一步 occupancy grid 分布，并让规划器对高不确定区域更保守。

**核心机制。** 输入为过去约 1 秒、10 Hz 的 `64×64` 二值 OGM。方法先按恒速预测未来 ego pose，把历史 LiDAR 变换到未来机器人局部坐标，显式消除自运动；ConvLSTM 提取动态占用，SOGMP++ 另用逆传感器 Bayesian map 累积静态几何。VAE 融合动态预测与静态图并自回归未来。规划时在第 6 步（0.6 秒）采样 8 个 OGM，以均值构造 prediction costmap、标准差构造 uncertainty costmap，两者都以软 Gaussian cost 进入局部规划。

**核心证据。** 模拟训练约 94,891 tuples，并在 OGM-Jackal/Spot 真实数据作跨机器人测试。去掉 ego compensation 明显变差；SOGMP++ 的静态图主要改善长时 SSIM。35 人模拟场景中，均值+不确定 costmap 的成功率 `.89`，去掉不确定层 `.86`，普通 DWA `.82`；45 人时 `.82/.79/.77`。Jetson TX2 上 SOGMP 约 `24.83 fps`，SOGMP++ `10.68 fps`。真实拥挤走廊只有一次“无碰撞到达”的定性视频。

**核心价值。** 它把“最可能占用”与“未来不确定区域”分别交给决策层；BlindAssist 可以让两者都与 route tube 相交，而不是只看一条最可能轨迹。

**读后感。** ego-motion 和静态/动态分支的消融很有说服力；所谓 uncertainty 则只是 VAE 样本方差，没有证明校准 coverage 或正确 abstention。它是一个占用表示参考，不是已经可靠的风险概率。

**取舍。** `借`future-local-frame 补偿、静/动态分支、均值+方差双 costmap；`不借`VAE 方差作为风险保证，也不外推机器人模拟成功率到人类可执行提醒。

---

## DR11 — RiskProp：碰撞锚点能弱监督时间风险，但单调先验不适合解除事件

**一手来源**：[CVPR 2026 论文](https://openaccess.thecvf.com/content/CVPR2026/html/Zou_RiskProp_Collision-Anchored_Self-Supervised_Risk_Propagation_For_Early_Accident_Anticipation_CVPR_2026_paper.html)；[官方仓库](https://github.com/xingyueye5/RiskProp)

**它真正问的问题。** 事故从哪一帧“开始危险”很主观；能否只相信碰撞帧和远离事故的初始帧，把风险从碰撞锚点向前传播，而不人工稠密标注 onset。

**核心机制。** 因果滑窗视频输出当前风险 `a_t=σ(z_t)`。事故视频只把初始帧标 0、碰撞帧标 1，安全视频全帧为 0；Future-to-Former Regularization 用停止梯度的后帧 logit 约束前帧：`Σ||detach(z_{t+1})-z_t||²`。Adaptive Monotonic Constraint 对任意后帧/前帧对惩罚风险下降，margin 随时间差和预测置信度变化。官方配置为 SlowOnly-R50、5 帧片段、10 fps；主要参数 `λ_reg=1.5`、`λ_mono=1.1`。

**核心证据。** CAP 有 11,727 段、约 219 万帧、58 类事故；Nexar 训练约 1,500 段、测试 1,344 段。CAP 的 mAUC@FPR≤0.1 `.483`、mAP `.890`、mTTA `1.207 s`；Nexar 分别 `.472/.870/.958 s`。从只用碰撞锚点到加 FFR、再加 AMC，CAP mAUC `.358→.474→.483`，说明主要收益来自后向时间正则，单调约束只贡献小增量。

**核心价值。** 碰撞/接触锚点、低假警约束下的首次报警提前量，以及不用主观逐帧风险标签的训练方式，都可用于事件 onset 研究。

**读后感。** 这是一种好用的弱标签时间正则，不是风险模型。所有正例最终都会碰撞，所以“风险应单调上升”在数据里合理；步行中用户停步、对方横穿后离开或相机恢复时，风险必须下降并产生 clear。照搬单调性会制造长尾 nuisance alert。

**取舍。** `借`碰撞锚点、FPR 受限 lead time 和轻量时间正则；`不借`全程单调 prior、汽车分数或未报告的部署阈值。BlindAssist 需要显式 onset–peak–release 生命周期。

---

## DR12 — Conformal Risk Tube Prediction：evaluator 比当前实现更值得借

**一手来源**：[论文](https://arxiv.org/abs/2603.23919)；[项目页](https://hcis-lab.github.io/CRTP/)；[官方仓库](https://github.com/HCIS-Lab/CRTP)

**它真正问的问题。** 逐帧二元风险会闪烁，且不能表达风险开始/结束边界和模糊区；项目因此为每个对象预测未来风险序列，再以 conformal buffer 生成时空风险管。

**核心机制。** 输入 3 帧 RGB、对象框/ID和遮挡 phantom box，4 fps，输出未来 8 步、四类风险的分数。每个风险类别和 horizon 独立以 `S_t=|g_t-p_t|` 校准分位数 `q_t`：`p≥1-q` 为风险、`p≤q` 为无风险，中间为 ambiguous；在线用 SAOCP 更新分位数。评估不只看帧准确率：Coverage 要求整段 GT 区间被管覆盖，Tube Volume 衡量保守度，Temporal Consistency 惩罚切换，Boundary Accuracy 衡量 onset/release 偏移，组合成 Risk-IoU。

**核心证据。** 数据约 1,000 个 CARLA 场景，交互/碰撞/障碍/遮挡四类可并发；单风险完整模型 Coverage `.851`、Risk-IoU `.637`，多风险 `.827/.569`。多风险从 Base→+STFA→+CACC 的 Risk-IoU `.527→.559→.609`；检测框扰动使 Coverage 降约 `.104`、管体积增 `5.18`。这些仍是模拟数据。

**仓库审计边界。** 论文描述 I3D 与 STFA，但仓库实际使用 timm ResNet50/partial-conv，相关损失不等同论文公式且截断前 12 个框。论文声称独立 calibration split，代码主要加载 training/validation，并在训练/验证期间更新、序列化 SAOCP；`coverage=0.7` 等关键语义没有在论文充分说明。于是“conformal guarantee”不能仅凭标题接受。

**核心价值。** 三态风险、区间 coverage、onset/release、切换次数和管体积，几乎正对 BlindAssist 的“不要闪烁、不要过早、要会解除、证据不足要不确定”。

**读后感。** 这是 30 项里“评测设计明显强于模型/仓库证据”的典型。即使最后完全不用其网络，risk tube evaluator 仍值得保留；但 calibration split 和实现不一致若不先修，统计保证就是空的。

**取舍。** `优先借`三态输出、边界/覆盖/切换/干预负担指标；`暂不借`其模型或覆盖保证。未来实现必须独立冻结 train/cal/test、风险类别和 α。

---

## DR13 — PIE：横穿意图不等于进入我的路径

**一手来源**：[ICCV 2019 论文 PDF](https://openaccess.thecvf.com/content_ICCV_2019/papers/Rasouli_PIE_A_Large-Scale_Dataset_and_Models_for_Pedestrian_Intention_Estimation_ICCV_2019_paper.pdf)；[官方仓库](https://github.com/aras62/PIE)

**它真正问的问题。** 从车载第一视角的行人 crop、框轨迹、场景和 ego 速度，预测行人是否有横穿意图，以及未来 0.5–1.5 秒 2D 框轨迹。

**核心机制。** 5 级观察者意图评分被缩放并舍入成二元 crossing 标签；意图模型用 VGG16/ConvLSTM/LSTM，轨迹模型用观测位置、预测速度和可选意图，编码器含时间注意力。任务通常使用标注框，真实 detector 失败不是主问题。数据还含遮挡、动作、看向、交通设施和 OBD/GPS 等上下文。

**核心证据。** 多伦多日间约 6 小时、53 段、6 条连续路线，约 91.1 万帧、1,842 名行人、74 万框。实验室与 AMT 汇集超过 700 人、27,630 个回答，标注一致性高。上下文+框轨迹的意图 Acc/F1 `.79/.87`，仅位置 LSTM `.63/.73`；1.5 s 轨迹 MSE 从位置 `636`，加意图/速度降到 `559`，使用 GT 意图与速度的 oracle 为 `473`。

**关键边界。** 意图是观察者共识，不是行人本人心理真值；每条 track 只有一个预关键点意图值。像素 MSE 受尺度影响，没有身体路径相交、碰撞、告警或 abstention 指标。论文按行人随机 split，而当前仓库默认已变成路线级 split，复现必须冻结模式。

**核心价值。** PIE 可作为真实第一视角行人动作/遮挡/ego-motion 的前置数据，并提供 `crossing` 与 `crossing-irrelevant` 的语义提醒：想过马路的人不一定与当前用户相关。

**读后感。** 数据和标签边界比 2019 模型更有价值。最危险的误用是把“会横穿”升级为“会撞我”；BlindAssist 必须再与 wearer route tube、时间和 closest approach 相交。

**取舍。** `只作数据/分层`，借用遮挡、动作、ego 速度和横穿无关负例；`不借`观察者意图为风险 truth，也不把 oracle 消融当部署性能。

---

## DR14 — RiskBench：最完整的是事件 evaluator，不是任何 baseline

**一手来源**：[ICRA 2024 论文](https://arxiv.org/abs/2312.01659)；[官方仓库](https://github.com/HCIS-Lab/RiskBench)

**它真正问的问题。** 不同风险识别方法使用不同数据和指标，无法比较；RiskBench 因此建立交互、碰撞、障碍、非交互场景，并测风险对象识别是否真正改变下游规划。

**核心机制。** 每个道路参与者/意外事件输出风险分数。定位报 precision/recall，非交互场景报 frame-level FA；`PIC` 对越接近关键点的错误施加更大惩罚。规划感知指标 `IR=|D_orig-D_post|/D_orig` 比较原场景与只保留模型认为风险对象后的理想规划器最近距离，并另报 collision rate。时间一致性要求关键点前 1/2/3 秒都持续识别正确。

**核心证据。** CARLA 6,916 个场景，测试 1,689 个，覆盖 4 类、6 种 actor 行为、14 地图、21 天气/光照。总体最佳简单 Range-10m F1 `53.6`，但 FA `15.2%`；学习法 RRL F1 `48.6`。QCNet 持续正确率从 1 秒前 `50.2%` 降至 3 秒前 `18.5%`，DSA 更低，直观展示逐帧看似可用的模型在事件时间线上会断裂。即便用 GT risk，planner 仍有非零碰撞，说明下游也不是 oracle。

**关键边界。** 全部是模拟，视觉基线还获得 GT tracklet。IR 使用特权 BEV planner并排除其无法处理的 Collision 场景；仓库一致性代码对缺失帧较宽松，可能轻微高估。CARLA 策略不能成为步行 truth。

**核心价值。** 风险对象、无风险负场景、提前量、持续正确、假警和下游行动后果，构成非常接近 BlindAssist 所需的事件评测骨架。

**读后感。** 表格清楚证明“轨迹更准”不自动变成“风险识别稳定”，更不自动让 planner 安全。这个 benchmark 最值得抄的是问题结构和反事实 evaluator，不是排行榜模型。

**取舍。** `优先借`taxonomy、PIC/lead time、持续正确、FA 和移除对象后的行动反事实；`不借`CARLA 数字、GT tracklet 或特权 planner 当 BlindAssist authority。

---

## DR15 — OF-VO：用速度障碍把相对运动转成候选动作

**一手来源**：[论文全文](https://arxiv.org/abs/2004.10976)；[作者项目页](https://gamma.umd.edu/ofvo/)

**它真正问的问题。** 在部分可观测且感知带噪的动态人群中，怎样把实例分割、光流和 LiDAR 的相对运动转成可解释的 velocity-obstacle 约束，再选择最接近期望方向的可行机器人速度。

**核心机制。** Mask R-CNN 得实例 mask，FlowNet2 得光流，LiDAR 投影后在 mask 内加权求 3D 位置，相邻帧差得速度。若相对速度 `v-v_i` 在某时刻使 `||p_i-(v-v_i)t||²` 小于双方半径平方，则落入速度障碍。论文以高斯误差和 `kσ` 扩张，再用 Cantelli 不等式给单约束概率下界；相机外、LiDAR 内的目标则枚举保守速度。最后与相机 FOV、部分观测和非完整运动学可行集相交，选择离 `v_pref` 最近的速度。

**核心证据。** Gazebo 五类场景各 200 次；真实 Turtlebot2 只有定性演示。`k=1` 时 Dynamic/Cross/Social 成功率约 `90/80/92%`，明显高于比较方法。`k` 从 0.1 增至 2，Cross 成功率约 `16→83%`，但时间 `10.1→23.67 s`，清楚展示保守度的安全—效率交换。

**关键边界。** 假设近处行人不突然变速、误差各向同性高斯、后方目标不会故意冲撞；作者承认局部冻结、光流大位移、FOV 外高速目标和速度抖动。单对象概率下界也不能自然变成多对象系统保证。

**核心价值。** 它把“目标在动”转成“哪些用户动作会进入碰撞锥”，比纯风险分数更接近可行动输出，并保留可调保守度。

**读后感。** 思路仍强，所谓概率保证却比表面弱。对 BlindAssist 最值得复用的是 collision-cone 几何与 `k` 的效率曲线，而不是重传感器管线或独立安全声明。

**取舍。** `借`相对运动→速度障碍→候选动作，以及不可行时降速/停；`不借`单对象高斯下界、LiDAR 栈或模拟成功率。多目标、头动和人体执行误差要重新定义。

---

## DR16 — Fast Incoming Objects：一个可解释的快速逼近辅助通道

**一手来源**：[BMVC 2016 论文 PDF](https://www.bmva-archive.org.uk/bmvc/2016/papers/paper146/paper146.pdf)

**它真正问的问题。** 仅凭移动、未标定的单目相机，能否在鸟、球或无人机等小型高速目标撞上相机前发现 looming，并给出图像平面的规避位置。

**核心机制。** 从稠密光流最小二乘估 focus of expansion，再以 `τ_i=||x_i-F||/||u_i||` 得像素 TTC。仿射全局运动补偿后，把图像分成 `20×20` cell；残差光流第 98 百分位经 sigmoid 形成来物置信度，TTC 取第 2 百分位。每格维护 15 帧背景 buffer 和 3 帧延迟，当前差异超过背景阈值或置信度 `≥0.75` 才判 incoming。倒高斯风险与“safe point”先验相乘，选图像平面安全点。

**核心证据。** 只有 8 个真实碰撞视频、1,367 帧、25 Hz；平均 F-score `.51`，优于比较基线，但仍很低。能在碰撞前约 10–40 帧触发，即 `0.4–1.6 s`。数据全是碰撞视频，没有长时间无事件假警率、闭环规避率或用户结果；光流噪声可让目标消失多达 15 帧。

**核心价值。** 它可作为类别无关的“快速逼近”独立候选通道，并明确要求先补偿 ego motion、再看 residual/looming，而不是把全局光流当风险。

**读后感。** 这是一个清楚、诚实的小 primitive：低 F-score 和短提前量决定它只能进融合层。图像里的 safe pixel 尤其不能翻译成可走方向；没有地面、身体或路线几何时，那只是画面空处。

**取舍。** `借`背景自适应、ego-motion residual、TTC 趋势和首次触发 lead time；`不借`safe pixel、仿射模型或独立防撞主张。应与身体 route tube、输入健康和事件 persistence 相交。

---

## DR17 — SelectiveNet：强迫系统报告“覆盖多少、接受后错多少”

**一手来源**：[ICML 2019 / PMLR 全文](https://proceedings.mlr.press/v97/geifman19a.html)；[作者代码](https://github.com/geifmany/SelectiveNet)

**它真正问的问题。** 与其训练一个全覆盖预测器再事后设阈值，能否把预测和拒答联合训练，在目标 coverage 下直接最小化被接受样本的风险。

**核心机制。** 模型输出预测 `f(x)` 与 selection probability `g(x)`。覆盖 `φ=E[g]`，选择风险 `R=E[l·g]/φ`；约束 `φ≥c` 被写成 `L_sel=R_hat+λ max(0,c-φ_hat)²`。三头结构另设辅助预测头，让被拒样本仍参与表示学习；总损失以 `α=0.5` 混合 selective 与 auxiliary loss。实际接受阈值必须用独立验证集的 `g` 分位数校准，默认 `g≥0.5` 并不保证目标 coverage。

**核心证据。** 在 CIFAR-10、SVHN、Cats-vs-Dogs 和混凝土回归上，指定较低 coverage 可显著降低接受集错误；例如 CIFAR-10 coverage `.95/.90/.80/.70` 时错误率约 `4.16/2.43/.86/.32%`。目标 `.75` 若不校准，实际 coverage 曾达到 `80.17%`，说明覆盖阈值校准不可省略。实验是静态 i.i.d. 通用任务，没有 OOD、时序事件或危险代价。

**核心价值。** 它提供一个必须画出的 risk–coverage 关系：系统发出多少动态提示，发出的那些错多少；而不是把一次 accuracy 藏在全覆盖平均里。

**读后感。** 最有价值的不是三头网络，而是纪律。BlindAssist 的拒答不能只追求低 selective risk，因为拒掉关键危险也可能最糟；coverage 必须按事件严重度、onset 和输入健康分层，并同时报告 critical miss。

**取舍。** `借`独立 selection head、coverage 校准、risk–coverage/AURC；`不借`通用 coverage 目标为安全目标。必须另计关键事件漏报、首次报警、重复负担和紧急 fallback。

---

## DR18 — Perceive With Confidence：最清楚的 fail-closed 统计边界

**一手来源**：[CoRL / PMLR 全文](https://proceedings.mlr.press/v270/dixit25a.html)；[官方软件](https://github.com/irom-princeton/perception-guarantees)

**它真正问的问题。** 对静态室内导航，怎样用独立校准环境为黑盒 3D 障碍检测选择膨胀半径，使 planner 只进入已知 free space，并考虑 planner 导致的闭环状态分布变化。

**核心机制。** 对每个校准环境，寻找一个最小 nonconformity `U_i`，使固定状态样本中所有可见 GT 障碍都被检测框加 `Δq` 包含；再用 order statistic/Beta 界选 `q_hat`，以数据集条件概率控制新环境 misdetection rate。初始障碍框保守膨胀；随着多视角观察，occupied 取交、free 取并，以 non-deterministic filter 收缩 unknown。FMT* planner 只在已知 free space 内规划，并以 inevitable-collision-state 约束动作。

**核心证据。** 400 个 3D-Front/PyBullet 校准环境、每环境 2,000 配置，100 个新仿真测试；另有 Unitree Go1+ZED2i 的 30 个真实布局、每布局 4–8 把椅子。`ε=.15` 时仿真选择 `q≈.75 m`，碰撞 `0%`、misdetection `7%`、goal success `90%`；硬件用仿真 `q≈.73 m` 时实证安全 `90%`，平均 conformal baseline 有 `50%` 碰撞。

**关键边界。** 定理依赖静态障碍、准确状态、校准/部署环境 i.i.d. 和固定有限状态样本；硬件布局并非严格从仿真分布抽取，所以硬件结果不是同一形式保证。RGB-D、VIO 和 safe planner 也远强于单目穿戴输入。

**核心价值。** “最坏环境遗漏→膨胀占用→只走已知 free→新证据才收缩 unknown”是非常适合 BlindAssist 的 fail-closed 结构；它还提醒我们校准必须按环境，而非随机混帧。

**读后感。** 这是形式边界最诚实的论文之一，也因此最不能随意迁移。其价值不是借一个 `ε` 数字，而是学习怎样把保证的条件逐条写出来；动态行人、开放街道和单目输入会破坏几乎全部原假设。

**取舍。** `借`环境级 nonconformity、独立 calibration、占用膨胀与 `UNKNOWN`；`不借`原 `ε` 保证。若重建，需动态事件真值、wearer route、分布单位和安全策略全部重新定义。

---

## DR19 — Camera Physical-State Monitoring：优化任务性能，不追求“好看”

**一手来源**：[DLR 官方论文页及全文](https://elib.dlr.de/147340/)；[官方代码](https://github.com/MaikWischow/Camera-Condition-Monitoring)

**它真正问的问题。** 相机模糊/噪声如何影响特定 detector，以及能否通过曝光与 ISO 的交换最大化下游检测 AP，而非最大化通用图像质量。

**核心机制。** 离线对 Sim/KITTI/Udacity 图像注入物理 blur/noise，运行 YOLOv4/7 和 Faster R-CNN，拟合 `(noise σ, MTF blur, AP)` 曲面。在线 CNN 从连续 patch 估 MTF 和噪声；利用 exposure 增加会加 blur、ISO 增加会加 noise 的关系，求使任务 AP 最大的比例 `α*`，成对调整 exposure/ISO 以尽量保持亮度。

**核心证据。** 噪声实验每数据集约 1,000 图，模糊约 150 图；硬件为 Allied Vision 相机和 Jaguar 4×4。停车场 YOLOv4 AP 从内置 auto exposure `26.08%`、固定手工 `47.54%` 提到框架调整后 `60.56%`。CNN blur 一次四 patch 约 `.24 s`，传统法约 13 秒；但真实移动硬件一次完整估计约 3 秒，不能称高频连续控制。

**关键边界。** 只建模 blur/noise，曲面强依赖相机、detector 和域；组合退化会互相破坏估计，噪声较大时 blur 不可信。没有导航事件、长时假警、控制震荡或参数变化对其他任务副作用评测。

**核心价值。** 相机健康应是独立 evidence channel；必要时主动缩短曝光以保护动态目标。最关键的问题不是“图像是否漂亮”，而是“当前输入是否仍支持路径风险任务”。

**读后感。** 这比通用 IQA 更工程化，也暴露一个重要成本：任务相关健康曲面需要本机相机和本任务校准，不能复用汽车 AP。它适合作门，不适合作安全结论。

**取舍。** `借`blur/noise monitor、任务相关校准和相机主动控制接口；`不借`汽车 IOPC、YOLO AP 曲面或 health=clear 语义。

---

## DR20 — GSHI：产品接口贴题，安全指数仍是自定义目标

**一手来源**：[2026 预印本全文](https://arxiv.org/abs/2605.05439)

**它真正问的问题。** 从单帧 RGB 同时估计 12 类退化的存在、严重度、空间坏区和全局健康，在 detector 明显失效前提前告警。

**核心机制。** EfficientNet-B2 多任务网络输出 degradation presence/severity、全局 `H` 和空间 uncertainty map `U`；对象可靠度是框内 `1-mean(U)`。训练退化包括雨雪雾、glare、vignette、运动/离焦模糊、噪声、曝光、压缩和遮挡。全局目标由手工乘积 `H=∏(1-s_i)^(w_i α_g)` 生成，lens occlusion、glare、blur 权重较高；网络同时直接回归 `H` 并受结构化公式监督。

**核心证据。** KITTI 上在线合成退化训练，冻结 YOLOv8n 定义“任务何时下降 20%”；DAWN 只作天气分类迁移。论文报告 Health MAE `.064`、issue mAP `.891`、平均提前 `.47±.25` 个归一化 severity 单位，DAWN balanced accuracy `84.2%`，RTX5090 `2.27 ms`。但训练样本数、split、optimizer/epochs 等关键复现信息未披露，也无官方代码。

**关键边界。** `H*` 与权重本来就是作者公式，低 MAE首先说明模型学会该公式，不证明公式等于真实安全。所谓提前量是合成 severity 差，不是秒/米；所有模式 100% trigger 也没有长期假警负担。defocus 与 detector mAP 高相关，lens occlusion 相关却较弱，说明手工 health 排序和任务失败并不总一致。

**核心价值。** 四输出 contract 很贴产品：全局健康决定本帧是否有资格发动态提示，空间坏区可判断路线区域是否被遮挡，退化类型/严重度则生成可理解的 fallback。

**读后感。** 接口设计值得借，`Safety-Critical` 数字暂时不值得信。它目前是合成退化诊断器，而不是已校准的安全指数；最好的用法是把它作为待证 hypothesis，与真实事件 coverage 对齐。

**取舍。** `借`四输出接口、空间坏区与 task-failure lead-time 协议；`不借`作者权重、`.9/.6` 阈值、440 fps 或功能安全表述。必须用 BlindAssist 自身输入/事件校准。

---

## DR21 — Evidence of Absence：声学接近证据与视觉缺席

**一手来源**：[论文全文](https://arxiv.org/html/2608.14952)

**它真正问的问题。** 这是一篇面向汽车盲路口的 2026-08 工作稿：当车辆被建筑遮挡、视觉没有共证据时，能否用发动机或轮胎声给出“可能有隐藏来车”的 advisory。它不是安静电动车检测器，也没有预测行人身体路径；隐状态主要是声源是否存在、方位和径向接近。

**核心机制。**

1. 56 麦克风阵列用 GCC/SRP-PHAT 估计声源方位。
2. 稳定发动机谱线存在时尝试 tonal Doppler；谱线不稳定时，以宽带能量 looming `d(log E)/dt = 2/TTA` 作为接近次序信号。
3. 在“隐藏危险源 / 可见无害源 / 无源”三个解释之间做贝叶斯竞争。视觉没有检测到目标并不直接等于不存在，而是经类别、方位和距离相关的视觉漏检率 `β_c(θ,d)` 进入似然；当视觉很差、`β` 接近 1 时，“没看见”几乎不提供反证。
4. 用 Neyman–Pearson 假警预算 `α` 设提醒阈值，再经 isotonic calibration 把证据映射为风险分数。输出仅是 advisory，不接管控制。

**核心证据。** 静态测试有 83 段录音（42 个接近、41 个无危险），动态测试 59 段。静态、`α=0.02` 时报告检测率 `0.88`、平均提前 `1.69 s`、约 `107` 次假警/小时；相对持续声学基线，假警数降低约 42%。但无危险音频只有约 `0.103 h`，所以假警率置信区间极宽。tonal Doppler 在 41 次驶过中的 40 次失败，主要因为发动机转速变化远大于多普勒效应；宽带 looming 与真实 metric TTA 的相关只有约 `0.07`，更适合作次序证据。平台一移动，AUROC 从约 `0.70` 降到 `0.56`，同一假警预算下检测率约 `0.29`。

**核心价值。** 最值得保留的不是麦克风阵列，而是“absence 只能相对于已知漏检模型解释”以及“在固定假警预算下只发 advisory”这两个推理接口。它恰好支持 BlindAssist 的关键语义：相机未见目标不能自动生成安全结论。

**读后感。** 论文的认识论很强，产品匹配却很弱。移动佩戴者、自身衣物/脚步/交通噪声以及安静电动车和自行车，正好击中它最薄弱的条件。标题容易让人以为这是跨模态缺失检测方案，细读后更像一个很诚实的反例：缺席证据只有在传感器可检出性已知时才有意义。

**取舍。** `借`视觉缺席的概率语义、固定假警预算和 advisory-only 输出；`暂不借`阵列声学方案。若未来重开，必须先做移动佩戴者、开放耳机和安静目标的独立可辨识性测试。

---

## DR22 — Indoor Imminent Danger Detection：去地后的未知障碍残差

**一手来源**：[IEEE 论文 PDF](https://ieeexplore.ieee.org/ielx7/6287639/8948470/09211506.pdf)

**它真正问的问题。** 系统试图在室内为视障者检测一米内的临近危险，同时避免把地板纹理或涂色当障碍。关键并不是 SSD 能识别多少类别，而是先找地面，再把“非地面点云”当作类别未知的障碍候选。

**核心机制。**

1. Intel RealSense D435i 的深度先做 decimation、空间、时间和孔洞填补滤波；IMU 用于估计朝向和设备高度。
2. 在潜在地面 ROI 内，以法向角约束和 `0.1 m` 距离约束做 RANSAC 地面平面拟合。
3. 将地面映射到 RGB 并在语义检测前遮掉，减少地面图案造成的误检。
4. 删除地面后，对身体附近的剩余点云做 passthrough 和欧氏聚类。这个通道不要求障碍属于已知类别。
5. 距离 `0.3–1 m` 时发低音空间提示，`<0.3 m` 或找不到路径时发连续/高音停止提示；SSD-MobileNet-v2 的场景描述由用户按需触发，避免持续播报类别。

**核心证据。** 原型由 Raspberry Pi 4 本地运行，约 3 Hz；论文给出的算法平均处理时间约 `358 ms`，按需 CNN 不计在主要实时链路内。缩小地面 ROI 使地面分割耗时降低约 `50.85%–73.73%`。实验受 COVID 限制，仅在一位作者家中用瓶子、椅子、异常物体和蒙眼行走者做定性场景，没有正式盲人参与者、碰撞率对照或室外结果。作者也明确承认不能检测向下楼梯，音频可能干扰盲人的环境听觉。

**核心价值。** `ground/free-space → residual obstacle` 是比“检测到 person/chair”更贴近风险的类别无关通道，可用于临时围挡、广告牌或训练集外物体；语义描述与紧急停止分离也符合选择性介入。

**读后感。** 这篇最有用的部分恰好不是论文强调的物体识别，而是几何残差。论文中“保证”未知障碍检测的措辞明显超过三个住宅场景能支持的证据。固定一米阈值也忽略用户速度、目标速度和制动时间，只能代表一个原型规则，不能代表风险。

**取舍。** `借`独立的类别未知残差通道和分级反馈语义；`不借`固定距离即危险、找不到路径即停止的充分性。室外阳光、稀疏细杆、深度孔洞和下行台阶必须另设评测。

---

## DR23 — Stereo + Motion Drop-off Detection：把负障碍与普通物体分开

**一手来源**：[IROS 2008 论文 PDF](https://homepages.inf.ed.ac.uk/msridhar/Papers/iros08_doffDetect.pdf)

**它真正问的问题。** 对机器人而言，坑、路缘和下行台阶不是“前方有一个凸起物”，而是地面突然消失。论文把局部安全图的状态显式分为障碍/悬空物、drop-off 边缘、地面、低于地面和 unknown，并把 unknown 当作不安全。

**核心机制。**

1. 相关匹配 stereo 是基线。
2. 全局 stereo 先按颜色分割，再在欧氏世界里为片段拟合垂直/水平平面，并以全局能量 `E(f)=Σ C(S,f(S))+Σ L_SS'·δ(f(S)≠f(S'))` 联合优化相邻片段的平面标签，最后把平面累积到 3D 栅格。
3. motion drop-off 通道匹配相隔 `N=5` 帧的近水平边缘，比较边缘上下区域的运动差 `δ_ab = δ_a - δ_b`；若差值越过阈值，就把该边缘视为潜在遮挡边界并投影到地面栅格。
4. 各通道融合到安全图；相对地面高/低超过约 `0.1 m` 的单元进入正障碍或负障碍状态，unknown 不被解释为自由空间。

**核心证据。** 四个数据集各约 500 对 stereo 图像，来自两个室内、两个室外场景，激光地图经人工清理作为真值。相关 stereo 的平均 precision/recall 约 `81%/93%`，全局 stereo 约 `92%/93%`。motion 通道发现了 5 个正前方 drop-off，同时有 7 个假阳性；可见的落差约 `10–15 cm`。实现非常慢：相关 stereo 约 2 秒/帧、全局 stereo 4.5 秒/帧、motion 6.7 秒/帧，而且 motion 需要多帧延迟。

**核心价值。** 它提供一个至今仍有用的失效分层：正障碍、负障碍和未知必须分别建模；一个高召回但高假警的 drop-off 通道可以由独立几何证据约束，而不是塞进通用 detector。

**读后感。** 机制老旧，问题分解却比很多新系统更诚实。它不把无深度返回当自由空间，也没有把所有危险强行归入物体类别。真正不能沿用的是累计栅格：作者指出若没有衰减或负证据，长时间运行最终可能把所有区域都标成占用。

**取舍。** `借`状态空间和独立负障碍评测；`暂不借`stereo/motion 实现。未来单目或深度方案也必须保留 `UNKNOWN`，不能用“没有凸起物”替代“没有坑”。

---

## DR24 — JRDB：拥挤与长遮挡的前置压力集，不是风险真值

**一手来源**：[JRDB 官方数据页](https://jrdb.erc.monash.edu/)；[原始论文全文](https://arxiv.org/html/1910.11792v4)

**它真正提供什么。** JRDB 是人类高度移动机器人的拥挤社会环境数据集，核心是 2D/3D 行人检测、持续身份和遮挡，而不是“谁将进入用户身体路径”的事件标签。

**核心机制/资产。** 传感器包括 360° stereo RGB（15 Hz）、两台 Velodyne 16 线 LiDAR（10 Hz）、两台 SICK、RGB-D、鱼眼、音频、IMU 和编码器。原始标注覆盖上排相机与 Velodyne：2D/3D 人框、跨帧 ID、遮挡程度、距离和方位。后来扩展动作、姿态、全景分割和轨迹预测任务。原始人工标注在 `7.5 Hz` 完成后线性插值到 `15 Hz`，因此不能把每个 15 Hz 框都称作独立人工真值。

**核心证据。** 原始版本有 54 段、约 64 分钟、30 个地点，约 240 万个 2D 框、180 万个 3D 框和 3500 条轨迹；33 段室内、21 段室外，32 段移动、22 段静止。论文基线显示长遮挡仍造成数千次 ID switch；3D 检测在 15 m 以外明显恶化。当前官方页还列出动作、pose 和 panoptic 扩展，但不同版本的样本口径必须随下载版本冻结。

**核心价值。** 它适合作为“如果连密集人群中的相对运动和遮挡恢复都做不到，就不应进入风险层”的前置压力测试，并可按遮挡、距离、室内外和平台运动分层。

**读后感。** JRDB 很容易因多模态和标签丰富而被误当成动态风险数据。细读后它更像一个强力故障放大镜：能揭示跟踪断裂，却没有用户 intended route、身体包络、closest approach、风险 onset 或 clear。用持久 ID 派生相对轨迹是合理的；从轨迹自行造“碰撞”标签则会把评测假设伪装成数据真值。

**取舍。** `只作评测/压力源`，用于人群检测、遮挡与轨迹连续性；不得承担 route-risk 功能真值。下载需按官方当时的账户、许可和版本重新核对。

---

## DR25 — Aria Digital Twin：穿戴式几何与时钟的 canary

**一手来源**：[Aria Digital Twin 官方页](https://www.projectaria.com/datasets/adt/)；[论文全文](https://arxiv.org/html/2306.06362)

**它真正提供什么。** ADT 把真实 Project Aria 采集与一个度量一致的数字孪生对齐，目标是设备、对象、人体和场景的高精度 3D ground truth。它提供几何与同步 authority，不提供出行危险事件。

**核心机制/资产。** 每段含一台 Aria 的 RGB、两台 outward monochrome、双 IMU、眼动相机及完整标定；真值包括设备与对象 6DoF pose、3D 人体骨架、3D gaze、深度、2D/3D 框、mask 和匹配的 photorealistic synthetic twin。构建流程以激光扫描场景/对象和 mocap 为几何基准，再与 Aria 时间线、坐标系对齐。

**核心证据。** 官方当前页给出约 200 段、400 分钟、两个室内真实场景和 398 个对象，其中 324 个静态、74 个动态。论文早期文本与当前发布页在静/动态对象计数上略有漂移，说明以后使用时必须冻结 release manifest，不能混合版本数字。活动主要是室内日常操作，不是开放道路行走。

**核心价值。** 它非常适合测试 BlindAssist 最容易被忽略的机械正确性：视频/IMU 时钟、坐标变换、头动、设备 pose、对象 pose 到当前帧投影、深度和 mask 的数据谱系。一个算法若在 ADT 的已知几何上都无法保持投影和时序一致，就不应讨论真实动态风险。

**读后感。** 这是 30 项里几何 authority 最强、也最容易被过度使用的资产之一。高精度 pose 会给人“风险也有真值”的错觉，但两个房间里的日常动作并不闭合身体路径碰撞事件。它最合适的角色是 adapter canary，而不是训练/证明风险模型。

**取舍。** `借`同步、坐标、头动和当前帧投影测试；`不借`其活动作为动态出行效果证据。数据下载条款、工具版本和 release 身份需要每次现场冻结。

---

## DR26 — REveL：高频跨模态运动数据比论文管线更有价值

**一手来源**：[REveL 官方项目页](https://uts-ri.github.io/revel/)；[论文全文](https://arxiv.org/html/2408.13394)

**它真正提供什么。** REveL 研究快速相对运动下的人员检测与 3D 跟踪，特长是事件相机、RGB、LiDAR、IMU 与 Vicon 的同步。它不是安静交通参与者数据：动态目标只有两名佩戴 Vicon 头盔的人。

**核心机制。**

1. RGB 路径用 YOLOv4；事件路径把 50 ms 事件分成 10 个时间片、两个极性张量，再交给预训练 RVT detector。
2. 两种 2D 检测都经 SORT 维持短轨迹。
3. LiDAR 投影进 2D 人框；取框中心方形区域内点的中位数作为 3D 位置。
4. 每条轨迹再用常速度 Kalman filter 平滑。Vicon 提供传感器和两名行人的 6DoF pose，可独立核对 ego/source 相对运动。

**核心证据。** 数据仅约 14.1 分钟、4 个 ROS bag，却含约 7.74 亿事件、2.2 万 RGB 帧、6700 个点云以及传感器/行人的高频 Vicon pose。RGB 约 23 Hz、IMU 1 kHz、LiDAR 约 7.9 Hz。论文的事件检测“真值”来自 YOLOv4 pseudo-label，并非独立人工标注；事件模型未针对该室内域微调。报告的 3D `XZ` MAE 约为 RGB `0.828 m`、事件 `0.983 m`，事件结果还只在人工挑选的约一分钟表现较好片段上评估。作者承认框中心裁剪容易混入背景，常速度模型不适合手持抖动，简单时间戳融合也不足。

**核心价值。** 数据的 Vicon 相对运动可用来检验 ego compensation、时间对齐和不同传感器在快速运动/模糊下的失效，不必把某个 detector 当真值。

**读后感。** 这是“数据比方法更值得留”的典型。传感器组合很漂亮，但论文管线把若干脆弱捷径叠在一起：pseudo-label、中心点取深度、常速度和选择性片段。若只看项目摘要，会高估 event camera 已解决快速接近；看完误差与选段规则后，应把它放回压力测试位置。

**取舍。** `只作评测/压力源`，用于同步、ego compensation 和模态退化；`不借`现有 2D→3D 跟踪链作为风险基线。没有 body corridor、closest approach 或事件 onset/clear 真值。

---

## DR27 — DynamicStereo / Dynamic Replica：动态时序深度的合成 oracle

**一手来源**：[论文全文](https://arxiv.org/html/2305.02296)；[官方仓库](https://github.com/facebookresearch/dynamic_stereo)

**它真正问的问题。** 给定一段 rectified stereo 视频，如何让每帧 disparity 在人物、动物和相机都运动时仍保持时间一致，而不是逐帧独立估计深度。

**核心机制。**

1. 多尺度 CNN 提取左右图特征，在 `1/16` 尺度加入四个 Space–Stereo–Time divided-attention block，使空间、双目和时间分别交换信息。
2. 沿极线构造 correlation volume，获得每个像素的 disparity 候选。
3. 以 coarse-to-fine 迭代更新 `D^(m+1)=D^m+g(D^m,φ)`；更新器使用在空间和时间维分解的 3D convolutional GRU。
4. 训练损失对所有帧和迭代加权：`Σ_t Σ_m γ^(M-m) ||D̂_t^m-D_t||`，`γ=0.9`；长视频推理用滑窗。

**核心证据。** Dynamic Replica 含 524 段合成视频，484 段训练、20 段验证、20 段测试，训练约 145,200 帧；分辨率 `1280×720`，随机 stereo baseline `4–30 cm`，含扫描人体、动物、深度、flow、foreground mask 和 camera parameters。论文在 Sintel 与 Dynamic Replica 的 temporal endpoint error 上优于逐帧 RAFT-Stereo，3D 时序更新也有消融增益。但真实视频主要是定性结果；作者明确展示了无纹理墙面失败，以及滑窗造成约 `1–2 s` 的低频深度摆动。推理约 `1.20 s/frame`，训练显存约 32 GB；数据解压约 2.2 TB、许可为 CC BY-NC。

**核心价值。** 它可作为动态深度/flow/ego motion 的合成 oracle，用来问“时序模型是否真的改善动态区域的一致性”，并提供精确 foreground 与 camera motion 分层。

**读后感。** 它与“动态出行”在词面上很接近，实际离手机助盲产品很远。模型过重、输入是 stereo、真实验证弱；真正值得的是数据中的可控真值。尤其滑窗低频摆动提醒我们：看起来更平滑的深度可能把风险阈值来回穿越，时序一致并不自动等于事件稳定。

**取舍。** `只作评测/压力源`；用作小规模 oracle/消融，不下载整库做无目的训练。`暂不借`模型架构，不把合成 temporal error 写成真实风险收益。

---

## DR28 — Spring + RobustSpring：把退化做成时间/双目一致的压力测试

**一手来源**：[Spring 论文全文](https://arxiv.org/abs/2303.01943)；[RobustSpring 论文全文](https://arxiv.org/html/2505.09368v2)；[官方 benchmark](https://spring-benchmark.org/)

**它真正提供什么。** Spring 提供高分辨率光流、disparity 和 scene-flow 真值；RobustSpring 在同一序列上加入 20 类 corruption，并尽量保持时间、双目和深度的一致，专门评估模型输出在输入退化下是否稳定。

**核心机制/资产。** Spring 来自 Blender 电影，47 个场景、6000 对 HD stereo，GT 以 UHD `3840×2160` 提供，同时含前/后向 flow、stereo disparity、scene flow、非匹配区和细结构。RobustSpring 覆盖亮度/对比度/饱和度、五类 blur、四类 noise、像素/JPEG/elastic，以及 spatter、frost、snow、rain、fog；16/20 种退化采用量身定制的跨时间、双目或深度一致生成。鲁棒性指标比较同一模型在 clean/corrupted 输入上的预测差异 `R_M^c=M[f(I),f(I_c)]`，而非直接比较预测与真值。

**核心证据。** Spring 有 37 个训练和 10 个测试场景，运动/双目真值规模约 2.38 万/1.2 万。RobustSpring 约 4 万帧、2 万 stereo 对，评测 17 个模型；雨、噪声和天气经常造成最大退化。作者也验证了很小的分层子采样可近似整体排序。但每种 corruption 只设一个强度，真实域转移检查有限；排名还会随平均数、中位数或 Schulze 聚合而变化。

**核心价值。** 它给相机健康/选择性风险一个可复跑的固定压力面：可画出退化强度下的事件 coverage、abstention、fragmentation 和 retained accuracy，而不是只展示几张雨天图片。

**读后感。** RobustSpring 最值得警惕的细节是：它的“robustness”测预测稳定，不测正确性。恒定输出会很稳定，却完全无用。所以低 `R` 不能被写成安全；必须和 clean accuracy、退化后的事件召回及 abstention 一起看。其优势是 corruption 有跨帧一致性，不会把生成伪影误当算法失败。

**取舍。** `借`固定 corruption 套件和双轴报告（正确性 × 稳定性）；`不借`单一总鲁棒分数作为晋级门。它只能评估感知退化，不提供 BLV 风险真值或移动端时延证据。

---

## DR29 — EgoTraj：佩戴者短期路线管，不是碰撞预测

**一手来源**：[论文全文](https://arxiv.org/html/2605.19004)；[官方仓库](https://github.com/yehiahmad/EgoTraj)

**它真正问的问题。** 从头戴 RGB、头部 6DoF pose、3D gaze 和场景上下文预测佩戴者自己未来 3.5 秒的轨迹。它没有预测旁人的未来，也没有身体碰撞标签。

**核心机制/资产。** 数据以 1.5 秒历史预测 3.5 秒未来，pose、gaze、场景语义分别编码后跨模态融合。相机为 30 Hz 方形 RGB，pose/gaze 约 50 Hz；位置线性插值、旋转用 SLERP 对齐，文件级同步握手的上界约 `5.893 ms`。gaze 通过每段 quadratic yaw/pitch→pixel calibration 映射。场景描述由 Qwen2.5-VL 以 1 Hz 生成，因此属于模型派生上下文，不是 native truth。

**核心证据。** 75 名参与者/会话、约 10.7 小时、115 万帧、总路程 46.73 km，7 个 waypoint，参与者自选路线；平均速度约 `1.25 m/s`。预测结果中常速度约 ADE/FDE `0.24/0.35 m`，完整 EgoCast 约 `0.16/0.28 m`。消融显示 scene 与 gaze 能改善结果，pose+scene+gaze 的一个设置约 `0.12/0.23 m`。但接近 90° 急转时所有模型都低估转向，active transition 的误差明显上升；数据中突然转向较少，输出还是单一确定轨迹，没有校准的不确定分支。

**核心价值。** 它是 30 项中最直接的 wearer-path 资产，可用于构造短期身体路线管，检验头动、历史、场景和 gaze 是否真的比常速度增量有效。

**读后感。** 这是一个重要拼图，但绝不是完整风险模型。只知道“我会往哪走”还不知道“对方会不会横穿”；确定性均值轨迹在急转时又恰好过度自信。gaze 的增益很吸引人，但对视障用户，视线行为与 sighted cohort 可能系统不同，不能把 gaze 升格为路线 authority。

**取舍。** `借`wearer-path baseline、同步检查和转向/静止分层；输出应扩成经校准的 route tube，而非单线。`不借`VLM 场景标签或 gaze 作为风险真值，也不得从 wearer ADE 推断避碰效果。

---

## DR30 — Open Source Soundscape：音频是可抢占的状态机，不是日志

**一手来源**：[官方仓库](https://github.com/microsoft/soundscape)

**它真正提供什么。** 这是 Microsoft Soundscape 原产品的开源子集，包含 iOS 客户端、OSM/PostGIS/GeoJSON 后端和 route authoring。第三方、专有与部署自动化部分被移除，因此不是可直接发布的完整产品，也不含动态危险检测。

**核心机制。**

1. iOS 端用 `AVAudioEngine` 把离散 callout 与持续 beacon 分开，通过 `AVAudioEnvironmentNode`/HRTF 和 listener heading 做头部相对空间化，另有 mono fallback。
2. callout 不是直接播放字符串，而有队列动作：普通 enqueue、紧急 interrupt-and-clear、hush/stop/fail；如果提示在排队期间已离开 live region，可取消陈旧提示。
3. callout history 抑制重复 POI；自动提示按距离、时间、运动和触发范围筛选，目的地提示随接近提高频率。
4. 多个 generator 有顺序和事件消费关系；后台运行、与其他音频混合以及来电中断都有显式状态。

**核心证据。** 这里的证据主要是可检查的工程接口与代码，不是新的随机用户试验。仓库明确说明它只开源原项目的一个非 turnkey 子集。代码中能看到离散/连续 player、队列抢占、重复抑制、位置仍相关性检查、空间 beacon 和路由编排，但没有 risk-priority 模型或动态目标事件 evaluator。

**核心价值。** 它把“输出通道是一种稀缺资源”落实为架构：紧急风险可抢占，非紧急叙述让行，陈旧提示可撤销，重复提示受抑制，持续方向 cue 与离散语言分离。这比在 detector 后直接 TTS 更接近用户真实负担。

**读后感。** 这是列表中最值得借鉴反馈工程、最不值得整体移植的项目。Soundscape 的价值在交互语义，不在 iOS/云技术栈；对 Android 重写这些小接口比搬整个服务更合理。它也提醒我们，事件 `clear` 和陈旧取消与首次告警同样重要。

**取舍。** `借`队列优先级、抢占、去重、陈旧取消、连续/离散分轨和后台共存语义；`不借`导航 POI relevance 作为风险 relevance，也不直接移植完整栈。

---

## 3. 精读后的重排：30 个候选不是 30 个同类方案

### 3.1 最值得保留的六个接口

1. **身体路线相关性**：DR01 的 collision point、DR06/08 的 clearance corridor、DR29 的 wearer-path history 应组合成 route tube；任何单一目标距离都不能替代它。
2. **未来危险证据**：DR09 的 binary horizon、DR10 的 stochastic occupancy、DR15 的 collision cone 是三种不同复杂度的候选。它们应做有控制的对照，不应同时堆成 model zoo。
3. **事件生命周期**：DR02 的分级升级、DR11 的碰撞锚点、DR12/14 的 onset/release/持续正确指标共同定义事件层；单调风险只能是开发对照，不能覆盖会解除的事件。
4. **独立输入健康与拒答**：DR17 强迫报告 risk–coverage，DR18 保留 unknown，DR19/20 检查输入能否支持任务，DR21 则证明“没有视觉共证据”只有相对于漏检模型才有意义。
5. **独立危险通道**：DR16 的快速 looming、DR22 的去地后未知残差、DR23 的负障碍不能被通用类别 detector 吞掉；每个通道都应有自己的可见性和失败语义。
6. **反馈调度**：DR30 的抢占、去重、陈旧取消，加上 DR02/06/07 的声音—触觉分工，才构成“选择性介入”；检测框后的直接 TTS 不构成产品策略。

### 3.2 按用途而非声量分层

| 角色 | 候选 | 精读后的判断 |
|---|---|---|
| 直接行为结果锚点 | DR01、DR02、DR06、DR07 | DR01 最强但分母受触发条件限制；DR06 是静态室内 corridor；DR02 样本极小；DR07 的“100%”不是零碰撞 |
| fail-closed / 事件 evaluator | DR08、DR12、DR14、DR17、DR18、DR30 | 优先借接口与指标；DR12 的论文—代码/校准 split 不一致，DR18 的形式保证不可跨域搬运 |
| 未来运动候选机制 | DR03、DR04、DR09、DR10、DR15、DR16、DR29 | 均只补一块；没有一项同时闭合 wearer route、target future、输入健康和 BLV 结果 |
| 相机/跨模态健康 | DR19、DR20、DR21 | DR19 是任务相关 blur/noise 工程；DR20 是待校准合成 health；DR21 在移动 ego 上严重降级 |
| 未知/负障碍 | DR22、DR23 | 问题分解值得借，现有室内 RGB-D/旧 stereo 实现与证据都不足 |
| 数据与压力源 | DR13、DR24、DR25、DR26、DR27、DR28 | 只能验证各自原生标签支持的前置能力，不得派生伪碰撞 truth |
| 暂缓的动作层 | DR05 | 上游相对运动、身体路径和人类执行误差未闭合前，优化器的 0% 模拟碰撞没有决策价值 |

### 3.3 精读后明确降级的项目

- **DR03**：从“助盲轨迹预测候选”降为“旋转补偿假设 + 数据压力源”，因为没有可核对的碰撞结果或用户协议。
- **DR07**：从“100% 避障系统”降为“复杂跨模态原型”，因为作者的 collision-avoidance rate 实际按检测到的障碍计，不是人体零接触。
- **DR10**：名称从 SCOPE 修正为 SOGMP/SOGMP++；其 VAE 样本方差不是校准不确定性。
- **DR12**：保留 evaluator，暂停采用 conformal guarantee；代码架构、损失和 calibration split 与论文叙述存在实质不一致。
- **DR20**：保留四输出 contract，健康指数本身降为待证标签；网络首先学的是作者手工公式。
- **DR21**：从“相机受阻时的声学补充”降为静态盲路口概念证据；移动平台和安静目标恰是其薄弱点。
- **DR22**：从“未知障碍方案”降为几何残差原型；只有单一住宅定性演示，没有 BLV 行走效果。
- **DR26/27/29**：分别保留同步真值、动态深度 oracle 和 wearer route；都不含动态风险事件 truth。

## 4. 读完 30 项后的核心判断

这些工作没有共同解决一个“检测问题”，而是在解决六个不同问题：我会往哪走、目标会往哪走、两条未来是否相交、当前输入是否值得信、事件何时开始/解除、以及怎样在不遮蔽环境声的前提下提示。把它们压成一个 object detector 分数，会同时丢失路线、时间、不确定性和交互语义。

最强的最小系统假设应当是：

`wearer route tube ∩ target/occupancy future ∩ time horizon` 形成候选风险；

`sensor health + calibration + event persistence` 决定它是否有资格告警；

`onset / escalation / release + interruptible feedback` 决定用户实际听到什么。

其中任一交集缺少 source-native truth，就只能报告该层 `NOT_EVALUABLE`，不能用 teacher 共识、检测分数或稳定输出补成“安全”。尤其应避免四种常见偷换：

1. `横穿意图` 不等于 `进入我的路径`；
2. `TTC to camera plane` 不等于 `physical body-route TTC`；
3. `预测稳定/健康分高` 不等于 `预测正确`；
4. `设备已触发事件中的接触下降` 不等于 `全程危险召回已建立`。

## 5. 仍然缺失、以后真正需要补的证据

- 穿戴视角下 source-native 的 wearer body/route tube 与 target future，含横穿、接近、离开、用户停步和转头；
- 完整时间轴的危险分母，而不是只审设备触发片段；
- 高位、细杆、临时障碍与 drop-off 各自的可见性/不可见性 truth；
- 输入退化与**任务事件失败**的关联，不是通用 IQA 或作者自定义 health；
- `first alert / escalation / clear / repeat / nuisance duration / critical miss` 的统一 evaluator；
- 开放式音频、触觉、白杖与环境声并存时的真实 BLV 反应、路线效率和认知负担。

因此，这份精读不会推荐立即实现 30 个候选中的任何一个完整系统。它留下的是一组可审计的机制部件、数据边界和停止误用的理由，供未来在明确授权、固定 truth 和单一失效层后使用。
