# TARO R0 独立并行研究路线指南

状态：`current design guide / WILD_LAB / NON_AUTHORIZING_REFERENCE / DYNAMIC_STATUS_OWNED_BY_TARO_CURRENT`

日期：2026-08-10

## 0. 文档职责与权限

本指南回答“为什么研究 TARO、怎样把它做成可证伪论文路线、每一步什么情况下继续或
停止”。它不是执行协议、数据授权、模型配置、训练计划或安全验证。

动态状态、唯一 successor、禁止动作与默认 App 影响只以 [TARO current](README.md) 为准。
本页的 `NON_AUTHORIZING_REFERENCE` 只表示“本指南不能启动执行”，不表示 TARO 从未运行；
P0/O0M 是否完成、O0R 是否可评估及当前是否暂停，必须读取 current 与对应签署结果，不能从本页
的设计阶段文字反推。
任何阶段只有在上一个阶段形成有效结果、下一个协议在 outcome 前单独冻结且 execution
authority 显式为 true 后才能运行；本指南列出的后续阶段不是预授权队列。

默认研究风格：`WILD_LAB / CANARY_LITE`。只有用户将来明确启动独立 Confirmation 或
Deployment 时才切换到 `EVIDENCE_TRACK`，且必须使用新的 parent/session/site-disjoint 数据。

## 1. 路线定义

### 1.1 需求场景与任务边界

TARO 面向一个窄而可计算的局部出行问题：用户已经借助白杖、定向行走技能或外部导航掌握宏观
方向，但仍需判断前方几步内，声明的 wearer path 与 body profile 是否具有足够净空。真实行人
环境中的临时占道、突出/悬空障碍与净空不足支持研究这一信息缺口；辅助技术通常与白杖等多种
工具共同使用，而不是替代既有技能。需求背景参考
[RNIB In My Way 2025](https://media.rnib.org.uk/documents/In_My_Way_-_Navigating_pedestrian_journeys_with_sight_loss_2025_PDF.pdf)
与 [WHO Assistive technology](https://www.who.int/news-room/fact-sheets/detail/assistive-technology)。

TARO 第一篇路线不回答全局导航、道路穿越、物体百科描述、自动控制用户身体、真实用户独立行走
或产品安全。需求只通过下列效果合同进入算法：

| 需求 | 计算对象与系统效果 | 主指标 | 算法约束 |
|---|---|---|---|
| 避免错误放行 | body/path-specific clearance interval | `false-clear`、query error | 完整区间进入 reducer，不能只看均值 |
| 减少无效阻断 | 在 false-clear 非劣下减少保守判断 | `false-block`、known coverage | all-`UNKNOWN`、coverage collapse 失败 |
| 暴露不确定性 | `UNKNOWN`、identifiability 与 reason code | interval calibration、错误高置信率 | 只更新可观测子空间 |
| 降低取证负担 | passive-first，必要时才考虑站定相机微基线 | risk reduction、prompt/time cost | 禁止要求身体迈步取证 |
| 保持任务及时性 | 在冻结时间/动作预算内形成 query 结论 | time-to-evidence、P95（后续阶段） | 同预算比较，不以无限观察换正确率 |

这些是目标效果和评价合同，不是 O0M 已经证明的用户效果。算法层、事件系统层和真实用户层必须
分开验证。

### 1.2 科学命题与组件

TARO 全称：

> **Task-directed Active Risk Observability**
> （任务定向主动风险可观测性）

TARO 有意把可替换的 factor encoder、残余 gauge posterior、observable-subspace update、
选择性不确定性与任务定向取证组合成一条受共同 clearance contract 约束的算法。它不要求每个
部件从零发明，而以 task-query identifiability 为主贡献，检验一个统一命题：

> 在声明的独立米制锚、有效 frame receipt、冻结的连续几何 factor 与 deterministic reducer
> 下，能否通过只更新局部可观测的残余 gauge，并在必要时选择一个受限的额外观测，使
> body/path-specific clearance 查询先变得可识别，即使完整 camera-scene state 仍不唯一？

两个内部组件分别承担：

- **GaugeFix**：对低维残余 gauge 建立 posterior，只更新可观测子空间；
- **PARA**：当 task query 仍受不可观测方向影响时，在相同帧数、延迟和动作预算下选择
  最有价值的被动历史帧或站定相机微基线。

PARA 是条件扩展而不是预设成功项：只有 passive posterior 和 action oracle 依次提供增量时才进入
学习。当前版本若 active branch 失败，passive-only 延续必须另立版本，不能在 outcome 后拆分联合
命题来回救结果。

### 1.3 组合式创新的成立条件

复用优秀模块不是 TARO 的弱点；不披露来源、没有共同问题定义、只靠堆叠提高 aggregate score 才是。
本路线把“缝得好”定义为：接口、调用顺序、停止条件、`UNKNOWN` 和证据合同共同产生单个模块或
朴素 cascade 不能提供、且可重复测量的任务行为。未来若要把组合本身写成论文贡献，必须同时满足：

1. 所有继承模块、权重、数据、接口改造和 provenance 显式登记；
2. 各模块共享非平凡的 body/path-specific clearance functional，而不是各自优化无关 proxy；
3. 调用存在条件依赖：只有当前 query 仍不可识别时才允许 PARA 消费额外证据；
4. 统一优化预冻结的非对称 query decision risk，并显式扣除时间、相机位移、提示和注意力成本；
5. 最终 authority 仍由固定 deterministic reducer 独占，组合不能把 learned confidence 伪装成三态；
6. 与 plug-compatible、同输入、同帧数、同延迟、同动作预算的强替代模块和朴素串联公平比较；
7. 用单组件、替换组件、联合组件和 factorial interaction 分离增益，联合收益不能由各模块单独结果推断。

这些条件是未来真实实验要证伪的 scientific contract，不是当前 synthetic O0M 已经证明的创新效果。

## 2. 明确的非命题

TARO 不研究、也不得宣称：

1. RGB-only 单目从无到有恢复可信绝对米制尺度；
2. 从零学习或替代有效 Camera2/ARCore 内参、crop、rotation、resize receipt；
3. 完整恢复全场景 3D、全部相机参数或全局 SLAM gauge；
4. 通用 next-best-view、全局 reconstruction coverage 或通用主动标定；
5. 通过要求用户向前/侧向行走来完成取证；
6. learned head 直接输出最终 clearance、free/blocked 或三态；
7. 未来像素生成、4D occupancy world model 或其他参与者的社会响应；
8. 真实视障用户安全、独立行走、产品有效性、默认 App 替换或部署认证。

## 3. 为什么这是实质不同的新路线

### 3.1 来自项目的诊断动机

Assistive Geometry A0 在三个 seed 上出现高度一致的 conservative clearance bias 与
false-block，但已有分析不能因果区分 depth scale、K/ray、support、boundary 或其他上游
表示误差；全部 truth-clear 支持又集中在一个 parent。因此 TARO 只能把“低维可修复 gauge”
作为待证伪候选机制，不能声称它已经解释 A0。

现有运行时 receipt 已能绑定设备、physical camera、crop、rotation、sensor-to-buffer、resize
与 K；TARO 的新变量是有效 receipt 之后的残余 uncertainty，而不是再造一个相机标定头。

当前 target atlas 的 explicit timestamp、pose transform、continuous boundary truth、完整 factor
schema 与 truth-clear factor bundle 仍缺失。底层 ARKitScenes raw 具备 K/pose/depth 结构并不等于
TARO schema 已物化或可执行。

### 3.2 与项目历史路线去重

| 历史/当前工作 | TARO 不重复什么 | TARO 的实质差异 |
|---|---|---|
| R2 factorization | 不重新定义 depth/support/boundary，也不输出 learned final state | 在冻结 factor/reducer 外建立 task-query identifiability 与 residual posterior |
| 已知高度/affine/ridge 尺度 | 不重跑静态全局 scale/offset 或换 seed/阈值 | 独立 metric anchor + 多帧观测子空间 + uncertainty/UNKNOWN |
| D0 temporal | 不把时序平滑或 future residual 重新命名为主动观测 | 只选择能消除 task-relevant unobservable direction 的证据 |
| AG-QSF | 不复活已关闭的 censor-data 路线 | 查询对象是当前 clearance 的局部可识别性，不训练 contact-survival head |
| AG-CBF | 不复活已关闭的 corridor-grid oracle | 不改变 corridor 表示，只对冻结 body-swept functional 做 observability |
| FRESH-TF/RCLE | 不使用 freshness/change 或 looming 代理任务风险 | 目标直接绑定声明的 clearance interval 与 factor posterior |

### 3.3 主要相邻文献与论文空位

- [Online Self-Calibration for VINS](https://arxiv.org/abs/2201.09170) 已系统分析完整 VINS
  自标定与退化运动；TARO 必须继承“欠激励时只更新可识别子空间”的事实。
- [AnyCam](https://arxiv.org/abs/2503.23282) 与
  [CalibAnyView](https://arxiv.org/abs/2605.14615) 已覆盖视频 pose/intrinsics 或任意多视图
  calibration；“视频标定”本身不是 TARO novelty。
- [PTC-Depth](https://arxiv.org/abs/2604.01791) 已使用已知位移、光流三角化与 Bayesian
  metric-scale update；“Bayesian scale smoother”本身也不新。
- [Task-Specific Bayesian NBV](https://arxiv.org/abs/2605.05095) 已直接优化下游任务视角；
  [GraspView](https://arxiv.org/abs/2511.04199) 已把 RGB-only active view 与在线米制对齐
  用于抓取。因此 TARO 必须证明 query-identifiability、人体 swept-volume、声明的 metric anchor、
  人类受限相机动作和 UNKNOWN 的联合差异。
- [SPARTA](https://proceedings.mlr.press/v305/dong25a.html) 已学习 approach-angle/state-conditioned
  traversability；“按方向或 profile 条件化可通行性”本身不是 TARO novelty。
- [CapNav](https://openaccess.thecvf.com/content/CVPR2026/papers/Su_CapNav_Benchmarking_Vision_Language_Models_on_Capability-conditioned_Indoor_Navigation_CVPR_2026_paper.pdf)
  已建立 capability-conditioned navigation benchmark；“按人体能力判断路径”不能单独作为贡献。
- [SCOPE](https://arxiv.org/abs/2608.04420) 已把 robot-inflated safety volume、未知 voxel 与主动
  viewpoint search 联结；“swept volume + active observation”也不是 TARO novelty。SCOPE 面向可控
  UAV 的已观测自由空间认证与执行规划，TARO 必须证明其不同对象：穿戴式部分米制 factor
  posterior、局部 task-functional identifiability、人类受限 camera-only 观察、校准 `UNKNOWN`
  及交互成本；TARO 不主张 certified path execution。
- [Fundamental limits for sensor-based robot control](https://journals.sagepub.com/doi/full/10.1177/02783649231190947)
  已用 task-relevant information potential 研究给定任务下传感器施加的性能上界；因此“信息与任务
  相关”本身不是新意，TARO 必须把对象收窄到 clearance functional 的局部可识别性与有限证据动作。
- [The Value of Sensory Information to a Robot](https://proceedings.iclr.cc/paper_files/paper/2025/hash/e1126028d9f1f69c13571ec462084d31-Abstract-Conference.html)
  通过在不同任务状态暂时 withholding observation 测量性能 regret；TARO 应继承这种反事实思想，
  报告去掉合法观测后的 query-risk 上升与所选动作相对同预算 oracle 的 sensing regret。
- [Perceive With Confidence](https://proceedings.mlr.press/v270/dixit25a.html) 已把 conformal
  calibration 与学习感知/规划联结；[Explore until Confident](https://explore-eqa.github.io/) 已用
  multi-step conformal prediction 校准 EQA 停止。故“conformal + confidence stopping”本身不是
  TARO novelty；TARO 的候选差异是 partial-metric factor posterior、body/path clearance、非对称
  false-clear risk、人类取证成本和 deterministic reducer，但仍须实验证明。
- [MapAnything](https://map-anything.github.io/)、
  [VGGT](https://openaccess.thecvf.com/content/CVPR2025/html/Wang_VGGT_Visual_Geometry_Grounded_Transformer_CVPR_2025_paper.html)
  与 [MASt3R-SLAM](https://openaccess.thecvf.com/content/CVPR2025/papers/Murai_MASt3R-SLAM_Real-Time_Dense_SLAM_with_3D_Reconstruction_Priors_CVPR_2025_paper.pdf)
  表明 feed-forward metric geometry 与 reconstruction prior 已很强；它们适合作为 proposal、teacher、
  initializer 或 upper bound，而不是 TARO 的独立 metric anchor、O0R truth 或三态裁判。

因此，可检验的论文空位不是“校准 + NBV”“人体条件化”或“扫掠体积认证”，而是：

> **under partial metric evidence and human-constrained sensing, full state can remain underdetermined
> while a body/path-specific task functional becomes locally identifiable; when it does not, bounded
> evidence selection must reduce query risk without hiding false-clear, coverage or interaction cost.**

高质量组合可以成为贡献，但论文必须证明共同 contract、条件调用与 interaction gain；仅把若干强模块
接成 pipeline，或分别引用它们的单项成绩，都不能建立 TARO 的联合主张。

## 4. 数学对象与可证伪条件

### 4.1 状态与观测

令冻结的连续几何 factor 为：

\[
f_t = \{D_t, U^D_t, S_t, U^S_t, B_t, U^B_t\},
\]

分别代表 metric-ish depth、support、boundary/evidence 及其 uncertainty。有效 camera receipt
定义 (K_{eff})，第一版不得把完整 K 作为自由变量。

低维残余 gauge 建议从最小状态开始：

\[
g_t =
[\log s,\ \delta\phi_s,\ \delta h_s,\ \delta\xi_{pose},\ \delta\tau] .
\]

- (s)：window-global metric scale residual；
- \(\delta\phi_s\)：support normal 的局部切空间残差；
- \(\delta h_s\)：support offset/height residual；
- \(\delta\xi_{pose}\)：有独立来源约束时的短窗 pose residual；
- \(\delta\tau\)：只有时间对齐证据充分时才开放的 clock residual。

可选 \(\delta K\) 必须作为独立 ablation，只有 receipt 缺失/冲突、camera switch、zoom/focus
或明确动态内参问题时才能开放；不得默认与 scale/support 一起自由优化。

### 4.2 Task functional

给定候选 wearer path (a)、身体 profile (b) 与几何状态 \(\theta=(f,g)\)：

\[
C(a,b;\theta)
= \min_{p\in V_b(a)} \operatorname{SignedDistance}(p,S_\theta),
\]

其中 (V_b(a)) 是冻结定义的 body-swept volume。输出必须是 clearance posterior/interval、
query identifiability 和 reason code，而不是 learned final state。

### 4.3 局部 task-query identifiability

令短窗信息矩阵为 (F\)，task functional 在当前线性化点的 Jacobian 为 (J_C\)。局部可识别
候选条件为：

\[
\mathrm{Null}(F) \subseteq \mathrm{Null}(J_C),
\]

即任何无法由当前观测区分的状态方向 (v)，都不能改变 task query：

\[
Fv=0 \Rightarrow J_Cv=0.
\]

这只是待检验的局部判据，不预先写成全局定理。P0 已在签署协议中冻结：

- 线性化点与状态参数化；
- eigen/singular-value 截断规则；
- (\lVert J_Cv\rVert\) 的数值容差及单位；
- min/swept-contact 非光滑点的处理；
- 判据与真实 query error/calibration 的验证方式。

若该判据不能预测 query 误差或 posterior calibration，TARO 的核心科学假设被削弱或证伪，
不能只解释为“模型没有调好”。

### 4.4 不确定性传播与三态边界

一阶诊断可使用：

\[
\sigma_C^2 \approx J_C\Sigma_\theta J_C^\top.
\]

由于 swept minimum 在接触切换处非光滑，正式实现应比较线性传播与小样本 Monte Carlo、
unscented transform 或 interval propagation。最终 clear/occupied/UNKNOWN 继续由冻结的
deterministic reducer 根据完整 interval 产生；TARO 不能把均值越阈值当成最终结论。

### 4.5 选择性风险校准候选——未来且非授权

只有未来获得独立、parent-disjoint 的 calibration split 后，才可在 protocol 中比较 split conformal、
conformal risk control（CRC）或其他预冻结 selective-risk calibration。候选 wrapper 位于 factor/query
posterior 与冻结 reducer 之间，只能校准 clearance residual、interval 或 selective decision risk；它不是
learned safety classifier，不能输出最终三态。

该 wrapper 不得修复无效 receipt、缺失 metric anchor、unsupported factor、错误时间绑定或数据角色
冲突，也不得通过缩小 `UNKNOWN` 率来宣称成功。任何 coverage/risk 保证只在明确写出的 calibration/
test 可交换性、loss、alpha、finite-sample correction 与数据分布假设下成立；设备、场景、body profile、
动作政策或人机交互分布发生 shift 时，不得外推为真实助行安全保证。当前没有合格 calibration source，
本候选未采用、未实现、未执行。

## 5. 计划架构

```text
RGB + valid K/crop/rotation/resize receipt + IMU/VIO + sparse tracks
    -> frozen/replaceable R2 factor encoder
    -> factor likelihoods and validity
    -> residual gauge factor graph / posterior
    -> TSVD observable-subspace update
    -> body/path-specific clearance posterior + query identifiability
        -> identifiable: optional pre-frozen calibration wrapper (future only)
            -> frozen deterministic reducer
        -> not identifiable: passive ring-buffer evidence selection
        -> still not identifiable and stationary: bounded camera micro-baseline
        -> no admissible evidence: UNKNOWN with reason
```

### 5.1 Receipt-first 层

`K_eff` 必须由 sensor K、active physical camera、crop、rotation、sensor-to-buffer 与 resize
确定性组合得到。任何缺失、非有限、shape 不匹配或跨帧 camera identity 变化都必须显式产生
invalid/UNKNOWN reason，不能回退 nominal K。

### 5.2 短窗测量因子

允许候选包括：

- sparse track reprojection residual 与 track survival；
- 有实际平移时的 sparse triangulated depth；
- 相邻帧 factor/point consistency；
- IMU gravity residual；
- support 像素的局部 plane/height-field residual；
- Camera2/ARCore K/pose prior；
- 后续协议明确允许的 scalar metric displacement 或稀疏 depth anchor。

禁止把物体常见大小、VLM 语义先验或大模型“看起来像多少米”当作独立几何锚。它们若被
研究，只能作为 disclosed learned prior，并必须与真正 metric anchor 分开报告。

### 5.3 可观测子空间求解器

候选实现边界：5–10 帧滑窗、robust IRLS/LM、SVD/TSVD，只更新通过可观测门的 state block。
不同状态必须使用不同动态先验：K/session-static、scale/slow、support/local、time-offset/rarely
open；不得给所有量套同一无约束 random walk。

输出至少包含：posterior mean/covariance、eigen spectrum、observable mask、metric-anchor validity、
source timestamps、update/freeze reason 和 query identifiability。

### 5.4 受限观测动作

动作空间按下列顺序开放：

1. `STAY`：当前帧，不增加成本；
2. `PASSIVE_HISTORY_SELECT`：最近 0.5–1.0 秒 ring buffer 中的合法历史帧；
3. `YAW/PITCH_MICRO_SCAN`：站定时小角度相机旋转，主要解除遮挡；
4. `LATERAL_CAMERA_MICRO_BASELINE`：仅移动手机/相机约 3/6/10 cm；
5. `SMALL_ARC`：只有前述 mechanics 支持后才作为独立候选。

纯旋转不得被计为 metric triangulation。任何动作的真实基线都以 frame-bound VIO/pose receipt
为准；计划动作、口头提示或目标位移不能代替实际执行。不得要求身体向前/侧向行走。

动作价值建议为约束优化：

\[
u^*=\arg\max_u
\mathbb E[\rho(I_C^t)-\rho(I_C^{t+1}\mid u)]
-\lambda E(u)-\mu T(u),
\]

其中 \(\rho\) 的 primary 应是预冻结的非对称 query decision risk，false-clear 权重与 loss matrix 必须
在 outcome 前冻结；interval width 只能作为并列诊断，不能替代决策正确性。(E/T) 是人机动作成本与
延迟，还应显式记录相机位移、提示次数和注意力负担。先过滤不允许/不可执行动作，再比较信息价值；
不能通过调一个任意加权和绕过动作边界。action scorer 只预测证据价值，最终三态仍由 reducer 产生。

动作价值至少用两个可证伪 regret 检查：

- **observation-withholding regret**：从同一合法 receipt 中去掉某项已允许观测，测量真实 query risk
  上升，避免把冗余传感信息误写成关键证据；
- **sensing regret**：所选动作相对同预算 one-step oracle 可实现 risk reduction 的损失，同时报告单位
  时间、单位相机位移、每次提示和注意力成本的 risk reduction。

当前 query 已可识别时仍触发动作的比例、证据不足后正确保留 `UNKNOWN` 的比例，以及动作执行失败/
拒绝，都是 primary guardrail；不能只报成功动作上的平均 interval 收窄。

## 6. 最小数据与接口合同

### 6.1 `TaroFrameReceipt`

P0 已冻结以下最小字段；机器 schema 与实际状态以签署的 P0 协议和结果为准：

- `source/session/parent/frame_id`、capture/site/device/mount identity；
- sensor timestamp、monotonic/cross-clock validity、`max_source_timestamp`；
- buffer/display/model shape、orientation、crop、resize、transform chain；
- active physical camera、K/distortion source、K validity；
- `T_world_camera` 或 relative pose、covariance、pose source/quality；
- IMU gravity、covariance、sync validity；
- sparse track ID、pixel pair/history、quality、age、visibility 与 rejection reason；
- metric anchor 类型、单位、validity 与 provenance；
- factor tensor identity、validity、uncertainty 与 producer version；
- body profile、path/query ID、action budget；
- data role、outcome access、selection influence、license/use scope。

### 6.2 `TaroFactorPosterior`

至少包含：

- 冻结 factor mean/uncertainty reference；
- residual state block mean/covariance；
- information/eigen summary 与 observable state mask；
- anchor validity、update/freeze/abstain reason；
- query-specific interval、identifiability 与 calibration fields；
- 完整 input provenance 和 causal timestamp ceiling。

### 6.3 `TaroObservationCandidate`

至少包含：

- action ID 与允许动作类型；
- intended camera-only delta、最大时间和是否要求身体移动；
- predicted visibility、track survival、gauge information 与 task value；
- predicted/realized baseline、actual receipt、failure reason；
- stopping decision 与 prompt count。

任何 `requires_body_motion=true` 的动作必须被第一版动作过滤器拒绝。

## 7. 阶段路线与 successor 条件

### P0：query/schema/protocol lock——已完成的非执行阶段

目的：冻结对象、数据角色、解析 fixture、负控、指标、预算和停止条件；不执行。

完成条件：

- 四个 TARO schema 可机器表达；
- analytic factor/query fixture 覆盖 clear、occupied、UNKNOWN 和非光滑接触切换；
- static、pure rotation、small-baseline、missing-anchor、wrong-K/time 的预期可观测性已声明；
- factor-oracle factorial arms、primary metric 与 failure scope 在 outcome 前冻结；
- execution authority 仍为 false，直到另行显式开放。

P0 的完成事实只由 [P0 lock result](TARO_P0_PROTOCOL_LOCK_RESULT_2026-08-10.md) 建立；本节保留其
设计职责，不提供新的执行权限。

### O0M：synthetic identifiability 与 factorial mechanics——已完成的机制 canary

O0M 只检验 measurement-only weak-subspace identifiability、fail-closed degeneracy、factorial
intervention purity、重参数化与确定性等解析 mechanics。其唯一正式 one-shot 已消费，签署结果为
[TARO O0M Synthetic Analytic Mechanics Result](TARO_O0M_SYNTHETIC_ANALYTIC_MECHANICS_RESULT_2026-08-10.md)。
它不能建立真实 factor causal headroom，也不能授权 O0R、G0/G1 或 A0/A1。

### O0R：真实 factor causal-headroom oracle

科学问题：如果 K/scale/support/boundary 某一层被换成准确 oracle，task-query bias 是否实质改善？

最小 factorial arms：

- frozen current anchor；
- receipt/K transform corruption control；
- scale-only oracle；
- support-only oracle；
- boundary-only oracle；
- scale+support；
- scale+boundary；
- support+boundary；
- all-factor oracle。

O0R 只回答 causal headroom，不训练 GaugeFix。若 all-factor oracle 仍不能改善 query calibration、
clearance/false-block Pareto，则 gauge 路线停止；若只有某个 factor 有 headroom，后续状态空间必须
收缩到该 factor，不能继续训练 full GaugeFix。

### G0：解析可观测性与退化运动 mechanics

场景矩阵至少覆盖：

- static、pure yaw/pitch、forward、lateral、small arc、mixed 6DoF；
- textured/textureless、single-plane/non-planar、near/far、epipole-near；
- ramp、stairs、thin obstacle、support occlusion；
- dynamic foreground、track loss、blur、rolling shutter 与 clock offset；
- valid anchor、missing anchor、wrong anchor、anchor shuffle。

目标不是平均参数误差，而是验证：observable mask、TSVD freeze 和 query-identifiability 能否预测
实际 query error/calibration。任何退化输入导致置信度上升或 UNKNOWN→clear 都是强失败。

### G1：被动 residual gauge posterior

输入只允许历史真实帧、有效 metadata/IMU/VIO、sparse tracks 与冻结 factor；不产生用户提示。

必须比较：

- no correction；
- constant global affine / EMA；
- 现有 known-height/ridge/spatial-calibration 类基线；
- VIO triangulation / PTC-Depth 类 scalar Bayesian scale；
- GaugeFix without TSVD；
- GaugeFix with TSVD/query gate；
- exact posterior/oracle ceiling。

若不能稳定超过简单 global affine/VIO anchor，或收益只来自一个 source/parent，停止 GaugeFix
learned/solver expansion，保留 receipt/diagnostic 工具。

### A0：主动观测 oracle

只在 G1 有有效 posterior 后开放。先使用离线 counterfactual view/replay，不训练 policy、不提示用户。

强基线：

- current/stay；
- best passive ring-buffer；
- random allowed action；
- fixed yaw sweep；
- fixed lateral micro-baseline；
- max baseline；
- max parallax；
- generic entropy/Fisher/log-det NBV；
- task-aware but gauge-agnostic scorer；
- TARO joint query/gauge oracle；
- ground-truth one-step oracle。

必须在相同帧数、wall-time、动作类型与动作成本预算下比较。若 active oracle 本身不优于
passive/max-parallax，PARA 终止；当前联合版本不得在 outcome 后直接降级，任何 passive-only
query-identifiability 延续必须另立路线版本。

### A1：compact observation-value scorer

只有 A0 通过后才允许。模型只评分 7–9 个冻结动作，输出 expected query-risk reduction、
visibility/track survival 与 uncertainty，不输出最终三态。

候选优先为解析 Fisher + 小 MLP/TCN，复用现有 feature，不增加第二次 dense encoder inference。
大 foundation model、VGGT/CUT3R/3DGS/生成式 world model 只能作为离线 teacher/upper bound。

### J0：联合 fresh Development

只有 O0R、G0/G1、A0/A1 分别通过后才可建立新的 `TARO_DEVELOPMENT`。目的：验证 joint route
是否在新 device/mount/site/session parents 上形成 query-calibration 与交互成本 Pareto，而不是把
两个单独模块的正结果相加。

第一版建议至少 12 个独立 parents，预冻结为 `8 FIT / 2 CHECKPOINT_SELECTION / 2 TRAIN_CANARY`
或另一个有依据的 parent-disjoint 结构；学习阶段使用固定 seed 集、聚合结果和
`selected_seed=null`，不得按最佳 seed 汇报。正式数字由 J0 协议在 outcome 前冻结。

### M0：移动端 shadow——非当前路线

只有 J0 质量通过、接口冻结和模型选定后才讨论：稀疏 track、posterior solver、action scorer 的
raw/task parity、延迟、内存、功耗与设备退化。M0 不改变默认 App，不执行真实助行提示。

## 8. 数据角色与采集设计

### 8.1 当前数据可承担的角色

- B1 consumed Selection/anatomy：`DIAGNOSTIC_ONLY`，只能帮助形成问题，不能选 factor、状态、
  threshold、action 或 gate；
- 历史 ARKitScenes B0 TRAIN：只能按既有 receipt 承担 mechanics/source-characterization；新的
  O0R contract 已另行冻结 24 个未打开、visit-disjoint 的 upsampling TRAIN parent，但在 truth-only
  preflight 前仍不能成为 TARO scientific outcome 或 Development promotion evidence；
- analytic synthetic：可验证坐标、interval、observability 与 mutation mechanics，不证明真实效用；
- Teacher/model-generated：只能作为明确 provenance 的 offline target/upper bound，不能冒充 sensor GT。

当前不分配 Confirmation，不创建 sealed TARO outcome。

### 8.2 未来 TARO 数据最低字段

每个候选 session 至少需要：

- 同步 RGB、K/crop/rotation/resize、IMU、VIO 6DoF 与 covariance；
- camera-body/mount identity 和实际相机高度变化记录；
- source-native 或独立 reference metric geometry；
- continuous depth/support/boundary target 与 validity/uncertainty 依据；
- sparse correspondences、track survival 与动态污染标签/诊断；
- body profile、候选 query/path；
- stay、yaw/pitch、左右 3/6/10 cm 或等价自然微基线 counterfactual views；
- 每个动作的实际基线、耗时、track survival、失败/拒绝原因；
- clear/occupied/near-boundary/UNKNOWN 的 parent-level 双侧支持。

truth-clear 与 truth-occupied 不得再次集中到单一 parent。建议在最终 8 个独立 Development
eval parents 中，两类 task support 各覆盖至少 6 个；具体门由未来协议根据数据审计冻结。

### 8.3 TwinScene 的可选角色

TwinScene 若未来独立立项，只能作为显式登记的离线 factor/action supervision source：

```text
real baseline -> rendered baseline -> single-variable rendered intervention
```

一个 twin 的两臂、同一 scene/camera path 与 asset family 不得跨 split；renderer artifact、effect-mask
外 treatment probe、cross-renderer 和少量真实物理 pair 必须单独过门。TwinScene 不证明 TARO
query-identifiability，也不获得 TARO freshness/Confirmation 身份。

### 8.4 Geometry-anchored minimal-pair 合同

未来 O0R/J0 若使用 counterfactual pairs，每一对只能改变一个 claim-relevant factor，并由 source-native
或独立 reference geometry 重新计算 query truth，而不是继承模型输出。优先 pair family 包括：

- 障碍边界刚好跨过/退出同一 body envelope；
- scene/path 不变，只改变预声明 body profile；
- 同一 frame 使用 valid K receipt 与冻结的错误 K mutation；
- 有实际平移的微基线与相同视角变化的 pure rotation；
- 同一 clearance boundary 的可见与受控遮挡状态。

另建 nuisance-invariance pairs：保持 geometry/query truth 不变，只改变光照、纹理、曝光或允许的成像
扰动。每个 family 报告 pair flip accuracy、truth-invariant consistency、pair-level `UNKNOWN`、高置信
false-clear 和 parent/site 分层。生成器、renderer 或 world model 可以提议 stress pair，但不能生成 truth；
同一 twin/asset/path 的两臂不得跨 split。O0M factorial mechanics 只证明 synthetic mutation 代码行为，
不证明未来真实 pair 的物理有效性或因果性。

## 9. Primary metrics 与统计

每个阶段只选择与科学问题匹配的 primary；不得用 aggregate score 掩盖 factor/query failure。

### 9.1 Factor/measurement

- log-scale error 与 posterior coverage；
- support normal angle、height/offset error；
- boundary localization/error interval；
- pose/time residual error；
- track coverage、identity/association、triangulation conditioning。

### 9.2 Query/decision

- clearance error、signed bias、interval score/coverage；
- query-identifiability precision/recall 与 calibration；
- false-clear、false-block、known coverage、UNKNOWN reason distribution；
- body/profile/path strata 与 near-threshold performance；
- transition 仅作与当前问题匹配的诊断，不把平滑当作几何正确。

### 9.3 Active observation

- realized asymmetric query-risk reduction；interval reduction 单独诊断；
- observation-withholding regret 与 sensing regret versus equal-budget one-step oracle；
- UNKNOWN→known 且 truth-consistent 的转化率；
- risk reduction per time、actual camera displacement、prompt 与 attention cost；
- actual baseline、track survival、prompt count、time-to-evidence；
- already-identifiable false-trigger rate、insufficient-evidence `UNKNOWN` retention；
- action refusal、执行失败和净效用（扣除时间/交互成本）。

### 9.4 统计规则

- parent/session/site macro 为主，frame-IID pooled 只作诊断；
- 报告 worst-parent 与预声明关键 strata；
- 使用 parent-cluster bootstrap CI；
- 固定 seed 集与聚合规则，不选择 best seed；
- UNKNOWN、缺失和 unsupported 永不作为 negative；
- 任何 outcome 后修改都建立新 evidence version，旧结果原样保留。

## 10. 拟议 kill gates

以下数字是原始路线指南为后续真实 O0R/G0/A0/J0 提出的审查起点，不是当前正式门，也不授予
执行权限。P0/O0M 已完成不等于这些 future gates 已获采用；任何重开版本必须在 outcome 前重新
论证并冻结。

| 阶段 | 拟议门 | 失败后关闭范围 |
|---|---|---|
| O0R factor headroom | all-factor 或受支持组合相对 anchor 的 query error/false-block parent-bootstrap 95% LCB `>0`，false-clear/known coverage 在预冻结 non-inferiority 内 | 无 headroom 则关闭 GaugeFix 科学机制；单 factor 有效则收缩状态空间 |
| G0 observability | missing-anchor、pure rotation、极小基线等退化集错误高置信更新率 `<=5%`，其余 freeze/UNKNOWN；anchor shuffle 必须显著破坏 metric 结果 | 关闭当前 observability rule/parameterization；若判据与 query error 无关，关闭 TARO 核心假设 |
| G1 mechanism | 可观测受控集拟议 scale `<=5%`、support normal `<=3°`、offset `<=5 cm`；仅在独立开放 `delta K` ablation 时要求 focal residual `<=3%`，且对应状态块须胜过 global affine/VIO baseline | 关闭对应状态块或 solver version，不自动关闭完整路线 |
| A0 action oracle | 相对 stay 的 unresolved query-risk median 降幅拟议 `>=20%`，LCB `>0`；scale/depth ambiguity 中 lateral 必须优于 rotation-only | oracle 不过则关闭 active PARA；passive-only 延续须另立版本 |
| A1 scorer | 拟议保留 oracle improvement `>=70%`，并胜过 max-parallax、generic Fisher 与 gauge-agnostic task scorer | 关闭 learned scorer，保留 analytic/passive route |
| J0 task Pareto | 拟议 interval width `-20%`、query error `-15%`、false-block relative `-20%`；false-clear不超过 `+1 pp`、known coverage下降不超过 `5 pp`，至少 `75%` parents 同方向 | 关闭 joint evidence version；按 factor/action诊断收缩，不事后降门 |
| M0 engineering | 拟议增量 P95 `<=10 ms/frame`、内存 `<=32 MB`、无新增 dense NPU invocation；功耗另测 | 只关闭 mobile claim，不否定离线算法 |

all-clear、all-occupied、all-UNKNOWN、coverage collapse、通过更敢报 clear 换取 false-block 下降均不得
通过。任何退化场景 UNKNOWN→clear、缺 metric anchor 仍输出高置信 meter scale，直接判机制失败。

## 11. 强制负控、消融与恶意反例

### 11.1 Gauge/anchor 负控

- no correction；constant affine；EMA；global ridge；
- anchor shuffle、wrong unit、wrong timestamp、wrong K/crop/rotation；
- remove metric anchor；pure rotation；small baseline；low texture；
- dynamic-track contamination；wrong track association；
- oracle gauge/factor upper bound。

若 anchor shuffle 后性能不下降，说明模型没有使用宣称机制，应停止对应 claim。

### 11.2 Active-view 负控

- current/stay；best passive history；random allowed action；
- fixed yaw、fixed lateral、max baseline、max parallax；
- generic entropy/Fisher/log-det；task-aware but gauge-agnostic；
- future-view one-step oracle。

### 11.3 因果与时间负控

- 每个输入/cache 写入 `max_source_timestamp <= anchor`；
- 删除、交换或随机化 anchor 后未来帧，当前 posterior 必须 bit-identical；
- 同一 current posterior 查询不同 candidate observation 时，base world/factor posterior 不变；
- teacher future 只能作 target，不能缓存成 student input；
- 同一 track/session/scene/twin family 不跨 split；
- UNKNOWN/invalid mutation 必须 fail closed。

### 11.4 组合归因与替换消融

- 固定 factor/reducer，只比较 `no GaugeFix`、GaugeFix、替代 posterior solver 与 oracle posterior；
- 在同一 posterior 上比较 `no PARA`、random/fixed/generic NBV、PARA 与 one-step oracle；
- 比较 GaugeFix-only、PARA-on-uncalibrated proxy、朴素 always-active cascade 与条件式 GaugeFix→PARA；
- 每一臂固定相同输入 receipt、frame/action/time budget、body/query roster 与 reducer；
- 报告主效应、interaction effect、worst-parent 与 coverage，不以 joint aggregate 掩盖任一组件 failure；
- teacher/foundation backbone 只能作为可替换 arm，不得同时改变 truth、receipt、预算或 reducer。

若联合臂不优于最佳同预算可插拔替代或朴素 cascade，组合创新 claim 失败；允许保留单组件负结果，
但不得在 outcome 后重命名为原联合路线成功。

## 12. 模型选择和 reducer 边界

- R2 factor backbone checkpoint 只能由各 factor 的 proper loss/error/calibration 与冻结 Pareto 规则
  选择，TARO task metric 不能拯救 factor failure。
- GaugeFix solver/student 可以根据 posterior/query-calibration primary 选择，但不能输出最终三态，
  也不能改变 factor backbone checkpoint。
- PARA scorer 可以监督 expected task-query value，因为这是其定义任务；最终 clear/occupied/UNKNOWN
  仍由同一 deterministic reducer 产生。
- reducer/task metric 只能在 factor/backbone selection 后用于 route-level diagnostic/Development
  decision；不得通过 learned shortcut 进入 factor graph。

## 13. 资源与端侧边界

第一版研究应优先：

- 128–256 sparse tracks；
- 5–10 帧短窗；
- `<20–30` 维低维状态；
- solver 1–2 Hz 更新；
- 7–9 个离散动作；
- 解析 Fisher/TSVD + 小 MLP/TCN；
- 重用现有 feature，不新增第二次 dense encoder。

AnyCam、CalibAnyView、VGGT/CUT3R、MapAnything、MASt3R-SLAM、DROIDCalib、3DGS 和生成式模型
可作为显式 provenance 的 factor/feature proposal、离线 teacher、initialization 或 upper bound。
比较时必须固定 source receipt、metric anchor、数据 split、frame/action budget 与 deterministic reducer；
它们不得改变 anchor validity、替 TARO/R2 选择 checkpoint、代替 O0R truth/injection adapter，或直接
输出最终三态。在同一 Snapdragon/HTP 真机通过 raw/task parity 与完整链路测量前，也不得写成端侧可用。

## 14. TwinScene 与 AC4D 的关系边界

```text
TwinScene --可选的离线 factor/action supervision--> TARO
TARO --未来且仅在独立 oracle 通过后可提供 current metric posterior--> AC4D
AC4D --不得反向作为 TARO observability 证据------------------------┘
```

- PARA action 是“站定时选择下一相机观测”；
- AC4D action 是“供未来风险查询的假设 wearer path”；
- 两者不能混为系统将执行的导航动作；
- TwinScene 与 AC4D 必须分别立项、分别过 oracle gate，当前不建立联合训练。

AC4D 将来若立项，必须先在 acceleration/turning、occlusion/reappearance、multi-agent、action
branching、1.5–3 s 多模态 strata 上超过 D44 + CV/CA Kalman + IMM；普通可见单目标 1 秒 ADE
改善不足以授权 world model。

## 15. 推荐论文叙事

### 可写的核心 claim

> Under declared metric anchors and a frozen factor/reducer interface, TARO tests whether
> a body/path-specific clearance functional can become locally identifiable before the full
> camera-scene state; only when it remains unresolved does TARO test whether passive-first,
> human-constrained evidence selection reduces query risk at bounded interaction cost.

### 必须提供的贡献证据

1. 从局部通行需求到 clearance/false-clear/false-block/coverage 的 task contract；
2. task-query identifiability 判据及退化运动 falsification；
3. metadata-first residual posterior 与 observable-subspace update；
4. 通过 oracle 后才开放的人类受限、camera-only、passive-first 观测策略；
5. 对 max-parallax、generic/task NBV、PTC-like scale 和 simple VIO/affine 的强比较；
6. 同预算 plug-compatible replacements、单组件/联合 factorial ablation 与 interaction attribution；
7. query calibration、false-clear/false-block/coverage、sensing regret 与 interaction cost 的 Pareto；
8. UNKNOWN、timestamp、anchor、split 与未来泄漏机器审计。

### 效果层级与 claim 边界

- **算法主证据**：query error、interval score/coverage、false-clear、false-block、known coverage、
  identifiability calibration 与 interaction cost；这是 TARO 首篇可直接回答的层级。
- **系统相关性**：提醒提前量、误提醒、重复和事件清除只能在独立事件系统评测中回答；query
  改善不能自动传播成这些结果。
- **真实用户效果**：碰撞/接触、停步、工作负荷、信任校准与独立出行需要单独的真实用户和场景
  证据；当前 synthetic O0M、未来 O0R 或设备性能均不能替代。

### 审稿人可能的否定与防守条件

| 攻击 | 何时成立 | 必须如何回答 |
|---|---|---|
| “只是 VIO/scale filter wrapper” | 只报 scale error | 用 task-identifiability、null-space/query error 和强 affine/PTC/VIO 基线 |
| “只是 task NBV 改名” | 只换 utility function | 证明 residual gauge、human constraint、UNKNOWN 与同预算 generic/task NBV 差异 |
| “A0 病因被事后猜中” | 用 consumed anatomy 调状态/门 | A0 只作诊断；新数据与 factorial oracle outcome 前冻结 |
| “通过保守 UNKNOWN 掩盖错误” | 只降 false-clear/false-block | 同时报 known coverage、interval calibration、all-UNKNOWN kill |
| “主动动作不现实” | 用计划基线或身体移动 | 使用实测 VIO baseline、track survival、prompt/time cost，拒绝 body motion |
| “组合系统不可归因” | 只展示联合臂或同时改 backbone/truth/预算 | 保留最小完整组合：冻结 factor/reducer + GaugeFix + conditional PARA；比较单模块、替换模块、朴素 cascade、联合臂与 oracle，报告 interaction gain |

## 16. 粗略预算与停止策略

以下只用于规划，不是承诺工期：

- P0 schema/protocol/analytic fixture：历史阶段，已完成；
- O0M synthetic mechanics：历史阶段，已完成并消费 one-shot；
- O0R factor oracle + G0 observability mechanics：若重开，约 1–2 周；
- G1 passive posterior prototype：约 2–4 周；
- A0 offline active-view oracle：约 1–3 周；
- A1 compact scorer：约 2–4 周；
- J0 新数据 Development：取决于采集/许可，通常另需数周。

任何阶段一旦 oracle 没有 headroom、核心判据不能预测真实 query error、简单基线等效，或收益
依赖 UNKNOWN/coverage collapse，即停止该分支。停止结果应保留为 negative evidence、fixture、
diagnostic 或 future baseline，不用换 seed、降门或扩大组合进行 after-outcome rescue。

## 17. 动态状态读取与 O0R 重开前置

动态状态、唯一 successor 和允许/禁止动作只读取 [TARO current](README.md)。P0 与 O0M 的签署
结果是不可回写的历史事实；已消费的 O0M one-shot 不得覆盖、删除或重跑。本指南没有隐含下一步，
也不能把后续阶段列表解释为执行队列。

本节原冻结的重开条件现已由 current 指向的
[ARKitScenes source-and-adapter contract](TARO_O0R_ARKITSCENES_SOURCE_AND_ADAPTER_CONTRACT_LOCK_2026-08-10.md)
在协议层满足，并由
[lock result](TARO_O0R_SOURCE_AND_ADAPTER_CONTRACT_LOCK_RESULT_2026-08-10.md) 静态签署。合同同时冻结：

- complete factor/query truth 与 truth-clear factor bundle；
- continuous boundary/uncertainty truth、target timestamp/pose；
- deterministic factor injection adapter；
- fresh paired outcome、parent/session/site 身份与数据角色；
- O0R arms、primary metrics、non-inferiority、预算、timeout 和 failure scope；
- 为什么该版本与已完成 synthetic O0M 的 claim 和 artifact root 完全隔离。

对应纯内存 mechanics 现由
[source-adapter implementation lock](TARO_O0R_ARKITSCENES_SOURCE_ADAPTER_IMPLEMENTATION_LOCK_2026-08-10.md)
签署：冻结 roster/asset-bound receipt、内部 residual uncertainty、FARO/candidate 双 extractor、
immutable common-support base、sparse boundary、TARO reducer 与 8×2 injection。该 PASS 只来自 synthetic
arrays，不等于 source/truth gate；不得下载 24 个 selected source body、物化 truth、运行 DepthART、
创建 scientific runner/checkpoint、主动提示或 Android 代码。后续用户 data-use authorization 与
materializer implementation lock 已由 current 另行登记；它们不改写本指南的非授权职责。

### 17.1 O0R 候选源只读能力映射（NON_AUTHORIZING）

下表只依据公开数据格式/项目文档做 metadata-level 映射；本次没有申请、下载、打开或统计任何
payload，也没有赋予数据角色。状态词只允许 `CANDIDATE_METADATA_MAPPED / GAP_OPEN / NOT_ADMITTED`：

| 候选源 | 文档显示的可映射能力 | 仍未关闭的 TARO 前门 | 当前状态 |
|---|---|---|---|
| [Aria Digital Twin](https://facebookresearch.github.io/projectaria_tools/docs/open_datasets/aria_digital_twin_dataset/data_format) | 穿戴式 raw/synthetic streams、timestamp、online calibration、6DoF trajectory、depth/segmentation、部分 skeleton/object GT | body-swept query truth、continuous boundary/uncertainty、自然 paired micro-action、parent/site 独立性、许可/角色、deterministic injection | `CANDIDATE_METADATA_MAPPED / GAP_OPEN / NOT_ADMITTED` |
| [ScanNet++](https://scannetpp.mlsg.cit.tum.de/scannetpp/documentation) | 高精度 laser/mesh、注册 DSLR、iPhone RGB-D、ARKit pose/intrinsics/IMU metadata | 穿戴式 mount/timing、动作交互与真实相机微基线 pair、body/query truth、boundary uncertainty、角色/adapter | `CANDIDATE_METADATA_MAPPED / GAP_OPEN / NOT_ADMITTED` |
| [ARKitScenes O0R contract](TARO_O0R_ARKITSCENES_SOURCE_AND_ADAPTER_CONTRACT_LOCK_2026-08-10.md) | 8 fit + 16 eval-candidate fresh TRAIN visits；registered FARO/AppleDepth、RGB、confidence、K、pose；truth/query/uncertainty/injection/statistics 已冻结；纯内存 adapter 44/44 tests PASS；24 × 3 preflight 与 data-use receipt 已绑定；HEAD/source/truth materializer 24/24 tests PASS | HEAD/Content-Length、source payload、真实 truth bundle、DepthART 与 O0R outcome 均未运行；implementation PASS 不等于 execution；手持域不能冒充穿戴式/active authority | `SOURCE_CONTRACT_ADAPTER_PREFLIGHT_AND_MATERIALIZER_IMPLEMENTATION_LOCK_FROZEN / HEAD_NOT_RUN / PAYLOAD_UNOPENED / SCIENTIFIC_NOT_RUN` |
| TwinScene future candidate | 可设计 exact factor/query interventions 与 matched nuisance pairs | renderer/cross-renderer/physical-pair validity、real anchor、independent truth、split family、许可与全部 O0R adapter | `GAP_OPEN / NOT_ADMITTED` |

任何后续 source audit 至少逐项记录：K/pose/IMU/timestamp、mesh/depth、boundary/uncertainty、factor/query
truth、paired observations、parent/session/site identity、license/use scope 与 deterministic adapter feasibility。
单个字段存在不等于 source 可用；缺失项保持 `UNKNOWN/GAP_OPEN`，不得当 negative，也不得由 teacher
或生成器填成 truth。只有 current 另立 outcome 前冻结的 source-specific contract，才可能改变状态。

### 17.2 当前重开状态

ARKitScenes R0 contract 的 24 个 identity 只从官方 Training metadata 与 pinned exclusion snapshot
产生，body/outcome 未打开。preflight 已冻结 72 个 exact HEAD target、environment、budget、
`47333152` trajectory risk 与 absent roots；用户 data-use receipt 已逐字绑定该 24 × 3 roster。随后
outcome-blind amendment 与 hash-bound materializer implementation 关闭 all-exact frame、fit-before-eval、
per-query lookup、original provenance、ndarray reload、atomic writer 与消费后 failure receipt seam，24/24 focused tests 与 6/6
validator mutation tests PASS；但 HEAD/GET 均没有运行，one-shot 没有消费，Content-Length 与远端可用性仍未知。

exact 72-URL、zero-body、hash-bound HEAD execution lock 现已冻结为 `AUTHORIZED_UNCONSUMED`，绑定
实现提交 `9c525103`、argv/environment/budget、exclusive root 与消费后 failure receipt。该锁提交后，当前唯一
successor 是 `TARO_O0R_ARKITSCENES_CONTENT_LENGTH_HEAD_EXECUTION`：只执行一次冻结 HEAD argv。
真实 O0R 仍须依次通过独立 HEAD execution、truth-only execution authorization、source/truth gate，再另行授权 baseline/factorial
execution；任一 gate 失败都终止该 evidence version，不得换 parent、降 denominator、先看 DepthART output
或回写本指南中的 gate。
