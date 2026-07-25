# USTRF 独立相机运动补偿 looming 信号 R0 目标（2026-07-25）

状态：`PREREGISTERED / RESEARCH_ONLY / NOT_EXECUTED`

当前可执行边界：`NEW_SOURCE_SESSION_AUTHORITY_AND_SPLIT_FREEZE`

最大权限：`CONTINUOUS_SIGNAL_AVAILABILITY_AND_CROSS_SOURCE_SEPARABILITY_ONLY`

## 一、结论与路线选择

[route-conditioned program closure R1](USTRF_ROUTE_CONDITIONED_PROGRAM_CLOSURE_R1_2026-07-25.md) 已永久关闭旧 dense、bbox-route、timing/token、causal lifecycle、120 episode / U0 与 architecture convergence 路线。旧 15 对窗口只保留为历史 discovery / falsification evidence；本目标的 producer、audit、配置、门禁和数据清单均不得读取它们。

下一阶段独立比较三条不继承旧 route field 或 lifecycle 的新假设：

| 路线 | 核心问题 | 优点 | 主要证据风险 | 当前排序 |
| --- | --- | --- | --- | ---: |
| 相机运动补偿后的 looming / collision cue | 去除头部转动、扫动和抖动影响后，局部径向扩张是否仍与真实几何接近强度稳定相关？ | 物理含义明确；连续量；可设计纯转动、静止目标和主动接近反事实 | 光流、相机姿态、深度与遮挡质量；全自运动消除可能错误抹掉“相机走向静止障碍”的真实接近 | `1` |
| 开放集未知障碍信号 | YOLO 类别外的视觉异常、地面突变或未知实体能否被发现？ | 直接研究 detector 未见类别；不依赖 COCO AP | “未知”与风险不是同义词；异常 truth、域偏移和开放集负分母很难冻结 | `2` |
| 短时未来占用预测 | 仅用历史帧能否预测未来数帧的空间占用变化，并识别风险区域正在形成？ | 与动态空间变化最直接；可覆盖无类别目标 | 需要新的 dense future truth；容易把未来帧、旧 route 或 lifecycle 偷渡进输入与评价 | `3` |

因此先执行 `EGOMOTION_COMPENSATED_LOOMING_SIGNAL_R0`。日常使用体验、提醒文案、触觉或现有 App 向导不再作为当前算法研究主线，也不因本目标发生任何改动。

## 二、独立假设与科学边界

### 2.1 主假设

在全新、按 source 与 session 隔离的视频上，因果、truth-blind 的旋转补偿局部径向扩张连续信号，相对 raw optical flow、bbox growth 和未补偿局部扩张，能更稳定地区分真实几何接近与纯相机转动/扫动，并在每个来源而非仅 pooled aggregate 上形成明确增益。

### 2.2 物理量不混同

本目标区分：

1. **旋转、扫动与抖动**：可产生大幅光流，但不必产生距离闭合，应由主候选补偿；
2. **相机向静止障碍前进**：虽由相机平移引起，却是真实几何接近，主候选必须尽量保留；
3. **目标相对背景主动接近**：在相机静止或运动时均可形成额外局部扩张；
4. **完整 6DoF 自运动消除后的 residual**：只表示目标相对静态场景的额外运动，不等于完整 collision cue。

因此主候选只冻结为**旋转补偿后的局部扩张**。oracle rotation arm 是旋转输入质量上界；使用 pose + depth/scene geometry 消除完整 6DoF 自运动的结果只作为 residual 诊断参考。若后者抹掉“相机前进接近静止障碍”，必须如实报告，不得因数值更干净而升级为主信号。

### 2.3 允许的 claim

R0 只允许回答：

> 一个连续、无 route、无 lifecycle 的局部扩张信号，是否相对对照具有跨来源稳定的几何接近可分性？

它不允许声称：

- 已估计米制 TTC、碰撞概率、可通行路线或报警等级；
- 已发现所有未知障碍；
- 已获得提醒、App、shadow、人体、安全或生产效果；
- oracle pose/depth、未来帧或 post-hoc truth 可作为在线输入。

## 三、数据角色与永久隔离

### 3.1 旧证据禁用

- producer、truth builder、audit、metric config 与 runner 对旧 15 对窗口的帧、边界、结果、阈值和派生统计读取数必须为 `0`。
- 独立的 `OLD_WINDOW_ADMISSION_FIREWALL` 是唯一允许读取旧数据身份标识的组件；它不得读取任何旧 outcome、窗口统计或候选结果。
- firewall 在新数据 payload 解码前冻结旧 source-family、session/member、逐帧 cryptographic content hash、可复原 decode hash 与 perceptual near-duplicate index。旧 source family/session、精确副本、转码副本、裁剪/缩放派生近重复均拒收。
- firewall 只向下游发布 `CLEAN_ROOM_ADMISSION_RECEIPT`、新数据 manifest SHA 与拒收计数；producer/audit 不接触 denylist 本身。缺少可复算 denylist 或 near-duplicate validator 即 fail closed。
- 旧 `11 event / 33 cell / 4.956min`、旧 route truth、C1–C3、T0、opener、clearance、reset lifecycle 和报警结果均不进入新分母。
- 历史 `11/31` JRDB global-affine canary只说明冻结 full-affine method/window availability failure；不得拿来选择本轮 motion model、质量门或接受阈值。

### 3.2 新数据最低结构

在读取候选信号结果前冻结 immutable manifest：

- 至少 `3` 个真实 source family；
- 每个 source 至少 `8` 个独立 session：discovery 至少 `4` 个，validation 至少 `2` 个，sealed holdout 至少 `2` 个；
- 一条长视频不得切段后冒充多个 session；
- 同一采集主体、场景连续段或相邻 burst 只能属于一个数据角色；
- 每个 source 都必须包含可评价的纯转动无接近、相机前进接近静止目标、相机静止目标主动接近和横向经过无持续闭合；
- 每个 `source × role × 必需反事实 cell` 至少覆盖 `2` 个独立 session；每个 session-cell 至少有 `10s` 连续可评价时长和 `20` 个有效、互不重叠的 500ms epoch；
- synthetic、模型生成或人工合成数据只可作 diagnostic，不得满足上述真实 source/session 分母。

每个 admitted session 必须绑定 source URL/来源、session identity、RGB member、capture timestamp、intrinsics、相机姿态/IMU可用性、深度或 scene geometry 可用性、对象/表面几何 truth、许可/使用限制与逐文件 SHA。缺失只使最小依赖 arm/单元 abstain；不得把缺失 pose、depth 或 flow 默认为零。

### 3.3 数据角色

```text
source/session metadata-only inventory
  -> immutable role freeze
  -> R0 discovery
  -> R0 terminal / optional frozen candidate
```

- discovery 可淘汰假设；表面成功只冻结一个 candidate。
- **R0 到 discovery terminal 为止**；validation 与 sealed holdout 只在 role freeze 中密封，不在 R0 打开。
- `LOOMING_DISCOVERY_CANDIDATE_FROZEN` 后必须另立 validation R1；R1 失败/成功终态与 one-shot holdout R2 必须在各自执行前独立预注册。
- validation 和 sealed holdout 永远不得参与 flow model、窗口、网格、质量门、归一化、缺失策略或 arm 选择。sealed holdout 一经打开，任何修改都必须建立新版本并使用新的 sealed data。

### 3.4 GPT / Codex 自主准入

source-native timestamp、intrinsics、pose、depth/geometry、文件 hash 和公开页面事实只按原始 receipt 核验，不由模型投票创造。凡 source discovery、场景/反事实 cell、视觉质量、隐私、非 source-native mask/identity 或数据准入需要模型判断时，必须绑定 [GPT / Codex 自主复核治理](../../AI_REVIEW_GOVERNANCE.md)：

- 先冻结 input bundle、允许字段、候选输出不可见性、prompt 与 SHA；
- 两个互不可见的新上下文分别生成结构化 receipt；
- 分歧时使用第三个全新上下文裁决；仍 abstain 的最小单元隔离/拒收；
- receipt 通过 `configs/ai_review_workflows_v1.json` 对应 workflow 与 `scripts/validate_ai_review_receipt.py` 复验。

模型共识不能补造相机姿态、深度、参与者同意、许可证授予或真实设备测量。

## 四、连续 truth 与 producer 隔离

### 4.1 独立连续 truth

评价 truth 固定为 source-native 3D 几何得到的非负闭合强度：

\[
G_t=\max\left(0,-\frac{d\log r_t}{dt}\right)
\]

其中 `r_t` 是相机到 source-native 对象表面或冻结空间单元的距离。truth 计算必须：

- 使用 source-native timestamp、pose、depth/3D geometry 和对象/表面 identity；
- 明确插值、缺失、遮挡、离屏和 identity 规则；
- 不使用候选 flow、detector bbox、route、alarm、lifecycle 或未来候选输出；
- 把 `G_t=0` 的纯转动、横向经过和无闭合片段保留在完整分母中。

`G_t` 只是几何接近强度，不是 collision probability。没有 route/body-envelope 时，不产生“必撞”或“安全通过”标签。

### 4.2 truth-blind producer

producer 只输出 fixed-grid / dense 连续 field、质量和 abstention：

- 不读取 3D approach truth、对象 mask/identity、事件窗口、source role outcome 或未来帧；
- 不使用 detector bbox 选择局部区域；
- 在 inventory 与逐 session signal ledger SHA 冻结后，独立 audit 才可把 truth ROI/空间单元 post-hoc 联结到 field；
- bbox-growth baseline 在独立 namespace 中运行，detector miss 必须记为 abstain，不能改变主 producer 的 grid 或 mask。

### 4.3 可复算评价原子

- 原子单位固定为 `source / role / session / truth-region / counterfactual-cell / 500ms epoch`；epoch 从 session 首个 capture timestamp 起按互不重叠 500ms 对齐。
- source-native truth region 通过冻结 intrinsics、pose 与 geometry 投影到图像。grid cell 只有中心落在 region 内且交叠面积 `>=50%` 才进入；投影不可核验时该原子 abstain。
- 每个 arm 的 epoch signal 是：先对同一 flow pair 内的合格 grid-cell 非负 score 取 median，再对 epoch 内有效 flow pair 取 median。每个 epoch 至少 `3` 个有效 flow pair、每 pair 至少 `4` 个合格 cell，且 truth-region 有效覆盖率 `>=50%`。
- epoch truth 是同一 500ms 内 source-native `G_t` 的 median。`G_t` 导数窗、传感器分辨率对应的 `epsilon_G` 和 tie 规则在 admitted payload 解码前冻结。
- concordance 的 comparable pair 必须来自同一 source/role、不同 session，且 `abs(G_i-G_j)>epsilon_G`；signal tie 计 `0.5`，truth tie 不进入分母。
- 每个无序 session pair 总权重相等，其内部所有 comparable epoch pair 等权；source concordance 是全部合格 session-pair concordance 的等权均值。任何单个 session 不得因帧数、对象数或时长获得更大权重。
- Kendall `tau-b` 使用同一原子、跨 session comparable-pair 与等权规则；它是共同方向诊断，concordance 是冻结主指标。
- CI 使用 `10,000` 次 session-cluster bootstrap，在 source 内整 session 重采样；pooled 值对 source 等权。per-source 少于 `4` 个 discovery session、少于 `6` 个合格 session pair或任一必需 cell 不满足分母时为 `NOT_EVALUABLE`，不能通过。

## 五、冻结比较臂

所有臂使用同一 RGB、timestamp、evaluation unit 和 common-support 报告；除 `BBOX_LOG_AREA_GROWTH` 外不依赖 detector 类别。

| Arm | 冻结定义 | 角色 |
| --- | --- | --- |
| `RAW_FLOW_ENERGY` | 原始 dense optical-flow magnitude 与未校正 divergence；不估计中心、不补偿相机运动 | 最弱 raw optical-flow 对照 |
| `BBOX_LOG_AREA_GROWTH` | 冻结 YOLO 输出的 past-only `0.5*d(log area)/dt`；无 route、无阈值选择 | 已知类别 detector/bbox 对照 |
| `UNCOMPENSATED_LOCAL_RADIAL_EXPANSION` | 在冻结多尺度网格内，从原始 flow 计算局部最佳径向中心、正 divergence 与各向同性；不做 ego compensation | 无运动补偿对照 |
| `ROTATION_COMPENSATED_LOCAL_EXPANSION` | 主候选；先用冻结的 causal camera-rotation estimator 去除旋转/扫动分量，再以同一网格和 kernel 计算局部径向扩张 | 待检验新假设 |
| `ORACLE_ROTATION_COMPENSATION` | 使用同步 source-native orientation、intrinsics 与 timestamp 精确去除 rotation flow，保留平移接近 | rotation compensation 的 oracle 输入上界 |
| `FULL_6DOF_RESIDUAL_DIAGNOSTIC` | 使用同步 source-native 6DoF pose、intrinsics 与 depth/static geometry 预测并去除完整自运动 flow | 目标相对静态场景运动诊断；不参与 collision-cue 排名 |

flow backbone、权重 SHA、输入分辨率、帧间隔、网格尺度、局部中心求解、平滑窗、rotation estimator、RANSAC/质量门、pose/depth 同步容差、`G_t` 导数估计器/窗口和所有 abstention rule 必须在 metadata-only/data-role freeze 后、解码 admitted RGB/geometry payload 或读取任何 `G_t` / arm outcome 前写入 versioned config 并 hash-bind。只允许用不进入任何数据角色的独立 engineering canary 验证解码与坐标合同。R0 只允许这一套冻结实现；不得按 source 加 patch 或从结果中挑光流模型。

`ORACLE_ROTATION_COMPENSATION` 的“上界”只指旋转输入权威与可用信息上界，不保证其 signal performance 高于其他 arm。`FULL_6DOF_RESIDUAL_DIAGNOSTIC` 可能有意消除静态障碍接近，禁止进入 go/no-go comparator 集合。

## 六、反事实矩阵

每个真实 source 必须至少覆盖：

| 条件 | truth 预期 | 主候选必须回答 |
| --- | --- | --- |
| 左右头部/相机转动，静态场景无接近 | `G_t≈0` | 相对未补偿 arm 明显抑制假扩张 |
| 相机前进，静止障碍距离闭合 | `G_t>0` | rotation-only 补偿后仍保留接近信号 |
| 相机静止，目标主动接近 | `G_t>0` | 保留局部扩张，不依赖 bbox 类别 |
| 目标横向经过，无持续距离闭合 | `G_t≈0` | 不把横向大 flow 当径向接近 |
| 相机抖动/短暂模糊，无接近 | `G_t≈0` 或 abstain | 质量不足时 abstain，不默认零运动 |
| 遮挡、形变或裁剪造成外观变大但无 3D 闭合 | `G_t≈0` | 暴露非刚体/成像伪扩张，不用阈值回救 |

full-6DoF oracle 若消除“相机前进接近静止障碍”的信号，只能证明其 residual 更接近“目标自身运动”，不能证明它更适合作为 collision cue。

## 七、先评价连续可分性

R0 不生成报警 threshold、precision/recall operating point、false alerts/min、risk tier 或在线 policy。

### 7.1 主要指标

按 source、session、反事实 cell 和 pooled-but-clustered 四个层级报告：

1. signal 与连续 `G_t` 的 Kendall `tau-b` 和 source-stratified concordance index；
2. 在完全相同 evaluable support 上，主候选相对 `UNCOMPENSATED_LOCAL_RADIAL_EXPANSION` 的 primary `delta concordance`；
3. 主候选相对 `RAW_FLOW_ENERGY` 与 `BBOX_LOG_AREA_GROWTH` 的逐臂 pairwise common-support delta；bbox arm 只作已知类诊断，不参与 primary go/no-go；
4. session-cluster bootstrap 95% CI、leave-one-source-out / leave-one-session-out 方向和 worst-source 数值；
5. availability、abstention、quality failure、motion magnitude、occlusion、range 与 blur 分层；
6. 纯转动 cell 的 signal suppression ratio；
7. 相机前进接近静止障碍与相机静止目标接近的 signal retention ratio。

缺失值不填 `0`。不同 support 的结果先各自披露，再报告 intersection-support 比较；不得用较窄 support 制造更高分。

suppression / retention 先按 session-cell 计算：

\[
\text{suppression}=1-\frac{\operatorname{median}(\max(S_{rot},0))}
{\max(\operatorname{median}(\max(S_{uncomp},0)),\epsilon_S)}
\]

\[
\text{retention}=\frac{\operatorname{median}(\max(S_{rot},0))}
{\max(\operatorname{median}(\max(S_{uncomp},0)),\epsilon_S)}
\]

`epsilon_S` 只能由不进入任何数据角色的 engineering canary 和数值精度预先冻结。若未补偿 median `<=epsilon_S`、有效 epoch 不足或任一 arm abstain，session-cell 为 `NOT_EVALUABLE`；必需 cell 因此不足分母时整个 source 不能通过。source 值对 session 等权，CI 同样整 session bootstrap。

### 7.2 冻结发现门

`ROTATION_COMPENSATED_LOCAL_EXPANSION` 只有同时满足以下条件才可冻结为 validation candidate：

- 每个 source 的 concordance point estimate 均 `>=0.60`；
- 相对 primary comparator `UNCOMPENSATED_LOCAL_RADIAL_EXPANSION`，每个 source 的 `delta concordance > 0`，worst-source point estimate `>=0.03` 且 worst-source 95% lower bound `>0`；
- session-cluster bootstrap 的 pooled `delta concordance` 95% lower bound `>0`；
- 相对 raw-flow 与 bbox 两个诊断臂的 pairwise common-support delta 在任一 source 不得为负；它们不替代 primary comparator；
- leave-one-source-out 和 leave-one-session-out 均不发生增益方向翻转；移除任一 session 后 `abs(delta_minus_session-delta_full)/max(abs(delta_full),0.01)` 不得超过 `0.50`，否则视为少数 session 支配；
- 纯转动无接近 cell 的 median signal 相对未补偿 arm 至少下降 `30%`；
- 两类真实接近 cell 的 median signal retention 均至少为未补偿 arm 的 `80%`；
- 任一 arm 的质量 abstention 不得通过删除困难 session 使 source/session 分母失守。

这些是独立研究冻结门，不是安全或产品接受门。数值在读取 outcome 前 hash-bind；若新数据 availability 使门不可评价，终态必须 fail closed，不缩小到好看的 source。

## 八、唯一合法终态

### `FAIL_CLOSED_NEW_DATA_OR_TRUTH_AUTHORITY_BLOCKED`

新 source/session 数、角色隔离、timestamp、intrinsics、pose/depth/geometry truth、许可/哈希、producer/audit 隔离或 common-support 任一不能闭合。不得回用旧 15 对窗口、旧 JRDB 32 帧或旧 route truth 补分母。

### `STOP_LOOMING_NO_CLEAR_INCREMENTAL_GAIN`

主候选相对 primary 未补偿对照没有满足冻结的增益/反事实门，或相对 raw-flow / bbox 诊断臂出现 source-level 负 delta。停止本实现，不调窗口、flow、网格、motion model 或 source rule 回救。

### `STOP_LOOMING_WORST_SOURCE_UNSTABLE`

pooled 结果表面良好，但任一 source 低于门、方向翻转或由少数 session 支配。停止路线；不得删除 worst source 或改为 pooled-only claim。

### `LOOMING_DISCOVERY_CANDIDATE_FROZEN`

全部 discovery 门通过。只冻结 config、model/hash、数据角色和一个 continuous candidate；R0 到此结束。只有另立 validation R1 后才可打开既定 validation session；不获得 alarm、App、lifecycle、shadow、人体、安全或生产权限。

未来 validation 与 sealed holdout 即使在各自独立 goal 中通过，也最多获得 `CONTINUOUS_SIGNAL_SEPARABILITY_REPLICATED`。要进入 causal runtime、报警或 App，必须再另立目标并重新定义任务 truth、身体/空间风险关系、延迟、失效与产品证据，不能自动继承本 R0。

## 九、实现和停止纪律

1. 当前第一动作仅做新 source/session metadata-only authority 与 split freeze；未通过前不下载超出有界 canary 的大包、不实现通用 framework。
2. 新研究代码只能进入 `scripts/research/egomotion_compensated_looming/`，输出只能进入 `artifacts.local/evidence/ustrf/egomotion_compensated_looming_r0/`。
3. producer、truth builder、audit 和 validator 使用独立进程与独立 receipt；validator 必须从原始 ledger 重建全部主要指标与 terminal。
4. 禁止修改 App、Kotlin、CameraX、YOLO 权重/NMS/confidence、旧 USTRF runner、route field、opener、lifecycle 或 feedback。
5. 任一 worst-source 不稳定、对照无明确增益或输入权威不足即停止；负结果即完成，不开启同轮参数搜索。
6. 开放集未知障碍和短时未来占用预测保持独立 roadmap；本 R0 结束后才可分别另立 goal，不共享 discovery outcome、阈值或接受证据。

## 十、R0 完成条件

R0 只有在以下事项全部闭合时才算完成：

1. 新数据 manifest、source/session cluster、三种数据角色和禁用旧 hash 已冻结；
2. frozen config 明确全部 arm、连续 field、质量/abstention 和反事实门；
3. producer 在 truth/outcome/旧窗口读取为 0 的条件下生成 hash-bound signal ledger；
4. audit 先复验 signal SHA，再联结独立 `G_t` truth；
5. validator 可重建 per-source、worst-source、cluster CI、common support、suppression/retention 和 terminal；
6. 只使用四个合法终态之一；
7. 形成日期化 result、机器 receipt、focused tests、研究索引和开发日志；
8. App、route、lifecycle、alarm threshold、shadow、人体与生产改动均为 `0`。
