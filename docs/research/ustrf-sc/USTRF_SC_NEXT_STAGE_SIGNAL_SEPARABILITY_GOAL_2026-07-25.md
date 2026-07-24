# USTRF-SC 下一阶段新信号可分性目标（2026-07-25）

状态：`GOAL / RESEARCH_ONLY / NOT_EXECUTED`

当前可执行边界：`ROUTE_CONDITIONED_SCALE_GROWTH_SEPARABILITY_R0`

最大权限：`SIGNAL_AVAILABILITY_AND_DISCOVERY_SEPARABILITY_ONLY`

## 一、目标

停止在已经证明不可行的 current-input policy family 内继续调整 TTL、qualification、renewal、association 或 opening timing。下一阶段改用“先证明新增信号具有可观测性与可分性，再允许工程化”的工作方式：

```text
冻结单一新信号
→ 审计可观测性上界
→ 扫描完整可分性前沿
→ 失败立即关闭当前角色
→ 表面成功只冻结 discovery candidate
→ 使用新 validation / sealed holdout 复验
→ 通过后才允许 causal producer
```

本目标不承诺尺度增长一定有效。它的首要价值是以半天至一天的离线工作，可信地支持或淘汰一个明确假设，避免再次先完成大规模在线工程、再发现输入没有可分信息。

## 二、父证据与已关闭方向

本阶段开始前必须重新绑定并验证以下父结果：

- [current-input policy feasibility bound R0](USTRF_CURRENT_INPUT_POLICY_FEASIBILITY_BOUND_R0_RESULT_2026-07-24.md)：终态 `CURRENT_INPUT_POLICY_FAMILY_NOT_FEASIBLE / VALID`；最大 coverage 仅 `8/11 = 24/33`，经验负机会不超过 2 时仅 `2/11 = 6/33`。
- [causal route-relative intrusion signal R0](USTRF_CAUSAL_ROUTE_INTRUSION_SIGNAL_R0_RESULT_2026-07-24.md)：终态 `SIGNAL_REJECT / VALID`；冻结的 radial/lateral/scale `2-of-3` 只覆盖 `7/11 = 21/33`，负暴露为 `43 / 4.956min`。
- 冻结 discovery 分母为 11 个 oracle-supported unique event、33 个 mechanically mapped supported candidate cell，以及 `4.95626851575min` 负暴露。`33/33` 与 `11/11` 必须同时报告，但不得冒充 33 个独立运行时事件。

旧 family 已由现有 `NOT_FEASIBLE / VALID` 收据关闭。本阶段不重复生成一个同义 `CURRENT_INPUT_POLICY_FAMILY_CLOSED` 实验，也不重新搜索：

- qualification duration；
- TTL；
- renewal；
- association threshold；
- opening timing；
- 已拒绝 `2-of-3` 组合的窗口、组合或零阈值。

只有增加当前 family 中不存在的候选无关、truth-blind、past-only 观测信号，才允许建立新的研究边界。

## 三、当前唯一可立即执行的垂直切片

### 3.1 研究问题

`ROUTE_CONDITIONED_SCALE_GROWTH_SEPARABILITY_R0` 只回答：

> 路线条件化的绝对尺度增长，能否单独作为 candidate-independent causal token 的 qualification 信号？

它不检验尺度信息对未来任意融合模型是否“有一点帮助”，也不授权把信号直接接入 opener。若失败，关闭的是本轮冻结定义作为 standalone token qualification 的资格，不得扩大成“尺度信息永远无用”。

### 3.2 冻结输入与非目标

保持不变：

- detector、置信度阈值、NMS 和预处理；
- T0 association 与 track/reset identity；
- causal route relation、route validity 和 reset；
- C1–C3、truth、event window、负暴露与 evaluator；
- opener、lifecycle、clearance 和提醒反馈；
- Android、Kotlin runtime、shadow、H2、人体和生产权限。

禁止：

- 同时扫描高度、面积、窗口长度、回归算法或质量门；
- 使用 event、truth、candidate、source/sequence identity、未来帧或 clearance 计算信号；
- 为改善结果增加 source whitelist、特殊 track 规则、qualification、TTL 或 renewal；
- 在同一 discovery 数据上选阈值后，再把同一数据称为 validation；
- 新建通用框架或在线 App 集成。

### 3.3 主信号定义

对 canonical source-frame 坐标中的归一化检测框，定义：

\[
S_t=\frac{1}{2}\log\left(w_t^{norm}h_t^{norm}\right)
\]

其中：

- `w_norm`、`h_norm` 必须由绑定 source frame 尺寸与 rotation receipt 的 canonical bbox 计算；
- 禁止使用显示层坐标或未绑定的 letterbox/crop 坐标；
- bbox 触及图像裁剪边界或被标记为严重截断时，该观测无效；
- `log(h_norm)` 只允许作为预先声明的诊断列，不能在查看结果后替换主信号。

每个 track/reset scope 的 `loomingScore` 固定为：

- past-only 600ms 窗口；
- 至少 5 个有效观测；
- 最大相邻观测间隔 150ms；
- 真实 timestamp 秒数作为自变量；
- 固定 Theil–Sen 稳健斜率；
- 不插值；
- relation gap、route unknown、track unobserved 或 reset 立即清空窗口。

上述选择构成一个新的、明确冻结的面积尺度假设，不冒充对上一轮 `log bbox height + 5-frame` 分量的纯归因。

### 3.4 两阶段执行隔离

**Producer 阶段**

- 只读取 canonical bbox、track/reset、route relation、route validity 和 timestamp；
- 为所有合格窗口输出 raw `loomingScore`、窗口起止时间、观测数、最大 gap、bbox 截断状态和 reset/abstention reason；
- 在 inventory 冻结前，truth、event window、oracle、负暴露和 candidate 解码数必须均为 0；
- 输出 41 条候选无关 sequence ledger，并绑定父输入和逐 ledger SHA。

**Audit 阶段**

- 在新进程中先复验 producer inventory 和全部 ledger SHA；
- 之后才 post-hoc 联结 11 个 supported unique event、33 个 supported cell 和冻结负暴露；
- 以全部实际观测 slope 的断点构造完整 threshold frontier；
- 输出完整 Pareto，不选择“最佳阈值”，不输出在线 policy。

### 3.5 主要门与完整披露

standalone qualification 的 discovery 可行点必须同时满足：

| 门 | 要求 |
| --- | ---: |
| supported unique event coverage | `11/11` |
| supported candidate cell coverage | `33/33` |
| 当前负暴露内 first opportunity | `<=2` |
| 首次合格延迟 | 通过输出前已冻结的 evaluator 数值门 |

若父 evaluator 尚无明确延迟数值门，必须在任何 signal outcome 生成前，以不读取 truth outcome 的独立理由写入配置并 hash-bind；不得事后使用“没有明显恶化”。

主要负机会口径保持为 `first opportunity per track-reset`，以便与父经验风险门一致。同时必须完整报告：

- first opportunity per activation interval；
- sequence-level negative incidence；
- opportunity duration；
- requalification count；
- full-sequence activation count；
- source、sequence 和 worst-source 分层；
- 经验点率与一侧 95% Poisson working UCB。

`<=2 / 4.956min` 只代表 discovery 数据上的经验 Pareto 筛选，不代表可信总体风险通过，也不提升 Evidence Maturity V2 权限。

### 3.6 唯一合法终态

#### `PURE_SCALE_GROWTH_NOT_SUFFICIENT_FOR_STANDALONE_TOKEN_QUALIFICATION`

完整 threshold frontier 不存在同时通过 coverage、经验负机会和延迟门的点。关闭本轮冻结尺度定义的 standalone qualification 角色；不调公式、窗口或阈值回救，不自动启动面积/高度/窗口变体。

尺度若未来作为辅助分量重新进入，必须建立新的预注册增量实验，明确：

- 主信号或基线；
- scale 的辅助角色；
- 增量指标和复杂度代价；
- 冻结消融；
- 未参与选择的 validation 数据。

#### `SCALE_GROWTH_DISCOVERY_CANDIDATE_FROZEN`

完整 frontier 至少存在一个 discovery 可行点。按预注册 tie-break 只冻结一个 threshold candidate；它只获得进入新 validation 数据的资格，不获得 producer、opener、shadow 或更高权限。

#### `FAIL_CLOSED_INPUT_OR_CONTRACT_BLOCKED`

父哈希、canonical bbox、timestamp、route/reset、frame membership、producer/audit 隔离或 validator 任一不完整。输出精确 gap/violation receipt 后终止，不缩小分母、不生成候选。

## 四、独立的证据补充工作

现有 11 个 supported unique event 与 `4.956min` 负暴露继续作为 discovery/falsification 数据，不承担最终接受职责。下一 validation 数据包从采集开始冻结为：

```text
discovery
validation
sealed holdout
```

初始目标为至少 3 个独立 session、合计约 15–20 分钟，并保持 session/sequence cluster 独立；不得把一条长视频切成多个独立样本。场景矩阵至少覆盖：

| 相机/用户运动 | 目标状态 | 研究作用 |
| --- | --- | --- |
| 相机静止 | 目标主动接近 | 测目标自身绝对扩张 |
| 相机前进 | 静止目标位于路线中 | 测自运动下真实接近 |
| 相机前进 | 目标平行移动 | 测长期共现 |
| 左右转向 | 静止背景人物 | 测旋转假扩张与扫入 |
| 相机前进 | 远处路线外静止目标 | 测全局扩张污染 |
| 相机静止 | 目标横向穿过路线 | 测尺度弱、横向关系强的事件 |
| 相机前进 | 目标同步远离 | 测净相对运动 |
| 多目标交叉 | 只有一人侵入路线 | 测动态背景与身份干扰 |
| 背景人物先出现 | 真目标后出现 | 测 opening preemption |
| 短暂擦过 polygon 边缘 | 无持续侵入 | 测瞬时关系假机会 |

采集必须使用受控相机装置或依法、合规、已获授权的正常视力采集流程，提醒链关闭；不得进行盲人独立行走或 human-facing 试验。目标 taxonomy、route truth、event ontology、许可/同意和数据角色必须在使用前形成真实收据；自动化或模型不得伪造这些来源事实。

10–15 分钟零负机会可以改善 pooled Poisson 界，但不能自动解决 source/session cluster 独立性、worst-source 或 Evidence Maturity V2 要求。

## 五、后继研究路线：必须另立独立 goal

下述路线是条件式 roadmap，不属于当前 R0 的执行权限。当前 R0 结束后必须根据 terminal 另立 goal，不得在同一轮自动继续。

### Gate 2：`EGO_MOTION_SIGNAL_AVAILABILITY_R0`

只审计背景运动信号能否稳定观测：

- source RGB 是否连续、timestamp 是否可靠；
- 排除所有 detector-observed person bbox 及固定扩张区域后，背景特征是否充足；
- 特征空间分布、RANSAC inlier ratio、重投影残差和变换条件数是否通过；
- route relation、frame gap 与 reset 约束下的可用窗口比例；
- 质量门后的最大 unique-event/candidate-cell coverage；
- worst-source availability 和不可用原因。

R0 只允许一个固定实现族：

```text
人物区域剔除后的背景稀疏 LK 光流
+ RANSAC 2D affine
+ 固定质量门
```

不同时加入 homography、dense flow、IMU、VIO、depth、multi-model voting 或 source-specific patch。质量不足时必须弃权，不能把 ego motion 默认为 0。可用性乐观上界不足时立即终止，不进入 separability。

### Gate 3：`EGO_MOTION_AWARE_EXPANSION_DECOMPOSITION_R0`

若 Gate 2 通过，同时保留：

\[
L_{absolute},\qquad
L_{ego},\qquad
L_{relative}=L_{absolute}-L_{ego}
\]

并输出 camera-motion quality、route relation、track/bbox quality、abstention reason 和 reset reason。

- `L_absolute` 保留“相机/用户接近静止目标”的信息；
- `L_ego` 解释全局转向、前进和抖动；
- `L_relative` 只表示目标相对背景的额外扩张；
- residual 不得单独解释为危险、TTC、距离或碰撞概率。

discovery 阶段只允许预注册的归因 arms：

1. absolute-only；
2. residual-only diagnostic；
3. absolute + camera-quality gate。

不得在同一 discovery 数据上任意组合多个阈值并宣称获得最终融合策略。

### Gate 4–5：冻结复验与工程化

只有冻结 candidate 在未参与选择的 validation 数据上通过，才允许一次性运行 sealed holdout。查看 holdout 后不得继续调参并保持 holdout 名义。

只有 validation/holdout 同时通过 coverage、风险、延迟、worst-source 和 cluster 门，才允许另立 causal producer goal；随后仍按独立阶段推进：

```text
causal producer
→ lifecycle / opener
→ full replay
→ Android shadow
→ 受控设备证据
→ human-facing review
```

任何 discovery 成功均不直接授权跨级。

## 六、防止再次进入死胡同的硬规则

1. 一个实验只回答一个问题，只扫描一个主要变量。
2. 先算输入可用性与乐观覆盖上界，再写实现。
3. 同一数据上的失败可以关闭假设；成功只能生成待验证候选。
4. 失败后禁止增加 TTL、renewal、特殊 source 规则或候选相关补丁回救。
5. separability 通过前不写 Android、Kotlin runtime、opener 或通用框架。
6. 每个新信号默认在半天至一天内形成 terminal receipt；超出预算先终止并解释缺口，不默认扩工程。
7. 纯尺度与自运动感知扩张两个新信息边界均失败后，暂停单目 bbox 特征搜索，重新审计 metric depth、IMU/VIO、route truth、事件 ontology 和任务拆分。
8. 负结果是项目进展，但不是算法能力或产品效果提升。
9. 只有未参与选择的数据上出现稳定可分性，才称为算法进展。
10. 只有 producer、shadow、设备链、反馈送达与后续受控证据闭合后，才讨论用户侧产品进展。

## 七、当前 R0 完成条件

只有以下事项全部满足，当前 `ROUTE_CONDITIONED_SCALE_GROWTH_SEPARABILITY_R0` 才算完成：

1. 配置冻结父哈希、信号定义、窗口、回归、质量门、延迟门、阈值断点和终态；
2. producer 在 truth/event/oracle/candidate 解码为 0 的条件下完成候选无关 ledger；
3. audit 在独立进程中先复验全部 producer SHA，再联结 truth 与负暴露；
4. 完整 slope breakpoint frontier 可由 validator 独立重建；
5. 同时报告 11/11 与 33/33，不扩大独立分母；
6. 主要 first-opportunity 风险口径和 interval/sequence/duration/requalification 完整披露均闭合；
7. focused tests 覆盖 reset、route unknown、relation gap、时间戳、截断、forbidden input、producer/audit 隔离、断点完整性和终态；
8. 结果只使用三个合法终态之一；
9. 不产生在线 policy、producer authority、opener、Android、shadow、H2、人体或生产授权；
10. 形成日期化结果、机器收据、验证收据，并更新研究索引与开发日志。

## 八、可直接启动的 `/goal`

```text
/goal 完成 ROUTE_CONDITIONED_SCALE_GROWTH_SEPARABILITY_R0，只审计路线条件化绝对尺度增长能否作为 standalone candidate-independent causal token qualification 信号。

先重新验证并绑定 current-input policy feasibility bound R0 与 causal route-relative intrusion signal R0 的父收据。保持 detector、NMS、T0 association、route relation、route validity、reset、C1–C3、truth、evaluator、opener、lifecycle、clearance、Android、H2 和生产权限不变；不再调整 TTL、qualification、renewal、association 或 opening timing。

主信号固定为 canonical source-frame 归一化 bbox 的 S_t=0.5*log(w_norm*h_norm)。只使用 past-only 600ms、至少5个有效观测、最大相邻 gap 150ms、真实 timestamp 和固定 Theil–Sen slope；不插值，bbox 触边/严重截断窗口无效，relation gap、route unknown、track unobserved 或 reset 立即清空。log(height) 仅作诊断，禁止结果后替换主信号。唯一扫描变量是 slope threshold，阈值集合使用全部实际 slope 断点。

严格分离 producer 与 audit：producer 只读 canonical bbox、track/reset、route relation/validity 和 timestamp，inventory 冻结前 truth/event/oracle/负暴露/candidate 解码必须均为0；audit 在新进程先复验全部 ledger SHA，之后才 post-hoc 联结11个 supported unique event、33个 supported cell和4.95626851575min负暴露。

输出完整 coverage-risk-delay Pareto，不选择最佳 policy。standalone discovery 可行点必须同时达到11/11、33/33、负暴露 first opportunity<=2和预先冻结的首次合格延迟门。主要风险单位为 first opportunity per track-reset，并同时完整报告 activation interval、sequence incidence、opportunity duration、requalification、full activation、source/sequence/worst-source、点率和Poisson working UCB。<=2只代表 discovery 经验筛选，不代表可信总体风险通过。

唯一合法终态：
1. PURE_SCALE_GROWTH_NOT_SUFFICIENT_FOR_STANDALONE_TOKEN_QUALIFICATION；
2. SCALE_GROWTH_DISCOVERY_CANDIDATE_FROZEN；
3. FAIL_CLOSED_INPUT_OR_CONTRACT_BLOCKED。

失败后不调公式、窗口或规则回救；成功也只冻结一个 discovery candidate，必须进入新 validation 数据，不写 Kotlin/Android、不接 opener、不进 shadow。所有证据写入 artifacts.local/ 并由独立 validator 重建；更新日期化结果、研究索引和开发日志。保持 mobileclip_blt.ts 及其他并行改动不动。
```
