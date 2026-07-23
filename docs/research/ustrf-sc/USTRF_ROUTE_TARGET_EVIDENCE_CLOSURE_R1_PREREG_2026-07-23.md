# route-target evidence closure R1 预注册（2026-07-23）

状态：`PREREGISTRATION_VALID / SEEN_REVIEW_BUNDLE_MATERIALIZED / ROUTE_ROLE_TRUTH_PENDING / HOLDOUT_UNOPENED / NO_CANDIDATE_RUN / ANDROID_SHADOW_CLOSED / H2_CLOSED`

## 决策

立即开启 `route_target_evidence_closure_r1`，但不开 tracker、detector 或 H2 新变量。父实验已证明固定 App YOLO11n FP16-320 在两来源覆盖 `3/3 + 12/12`、critical miss `0`；T0–T3 却都只有 `14/15`、负窗 `22` 次提醒（`8.620/min`）、repeat `12`，association 只改变 fragmentation。因此本轮只回答三件事：行人是否进入/占用/穿越当前因果路线，何时对同一风险事件只交付一次，何时确认离场并终止事件。

本预注册由 `configs/ustrf_route_target_evidence_closure_r1.json` 与 fail-closed validator 共同冻结。当前 15+15 窗口只有 oracle 故障归因权限；没有候选选择、标量调参或 shadow 权限。seen 盲审 bundle 已重算 `4,594/4,594` RGB 哈希并联结 route receipt，共带入 `3,745` 个既有 target/negative seed boxes；bundle SHA-256 为 `e067e64148a313fff60a90d215acc730113b40446591a1506cc9209f7a22502d`。seed 不代表全体 person 已闭合，每帧仍要求 full-frame discovery。新鲜 holdout 尚未发现、下载、解码或物化。

## 逐人路线角色真值

每个可见 person 必须有稳定 `person_id`、逐帧 bbox/visibility、因果 route receipt 与 age、以及下列互斥角色：

- `route_intersecting`：当前脚点/占用区域与当前因果路线相交；
- `approaching_route`：仅用截至当前帧的证据，显示其到路线的有符号间距持续减小；
- `adjacent_safe`：路线外且没有进入趋势；
- `receding`：此前接近/相交后正离开；
- `cleared`：已物理通过或离开，并进入当前事件终态。

路线 invalid/stale 或视觉证据不足时必须 abstain，不能强塞五态；missing/occluded 也不能冒充 `cleared`。真值同时冻结 person-bound `risk_event_id`、alertable start、可选 intersection、passed/clear、critical、重新进入与路线切换规则。审核时隐藏 detector、candidate alert 与候选结果；两次独立模型不一致时由 fresh third-model 仲裁，仍不确定则只隔离该 person episode。

## 三条单接缝 oracle

| Arm | 只替换 | 固定不变 | 防混因要求 |
| --- | --- | --- | --- |
| O1 oracle person | 所有目标与共现 person 的 truth boxes/IDs | 当前 route-hit 与 event kernel | 不能只喂目标；没有全体稳定 ID 时负例 FA 不具解释权 |
| O2 oracle relation | 单帧 detection 唯一匹配后的 route role | detector、分数、NMS、association timing、event kernel | 不导入跨帧 truth ID；未匹配/多义 detection 直接 unknown |
| O3 oracle lifecycle | truth event grouping、one-shot latch、terminal clear | 当前 evidence stream、eventKey、age 与 route state | truth 不得凭空造 alert；首次交付仍须当帧 current evidence 成立 |

三臂与当前 T0 对照使用完全相同的 hash-bound input ledger，每臂仅替换一个 seam。O1 定位 person observation 上界，O2 定位路线关系，O3 只定位重复与清除上界。

## 三个预注册结构候选

1. `C1_CAUSAL_ROUTE_RELATION_FSM`：逐 person 的五态+unknown 因果状态机；仅 approaching/intersecting 可累计既有 2 帧 alert evidence，adjacent/unknown veto。
2. `C2_ROUTE_OCCUPANCY_EPISODE_FSM`：在 C1 上以同一 causal route segment 的占用/切入 episode 维持 one-shot latch；track fragment 可归并到未清 episode，同时显式审计多人错误合并。
3. `C3_DUAL_KEY_CLEARANCE_FSM`：在 C2 上使用 person lineage + route episode 双键；结束同时需要 person receding/cleared 与 route segment released，unknown 只暂停、不结束。

三者沿用 `.35/.45`、当前 detector、当前 association 输入、2-frame alert 与 3-frame clear；禁止在 seen 窗口调 route margin、阈值、NMS 或任何标量。候选实现、config、hash 与词典序全部冻结后才可一次性解封 holdout。

执行接缝同样属于冻结合同：正式 App `yolo11n_fp16_320.tflite`、COCO labels、Android Canvas `ImagePreprocessor` exporter 与 T0 配置必须逐文件哈希绑定；holdout truth/window 完成两来源 admission 并冻结哈希之前，App detector 和 C1–C3 都不得运行。Detector evidence 必须来自设备 CPU-4-thread canonical raw tensor，不接受已知非逐像素等价的 host PIL reconstruction。解封后每个候选对每条完整 sequence 只运行一次，状态不得在正负评分窗口边界重置；truth 只能在候选状态更新之后用于告警归因和指标计算。

## sealed holdout 与选择门

holdout 必须来自两个独立 provenance/session family，与当前 LILocBench parent sequence、person/route trace、相邻帧和近重复全部断开。每来源至少 10 个正事件、2 个 critical、10 个 matched negative 窗口和 10 分钟可评分负暴露；每来源都必须包含共现 person。来源纳入不得查看当前 kernel alert 或 detector 难度，truth reviewer 不看候选输出；每候选只运行一次，失败后不得换源、换窗、改 role 或重放挑结果。

逐来源及各指标 worst-source 必须同时满足：event recall `>=0.90`、critical miss `=0`、false alerts/min `<=0.50`、clearance `>=0.90` 且 P95 `<=1500ms`、repeat `=0`、event regeneration `=0`、evidence age P95 `<=200ms`、unknown/stale route active alert `=0`。false alert 同时计负窗提醒与正窗中由 `adjacent_safe/receding/cleared` 共现者触发的交付；clearance 的提前结束和 window-end censored 都按失败，不能只在成功子集上报告 P95。

没有候选全门通过时，结论固定为 `STOP_NO_ANDROID_SHADOW_KEEP_H2_CLOSED`，不选择“相对最好”。只有合格胜者才允许另开 production-isolated Android shadow；在此之前不修改 App，也不启用深度、TTC 或 route-risk flip。

## 当前可执行顺序

1. validator 绑定父 result、target attribution、association-only 与 frozen person truth 的 SHA-256；
2. 生成全体 person 的 candidate-hidden route-role review bundle，双模型+第三模型仲裁并冻结 seen truth；
3. 运行 T0 + O1/O2/O3，仅作故障归因；
4. 冻结 C1/C2/C3 实现与 hash；
5. 自动发现并物化两来源 sealed inventory，一次性评测；
6. 全门通过才进入独立 Android shadow，否则保持停止。

本轮全部证据只有 benchmark research authority，不提供训练、App、生产或真实辅助安全授权。
