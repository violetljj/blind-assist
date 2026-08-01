# HFTF 候选未来可通行场支线

当前状态：

`CANDIDATE_SIDE_LANE_ACTIVE / DEVELOPMENT_STANDARD /
H1_MULTI_HEIGHT_PROXY_NOT_SUPPORTED_STOP /
R2_POINT_SUPPORT_PROXY_BURNED /
STAGE_B_R3_SOURCE_OR_REFERENCE_NOT_EVALUABLE /
R3_1_REFERENCE_OPPORTUNITY_COHORT_NOT_EVALUABLE /
R4_STAGE_B_SPLIT_SOURCE_TEACHER_MECHANICS_SUPPORTED /
C0_EGOWALK_MEDIA_TRANSPORT_NOT_EVALUABLE /
C0_1_STAGE_C_SOURCE_TRANSPORT_FEASIBILITY_SUPPORTED /
D0_SEMANTIC_INDEPENDENT_LABEL_READINESS_SUPPORTED /
D1_CAUSAL_FUTURE_LABEL_MECHANICS_SUPPORTED /
STAGE_C_FRESH_FOOT_GROUND_STUDENT_CANARY_E0_FROZEN /
E0_FRESH_SOURCE_LOCK_VALIDATED /
E0_FRESH_MEDIA_TRANSPORT_SUPPORTED /
E0_FRESH_TEACHER_MECHANICS_NOT_EVALUABLE /
STAGE_C_FOOT_GROUND_STUDENT_CANARY_E0_1_FROZEN /
E0_1_FOOT_GROUND_STUDENT_CANARY_NOT_EVALUABLE /
STAGE_C_MULTI_SOURCE_EVALUATION_QUALIFICATION_E0_2_FROZEN /
E0_2_FIXED_BATCH_TEACHER_MECHANICS_NOT_EVALUABLE /
EGOWALK_FOOT_GROUND_STUDENT_SOURCE_ROUTE_CLOSED /
INNOVATION_NOT_EVALUABLE /
RESEARCH_MAINLINE_UNCHANGED / DEFAULT_APP_UNCHANGED`

## 当前结论

HFTF（Human-Centric Future Traversability Field）已作为独立候选支线立项，但没有
替换当前双环研究主线，也没有进入 Android、提醒或默认 App。

本支线检验的不是“再换一个主模型”，也不把场的基础维度包装成新发明。它把历史
USTRF 已出现的方向、距离、高度、身体/头部、动态和 uncertainty primitives 收窄成一个
版本化候选合同：

`F(theta, distance, horizon, height_band) -> risk_score + known_score`

其中 `height_band` 至少区分 `foot / body / head`，`horizon` 同时包含 current 与短时
future。教师可以在训练时使用 metric depth、pose 与 future frames；student 推理只能
使用当前及历史 RGB。任何几何派生标签都是 proxy，不是人类事件或安全真值。

真正待检验的新信号假设是：一个 **action-agnostic、history-RGB、显式短期未来** 的
分层 cell predictor，能否产生 current-only/历史 USTRF 没有证明的表示增量。

通用 H0 最初只准入静态 metric projection。随后 source-specific verifier 固定官方
SANPO loader Git commit 与 `common.py` hash、在线 GCS object generation/MD5/CRC32C
和本地 source bytes，复算出 pose row `frame_num` 与同编号 RGB/depth/mask 的绑定。
metric-depth reprojection 在发现会话上从 48 个轴/方向假设中唯一选择
`p_world = R_xyzw @ p_opencv_camera + translation_m`。

H0.2 又在 outcome-blind 规则选出的三个独立 official train sessions 上冻结并复现该
公式；三会话均 rank 1，median relative depth error 为
`0.000369–0.000763`。每帧 semantic-ground + metric-depth 的确定性局部平面拟合均得到
`+Z` source-derived vertical，三会话 camera-to-local-ground proxy 中位数为
`1.229–1.307 m`。cohort 终态为
`HFTF_H0_2_INDEPENDENT_SESSION_REPLICATION_ADMITTED`，只授权 H1 geometry teacher
canary。

这不是官方或物理 camera-to-person calibration。SANPO feature label 的
`right_handed_y_up` 与这四个回放 source-derived `+Z` vertical 有局部冲突；当前只对
这些 evidence versions 使用 source-derived frame，不外推到其他 SANPO 版本，更不声称
官方标签整体错误。精确 capture timestamp、真实人体尺寸、participant event truth 与
student/effect 仍为 `NOT_EVALUABLE`。

H1 R0 已按 outcome 前冻结的数值协议一次执行并关闭为
`H1_GEOMETRY_TEACHER_NOT_EVALUABLE`。四个 source sessions 与 usable anchors 过门，
但 360° anchor-centric field 的 current/near/far required-cell known coverage
只有 `5.62%–9.68% / 0.54%–6.13% / 0%–4.25%`，低于冻结的
`.15/.10/.10`。因此 multi-height/future 非冗余不能评价；R0 sessions 已烧毁，不在
同一数据上调 known/coverage 门救援。

H1 R1 已在任何新 teacher outcome 计算前冻结。它不改 R0 的 known、UNKNOWN、
denominator 或数值门，只把不可由单目 observation 支持的 360° 输出合同改写为
camera-forward `[-45°, +45°]` 的 6-bin locomotion sector。R1 使用排除 R0 burned
sessions 后按 official train session ID 字典序选择的四个全新 source sessions；四者
source authority 已通过并绑定精确 report/manifest/spec/pose hashes。

R1 正式一次性结果仍为 `H1_GEOMETRY_TEACHER_NOT_EVALUABLE`，但定位发生了变化：
4/4 current known coverage 达到 `22.07%–36.77%`，全部越过 `.15`，说明 R1
evidence version 的 current support 可评价；由于 cohort 同时改变，不能把与 R0 的
差异单独归因于 sector。`00c2a1cd` 的 near/far 仅 `3.34%/0%`，在
future coverage 顺序门停止。burn 后诊断显示该 source 的 `0.4/0.8 s` camera
translation 中位数约 `3.60/7.14 m`，明显高于另三者的
`0.74–0.93/1.49–1.87 m`。这只形成 ego-motion/future-view support 的下一机制假设，
不构成因果确认；四个 R1 sessions 也已永久 burned。

R2 的 source-preparation 合同已在新 teacher outcome 前冻结：future field origin 只由
anchor 前 `400 ms` 的历史位姿速度在 local-ground plane 上外推，future pose 不选择
origin/方向；其余 R1 bins、UNKNOWN、denominator 与所有数值门保持不变。新 cohort
按排除 R0/R1 后的 official train ID 字典序冻结为
`03694304/03b6dc99/03c87279/03d70593`。该阶段只授权获取与 source authority，没有
提前运行 teacher。

随后四个 R2 source authority 全部通过，完整 report/manifest/spec/pose hashes 绑定
到正式 protocol；history selection、ground-tangent velocity 与 horizon-origin
advection runner 先提交推送，才执行以下一次性结果。

R2 已一次性执行并关闭为 `H1_MULTI_HEIGHT_PROXY_NOT_SUPPORTED_STOP`。4/4
source/mechanics validity 全过，worst current/near/far coverage 为
`20.42%/18.47%/11.91%`，因此 future-view support 不再是该 evidence version 的
blocker。`03c87279` 的 height disagreement 仅 `2/684=0.292%`，低于 2%；future
diagnostic 也只有 3/4 过门，不能越过 height 顺序门解释。R2 sources 已 burned。

重新对照最初 HFTF 构想后，确认 R0–R2 teacher 只做 angular-cell point counts，没有
人体横向宽度/安全余量膨胀、swept candidate trajectories 或 foot step/drop。因此 R2
只关闭当前 point-support proxy，不能据此删除 multi-height 并直接进入更容易的
single-height future。当前回到原始 Stage B：已冻结 Development-only swept-envelope
label-mechanics canary，在 burned R2 sources 上先把人体包络监督做对；通过后才允许
fresh-source formal R3。

D0 已在四个 burned R2 sessions 上完成并准入 fresh R3。7/7 structural canaries、
4/4 source binding 与 UNKNOWN→SAFE 防火墙全部通过；四个 sessions 均有三高度 known
与 height-specific outputs，相对旧 point-support 新增 209 个 swept-collision
cells。该结果仍只是 mechanics audit。真实来源没有检出 ground step/drop risk，
3,600 个 foot cells 中 2,905 个为 ground-UNKNOWN；因此下一 formal R3 必须以全新
sources 和独立冻结的高密度 geometry reference 评价 reference-relative 增益，不能把
“输出更多 collision”本身当成 Stage B 成功。

R3.1 随后按冻结规则用完 40-session SANPO-Synthetic train screening budget，终态为
`R3_1_REFERENCE_OPPORTUNITY_COHORT_NOT_EVALUABLE`：`0/4` source 同时满足 obstacle
与 ground opportunity。34 个完成 dense reference ground 计算的 source 合计
ground-risk cells 为 0；与此同时 29/34 通过全部 obstacle opportunity checks。这把
blocker 定位为当前 semantic-ground-only source representation 无法提供台阶/落差机会，
而不是 swept-envelope obstacle effect 的负结果。不得扩大同一队列、降低门槛或用
零 ground opportunity 冒充 agreement。

唯一 successor 是 split-source Stage B：在新的 SANPO sessions 上做
reference-only obstacle qualification 与原 R3 effect gates；foot-ground 则改用带
解析或 source-native surface elevation truth 的独立 metric terrain source。两个
source role 都通过前不授权 Stage C；通过也只形成 split-source teacher mechanics
Development support，不形成自然 prevalence、人类事件、student utility 或安全结论。

R4 已按该 split-source 合同执行并达到
`R4_STAGE_B_SPLIT_SOURCE_TEACHER_MECHANICS_SUPPORTED`。新 SANPO challenge cohort 的
obstacle candidate/baseline primary F1 为 `.98756/.76000`，delta `+.22756`；
4/4 session 和 foot/body/head 均为正，precision delta `+.37792`、recall delta
`-.00493`。解析 terrain 的 20 risk、16 safe、6 UNKNOWN 全部正确/弃权，candidate
F1 `1.0`，比最佳 endpoint baseline 高 `+.25`。前者只外推到 reference-qualified
challenge cohort，后者只属于 controlled mechanics。

Stage C C0 source-feasibility contract 现已在任何新媒体内容或 geometry outcome 前
冻结。SANPO 继续承担 causal obstacle/future teacher source role；EgoWalk 只承担
natural RGB/depth/pose transport 与 semantic-independent surface observability
canary。EgoWalk 239 条 pose metadata 中按完整性、5 Hz timeline、无回零重定位、
文件绑定与体积的结果无关规则，冻结两个不同录制日期的最小 cohort：
`2024_08_15__19_45_11 / 2024_07_11__12_33_57`。冻结时 RGB/depth media 尚未下载或
打开。

C0 即使成功，也只授权冻结 Stage C label-and-student canary protocol；不授权正式
label execution、student training/effect、切换研究主线或修改 App。

C0 现已按冻结门关闭为 `C0_EGOWALK_MEDIA_TRANSPORT_NOT_EVALUABLE`。两条 source 的
pose/RGB/depth 文件 hash、`647/664` 帧完整 decode、ordinal PTS 与全部 natural depth
support 门均通过，但 RGB/depth container nominal rate 均为 100 Hz，不满足原协议的
5 Hz reported-rate 门。dataset meta 与 parquet timestamp 仍一致指向 5 Hz。

该失败保留不覆盖；最小 C0.1 successor 已冻结，只在同一 consumed cohort 上把物理
timeline authority 改为 parquet frame/timestamp + `meta/info.json`，container nominal
rate 仅记录。其余 gate 与权限全部不变。

C0.1 现已通过：两条 source 的 parquet delta 均为 `198/200/201 ms`，有效 rate
`5.0 Hz`，pose/RGB/depth 为 `647/647/647` 与 `664/664/664`；原 surface
observability 继续全过。当前 claim ceiling 只是 consumed source schema repair 与
natural depth support。唯一新权限是冻结 Stage C label-and-student canary protocol；
在 semantic-independent depth reader 产生 known/UNKNOWN 与 natural opportunity
之前，不能训练 student。

Stage C D0 semantic-independent label-readiness 现已在 formal report 前冻结。两条
EgoWalk 都明确为 consumed calibration；近于 `1.2 m` 的未观测区域保持 UNKNOWN，
可评价断面固定为 `1.4–3.0 m`。reader 以 depth-only ground-plane RANSAC 和 horizontal
support modes 输出 ground-continuity proxy；semantic class、annotation 与 RGB outcome
均不参与 formal mechanics。D0 通过也只允许冻结 fresh-source label/student canary，
不直接训练。

D0 已达到 `D0_SEMANTIC_INDEPENDENT_LABEL_READINESS_SUPPORTED`：265/265 formal
frames 有 ground plane，outdoor/indoor direction known fraction 为 `.918/.782`；
7 个 risk proxies 分布于 7 frames/4 directions，UNKNOWN→SAFE 为 0，七个 structural
canaries 与第二遍 byte-determinism 全过。claim 仍只到 consumed geometry-label
readiness。下一步必须先冻结/验证 phone-causal future-label mechanics，再设计 fresh
train/dev/held-out student canary。

Stage C D1 causal future-label mechanics 已在 formal report 前冻结。history-only pose
决定 `.4/.8 s` advected origin，grid orientation 固定 current yaw；future pose 只可
重投影 future depth observation。consumed calibration 中 future observation 相对
current-only 新增 known cells 为 outdoor `186/280`、indoor `303/490`，known loss 0。

D1 正式双运行现已达到 `D1_CAUSAL_FUTURE_LABEL_MECHANICS_SUPPORTED`。两条 source
的 history-speed eligibility 均为 1.0；outdoor candidate known fraction 为
`.9266/.8766`，indoor 为 `.7954/.7588`，future 新增 known cells 与校准披露一致，
known loss 和 UNKNOWN→SAFE violation 均为 0。24 个 future risk-proxy cells 覆盖
5 个方向，全部 structural canaries 与第二遍 byte-determinism 通过。

这只支持 consumed source 上的 causal future teacher-label mechanics。唯一新权限是
冻结 fresh session-disjoint teacher corpus + student canary protocol；不得把 proxy
解释为 hazard/safe truth，也不授权 acquisition、corpus generation、student training/
effect、主线、Android/App 或安全 claim。

Stage C E0 现已在任何 fresh RGB/depth 或 geometry-label outcome 前冻结。从 95 条
healthy inventory 排除两条 consumed source 后，按总字节升序与 recording-date
互斥规则锁定 6 条 fresh trajectory，角色固定为 `4 train / 1 dev / 1 heldout`。
E0 只检验 5 方向、`[0,.4,.8] s` 的 foot-ground known/risk proxy；三个等参数
MobileNetV3-Small arms 分别是 single-frame future、history current-only 与 history
future。source/transport/teacher/opportunity 门全部通过前不训练，heldout 不足也不得
换样。body/head、完整距离场、事件效果与主线仍未授权。

E0 source-lock validator 已正式双运行并达到 `E0_FRESH_SOURCE_LOCK_VALIDATED`：
parent/inventory/metadata/预训练权重 hashes、六条选择、角色、日期与 18 个文件绑定
全部复算一致，payload byte-exact。报告 SHA-256 为
`9e3ce8793597907dbe87e6a9c57d9f3f9ffcfb1510f078ea31e01148eab046dc`。
当前只授权获取合同精确绑定的六条媒体；teacher corpus 与 student training 仍需后续
transport/teacher/opportunity 门。

六条 exact media 已获取并 burned，18 个 source files 合计 956,183,459 bytes，
acquisition manifest SHA-256 为
`8b19ff024ed6eb8d1ed0afdeeffad78025af9a3c623c6df9c598b5a8161ffdc3`。
transport 正式双运行达到 `E0_FRESH_MEDIA_TRANSPORT_SUPPORTED`：六条
pose/RGB/depth count 均逐 source 相等，全部 parquet physical rate 为 5.0 Hz，PTS
严格递增，payload byte-exact；report SHA-256 为
`a2a0c3e739d93c79afb613727a4946fb7967c087cfdeb49c9539ecb5e66c9ac7`。
当前只新增 teacher mechanics + role-opportunity audit 权限，尚不授权 corpus/training。

E0 teacher-opportunity 正式双运行已在训练前停止为
`E0_FRESH_TEACHER_MECHANICS_NOT_EVALUABLE`。三角色 risk/no-risk opportunity 全过：
train/dev/heldout 分别有 `27/8/36` 个 risk cells 与 `22/4/19` 个物理 risk anchors；
但 `.8 s` candidate known fraction 仅 2/6 source 达到冻结 `.70` 门，另外四条为
`.6015–.6857`。`.4 s` 则 6/6 通过。不得降低 E0 门或在 burned dev/heldout 上训练；
唯一 successor 是另冻 `.4 s`-only E0.1，并使用全新 dev/heldout。

E0.1 已在新评价媒体前冻结：原四条 E0 train 只作为 consumed training data，原
dev/heldout 不再使用；从排除全部八条 consumed 后的 healthy inventory 按相同
outcome-independent 规则锁定 `2024_12_01__15_29_33` 为 dev、
`2024_07_10__11_01_46` 为 heldout。三臂只输出 `[current,.4 s]`，模型、训练、
阈值与 `.03` superiority margin 已冻结。新 source transport/teacher/opportunity
全过前不生成 corpus 或训练。

E0.1 已在训练前停止为 `E0_1_FOOT_GROUND_STUDENT_CANARY_NOT_EVALUABLE`。
新 dev/heldout 的 `.4 s` known fraction 为 `.9329/.8312`，全部 mechanics 通过；
dev 有 4 risk cells/4 anchors，但 heldout 只有 1/1，低于冻结 2/2。不得降门或换一条
已知更有利的单 source。唯一 successor 是一次性固定 3 dev + 3 heldout、与全部
consumed dates 互斥的 E0.2；若仍无 role opportunity 就关闭该 source route。

E0.2 现已在任何新媒体前一次性冻结 3 dev + 3 heldout；六个 recording dates 与全部
十个 consumed dates 互斥，角色按 metadata 排序位置交替分配，总媒体
1,232,000,737 bytes。每角色必须覆盖至少 4 risk cells/4 anchors/2 sources/2
directions；学生合同完全复用 E0.1。固定 batch 任一门失败即关闭该 EgoWalk
foot-ground student source route，不再扩大。

E0.2 固定 batch 已关闭为 `E0_2_FIXED_BATCH_TEACHER_MECHANICS_NOT_EVALUABLE`。
role opportunity 大幅通过：dev/heldout 各 `35/37` risk cells、`32/32` physical
anchors、5 directions；但 3/6 source 的 `.4 s` known fraction 为
`.3257/.6515/.5000`，低于 `.70`，其中一条 plane known 也低于 `.95`。因此
`EgoWalk + D0/D1 foot-ground reader + RGB student` source route 关闭，不训练、不再
扩源。HFTF 本身保留，下一路线转向 R4 已有 reference 支持的 SANPO body/head
obstacle temporal student；foot-ground 不得混入该结论。

## 当前真源

- [R0 候选支线章程](HFTF_CANDIDATE_LANE_CHARTER_R0_2026-08-01.md)
- [R0 机器可读合同](HFTF_CANDIDATE_LANE_R0_2026-08-01.json)
- [H0 来源可行性结果](HFTF_H0_SOURCE_FEASIBILITY_RESULT_2026-08-01.md)
- [H0.1/H0.2 SANPO source-specific 结果](HFTF_H0_1_H0_2_SANPO_PROXY_AUTHORITY_RESULT_2026-08-01.md)
- [H1 geometry teacher canary R0 protocol](HFTF_H1_GEOMETRY_TEACHER_CANARY_PROTOCOL_R0_2026-08-01.md)
- [H1 R0 machine-readable protocol](HFTF_H1_GEOMETRY_TEACHER_CANARY_PROTOCOL_R0_2026-08-01.json)
- [H1 R0 result](HFTF_H1_GEOMETRY_TEACHER_CANARY_RESULT_R0_2026-08-01.md)
- [H1 R1 forward-sector protocol](HFTF_H1_FORWARD_SECTOR_GEOMETRY_TEACHER_CANARY_PROTOCOL_R1_2026-08-01.md)
- [H1 R1 machine-readable protocol](HFTF_H1_FORWARD_SECTOR_GEOMETRY_TEACHER_CANARY_PROTOCOL_R1_2026-08-01.json)
- [H1 R1 result](HFTF_H1_FORWARD_SECTOR_GEOMETRY_TEACHER_CANARY_RESULT_R1_2026-08-01.md)
- [H1 R2 causal-advected source preparation](HFTF_H1_CAUSAL_ADVECTED_ORIGIN_SOURCE_PREPARATION_R2_2026-08-01.md)
- [H1 R2 machine-readable source preparation](HFTF_H1_CAUSAL_ADVECTED_ORIGIN_SOURCE_PREPARATION_R2_2026-08-01.json)
- [H1 R2 causal-advected teacher protocol](HFTF_H1_CAUSAL_ADVECTED_ORIGIN_GEOMETRY_TEACHER_PROTOCOL_R2_2026-08-01.md)
- [H1 R2 machine-readable teacher protocol](HFTF_H1_CAUSAL_ADVECTED_ORIGIN_GEOMETRY_TEACHER_PROTOCOL_R2_2026-08-01.json)
- [H1 R2 result](HFTF_H1_CAUSAL_ADVECTED_ORIGIN_GEOMETRY_TEACHER_RESULT_R2_2026-08-01.md)
- [目标一致性与 swept-envelope 修复](HFTF_OBJECTIVE_ALIGNMENT_AND_SWEPT_ENVELOPE_REPAIR_2026-08-01.md)
- [Stage B swept-envelope mechanics protocol](HFTF_STAGE_B_SWEPT_ENVELOPE_LABEL_MECHANICS_CANARY_D0_2026-08-01.md)
- [Stage B machine-readable protocol](HFTF_STAGE_B_SWEPT_ENVELOPE_LABEL_MECHANICS_CANARY_D0_2026-08-01.json)
- [Stage B swept-envelope mechanics D0 result](HFTF_STAGE_B_SWEPT_ENVELOPE_LABEL_MECHANICS_RESULT_D0_2026-08-01.md)
- [Stage B reference metric pilot D1](HFTF_STAGE_B_REFERENCE_METRIC_PILOT_D1_2026-08-01.md)
- [Stage B machine-readable D1 pilot](HFTF_STAGE_B_REFERENCE_METRIC_PILOT_D1_2026-08-01.json)
- [Stage B D1 pilot result](HFTF_STAGE_B_REFERENCE_METRIC_PILOT_RESULT_D1_2026-08-01.md)
- [Stage B R3 source preparation](HFTF_STAGE_B_SWEPT_ENVELOPE_REFERENCE_COMPARISON_SOURCE_PREPARATION_R3_2026-08-01.md)
- [Stage B machine-readable R3 source preparation](HFTF_STAGE_B_SWEPT_ENVELOPE_REFERENCE_COMPARISON_SOURCE_PREPARATION_R3_2026-08-01.json)
- [Stage B R3 formal protocol](HFTF_STAGE_B_SWEPT_ENVELOPE_REFERENCE_COMPARISON_PROTOCOL_R3_2026-08-01.md)
- [Stage B machine-readable R3 formal protocol](HFTF_STAGE_B_SWEPT_ENVELOPE_REFERENCE_COMPARISON_PROTOCOL_R3_2026-08-01.json)
- [Stage B R3 formal result](HFTF_STAGE_B_SWEPT_ENVELOPE_REFERENCE_COMPARISON_RESULT_R3_2026-08-01.md)
- [Stage B R3.1 reference-only opportunity qualification](HFTF_STAGE_B_REFERENCE_ONLY_OPPORTUNITY_QUALIFICATION_R3_1_2026-08-01.md)
- [Stage B machine-readable R3.1 qualification](HFTF_STAGE_B_REFERENCE_ONLY_OPPORTUNITY_QUALIFICATION_R3_1_2026-08-01.json)
- [R3.1 source-pool burn ledger](HFTF_R3_1_SOURCE_POOL_BURN_LEDGER_2026-08-01.json)
- [R3.1 inventory candidate plan result](HFTF_R3_1_INVENTORY_CANDIDATE_PLAN_RESULT_2026-08-01.md)
- [Stage B R3.1 qualification result](HFTF_STAGE_B_REFERENCE_ONLY_OPPORTUNITY_QUALIFICATION_RESULT_R3_1_2026-08-01.md)
- [Stage B split-source validation R4](HFTF_STAGE_B_SPLIT_SOURCE_VALIDATION_R4_2026-08-01.md)
- [Stage B machine-readable R4 protocol](HFTF_STAGE_B_SPLIT_SOURCE_VALIDATION_R4_2026-08-01.json)
- [R4 source-pool burn ledger](HFTF_R4_SOURCE_POOL_BURN_LEDGER_2026-08-01.json)
- [Stage B split-source validation R4 result](HFTF_STAGE_B_SPLIT_SOURCE_VALIDATION_RESULT_R4_2026-08-01.md)
- [Stage C source-feasibility C0](HFTF_STAGE_C_SOURCE_FEASIBILITY_C0_2026-08-01.md)
- [Stage C machine-readable C0 protocol](HFTF_STAGE_C_SOURCE_FEASIBILITY_C0_2026-08-01.json)
- [Stage C C0 result](HFTF_STAGE_C_SOURCE_FEASIBILITY_RESULT_C0_2026-08-01.md)
- [Stage C source-feasibility C0.1](HFTF_STAGE_C_SOURCE_FEASIBILITY_C0_1_2026-08-01.md)
- [Stage C machine-readable C0.1 protocol](HFTF_STAGE_C_SOURCE_FEASIBILITY_C0_1_2026-08-01.json)
- [Stage C C0.1 result](HFTF_STAGE_C_SOURCE_FEASIBILITY_RESULT_C0_1_2026-08-01.md)
- [Stage C semantic-independent label readiness D0](HFTF_STAGE_C_SEMANTIC_INDEPENDENT_LABEL_READINESS_D0_2026-08-01.md)
- [Stage C machine-readable label readiness D0](HFTF_STAGE_C_SEMANTIC_INDEPENDENT_LABEL_READINESS_D0_2026-08-01.json)
- [Stage C label readiness D0 result](HFTF_STAGE_C_SEMANTIC_INDEPENDENT_LABEL_READINESS_RESULT_D0_2026-08-01.md)
- [Stage C causal future-label mechanics D1](HFTF_STAGE_C_CAUSAL_FUTURE_LABEL_MECHANICS_D1_2026-08-01.md)
- [Stage C machine-readable future-label D1](HFTF_STAGE_C_CAUSAL_FUTURE_LABEL_MECHANICS_D1_2026-08-01.json)
- [Stage C causal future-label mechanics D1 result](HFTF_STAGE_C_CAUSAL_FUTURE_LABEL_MECHANICS_RESULT_D1_2026-08-01.md)
- [Stage C fresh foot-ground student canary E0](HFTF_STAGE_C_FRESH_FOOT_GROUND_STUDENT_CANARY_E0_2026-08-01.md)
- [Stage C machine-readable fresh student canary E0](HFTF_STAGE_C_FRESH_FOOT_GROUND_STUDENT_CANARY_E0_2026-08-01.json)
- [Stage C fresh foot-ground student canary E0 result](HFTF_STAGE_C_FRESH_FOOT_GROUND_STUDENT_CANARY_RESULT_E0_2026-08-01.md)
- [Stage C foot-ground student canary E0.1](HFTF_STAGE_C_FOOT_GROUND_STUDENT_CANARY_E0_1_2026-08-01.md)
- [Stage C machine-readable foot-ground student canary E0.1](HFTF_STAGE_C_FOOT_GROUND_STUDENT_CANARY_E0_1_2026-08-01.json)
- [Stage C foot-ground student canary E0.1 result](HFTF_STAGE_C_FOOT_GROUND_STUDENT_CANARY_RESULT_E0_1_2026-08-01.md)
- [Stage C multi-source evaluation qualification E0.2](HFTF_STAGE_C_MULTI_SOURCE_EVALUATION_QUALIFICATION_E0_2_2026-08-01.md)
- [Stage C machine-readable multi-source qualification E0.2](HFTF_STAGE_C_MULTI_SOURCE_EVALUATION_QUALIFICATION_E0_2_2026-08-01.json)
- [Stage C multi-source qualification E0.2 result](HFTF_STAGE_C_MULTI_SOURCE_EVALUATION_QUALIFICATION_RESULT_E0_2_2026-08-01.md)
- [可执行审计 Module](../../../scripts/research/hftf/README.md)

## 与历史 USTRF-SC 的边界

HFTF 与已关闭的 USTRF-SC 都涉及 dense risk、`g=(theta,rho,z)`、人体尺度、
foot/body/head、metric depth、时序、dynamic 和 uncertainty；这些全部视为**继承的历史
primitive**，不能作为仓库内新增因果变量。允许重开的新增信号假设只有：

1. action-agnostic 地一次输出全部候选方向，不从 RGB 猜测用户意图路线；
2. history-only RGB student 显式预测短期 future layered cells，而不是只消费
   current geometry 或既有 route/lifecycle；
3. action policy 与 representation evaluation 隔离，先用 current-only、single-frame
   和历史 USTRF primitive 做直接基线，证明 future representation 的独立增量。

历史 USTRF 的 15 对窗口、关闭终态和限制保持不可变，不得作为 HFTF 的 fresh selection
或重新包装为新证据。

## 创新性上限

当前创新性终态是 `NOT_EVALUABLE`。尤其 [AgniNav](https://arxiv.org/abs/2606.10903)
已经使用身体碰撞包络、RGB-D 高度条件标签、单目 RGB student、64 个极坐标 bin 与
边缘端部署；[AI Guide Dog](https://arxiv.org/abs/2501.07957) 已预测一秒后的
`LEFT/RIGHT/FRONT`；Google [Running Guide Agent](https://blog.google/innovation-and-ai/models-and-research/google-deepmind/running-guide-agent/)
也公开了端侧快速路径与低频语义推理的双路径。因此 HFTF 不声称“身体包络、几何教师、
极坐标 student、未来方向或双路径架构”本身首次出现。

当前只保留一个待检验的组合新意：助盲行人场景下
`foot/body/head × current/short-future × phone-causal × selective-abstention`
的统一输出合同。系统检索、直接基线和消融完成前，不使用“首次”“世界模型”或
“已确认革新”等表述。

最低直接相关工作还包括 [EgoNav](https://arxiv.org/abs/2403.19026)、
[Egocentric Future Localization](https://openaccess.thecvf.com/content_cvpr_2016/html/Park_Egocentric_Future_Localization_CVPR_2016_paper.html)、
[Navigation World Models](https://arxiv.org/abs/2412.03572) 与
[NavWM](https://arxiv.org/abs/2606.24101)。即使未来 H3 utility 胜出，创新性仍保持
`NOT_EVALUABLE`，直到这些直接比较与系统检索独立完成。

## 晋级原则

“超过主线”必须发生在同一 parent-event ledger、canonical decision kernel、输出语义、
设备预算与预先冻结 margin 下。各 arm 的 source adapter 可以不同，但必须在 outcome
前冻结并计入候选系统。HFTF 只有在 source-held-out 评价中改善至少一个 co-primary，
且不实质损害其余 co-primary、假提醒、错误方向、弃权覆盖、最差 source 和设备成本，
才能从候选支线晋级为研究主线。teacher agreement 单独不能触发晋级。

研究主线晋级不等于正式 App 替换；后者仍需独立 Confirmation、设备、发布和安全边界。
