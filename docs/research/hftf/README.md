# HFTF 候选未来可通行场支线

当前状态：

`CANDIDATE_SIDE_LANE_ACTIVE / DEVELOPMENT_STANDARD /
H1_MULTI_HEIGHT_PROXY_NOT_SUPPORTED_STOP /
R2_POINT_SUPPORT_PROXY_BURNED /
STAGE_B_D1_REFERENCE_METRICS_READY / R3_SOURCE_PREPARATION_FROZEN /
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
