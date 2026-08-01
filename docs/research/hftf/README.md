# HFTF 候选未来可通行场支线

当前状态：

`CANDIDATE_SIDE_LANE_ACTIVE / DEVELOPMENT_STANDARD /
HFTF_H0_SOURCE_FEASIBILITY_PARTIAL / INNOVATION_NOT_EVALUABLE /
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

第一轮来源审计已产生有效终态
`HFTF_H0_SOURCE_FEASIBILITY_PARTIAL`：现有 25-frame、单 session
SANPO-Synthetic replay 可支持静态 metric geometry projection canary，但没有显式
HFTF pose-frame/time binding sidecar、结构化 camera-to-body/ground contract 或独立
parent-event ledger，因此
multi-height teacher、future teacher 与 student-effect evaluation 均为
`NOT_EVALUABLE`。

这里的 `SANPO-Synthetic` 是 replay manifest 的声明身份，不是 H0 已完成的密码学来源
认证。即使补齐 hash-bound pose/body sidecar，通用 H0 也只检查结构，不会给 sidecar
自签 multi-height/future 权威；进入 H1 前仍需 source-specific mapping/calibration
verifier。

## 当前真源

- [R0 候选支线章程](HFTF_CANDIDATE_LANE_CHARTER_R0_2026-08-01.md)
- [R0 机器可读合同](HFTF_CANDIDATE_LANE_R0_2026-08-01.json)
- [H0 来源可行性结果](HFTF_H0_SOURCE_FEASIBILITY_RESULT_2026-08-01.md)
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
