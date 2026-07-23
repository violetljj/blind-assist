# USTRF route-target evidence closure R1 实验总结与 V2 交接（2026-07-23）

状态：`R1_CLOSED_DATA_BLOCKED / SOURCE_SEARCH_STOPPED / CANDIDATES_UNRUN / V2_GOVERNANCE_VALID / CURRENT_LEVEL_L0`

## 阶段结论

R1 没有选出可进入 Android shadow 的路线目标事件算法，也没有证明系统已经能够可靠理解行人与用户路线的动态关系。本轮真正完成的是故障边界收缩：detector 已不再是当前首因，单纯修改 association 也不能改变 event recall、误提醒、repeat 和 clearance；oracle lifecycle 能在保持 `14/15` recall 的同时把当前口径的 false alert 与 repeat 降为 0，说明事件生命周期是主要可修复缺口，但仍有一个独立的上游 evidence 缺失事件，不能靠 cooldown、clearance 或 tracker 修复。

三个结构候选 C1–C3 已在候选输出不可见时冻结，分别面向连续路线关系、同一事件只提醒一次和 terminal clear。由于两轮 holdout、两条 0327 canary、NavWareSet Stage A、REveL 及外部候选清单均未形成两来源选择权，C1–C3 从未运行。最终决定保持 `DATA_BLOCKED_STOP_SOURCE_SEARCH`，这表示 R1 的完整 holdout 准入失败，不表示已有数据和局部指标证据全部无效。

## 实验问题与固定边界

本轮回答三个相互关联的问题：画面中的真实行人是否正在接近、进入、占用或穿越用户路线；同一风险事件何时只交付一次提醒；行人何时已在持续新鲜观测下离开路线并允许事件结束。逐人路线角色固定为 `approaching_route`、`route_intersecting`、`adjacent_safe`、`receding` 和 `cleared`。seen 数据只用于故障归因；候选选择必须依赖新鲜、两来源、包含真实共现行人的 sealed holdout。

实验期间没有回调 detector `.35`、NMS、tracker 或其他标量门，也没有打开深度、TTC 和 route-risk flip。missing、occluded、stale 或 unknown 不得被解释为 clear。候选路线只允许使用当前及过去输入，未来轨迹只能服务离线注释。任何来源或指标未满足真值合同，都不能由 pooled 结果、代理轨迹或扩大下载量补齐。

## 实验经过

| 阶段 | 执行内容 | 主要证据 | 阶段判断 |
| --- | --- | --- | --- |
| Seen truth 与 oracle 归因 | 对 15+15 窗口建立逐人身份提议、五态路线角色代理和 person-bound lifecycle，运行 T0 与三条 oracle 臂 | O1 因 34 个共现身份片段无法闭合而 `not_evaluable`；O2 的路线关系代理增加抖动、repeat 和误提醒；O3 保持 `14/15` recall，同时把 false alert 与 repeat 降为 0 | lifecycle 是主要可修复缺口；另有一个独立 evidence miss；路线关系代理不能当作 oracle 真值 |
| 候选冻结 | 预注册 C1 连续路线关系、C2 episode-level 单次提醒和 C3 lineage+episode clear；冻结实现与入口 | detector、association、阈值、H2 均未改变 | 只获得待评实现，没有候选效果结论 |
| 首组 CrowdBot holdout | 物化两来源 16 条序列、22,856 RGB，完成双视觉人物 pass、因果路线、投影角色与窗口冻结 | 只有 9,210/22,856 帧达到旧的 all-person role complete；6,340 个事件 proposal 全部隔离；负暴露仅 `0.336/1.126min` | 来源准入 `0/2`；未运行候选 |
| CrowdBot replacement | 物化 `0410 mds + 1203 shared-control` 共 23 条序列、34,779 RGB，并修正视觉事件生成与 false-alert 分母合同 | 最终仅接受 2 个事件；逐来源负暴露 `2.948/2.585min`，均没有同序列等长 matched negative | 来源准入仍为 `0/2`；证明 LiDAR/pose 容量不能预测相机身份连续性与 terminal clear |
| 0327 reject-only canary | 只下载冻结的 event/negative 两条 canary，共 4,422 RGB；完成因果路线与两条视觉人物 pass | `0 positive / 0 critical / 0 matched negative / 0.0764min negative exposure` | 拒绝 0327，停止剩余 11 条下载 |
| NavWareSet Stage A | 只取得 181,596,612-byte 微型样本，先修正 weak ETag 不能冒充 content SHA 的完整性问题，再解码 robot/GRS 时间轴 | 两个原始时间区间相隔 `14.079947295s`，没有注册 temporal offset | Stage A 拒绝；约 5.97GB Stage B 与其他 scenes 均未下载 |
| REveL 最终审计 | 复核本地 `dynamic` 的 8,580 RGB、helmet identity、sensor/person Vicon 与 calibration，并审计匿名 session 2 清单 | `dynamic` 已被多轮开发实验查看；排除后全部未查看 footage 总时长上界约 `7.903min`；小包不含完整 route/identity/clear 链 | 仅保留 development、identity alignment 与回归权限，不得成为 sealed holdout |
| 外部清单终审 | 有界审计 JRDB、Oxford-IHM、KTP/IAS-Lab、FLOBOT、THÖR-MAGNI 与 SCAND，不增加第三条搜索路线 | 没有来源同时提供匿名可哈希微型样本、相机稳定全人身份、原生因果路线、同人连续到 clear 和可评分共现负暴露 | 可用来源 `0/2`，最终审计新增下载 `0 bytes`，停止来源搜索 |

## 结果怎样帮助算法工作

现有证据支持调整研发重心。事件提醒应绑定连续的 hazard episode，而不是单帧目标或可重建的 track ID；clear 必须依赖同一人物在新鲜观测下持续离开路线，不能由检测消失推断。上游 route-target evidence 与下游 lifecycle 需要分开评分，因为 O3 能消除误提醒和重复交付，却不能创造漏失的第 15 个事件。后续实现仍应围绕 C1–C3 的结构差异展开，而不是继续替换 detector、提高 tracker 复杂度或全局降低风险阈值。

此前下载的数据仍有用途。CrowdBot 两轮物化揭示了 metadata/LiDAR 容量与 camera-visible identity continuity 之间的差距；0327 证明大规模来源在微型相机 canary 上可以被低成本淘汰；NavWareSet 暴露跨模态时间合同必须在大包下载前验证；REveL 可继续服务身份、Vicon 对齐和输入回归。这些数据只能承担各自已证明的 development、diagnostic 或 source-rejection 角色，不能因“已经下载”而升级为 selection、confirmation 或 shadow lockbox。

## R1 的停止状态

R1 的来源搜索已经结束。最终收据为 `artifacts.local/evidence/ustrf-route-target-evidence-closure-r1/source-search-final-bounded-decision-r1.json`，SHA-256 为 `a4034d6ca4cfb870efb35a2b886a29f1f370447e2a8a60d463196b1041069ec8`。收据记录可用来源 `0/2`、新下载 `0 bytes`、来源搜索和微型 canary 未授权、sealed holdout 未授权、候选输出未执行，Android shadow 与 H2 关闭。

原始 R1 结果文档 `docs/research/ustrf-sc/USTRF_ROUTE_TARGET_EVIDENCE_CLOSURE_R1_RESULT_2026-07-23.md` 已被 V2 配置以 SHA-256 `c2d306d22b618236fb7cba1b6998f907a019be5886b80a458c33914afc1ef8b3` 绑定。本总结是追加的解释与交接，不修改 R1 原始结论，也不把 R1 的失败追溯改写为新标准下的成功。

## 接纳 Evidence Maturity V2

新的证据成熟度标准位于 `docs/research/ustrf-sc/USTRF_ROUTE_TARGET_EVIDENCE_MATURITY_STANDARD_V2.md`，机器合同为 `configs/ustrf_route_target_evidence_maturity_v2.json`。当前验证结果是 `VALID_EVIDENCE_MATURITY_STANDARD_V2`，最高授权仍为 `L0_ENGINEERING_DIAGNOSTIC`；R2 metric eligibility mask 尚未冻结，R2 候选没有运行，也没有候选选择或 Android 权限。

V2 接纳的核心变化是按指标分别建立有效分母。event recall 不再被缺失 terminal clear 连带抹除，clearance 仍必须有真实 terminal clear 和 post-clear follow-up；false alerts/min 使用完整序列中 route known 且所有 route-relevant person 已解决的负暴露；右删失、身份丢失和路线任务变化分别记录，不能写成成功、失败或 `0ms clearance`。低样本只允许输出 `evaluable_underpowered / estimate_only`，空分母必须是 `not_evaluable + null`，不能把观察到零失败等同于置信边界充分。

现有 R1 数据按 V2 重新定位：

| 数据 | 当前角色 | 最高可申请层级 | 禁止用途 |
| --- | --- | --- | --- |
| LILocBench 15+15 seen | seen / diagnostic | 重新冻结逐指标 eligibility 后最高 L1 | R2 selection、confirmation、shadow lockbox |
| CrowdBot 首组与 replacement | development / partial metric evidence | 重新审计事件删失与分母后最高 L1 | 用已查看数据选择胜者或确认泛化 |
| 0327、NavWareSet、REveL、外部清单 | source-rejection / prescreen regression | L0 | 候选效果结论、选择或 Android 晋级 |

## 下一独立边界

下一项工作固定为 `R2-L1 metric eligibility materialization`，不是恢复无界来源搜索。开始任何 exploratory candidate profile 前，需要完成以下接纳步骤：

1. 在候选和 App 输出不可见时，为每个数据来源和每项指标冻结 eligibility mask、删失状态、排除原因计数与分母收据。
2. 每项指标输出 `support_status`、`result_status`、numerator、denominator、value、CI、`bound_sufficient`、`gate_result` 和 `ineligible_reason_counts`。
3. 逐来源验证人物身份连续性、因果路线有效性、terminal clear 与 post-clear follow-up；missing、unknown 和 identity loss 保持 fail closed。
4. 将 truth、mask、分母、协议和工具版本绑定哈希，再决定是否允许 C1–C3 各执行一次 L1 exploratory profile。
5. L1 只允许点估计、置信区间、局部故障归因和安全 veto 淘汰，不选择胜者，也不打开 Android。
6. 若以后需要 L2 选择权，必须使用新的 fresh-selection 数据，并满足 V2 的逐指标分母、两独立 session family、逐来源门和 worst-source 规则。

V2 合同验证入口：

```powershell
python scripts/run_research_tool.py ustrf-route-target-evidence-closure validate_evidence_maturity_v2.py `
  --config configs/ustrf_route_target_evidence_maturity_v2.json `
  --repo .
```

接纳标准以后，R1 的 `DATA_BLOCKED_STOP_SOURCE_SEARCH` 仍作为历史结论保留；V2 只允许回收已经真实闭合的局部指标证据，不授予候选胜者、Android、人体效果或生产安全结论。
