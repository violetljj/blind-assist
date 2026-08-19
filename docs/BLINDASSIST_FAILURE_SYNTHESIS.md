# BlindAssist Failure Synthesis / Global Reckoning

状态：`BLINDASSIST_FAILURE_STRUCTURE_IDENTIFIED / PRIMARY_CAUSE_PAIR=H3_TARGET_SUPERVISION+H4_POLICY_OBJECTIVE / H3_VS_H4_CAUSAL_SPLIT_UNRESOLVED`

证据截止：2026-08-17。范围是已有 Git 历史、current、冻结 result/protocol 与只读 artifact
绑定；没有重跑旧 benchmark，没有修改算法或冻结终态。机器摘要见
[`latest.json`](research/failure-synthesis/latest.json)，因果层和
Oracle 证据分别见[因果失败模型](BLINDASSIST_CAUSAL_FAILURE_MODEL.md)与
[Oracle Ladder](BLINDASSIST_ORACLE_LADDER.md)。

## 结论先行

BlindAssist 持续达不到核心任务门槛，不是因为“还没找到足够好的 encoder”。项目长期把大部分实验
预算投向 representation、depth、geometry、ranking 和局部时序，而产品 gate 需要的是稳定的
`现在是否应提醒 / 是否会挡路 / 何时清除 / 不确定时怎么办`。现有 source mask、depth、pose、
teacher proxy 和几何 truth 并不等于该 actionability truth；当前 reducer/policy/evaluator 又把多个
危险类型压入 clearance/三态/scalar-risk 合同。因此代理改善经常不能传到事件决策。

当前最可信的竞争性解释是：

1. **H3 TARGET / SUPERVISION FAILURE**：首要瓶颈。完美四类 mask 进入当前 soft adapter 后仍有
   `12/14` false-alert events、只清除 `4/16` positives；中心障碍 label readiness 也未达到可驱动告警
   的门。现有监督回答“看见什么/几何是什么”多于“现在该做什么”。
2. **H4 DOWNSTREAM OBJECTIVE / POLICY FAILURE**：次要且与 H3 紧密耦合。oracle box 能恢复
   `2/2` 正事件却产生 53 个误提醒帧并 `0/2` 清除；深度与 task head 多次降低 false-clear，却把
   false-block 或 UNKNOWN 推高。当前转换合同不能可靠把较好 representation 变成可用决定。
3. **H2 REPRESENTATION CEILING**：有实证，但不是已证明的总根因。DepthART、Assistive Geometry、
   obstacle logit、pose analytic 和 Q-Plane 都暴露跨 parent 的 scale/plane/obstacle-coverage 不稳；
   但 source-depth oracle 在 Q-Plane cohort 为零误差，且 oracle mask 曾在极小 cohort 通过，说明
   至少部分数据上 upstream information 存在。
4. **H1 INPUT / OBSERVABILITY CEILING**：尚未排除，但当前排名最低。单帧 RGB 的 metric scale、
   遮挡后几何、运动与事件结束确有不可观测风险；然而项目还没有在同一 fresh parent cohort 上完成
   `single RGB vs causal clip vs clip+pose/depth` 的人类/actionability upper-bound 对照。当前纯 RGB
   SVRF O0 也尚未运行，不能把 observability ceiling 写成已建立事实。

正式判断：`SEARCH_CONCENTRATION / WRONG_LEVEL_OPTIMIZATION` 已成立；
`PROXY_TARGET_ALIGNMENT_NOT_ESTABLISHED` 已成立。下一笔预算不应再用于新网络，而应先做
“完美 actionability / 完美 geometry / 当前 policy”三段替换实验，分开 H2、H3、H4。

## 1. 研发时间线与累积的失败约束

表中“falsified”只在证据实际覆盖的范围内成立；`NOT_EVALUABLE` 不冒充算法反证。

| Stage | Hypothesis | Intervention / expected mechanism | Metrics | Result / verdict | What this actually falsified | What remained unresolved | Evidence |
|---|---|---|---|---|---|---|---|
| 早期 YOLO11n + bbox rules | 类别、框位置/面积和阈值足以生成有用风险 | 检测、稳定器、相对几何与告警规则 | AP、risk miss/FP、回放与延迟 | 成为 incumbent，但没有产品/安全证明 | 规则不能补出 detector 未观察到的路线占用、事件身份和清除信息 | 新输入表示或 actionability target 的上限 | [2026-08-01 retrospective](research/ALGORITHM_RESEARCH_RETROSPECTIVE_2026-08-01.md) |
| Detector swap | 更快/更新 detector 会带来无回归风险收益 | YOLO26n 对 YOLO11n | AP/recall、critical miss、false alert、P95 | 更快但风险级门未过，保留 YOLO11n | 通用 AP/速度或单项 FN 改善不能授权替换 | 是否需非 bbox 信息 | [detector benchmark](DETECTOR_BENCHMARK.md) |
| MiDaS / early depth fusion | 加单目深度即可改善物理风险 | 端侧 depth + 保守融合 | miss、FPR、distance、P95 | critical miss `9→7`，但 FPR `.037→.185`，total P95 `56→292 ms`，拒绝 | “有深度信号”不等于净事件/设备收益 | 深度表示、尺度锚和 policy 各自贡献 | [detector/depth benchmark](DETECTOR_BENCHMARK.md) |
| Information ceiling D0 | post-YOLO rules 是主要缺口 | 当前 YOLO / oracle box / oracle mask 三臂 | event hit、false alert、clear | YOLO `0/2`；box `2/2` 但 53 FP frames、`0/2` clear；mask `2/2`、0 FP、`2/2` clear | 同一失败上继续堆 post-YOLO rules；框存在本身不提供行动性/清除 | mask 几何、source policy、taxonomy 中谁产生收益；cohort 仅 3 events | [three-arm D0](research/dual-loop/INFORMATION_CEILING_THREE_ARM_D0_RESULT_2026-08-01.md) |
| Reference/actionability construct | 更丰富像素/几何 reference 能直接成为动作 truth | swept envelope、dense reference、event adapter | proxy F1、event hit/false/clear | burned D1 proxy F1 `+.1587–+.1720`；fresh R3 因一 session 无 positive 而 NOT_EVALUABLE；R3.1 `0/4` qualified，34 ground reports 有 0 risk cells | proxy agreement 不能越过 opportunity/ground truth 缺失 | source-native actionability 和真实 prevalence | [D1](research/hftf/HFTF_STAGE_B_REFERENCE_METRIC_PILOT_RESULT_D1_2026-08-01.md), [R3](research/hftf/HFTF_STAGE_B_SWEPT_ENVELOPE_REFERENCE_COMPARISON_RESULT_R3_2026-08-01.md), [R3.1](research/hftf/HFTF_STAGE_B_REFERENCE_ONLY_OPPORTUNITY_QUALIFICATION_RESULT_R3_1_2026-08-01.md) |
| Segmentation / labelability / RISKSEG | dense mask 与更高像素质量会带来事件收益 | PIDNet-S、failure atlas、gating、truth-mask soft adapter | mIoU/boundary、event recall、false events、clear、P95 | RISKSEG `0/3` seeds 过事件门；learned false events `13/14,13/14,14/14` vs YOLO `6/14`，设备 P95 `77.374 ms` PASS；perfect four-class IDs 仍 `12/14` false、`4/16` clear | 性能、mIoU、argmax adapter 或更大同标签模型不足以解决 actionability | 新 target 是否可稳定标注；policy 是否仍错 | [RISKSEG R0](research/dual-loop/RISKSEG_R0_FINAL_RESULT_2026-08-01.md), [R1 P0](research/dual-loop/RISKSEG_R1_P0_SOFT_DENSE_ADAPTER_AUDIT_RESULT_2026-08-01.md), [failure atlas](research/dual-loop/DUAL_LOOP_SEGMENTATION_FAILURE_ATLAS_AND_RESIDUAL_LABELABILITY_R0_RESULT_2026-08-01.md) |
| RCLE / motion-temporal diagnostics | egomotion compensation 或周期结构可稳定解释 false risk | controlled translation/depth oracle、自然 session time diagnostics | rotation boundary、coverage、frequency alignment | Stage B rotation `0/8`、18 coverage failures；自然 session flow-at-pose-frequency `R²=.020–.035`，合法 HOLD/NOT_EVALUABLE | 当前 motion proxy/fit 不能被当作稳定因果解释 | 更好的 truth、同域 motion measurement 与真实 event effect | [RCLE Stage B](research/rcle/RCLE_PERIODIC_SELF_MOTION_COUNTERFACTUAL_R2_QMS_R1_STAGE_B_TRANSLATION_DEPTH_ORACLE_OBJECT_APPROACH_RESULT_R1_2026-07-29.md), [temporal diagnostic](research/rcle/RCLE_TEMPORAL_STRUCTURE_DIAGNOSTIC_R1_RESULT_2026-07-28.md) |
| HFTF future labels / temporal student | causal history RGB 可学习 future geometry proxy | future-label teacher mechanics、F0.1、P3 temporal head | support、risk F1、clearance delta、transition | future teacher mechanics PASS；F0.1 risk F1 median `.173267 < .6`；P3 多项改善但 clearance-delta 仅 `1.5646% <5%` 且 1/3 parent 回归，MIXED | future teacher mechanics 不等于 student/event effect；当前 direct RGB→proxy 时序学生不足 | temporal upper bound、target quality、clip observability | [future mechanics](research/hftf/HFTF_STAGE_C_CAUSAL_FUTURE_LABEL_MECHANICS_RESULT_D1_2026-08-01.md), [F0.1 heldout](research/hftf/HFTF_STAGE_C_SANPO_HELDOUT_EFFECT_RESULT_F0_1_2026-08-01.md), [P3](research/hftf/P3_TEMPORAL_DEVELOPMENT_SCREEN_R0_RESULT_2026-08-06.md) |
| DA2 comparison | 蒸馏/轻量/混合精度能保留 canonical task envelope | P1 truth-referenced gate、A1–A5S | depth、clearance、false-clear/block、transition | 无 P2 arm 全过；A2 raw AbsRel `10.53%`、false-clear `2.86%`，但 temporal clearance `.221 m` 和 state 失败 | 速度、depth accuracy 或低 false-clear 不能单项晋级 | canonical 本身与产品 actionability 的对齐 | [DA2 closure](research/hftf/DAV2_P1_P2_EXECUTION_CLOSURE_2026-08-05.md) |
| A3 collapse audit | 极低 false-clear 意味更安全 | temporal mobile student | task gates | false-clear `.095%`，但 clearance MAE `1.118 m`、collision agreement `38.38%`、几乎全 occupied | false-clear 可由封路塌缩购买；必须联合 false-block/coverage | 合理 utility trade-off 与用户代价 | [A3 result](research/hftf/DAV2_TEMPORAL_MOBILE_STUDENT_A3_R0_RESULT_2026-08-05.md) |
| DepthART R0 / DA2 | 更强 metric depth 能通过 task-quality admission | frozen DepthART vs DA2 | clearance、false-clear/block、temporal | clearance `.1582/.3804 m`、false-clear `6.76/24.25%` 改善；false-block `3.10/0.48%` veto | generic/clearance improvement 不保证 task Pareto | metric adapter、boundary geometry与 policy 的责任分解 | [DepthART R0](research/hftf/DEPTHART_ADMISSION_R0_RESULT_2026-08-07.md) |
| G4-D / HTP deployment | 可转换/可执行即能保留模型数值 | PyTorch→ONNX→SM8650 HTP strict parity | max/mean absolute error | direct/context bit-exact，但 HTP vs PyTorch `max_abs=1.435607`、`mean_abs=1.070408`，strict FAIL | QAIRT 2.47 SM8650 standard-float path 的当前 strict route；部署 mechanics 不能替代 task quality | 不同 runtime/hardware；与产品根因基本正交 | [G4-D result](research/hftf/DEPTHART_ADMISSION_R1_A3_RESULT_2026-08-07.md) |
| Assistive Geometry B1/A0 | depth-only task geometry 可保守又可通行 | 3 seeds、1,200-frame selection、deterministic reducer | MAE、false-clear/block、transition | false-clear `.0241` PASS；MAE `.3152 m`、false-block `.7501`、transition `.7728` 全 `0/3` PASS，永久关闭 | 增 seed/epoch/loss 或最好 checkpoint；问题在 threshold 前的系统性 conservative geometry | depth scale vs support/ground 的因果分解 | [B1/A0 evaluation](research/assistive-geometry/BLINDASSIST_ASSISTIVE_GEOMETRY_B1_A0_DEVELOPMENT_EVALUATION_RESULT_2026-08-09.md), [anatomy](research/assistive-geometry/BLINDASSIST_ASSISTIVE_GEOMETRY_B1_A0_FAILURE_ANATOMY_RESULT_2026-08-09.md) |
| R2 / SuperTeacher / label frontdoor | source-native continuous factors + reducer seam 能解决 unified head | factorized labels、SuperTeacher、session anchor、deterministic reducer | coverage、seam gates | label frontdoor与 seam PASS；12-frame seam 输出 18 CLEAR/90 UNKNOWN；明确不证明 learnability或 utility | 工程 seam 和 label availability 不是跨传感器任务成功 | factor reliability、target semantics与跨 parent generalization | [F1 frontdoor](research/assistive-geometry/BLINDASSIST_ASSISTIVE_GEOMETRY_R2_F1_SOURCE_NATIVE_LABEL_MATERIALIZATION_AND_FRONTDOOR_RESULT_2026-08-11.json), [SuperTeacher landing](research/assistive-geometry/BLINDASSIST_ASSISTIVE_GEOMETRY_R2_SUPERTEACHER_TO_AG_LANDING_RESULT_2026-08-12.json) |
| DepthART D1/D2/D3 | learned task head/risk ranking 能同时修 false-clear 和 false-block | fixed mixed, task-evidence head, direct veto, bounded UNKNOWN | MAE、false-clear/block、coverage | D2 head MAE `.436153→.279311`、false-clear `.207949→.085991`，但 false-block `.262735→.376138`；D3R4/R5 降 false-clear却把 fresh false-block推至 `46.78%/27.92%`；D3R6 2% deferral PASS但后验不优于随机期望 | direct state/veto 可用一类错误换另一类；D3R6 gate PASS 不证明 ranking 增量 | 是否存在真正 task-aligned score；policy trade-off | [D2](research/hftf/DEPTHART_TASK_PRESERVING_D2_DEVELOPMENT_QUALITY_RESULT_2026-08-12.md), [DepthART current](research/hftf/README.md) |
| Assistive Geometry factor-wise landing | 组合 depth/support/boundary/obstacle 可安全落到 task | oracle、LOPO calibration、interaction、pose analytic | parent coverage、double thresholds、task cells | oracle headroom存在；learned correction coverage低；support会 fail-open；obstacle undercoverage；interaction与 pose analytic均 `0/6` safe threshold pairs | 同一 RGB factor observable 上继续堆浅层 selector；support 创建 validity | source-native obstacle supervision或不同 representation | [AG current](research/assistive-geometry/README.md) |
| Q-Plane O0-A | query-local ray-plane residual 有表示上界 | A0–A5 与负控 | parent-macro MAE、false-clear/block、gap closure | A4 MAE `.17284 m` 差于 global scale `.14590 m`，gap closure `-12.20%`，关闭 | 当前 Q-Plane family / O0-B / learned head | source oracle为何能过、可学表示是否存在 | [Q-Plane result](research/assistive-geometry/BLINDASSIST_BA_CLEAR_QPLANE_O0A_REPRESENTATION_HEADROOM_RESULT_2026-08-14.md) |
| TARO observability/ranking | 同预算主动选帧能更早获取 task evidence | oracle、pose/RGB retention、R14–R38 scorer | parent-macro、opportunity strict wins | 多个 consumed canary 有 signal；fresh R36无增量；R38 ranker `22.0682 < 25.0227 generic`、strict wins `2/5`，有效 FAIL并关闭 | 在同一 scorer family 上继续 rescue、容量/loss/gate 回调 | task observability headroom 与可迁移 scorer 之间的缺口 | [TARO current](research/taro/README.md) |
| SATOM / metric-ground test | 新绝对 range + memory 可绕过 RGB ceiling | frozen real E0 | ground-height observability前门 | arm metric 前因 DepthART ground-height不稳而 NOT_EVALUABLE | 不能反证 sensing/memory；能反证在已开 Bonn 输出上事后修高度 | 独立 metric ground source或 height-free task | [SATOM current](research/satom/README.md) |
| SVRF current | height-free纯 RGB相对风险可能绕过 metric target | scale-free O0 frozen | parent ranking/coverage（未运行） | `REAL_O0_NOT_RUN` | 什么都未被科学反证 | 纯 RGB observability、relative-risk target、跨域可迁移性 | [SVRF current](research/svrf/README.md) |

## 2. Falsified / weakened hypothesis registry

### F1 — “只要 generic depth / clearance 更准，产品 gate 会自然改善”

`STATUS: STRONGLY_WEAKENED`

- Evidence：DepthART R0、DA2 A2、D2 head、P3 都出现 proxy 改善但联合任务门失败。
- Counterevidence：Q-Plane 的 source-depth oracle为零误差，说明完美 upstream 仍可能有价值。
- Scope：当前 monocular/reference/reducer/evaluator contracts。
- Reopen：同一 fresh cohort 上，oracle geometry→冻结 policy PASS，且 estimated geometry→同 policy FAIL。

### F2 — “降低 false-clear 本身就是安全/utility 改善”

`STATUS: FALSIFIED`

- Evidence：A3 用近乎全 occupied 得到 `.095%` false-clear；B1/A0 和 D3 direct veto 均以巨大
  false-block 代价降低 false-clear。
- Counterevidence：有界 UNKNOWN deferral 可封顶代价，但 D3R6 ranking 增量仍未建立。
- Scope：把 false-clear 单独作为研发驱动指标。
- Reopen：必须联合 false-block、known coverage、transition、parent worst-case 和 event utility。

### F3 — “更高 mIoU / perfect four-class mask 足以解决 actionability”

`STATUS: FALSIFIED_FOR_CURRENT_TARGET_AND_POLICY`

- Evidence：RISKSEG R0 像素最佳 seed没有更好 event result；truth-mask soft adapter仍 12/14 false events。
- Counterevidence：90-frame oracle mask曾支配当前链，但混合了 corridor/source policy且仅3 events。
- Scope：当前四类 target + adapters + event policy。
- Reopen：新的 actionability labels、独立 parent cohort、先通过 label stability 和 oracle-policy gate。

### F4 — “增加 temporal machinery 会自然带来事件收益”

`STATUS: STRONGLY_WEAKENED`

- Evidence：F0.1、P3、RCLE自然 session、frame-veto density结果未形成稳定事件增量。
- Counterevidence：causal future labels增加 support；P3多项 development metrics向好。
- Scope：当前 RGB→geometry-proxy students、flow/motion proxies和缺 event truth 的设置。
- Reopen：先做同 cohort single-frame vs causal clip upper bound，再决定是否训练。

### F5 — “当前表示没有任何 headroom”

`STATUS: FALSIFIED_AS_A_BLANKET_CLAIM`

- Evidence：mask oracle、小规模 factor-wise oracle、source-depth oracle显示局部/上界信息存在。
- Counterevidence：learned transfer、fresh ranking和跨 parent calibration普遍失败。
- Scope：不能把局部 oracle headroom外推为可学、可迁移或产品可用。
- Reopen requirement：不适用；正确状态是部分支持而非全局断言。

### F6 — “RGB 输入本身已经被证明不够”

`STATUS: UNRESOLVED`

- Evidence for：metric scale/ground-height、遮挡、运动和清除需要额外信息；pose能恢复 3/6 parent geometry。
- Counterevidence：oracle mask来自图像标注；设备 RGB pair获得 positive-evidence signal；SVRF尚未运行。
- Scope：single RGB 对 task actionability 的信息上限。
- Reopen/close：必须执行 blinded single-frame/clip/sensor reveal upper-bound。

### F7 — “部署/运行性能是当前科学主瓶颈”

`STATUS: FALSIFIED_FOR_RISKSEG; ROUTE-SPECIFIC_ELSEWHERE`

- Evidence：RISKSEG QNN/HTP 10分钟性能PASS但事件质量FAIL；G4-D只关闭特定QAIRT路径。
- Counterevidence：MiDaS与某些DepthART路径确有工程阻塞。
- Scope：不能用平台成功或失败解释全部科学结果。
- Reopen：只有候选先通过效果门后，部署才重新成为晋级瓶颈。

## 3. Root cause ranking

| Rank | Root cause | Prior | Supporting evidence | Contradicting evidence / anomaly | Experiments explained | Next diagnostic information value | Cost |
|---:|---|---|---|---|---|---|---|
| 1 | H3 target/supervision failure | High | perfect class mask仍不区分 should-alert/cleared；label readiness低；ground opportunity为零；source/pseudo/geometry truth混杂 | 小型oracle-mask cohort曾全过；source-native factor labels可物化 | segmentation、SuperTeacher、reference、RISKSEG、部分geometry transfer | Very high：actionability adjudication + label stability | Medium |
| 2 | H4 downstream objective/policy/evaluator failure | High | oracle box恢复目标却无法清除；false-clear/block冲突；canonical曾被用作truth且自身false-clear高；统一三态/scalar risk压缩多类危险 | 当前mask oracle在小cohort可过；新gate已加入coverage/worst-parent | bbox rules、A3、B1/A0、D2/D3、RISKSEG adapter | Very high：oracle geometry→current policy 与 direct action oracle | Low–medium |
| 3 | H2 representation ceiling | Medium-high | depth scale/plane、obstacle undercoverage、Q-Plane、TARO fresh transfer、0/6 threshold pairs | source-depth oracle零误差；oracle mask和factor oracle显示上界 | DepthART、AG、Q-Plane、TARO、SATOM preflight | High：perfect geometry substitution | Low |
| 4 | H1 input/observability ceiling | Medium | monocular scale、occlusion、motion、event end；pose恢复部分geometry | 未做直接upper bound；纯RGB O0未运行；image-derived oracle有signal | temporal/metric-ground失败的一部分 | High：single RGB vs clip vs sensor reveal | Medium |

H1 排名第四不是“RGB 足够”，而是证据尚未把 input ceiling 与 target/policy failure 分开。H2 排在
H3/H4 后，是因为项目已经证明“representation 改善不传导”，却尚未证明“给 policy 完美 geometry 就能过”。

## 4. Search concentration / wrong-level optimization

历史矩阵见[因果模型](BLINDASSIST_CAUSAL_FAILURE_MODEL.md#历史干预矩阵)。结论明确：绝大多数
有 outcome 的实验干预了 representation；真正改变 target/actionability truth 的实验极少，改变
downstream policy 并用独立 action truth 评价的实验更少，真实用户/产品 evaluator 为零。

因此当前研究史符合：

```text
SEARCH_CONCENTRATION / WRONG_LEVEL_OPTIMIZATION
```

这不是说 representation 不重要，而是没有先做 oracle decomposition，就无法知道 representation 是否是
总系统的 binding constraint。B1/A0→R2→interaction→pose→Q-Plane 的连续失败链尤其说明，同一层级的
新名字不能替代跨层因果干预。

## 5. Proxy-target alignment verdict

正式状态：`PROXY_TARGET_ALIGNMENT_NOT_ESTABLISHED`。

以下 proxy 不再有资格单独驱动研发或晋级：generic depth AbsRel、scale-aligned AbsRel、clearance MAE、
mIoU/boundary F1、geometry-proxy F1、frame alert density、false-clear、transition agreement、ranking macro、
HTP partition/latency。它们只能作为解释变量或 guardrail；候选必须同时报告 actionability/event outcome、
false-clear、false-block、UNKNOWN/coverage、transition/clear 和 parent/session worst case。

## 6. 产品目标诊断

当前证据支持 `CURRENT TASK FORMULATION IS PARTIALLY MIS-SPECIFIED`，但还不足以签署
`CURRENT PRODUCT TARGET NOT SUPPORTED BY AVAILABLE INFORMATION`。

- 单帧 monocular RGB 被要求输出 metric-ish clearance，却缺稳定尺度锚；这是输入/目标不匹配风险。
- “深度预测”反复不等于“安全通行判断”。
- false-clear 与 false-block 不是同一 scalar risk 上可随意互换的对称误差；A3/B1/D3已给出结构冲突。
- transition gate暴露的是事件生命周期和稳定性，不只是 temporal smoothness。
- obstacle、台阶/落差、动态接近、悬空物、窄通道的可行动语义不同；统一三态/单标量可能丢失决策条件。
- 对盲人有用的最终输出应先定义为受限场景中的 action recommendation / traversability / collision-likelihood
  事件合同，再决定需要何种 geometry；本轮不启动该新路线。

## 7. 最高价值诊断与预算决定

D-ORACLE-1 已被用户指定为唯一 P0，并已冻结为严格三臂 matched causal ladder；完整合同见
[protocol](research/failure-synthesis/D_ORACLE_1_MATCHED_CAUSAL_LADDER_PROTOCOL_2026-08-17.md)与
[Oracle Ladder](BLINDASSIST_ORACLE_LADDER.md#唯一-p0-diagnostic)：

```text
A = Direct Action Oracle
B = Perfect Source Geometry -> Current Policy
C = Estimated Representation -> Same Current Policy
```

只用 matched parent utility 计算 `G_downstream=U(A)-U(B)` 与
`G_representation=U(B)-U(C)`；B/C 后的 policy/config/threshold/coverage/evaluator/denominator逐hash相同。
parent-local geometry derangement只作机制control，不是第四竞争arm。

H3/H4不在本轮拆分。只有 `A materially > B` 后才允许另立小型D-ORACLE-2；其arms现在不定义。
single-RGB/causal-clip/sensor-reveal和policy frontier均降为未冻结backlog。

在 D-ORACLE-1 定位 representation headroom 之前，**不再给新 encoder/loss/fusion/depth model/selector
或SVRF执行分配预算**。

## 8. Continue / pivot / stop

- **继续当前 representation 路线**：仅当 perfect geometry→current policy PASS，estimated representation→
  same policy FAIL，且 failure跨 parent稳定。
- **切换 representation**：满足上一条件后，旧 family 已达到预声明停止条件；新表示必须改变可恢复信息，
  不能只是 Q-Plane/selector 改名。
- **切到 temporal**：causal clip upper bound显著高于single RGB，且 label stability与action outcome均提高；
  只有proxy improvement不够。
- **修改 product objective / policy**：direct action oracle PASS，但 perfect geometry→current policy FAIL；或
  perfect-geometry frontier不存在联合可行点。
- **停止研究线**：同一 input/target/policy 条件下连续两个独立 parent-disjoint结果未越过预冻门，或 oracle
  显示该层不是 binding constraint；已关闭的 B1/A0、Q-Plane、TARO R38 和当前 obstacle selector family
  不得再以新名字重开。

## 9. 当前知识状态

```text
ESTABLISHED:
- No candidate has earned default-App, product, or safety authority.
- Proxy improvement does not reliably transfer to task/event gates.
- False-clear can be improved by pathological over-blocking or bounded abstention.
- Current four-class segmentation target plus current adapters/policy is not sufficient.
- Research effort has been concentrated at the representation layer.

STRONGLY SUPPORTED:
- Target/supervision mismatch and downstream objective/policy mismatch are the primary cause pair.
- Current monocular metric-depth/geometry representations are not cross-parent reliable enough.

FALSIFIED / STRONGLY WEAKENED:
- Bigger/newer model, better generic depth, higher mIoU, lower false-clear, or faster deployment alone will solve the task.
- Adding temporal machinery without an event/actionability upper bound is a justified next step.

UNRESOLVED:
- H3 target failure versus H4 policy/evaluator failure as the single primary cause.
- The independent ceiling of single RGB and the incremental value of causal temporal information.
- Whether perfect source-native geometry can make the current product gate pass.

PRIMARY BOTTLENECK HYPOTHESIS:
H3 TARGET / SUPERVISION FAILURE.

SECONDARY BOTTLENECK HYPOTHESIS:
H4 DOWNSTREAM OBJECTIVE / POLICY FAILURE.

HIGHEST-VALUE NEXT EXPERIMENT:
D-ORACLE-1 ACTIONABILITY→GEOMETRY→REPRESENTATION LADDER (UNIQUE P0; PROTOCOL FROZEN).

DO NOT SPEND MORE BUDGET ON:
new encoder/loss/fusion/depth/selector variants; old B1/A0, Q-Plane, TARO or obstacle-router rescues;
proxy-only wins; deployment optimization before task admission.
```

## 10. Retrospective decision audit

如果今天带着全部现有知识重新开始，**不会选择现在这条“YOLO rules → generic depth → segmentation
architecture → temporal proxy → increasingly elaborate geometry/ranking”研发路线**。

正确起点应是：先冻结 parent-event actionability contract；做 current / oracle box / oracle mask / oracle
geometry / direct-action oracle ladder；只在确认哪一级从 PASS→FAIL 后才研发对应层。

最早的关键偏航点不是某个具体模型，而是把“可获得的 source truth（类别、mask、depth、pose）”当成
“产品所需 truth（应该提醒、何时清除、是否可通行）”，随后又用 proxy improvement 选择 representation。
当时证据不足以发现，是因为缺少 parent-event truth、清除标签、独立 session、source-role账本和
matched oracle controls；早期只能看到局部帧级数字。现在证明应换决策顺序的证据是：oracle box/mask
分裂、perfect four-class mask仍失败、DepthART/DA2/D2多次 proxy→task断裂，以及 B1/A0、Q-Plane、TARO
在更复杂表示下仍跨 parent失败。

这不是建议放弃 BlindAssist，而是建议停止把“下一模型”当作默认研究动作。下一步必须先回答：
**给定完美 actionability 或完美 geometry，当前 gate 是否可通过？**
