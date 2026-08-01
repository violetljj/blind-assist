# HFTF Stage C D3-Q0 筛选与封存效果执行合同

状态：
`FROZEN_AFTER_D3_Q0_ROSTER_BEFORE_ANY_D3_MEDIA_SUPPORT_OR_TRUTH`

## 结论

40-slot metadata roster 已锁定，但本文件冻结时仍未打开任何 D3 slot 的 pose、depth、
mask、support 或 future truth。本合同把后续资格筛选、首 6 个合格 source 的
future-blind prediction，以及一次性 sealed-truth effect evaluation 同时锁定；不得在
看到任何 D3 outcome 后再修改阈值、轴、顺序、实现或效果门。

本阶段是 `THESIS_DEVELOPMENT` 条件挑战集实验，不是自然分布 prevalence、人体事件
truth、独立安全验证、生产准入或默认 App 变更。几何教师继续只是 SANPO synthetic
proxy。

## 冻结输入与实现

机器可执行 JSON 绑定以下不可变父级及其 SHA-256：

- D3-Q0 reference-and-support-only 主协议；
- 已封存的 40-slot metadata roster 与 tracked roster result；
- D2 的 `NOT_EVALUABLE` result、mechanics contract、D2/D2.1 设计；
- G0 signed-clearance 定义与 swept-envelope mechanics；
- D3 common/state、next-slot runner、selector-only aggregator、selected-six
  future-blind preprocessor、sealed-effect evaluator；
- D2 transport dependency、mechanics primitives、G0 geometry primitives 与 SANPO
  pose/ground authority；
- 20 个状态机测试和 15 个 pipeline/firewall 测试。

所有执行入口在正式运行时必须验证 exact hashes、tracked-clean 状态以及
`HEAD == origin/master`。canonical screening root 固定为：

`artifacts.local/evidence/hftf/stage-c-d3-q0-screening-20260802`

所有预计算路径及其 `.tmp` 形式必须短于 240 字符；slot directory 只使用 roster
index 与 session-ID hash token，不在路径中暴露完整 session ID。

## 逐 slot 资格筛选

slot 只能按冻结的 session-ID 字典序逐个消费。每个 slot 在首次 pose/media request
前先排他写入并 `flush + fsync` attempt；中断、下载、内容、mechanics 或写入失败都产生
固定 failure terminal 并消耗该 slot。不得重跑、替换、跳序、扩充到第 41 个或依据
已见 outcome 改变顺序。

任何 selector/failure 都必须绑定并由 state scanner 重验该 slot 的 durable attempt；
没有合法 attempt 的 receipt 不能被聚合。attempt 写入前的空 slot directory 不代表
source 已打开，可安全完成同一个 attempt；durable attempt 后的孤儿状态必须直接封存
failure，不能重开输入。遗留 `.tmp` 不删除，改名为 `.orphan` 后再写正式 failure。

每个 slot 只下载 1 个 pose CSV，以及 normalized frame `2..12` 的 11 个 mask 和
11 个 depth；RGB 下载和读取均为 0。qualifier 对 7 anchors、body/head、
`.4/.8 s` 四个 strata 计算：

- persistence support known；
- causal-advected support known；
- future truth signed-clearance 与 truth known；
- 三方 common-known 上的 coverage、truth-risk、truth-safe counts。

每个 stratum 的冻结 denominator 为 `7 × 6 × 6 = 252`。四个 strata 都必须同时满足
common-known coverage `>= .10`、truth-risk cells `>= 5`、truth-safe cells `>= 20`，
且 UNKNOWN→SAFE violation 必须为 0。

qualifier 不得计算或写入 persistence/advected 两个候选臂的 clearance、MAE、F1、
confusion、delta 或 parent improvement。含逐 cell support 与 future truth 的 payload
必须先 durable，随后才可写 closed-list selector。selector 与 aggregator 不得读取
sealed payload，只能读取 selector/failure receipt。

首 6 个合格 selector 出现时立即停止并写出：

`D3_Q0_REFERENCE_SUPPORT_OPPORTUNITY_COHORT_QUALIFIED`

40 slots 用尽仍不足 6 个时写出：

`D3_REFERENCE_SUPPORT_OPPORTUNITY_COHORT_NOT_EVALUABLE_BUDGET_EXHAUSTED_NO_EXPANSION`

## 选择后 future-blind prediction

只有 exact first-six selection durable 后，preprocessor 才可运行一次。它只能读取：

- pose normalized indices `0..8`；
- current/history depth 与 mask normalized indices `2..8`。

它不得读取 future-only `9..12` pose/media，也不得读取 sealed payload。42 个 anchor
prediction records、42 个 points arrays 与 84 个 horizon records 必须在 completion
前全部 durable。任何中断保留 partial artifacts，并禁止第二次 preprocessor。

## 一次性 sealed effect

effect evaluator 必须先完整验证 selection、42 个 predictions、points 与 completion，
再 durable 写入 effect attempt 和 sealed-payload-open-once receipt。之后每个 selected
payload 只读一次，不得重新打开 future media。

aggregator 同样必须在第一次读取 selector/failure receipt 前 durable 写入
aggregate attempt，selection/budget terminal 反向绑定该 attempt。aggregator、
preprocessor 或 evaluator 若在 attempt 后硬终止，下一次调用只能从既有 attempt/open
receipt 封存 no-rerun failure，不得重新读取 selector、prediction 或 sealed payload。

在计算 effect 前必须重算：

- selector 的四 strata 资格；
- predicted basis；
- qualifier support 与 formal prediction support 的 exact equality；
- UNKNOWN 不能带 numeric clearance；
- D2 opportunity adequacy。

任一不一致都停止为：

`D3_NOT_EVALUABLE_QUALIFICATION_RECOMPUTE_MISMATCH_NO_REPLACEMENT`

selection 与 prediction completion 尚未 durable 时，过早调用 evaluator 只拒绝执行，
不创建 attempt。两者已 durable 后，pre-truth validation 失败必须封存为
`D3_Q0_EFFECT_PRETRUTH_VALIDATION_FAILED_NO_RERUN_NO_REPLACEMENT`；effect attempt
或 open-once receipt 之后的任何中断必须封存为
`D3_Q0_SEALED_PAYLOAD_EFFECT_INTERRUPTED_NO_SECOND_OPEN_NO_REPLACEMENT`。两类失败都
禁止第二次 evaluator、第二次 sealed payload open、换源或同 cohort 调参。

只有上述检查全部闭合，才复用 D2 完全相同的 parent-session macro estimand 与效果门：
macro MAE relative reduction `>= .10`、absolute reduction `>= .03 m`、body/head 与
`.4/.8 s` 分层均 noninferior、6 个 parents 至少 5 个严格改善、parent-macro risk-sign
F1 delta `>= .03`、UNKNOWN→SAFE 为 0。

效果终点只能是：

- `CAUSAL_SIGNED_CLEARANCE_TRANSPORT_SUPPORTED_FOR_RGB_STUDENT_PROTOCOL`
- `CAUSAL_SIGNED_CLEARANCE_TRANSPORT_NOT_SUPPORTED_STOP`

正终点也只允许另冻 RGB-student protocol；不直接授权训练、执行、reserved official
test、切换研究主线、修改 Android/default App、生产或安全声明。

## 执行顺序

所有命令均从仓库根目录运行，并且只能在本合同已提交、推送、远端一致、formal
contract validation 与独立审计通过后开始：

```powershell
E:\codex-tools\bin\blindassist-python.cmd scripts/research/hftf/run_stage_c_d3_q0_next_slot.py `
  --contract docs/research/hftf/HFTF_STAGE_C_D3_Q0_SCREENING_EFFECT_EXECUTION_CONTRACT_2026-08-02.json `
  --retries 3
```

runner 返回 `aggregate_required=false` 时继续执行唯一 next slot；只有返回
`aggregate_required=true`、即达到首 6 个合格 source 或 40-slot budget terminal 后，
才运行 selector-only aggregator：

```powershell
E:\codex-tools\bin\blindassist-python.cmd scripts/research/hftf/aggregate_stage_c_d3_q0_screening.py `
  --contract docs/research/hftf/HFTF_STAGE_C_D3_Q0_SCREENING_EFFECT_EXECUTION_CONTRACT_2026-08-02.json
```

只有 exact first-six selection 成立时才按顺序运行：

```powershell
E:\codex-tools\bin\blindassist-python.cmd scripts/research/hftf/preprocess_stage_c_d3_q0_selected_future_blind.py `
  --contract docs/research/hftf/HFTF_STAGE_C_D3_Q0_SCREENING_EFFECT_EXECUTION_CONTRACT_2026-08-02.json

E:\codex-tools\bin\blindassist-python.cmd scripts/research/hftf/evaluate_stage_c_d3_q0_sealed_effect.py `
  --contract docs/research/hftf/HFTF_STAGE_C_D3_Q0_SCREENING_EFFECT_EXECUTION_CONTRACT_2026-08-02.json
```

本 Markdown 是人读说明；执行权威是同名 JSON 与其中绑定的 exact bytes。
