# P1-AMRM0 matched Development canary result

日期：2026-08-22（Asia/Hong_Kong）

终态：`P1_AMRM0_MEMORY_POISONING_FAIL`

## 回答

本 canary **没有建立 verified multi-view referent memory 的 identity value signal**。

在完全相同的 consumed P1-D0 15 episodes、1,724 frames 与 P1-A4 exact candidate stream 下，AMRM0 的
identity precision 从 `85/897 = 9.48%` 上升到 `79/742 = 10.65%`，identity coverage 从
`897/926 = 96.87%` 降至 `742/926 = 80.13%`。但关键反证同时成立：

- wrong-instance reacquisition：`12 -> 38`；
- verified-bank poisoning：`17` 次，违反 zero-poisoning hard gate；
- newly accumulated verified KF 对 true reacquisition 的贡献：`0`；
- correct identity commitments：`85 -> 79`；
- true same-instance reacquisition：`0 -> 15`，但全部仅由 original binding/context 支持，不能归因于多视角积累。

因此 precision 的小幅上升主要来自更强 abstention，而不是 verified multi-view memory 建立了可靠的同实例身份能力。

## Matched contract

- baseline：sealed P1-A4 continuous-correspondence output；
- AMRM：只能对同一 candidate bbox commit 或输出 `NONE`；
- candidate availability：926 candidate / 798 explicit NONE，exact parity；
- added candidates / frames / global search：全部为 `0`；
- post-initialization ground-truth reads：`0`；
- verifier：继承 P1-A2 frozen DINOv2-S dense gate，无阈值搜索；
- claim ceiling：`CONSUMED_ADT_MATCHED_DEVELOPMENT_SIGNAL_ONLY`。

## Risk-coverage 与诊断

| 指标 | P1-A4 baseline | AMRM0 |
|---|---:|---:|
| identity precision | 9.48% | 10.65% |
| identity coverage | 96.87% | 80.13% |
| wrong identity commitments | 812 | 663 |
| correct identity commitments | 85 | 79 |
| wrong-instance reacquisition | 12 | 38 |
| true same-instance reacquisition | 0 | 15 |
| false continuity | 792 | 625 |
| honest abstention on wrong candidate | 24 | 173 |
| false-loss frames | 121 | 207 |

AMRM0 的 timely stale 为 `6/16 = 37.5%`。最终 verified bank 共 35 个 coverage cells，中位每 episode 为 2；
其中 17 次新增观察实际上属于错误实例。Scale-change 与 2D bearing-change reacquisition 在本次数据中分母均为 0；
physical viewpoint truth 不存在，固定为 `NOT_EVALUABLE_MISSING_VIEWPOINT_TRUTH`。

冻结终态之后的 descriptive attribution 显示，663 个 wrong commitments 中 `642/663 = 96.83%` 是 background
candidate，21 个是其他 ADT instance；这不改变终态。它把第一层失败进一步定位为：在 A4 proposal 已漂移到背景时，
AMRM0 的 target/context identity authority 仍大量放行，而不是多视角 bank 弥补了 proposal 漂移。

## Failure structure 与唯一后继

直接失败层是 **verified admission authority 不成立**：原始 target/context dense support 并不构成独立的实例身份确认，
所以错误候选仍能被写入 verified bank。与此同时，新积累 KF 没有贡献任何正确重捕获，当前证据不支持继续扩展 AMRM。

唯一后继是 outcome-preserving failure autopsy：定位 17 次 poisoning 的候选/场景结构，以及 original-binding/context
为何共同放行错误实例。禁止调 retrieval 阈值、增加 detector/prompt/frame、启动 AMRM1/2/3、VLM、VIO、SLAM、
scene graph 或 geometry。只有尸检形成新的、单变量可证伪假设后，才另行决定是否开新实验。

## Evidence identity

- implementation commit：`ea277c9ede2057f460606576bc458ee30733fed7`
- manifest SHA-256：`ccc36e7dc1382ef7028f0c2efcaf230420ae69e410438bb0328d177f26ce405e`
- prediction SHA-256：`73839296e990328a2fb8d9ba3a6d7263e8f38003bc56f502963d16aa8d000526`
- trace SHA-256：`54ed59da0ca9853e0f95658f560f615c9b2fb7bc5bb8eb6ef164b2342b282c4b`
- local result：`artifacts.local/evidence/p1_amrm0_matched_canary_v1/result.json`
