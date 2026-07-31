# Central obstruction Agent label readiness D0-A1 entry

状态：`ACTIVE / R2_LOCK_FROZEN / PRIMARY_CALIBRATION_COMPLETE /
FRESH_ISOLATED_PASS_REQUIRED / D0_A2_NOT_AUTHORIZED`

时间：2026-07-31（Asia/Hong_Kong）

## 结论

D0-A1 已正式进入。R2 在 4 个明确排除于 production 的 calibration source 上冻结
11 个 clip、55 个 observation、中央 ROI、三态 prompt、parent-event/matching、
歧义分层、风险分层 audit 和 readiness 数值门；production source overlap 为 0，
candidate output access 为 false。输入 producer 与独立 validator 均为 `VALID`。

当前 primary calibration pass 已完整覆盖 55/55 observation，但 R2 是对 invalid R1
标签的零变更时间戳修复转录，并如实披露为
`PRIMARY_CURRENT_TASK_NON_ISOLATED_TIMESTAMP_REPAIR_TRANSCRIPTION`。因此只能形成
prompt calibration 与初始 parent-event 诊断，不能计算 agreement、不能声称
consensus、不能判定 D0-A1 readiness，也不授权 D0-A2 或 D0-B。

当前唯一下一动作是：在不知道 primary label 的 fresh isolated context 中，对同一
R2 prompt、同一 55 个 observation 运行第二 pass；不得修改 R2 输入、prompt、ROI、
事件规则或门槛。

## 排除式 calibration bundle

| source | calibration role | clip / observation |
| --- | --- | ---: |
| JRDB 32-frame diagnostic window | partial-image-sequence stress/calibration | 1 / 5 |
| Ulm public walk | stable street、foreground、upward-turn stress | 3 / 15 |
| Alicante public walk | pedestrian/pillar/marina/upward-view stress | 4 / 20 |
| Burwell edited public video | title card、barrier、fence/scene-cut stress | 3 / 15 |

这些 source 在 D0-A0 R3 reuse-role ledger 中均为
`ADMIT_D0_A_CALIBRATION_ONLY`。D0-A0 的 6 个 production-labeling session 未进入
本轮，calibration observation 永久不得计入 D0-A4 readiness 或后续事件效果分母。

## R0 → R1 calibration learning

R0 输入包先冻结并通过 payload/ROI validator；source-only inspection 随后发现原
prompt 的“scene element 占据 ROI”过宽，会把普通背景建筑也当成中央阻塞。R0 没有
生成标签，不作 readiness 证据，也未覆盖。

R1 只作 observation-definition 修订：

- positive 必须是前景/中景实体实际遮挡后方场景，或终止中央视线；
- 背景建筑、地面、天空、水面、远景、纹理/阴影和“仅出现在 ROI”均不够；
- `NO_VISIBLE_CENTRAL_OBSTRUCTION_EVIDENCE` 仍不表示 clear/safe/free-space；
- 连续、同原因的 `NOT_EVALUABLE` 保持一个 parent event，不再错误拆成单帧事件。

该修订发生在任何 raw label 写入前，且没有读取 candidate、YOLO、分割、深度、risk、
feedback 或旧 review label。

R1 首次 primary 写入把 submission time 手工估计为晚于 validator 的未来时刻，封存为
`INVALID_REVIEW_TIMESTAMP_ORDER`。R2 仅更新 evidence identity/output root、用 host
clock 写入真实时间，并增加 future-submission rejection；prompt、ROI、source、clip、
frame、label、event、audit 与 readiness threshold 均未改。R2 primary 明确披露读取
了 invalid R1 predecessor，55 个 label 原样转录、变化为 0，仍不是独立 pass。

## R2 primary calibration diagnostics

| 指标 | primary-only 结果 |
| --- | ---: |
| observation | 55 |
| primary parent event | 18 |
| `VISIBLE_CENTRAL_OBSTRUCTION_PRESENT` | 28 |
| `NO_VISIBLE_CENTRAL_OBSTRUCTION_EVIDENCE` | 12 |
| `NOT_EVALUABLE` | 15 |
| claim-critical observation | 43 |
| low-risk observation | 12 |
| 冻结 low-risk independent-audit target | 4 |

质量状态为 `STABLE=28`、`TURNING=10`、`OCCLUDED=12`、
`OTHER_NOT_EVALUABLE=5`；`BLURRED/DARK=0`。来源、clip、observation、三态覆盖、
至少三种质量状态与 `NOT_EVALUABLE <= 0.40` 的 precondition 均通过，但这只证明
pilot denominator 足以进入独立复核，不证明标签可靠。

## R2 lock 与门

- ROI：normalized `xyxy=[0.25, 0.15, 0.75, 0.95]`；native ROI 任一边小于
  64 px 时 fail closed。
- claim-critical：所有 positive、`NOT_EVALUABLE`、非 `STABLE`、scene-cut
  邻接项和 terminal-changing item，要求两个 fresh isolated pass。
- low-risk：稳定 negative 只做 primary；D0-A2 后每 source 最少 1 个、总体 20%
  进入冻结抽样 audit。
- D0-A1 readiness：overall label agreement `>=0.80`、claim-critical
  agreement `>=0.80`、parent-event match `>=0.75`、boundary delta P95
  `<=1 observation`、unresolved `<=0.10`、`NOT_EVALUABLE <=0.40`，且所有
  denominator 非零。缺 isolated evidence 必须 `NOT_READY`。

## 证据身份

本地 root：
`artifacts.local/evidence/central-obstruction-agent-label-readiness-d0-a1-r2/`

| artifact | SHA-256 |
| --- | --- |
| `d0a1_lock_r2.json` | `7d3c9061c278fb96c778de8cad839d92486d891869ab2a07859945593a972026` |
| `d0a1_review_prompt_r1.md` | `7673d49aba86012f6935d7ebe44bd42b3be373f49aa7706e38adb50c6c1474c8` |
| `pilot-input-manifest.json` | `e908ec19d4101a9f5f8214f2a41317c28c88e5169038cfd4218ae10c71f82bf7` |
| `pilot-input-receipt.json` | `0968e5c6ea59cc84c68b21c5b24debf21a73c9ef10dfaa259587d6ff2f354625` |
| `pilot-input-validation.json` | `ca3c074c15d1c43fb7d28f5730d9f2e6d6260513c9ca26fc088e996e21a5d39b` |
| `primary-review.json` | `74d77bd5c4db57e7b78796f8bba8092fa30671a4494a28c65e18ad544f7bc68b` |
| `primary-parent-events.jsonl` | `ed46f765da316d31d584b49a4c83e8d63013677514183409d65a744afa610c0d` |
| `primary-review-validation.json` | `fffedc2b2e3405cf3d4e534cd27022bdd66c6be780ae20305ff99e7c63e2aa8e` |

## Claim ceiling

当前只支持：D0-A1 R2 的 calibration 输入、ROI、prompt、事件/审计规则、数值门与
一个非隔离 primary pass 已形成且可复算。它不是 independent agreement、model
consensus、人类/客观真值、production label、模型增量、可通行性、产品或安全证据。
