# Central obstruction Agent label readiness D0-A1 result

状态：`COMPLETE / VALID / AGENT_LABEL_PROTOCOL_NOT_RELIABLE /
D0_A2_NOT_AUTHORIZED / D0_B_NOT_AUTHORIZED`

时间：2026-07-31（Asia/Hong_Kong）

## 结论

D0-A1 R2 已完成冻结的 fresh isolated second pass、材料分歧裁决与最终 readiness
复算。终态是 `AGENT_LABEL_PROTOCOL_NOT_RELIABLE`，不是 `READY`。

55 条 observation 的 raw reviewer agreement 为 `47/55=0.8545`，claim-critical
并集 agreement 为 `39/47=0.8298`，二者均通过冻结的 `0.80` 门；matched event 的
boundary delta P95 为 `1 observation`，也通过门。但是两遍 raw review 形成的
parent event 分别为 18 与 19 个，只匹配 12 个，match rate
`12/19=0.6316 < 0.75`。因此可见阻塞 observation 在本 calibration 上尚可判，但
当前五点 clip/event 规则不能稳定复现 parent-natural-event 边界，不能扩到 D0-A2。

8 个 label disagreement 已全部由 fresh third Agent 处理：7 个形成 adjudicated
label，1 个按冻结规则隔离为 `NOT_EVALUABLE`。裁决只关闭 unresolved item，不覆盖或
“修复”两遍 raw event-match；最终 unresolved 为 0，canonical
`NOT_EVALUABLE=12/55=0.2182`，但 event-match 门仍失败。

## 隔离与执行顺序

1. 主任务冻结只含 prompt、manifest、11 张 contact sheet 与 55 张 observation PNG
   的隔离说明；不向 second Agent 提供 primary label、计数、current 结论或模型输出。
2. fork-none fresh Agent 查看 11/11 contact sheet、55/55 observation 后先封存 raw
   second review；SHA-256 在任何比较前固定。
3. validator 才读取两遍 review，按两遍 label/quality 的 claim-critical 并集计算
   agreement，冻结 8 个 material disagreement packet。
4. 另一个 fork-none fresh Agent 只读取 prompt、8 项 packet、对应图像与 contact
   sheet；它不知道 aggregate threshold/result，只输出第三方裁决。
5. finalizer 保留 primary、second、adjudication 三份 raw 输出不可变，派生 canonical
   calibration ledger 和最终 terminal。

second pass 的 label 为 `PRESENT=34 / NO_EVIDENCE=9 / NOT_EVALUABLE=12`；quality
为 `STABLE=25 / TURNING=13 / BLURRED=1 / DARK=3 / OCCLUDED=1 /
OTHER_NOT_EVALUABLE=12`。最终 canonical label 仍为 `34/9/12`，共 19 个 parent event。

R2 primary 是 invalid R1 label 的零变更时间戳修复转录，只有 second pass 是本轮
fresh isolated context。因此本结果不升级为“两份全新隔离 raw review”的强共识；
即使采用更宽松的 primary-versus-isolated 比较，冻结 event-match 门仍明确失败。

两个 raw Agent 对 hash 字段分别使用 `review_prompt_sha256` 与 `packet_sha256` 命名；
validator 只接受这些显式 alias 且要求值与冻结 prompt/packet SHA-256 完全相等，
没有改写 label、quality、rationale 或 timestamp。alias 与 isolation/context mutation
均有 focused regression test。

## 冻结门结果

| gate | 结果 | threshold | pass |
| --- | ---: | ---: | --- |
| overall observation label agreement | `0.8545` | `>=0.80` | 是 |
| claim-critical union label agreement | `0.8298` | `>=0.80` | 是 |
| parent-event match rate | `0.6316` | `>=0.75` | **否** |
| matched boundary delta P95 | `1` | `<=1` | 是 |
| post-adjudication unresolved fraction | `0.0000` | `<=0.10` | 是 |
| canonical `NOT_EVALUABLE` fraction | `0.2182` | `<=0.40` | 是 |
| coverage / isolated completion / adjudication completion | complete | required | 是 |

所有门必须同时通过；不得用 observation agreement、canonical labels 或第三方裁决
替代失败的 raw parent-event reproducibility。

## Evidence identity

本地 root：
`artifacts.local/evidence/central-obstruction-agent-label-readiness-d0-a1-r2/`

| artifact | SHA-256 |
| --- | --- |
| `isolated-second-review.json` | `47049587c7d519a969f86d99035fd1a05b5e744ada86e46aec1f796eb82bd930` |
| `isolated-second-parent-events.jsonl` | `24550116d19b0b99def19e012ae4beb9bd10507c9c13ce90a45ef0df9ad7971f` |
| `d0a1-initial-agreement.json` | `c44feab3c55ff99c1d1c25a7ba5646c28e020a5b286f27aca9dc1ba2a756be64` |
| `d0a1-adjudication-packet.json` | `339c84ad4b88cc94fd7b323f744fd335df8decb249ec50bbe888b9c96fd1f13d` |
| `adjudication-review.json` | `66f902315e1dfc83de17e91bd89551639967b5f37610403c3ceea1fda3ac6671` |
| `canonical-calibration-labels.jsonl` | `79ea6f39d00d96b05403b813ac92bf78d42aa12fc87d4072d455e2ca19ac352d` |
| `canonical-calibration-parent-events.jsonl` | `cc4e1434d83b56eef87bbe85896c50f101d5e5676e5f78b349fb63ca385262b2` |
| `d0a1-final-readiness.json` | `131b799bedf7dc4e27e63b5583205adb114a9e5af7f307a228bf005e32acad40` |
| `d0a1-final-validation.json` | `a35007909530622a6700ef748fa1860f2bcbc472e57748ef1b8979bc685445f0` |

R2 input、primary 及其既有 SHA-256 保持不变；formal output 均 write-once，未覆盖
R0/R1/R2 predecessor。

## Failure learning 与下一边界

支持的推断是：当前 observation wording 在 observation 粒度有一定稳定性，但
foreground/midground 深度次序、近距离表面、turn/edit view 与五点稀疏采样共同使
parent-event 边界不够可复现。不能由此断言单 RGB 的中央阻塞定义普遍不可行，也不能
把失败归因于任何候选模型，因为 reviewer 从未读取 candidate output。

当前唯一合法 successor boundary 是：另立新的 D0-A 版本，只在这些已烧毁的
calibration/disagreement stress cases 上修改 observation definition 或 event workflow，
预先冻结新的 falsifier 后再决定是否重跑 calibration。不得在 R2 上调门、重切 clip、
改 label 救援，不得启动 D0-A2、D0-AT、D0-B、模型效果评价、Android 或人体工作。

## Claim ceiling

这是 `CANARY_LITE / Agent labelability calibration` 的失败诊断，不是人类或客观真值、
模型 B 增量、可通行性、碰撞风险、真实用户效果、产品或安全证据。
