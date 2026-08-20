# BA-ADT YOLOE visual-prompt candidate canary result

状态：`VALID / DEVELOPMENT_ONLY / PROPOSAL_BOTTLENECK_CONFIRMED / YOLOE_26N_VISUAL_PROMPT_NOT_SUPPORTED / NO_DINOV2 / SKY_DISABLED`

## 问题与单变量实现

本轮先用已消费的 `Apartment_release_clean_seq136_M1292 / Carrot_A (uid 4917588638317799)` 对
instance-redetection R1 做 GT-isolated failure accounting，再只问一个问题：保持 TargetMemory、2-of-3
confirmation、弱时空先验、memory quarantine、flow5 和 evaluator 不变时，以 `YOLOE-26n` visual prompt
替换 LOST candidate generator，能否提高 candidate coverage 和正确重捕获。

R1 诊断重放只在 RGB observer 中增加 proposal/verifier trace；ADT GT 仍只进入独立 accountant。R1 原
evaluator 指标逐项重现。失败定义为：正确 proposal 从未出现是 `NO_CANDIDATE`；出现但未过现有
TargetMemory/阈值是 `CANDIDATE_REJECTED`；正确且 eligible 但未完成连续确认是
`CONFIRMATION_FAILED`。

YOLOE arm 继续用 YOLO11n 做正常 detector/flow tracking，只在 LOST 时调用 `yoloe-26n-seg.pt`。visual
prompt 来自首次确认的 RGB anchor bbox，不使用 GT；现有 handcrafted TargetMemory 仍是唯一 identity
verifier。运行环境为 `ultralytics 8.4.52`、CUDA:0。官方 visual-prompt 与静态导出语义见
[Ultralytics YOLOE documentation](https://docs.ultralytics.com/models/yoloe)。

## R1 failure accounting

| Item | R1 diagnostic replay |
|---|---:|
| Eligible reacquisition opportunities | 10 |
| Successful / failed | 5 / 5 |
| `NO_CANDIDATE` | **5** |
| `CANDIDATE_REJECTED` | **0** |
| `CONFIRMATION_FAILED` | **0** |
| Candidate recall during GT-visible LOST search | **34 / 405 = 0.0840** |

五个失败窗口中 best candidate IoU 全为 `0.0`。这支持“当前首要瓶颈是 proposal generation”，不支持
先增加 DINOv2 identity verifier。

## YOLOE canary result

| Metric | R1 class-YOLO candidates | YOLOE-26n visual prompt |
|---|---:|---:|
| Candidate recall during GT-visible LOST search | **0.0840** (34/405) | **0.0686** (29/423) |
| Failure A / B / C | **5 / 0 / 0** | **5 / 0 / 0** |
| Localization recall, IoU >= 0.10 | **0.6203** | **0.6034** |
| Mean IoU on GT-visible frames | **0.4743** | **0.4637** |
| Correct reacquisition within 30 frames | **0.4000** | **0.2000** |
| Correct reacquisition within 90 / 180 frames | **0.5000 / 0.5000** | **0.5000 / 0.5000** |
| Median successful reacquisition delay | **26** | **31** |
| Longest localization dropout | **159** | **164** |
| GT-invisible false-visible | **0.0073** | **0.0073** |
| Correct / wrong / unresolved instance-redetection | **13 / 0 / 0** | **9 / 0 / 0** |

两个 arm 的 LOST 状态轨迹不同，所以 candidate-recall 分母是各自实际 GT-visible LOST search frames；该
指标用于端到端候选机制诊断，不解释为固定帧集合上的独立模型 benchmark。所有十个 opportunity 的
逐窗口 best IoU、first-valid-candidate latency 和分类保存在 machine output 中。

结果不支持 YOLOE-26n 单 visual prompt：它没有使任何失败窗口产生正确 proposal，反而降低 candidate
recall、@30 reacquisition 和整体 localization recall。`wrong-instance=0` 只说明本 sequence 上未观察到
错误重接；它不抵消 proposal negative，也不证明相似实例鲁棒性。

## Evidence identity 与决策

```text
yolo11n.pt                    sha256 0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1
yoloe-26n-seg.pt              sha256 1741c1f8da3cea47e2c01829c334a50dc0b9bbd05e685b90a3ce84fae32c8c1b
R1 failure_accounting.json    sha256 7d7b8672fd0e53c272dd0829a6b0f346c5a31070f8f2832443828b4d4c888590
YOLOE rgb_observations.json   sha256 aa036b21502239c9948f4766dd02ddb701ae03c52f1f6010bd5d9f14fceec109
YOLOE evaluation.json         sha256 c8d485f99ed6f7e3940c63ee8a7dec44ec597c74bf0ec80647fd16dcd6b7b084
YOLOE failure_accounting.json sha256 328f029191c56be2345b66b33bc0aa70116f34ab79da2a9e5422b4a571ba5155
```

机器输出位于 ignored `artifacts.local/evidence/ba_adt_candidate_accounting_r1/` 与
`artifacts.local/evidence/ba_adt_yoloe_visual_prompt_canary_1/`。

停止继续堆 verifier；DINOv2、SAM、Sky、Android/default-App 均不进入。唯一 successor 是
`ADT1_REAPPEARANCE_OBSERVABILITY_DIAGNOSTIC_R3`：在同一 consumed Development sequence 上只诊断五个
`NO_CANDIDATE` 窗口的目标像素尺度、visibility/遮挡与 RGB 可辨识性，判断是小目标/遮挡边界还是该
visual-prompt 机制不适配。它不是新 detector canary，也不产生导航、产品或安全结论。
