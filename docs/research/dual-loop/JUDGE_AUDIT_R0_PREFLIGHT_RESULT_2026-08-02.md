# 裁判审计 R0：preflight 结果（2026-08-02）

## 结论

候选数量门已满足，但裁判审计尚未进入可判定阶段：当前有 78 个去重 source-session 候选，落在 50–100 的目标范围内；不过它们仍来自 source-mask discovery，八类 RGB 场景、原子真值、`UNKNOWN` 动作标签、相似框物理反事实、盲审稳定性和 oracle 两路径 trace 都尚未冻结。因此终态是：

`HOLD_JUDGE_AUDIT_COHORT`

这不是“模型失败”，也不是“没有信号”；是数据/裁判证据尚未达到可评价门。

## 当前证据

机器可读结果：[judge-audit-preflight-v6.json](../../../artifacts.local/evidence/eval-validity-r0/judge-audit-preflight-v6.json)。

| 项目 | 当前值 | 解释 |
|---|---:|---|
| 去重 source-session 候选 | 78 | 足够开始小型盲审，但尚未成为 event truth |
| selection discovery 记录 | 109 | 含 47 条跨 discovery 重复 session |
| normal discovery 记录 | 16 | 严格 normal source-mask shortlist |
| profile | 31 center / 15 lateral / 18 step / 14 normal | 只是 mask geometry profile |
| discovery arm | 78 source-mask | 正式至少还需一个独立 discovery arm |
| formal discovery mix | `NOT_ESTABLISHED` | 当前没有 random RGB、motion-temporal 或 metadata-only arm |
| 八类覆盖 | 0 类已确证 | 需要 RGB 盲审，不能由 profile 映射 |
| causal/retrospective reviews | 0 | 尚未产生动作真值 |
| 相似 YOLO 框反事实对 | 0 | 尚未测试物理风险敏感性 |
| oracle mask/depth/geometry/trajectory | 0 | 当前没有统一比较输入；正式 oracle 尚未开放 |

候选扫描本身只使用了有界的官方 train session；没有下载全量数据、训练模型、调参或改动 Android/默认 App。

另外完成了 1 个 50 帧 RGB/mask 链路探针：`rgb-probe-wz9-v1`，50 个 source frame、50 个唯一 RGB hash、50 个 pending review，`benchmark_ready=false`。它只证明下载/对齐/完整性链路可用；它是 burned calibration asset，不计入裁判分母，也不能填充正式八类配额。

当前准入状态明确为 `formal_review_access=false`。尚未开始的 burned calibration pilot 要求
8–12 个事件、3–4 对反事实、2 名 causal reviewer 加 1 名 retrospective reviewer；只提交
primitive fields，由机器规则派生 actionability。pilot 通过前不能冻结正式 cohort。

## 后续唯一正确动作

先从 78 个候选中 output-blind 抽取并烧录 8–12 个 pilot 事件；完成两名独立 causal reviewer
和一名 retrospective reviewer 后封存 review 与 bundle hash。此时不打开 YOLO/oracle trace，
也不直接生成 pair。review 封存后，才允许 YOLO 以 `SELECTION_ONLY` 读取框、尺度、位置和
标签无关的 `selection_time_slot`；deterministic 枚举并按冻结排序取前 3–4 对。pair-builder
不得读取 `reviewed_event_phase`、`reviewed_motion_relation` 或任何 derived label。若不足时
保持 `NOT_EVALUABLE/HOLD`，不得回头挑样本。之后才生成 native/system-chain traces 并运行四项
judge audit。depth、geometry、trajectory 缺失时保持 `NOT_EVALUABLE`。若 native oracle
能区分物理任务而 system chain 不能改善当前 YOLO，则报告
`FLAG_EVALUATION_STACK_CEILING_SUSPECTED`，不能简化为“oracle 没用”。

## Burned calibration pilot：当前执行收据

已按冻结的 output-blind screening 顺序抽取 8 个事件，并生成
`judge-burned-pilot-v1`：两份 causal 前缀可见 RGB packet、一份 retrospective 全序列 RGB
packet，以及隔离的 custodian review map。packet 只含 RGB 和 primitive response shape；不含
source mask、类别、source-session、YOLO、模型、oracle 或 action label。输入 RGB 逐帧通过
`continuous-native-input-plan-v2` 的 native MD5 receipt；本次使用的是 pilot-only 的
plan-bound staging asset，尚未声称正式 materialization/data-admission 已完成。

三个独立 primitive review 已封存，selection-only candidate universe 已生成；但 865 个
candidate 经冻结相似度门后没有 eligible pair，native/system-chain oracle 也没有生成。四项
审计报告见 [JUDGE_AUDIT_R0_BURNED_PILOT_RESULT_2026-08-02.md](JUDGE_AUDIT_R0_BURNED_PILOT_RESULT_2026-08-02.md)，当前 pilot 终态为
`STOP_JUDGE_AUDIT_FAILED`：test 4 的 visibility primitive 在 480 个 causal frame
comparisons 上完全不一致。test 1 为 `PASS`，test 2/3 为 `NOT_EVALUABLE`；这不构成模型、
YOLO 或 oracle 的优劣结论。pilot 不会填充正式八类 coverage，正式 `formal_review_access`
继续为 false。
