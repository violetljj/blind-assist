# RISKSEG-R0 event-eval 数据门结果

状态：`COMPLETE / HOLD_EVENT_EVAL_DATA / EVENT_TRUTH_NOT_FROZEN /
PIDNET_PREFLIGHT_NOT_STARTED / TRAINING_NOT_STARTED / DEFAULT_APP_UNCHANGED`

日期：2026-08-01（Asia/Hong_Kong）
执行者：`violjjet`

## 结论

`RISKSEG-R0` 已按顺序执行到新 event-eval 数据门，并在这里 fail closed。两路隔离
RGB-only review 只有 14 个 parent events 达到同桶一致：

| bucket | 当前同桶一致 shortlist | 合同下限 | 缺口 |
|---|---:|---:|---:|
| `blocking_obstacle_positive` | 7 | 8 | 1 |
| `boundary_level_change_positive` | 2 | 8 | 6 |
| `parallel_curb_negative` | 1 | 7 | 6 |
| `normal_walkable_negative` | 4 | 7 | 3 |
| 合计 | 14 | 30 | 16 |

因此没有冻结 event truth，不启动 PIDNet-S TFLite/QNN/SM-S9280 预检，不训练模型，
也不打开 YOLO、PIDNet 或 truth-mask oracle 输出。默认 App 保持 YOLO baseline。
这是合同定义的 `HOLD_EVENT_EVAL_DATA`，不是等待训练授权。

## 数据与隔离审计

排除 520 train/dev 与固定 90-frame regression 的并集后，禁用集合为 11 个 native
source sessions。固定 90-frame 集自身只有 3 个 parent events，并与 train 共享两个
sessions，仍只作 contaminated non-gating regression smoke。

本地盘点找到 27 个 session 的完整 RGB/source-mask 素材和 29 个唯一候选窗口；目录名与
mask profile 只作 shortlist。首轮两路盲审严格要求正例同时具有可辨的 alertable 与
passed phase，负例全窗无行进区侵入，结果只有 9 个同桶一致事件。

随后执行 SANPO official test split 的 output-blind sparse mask 扫描：

- 48 个 source sessions；
- 原始 51 个候选 / 30 个 sessions；
- 排除合同禁用 session 后为 44 个候选 / 26 个 sessions；
- 14 个 boundary broad candidates 进入精确窗口 mask gate，13 个通过、1 个拒绝。

本轮新物化 9 个完整 draft，共 750 RGB + 750 source-mask frames；每份
`manifest_validation.ok=true`，但在事件真值冻结前全部
`benchmark_ready=false`。新窗口按 50 帧或 100 帧全窗生成 contact sheet，并由两路互不
可见的 reviewer 在 PIDNet/YOLO/oracle 输出关闭时复核。三批新增 review 最终只增加
3 个障碍、1 个落差和 1 个正常事件；长窗不能自动解决持续侵入或多事件重叠。

一个 `-PqSD...` 下载尝试因 leading-hyphen CLI value 未物化。本轮不重跑：即便它补齐
唯一的障碍缺口，也不能弥合落差 6、平行路沿 6、正常通行 3 的缺口，不会改变
`HOLD_EVENT_EVAL_DATA`。

## 证据边界

14 个事件只是“两路 RGB review 同桶一致 shortlist”，不是 frozen truth：正例区间仍需
裁决绑定，所有事件仍需进入统一 candidate index、review receipts、adjudication 与 cohort
validator。没有用目录名、source-mask 类别、模型输出或单路 review 替代真值。

本地 evidence：

- `artifacts.local/evidence/riskseg-r0/event-eval/discovery-v1/batches/`
- `artifacts.local/evidence/riskseg-r0/event-eval/review-v1/full/`
- `artifacts.local/evidence/riskseg-r0/event-eval/new-review-batch-01/`
- `artifacts.local/evidence/riskseg-r0/event-eval/new-review-batch-02/`
- `artifacts.local/evidence/riskseg-r0/event-eval/new-review-batch-03/`

关键 index 与 scan SHA-256 已写入
[machine result](RISKSEG_R0_EVENT_EVAL_DATA_GATE_RESULT_2026-08-01.json)。

## 后继边界

只有取得至少 `1/6/6/3` 个新的、同样 source-session-disjoint 的
`blocking/boundary/parallel/normal` parent events，并重新完成 output-blind 双审与必要
裁决，才允许再次运行 cohort freeze validator。不得在已拒绝窗口上放宽 passed/侵入标准，
不得用随机帧拆分、同 session 重复窗口、像素 IoU 或 90-frame regression 替代事件门。

当前停止点之后，PIDNet-S 预检、三 seed 训练、三臂事件评价和默认 App 替换全部未执行。
