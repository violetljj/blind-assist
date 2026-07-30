# dual_loop_d0_egomotion_error_attribution_r1

状态：design review pass / implementation complete / implementation lock pending /
formal not authorized

## 研究问题与版本

`D0_EGOMOTION_ERROR_ATTRIBUTION_R1` 只在已经烧毁的单个 REveL Dynamic capture
内，为下一次受控实验给出 operational priority。它不识别 dominant causal
mechanism，不产生有效性、泛化、产品或安全证据。

## 稳定 Interface

稳定根 Adapter：

```powershell
E:\codex-tools\bin\blindassist-python.cmd `
  scripts/run_dual_loop_d0_egomotion_error_attribution_r1.py `
  freeze-dependency `
  --natural-events artifacts.local/evidence/dual-loop/target-track-causal-radial-geometry-lite-r0/input-freeze/natural_events.jsonl `
  --output artifacts.local/evidence/dual-loop/d0-egomotion-error-attribution-r1/input-freeze/dependency_receipt.json
```

实现锁校验、一次性 producer 与完全独立 validator 也只通过根 Adapter 暴露：

```powershell
E:\codex-tools\bin\blindassist-python.cmd `
  scripts/run_dual_loop_d0_egomotion_error_attribution_r1.py `
  create-implementation-lock --repository-root E:\linnan\linnan `
  --output <implementation_lock.json>

E:\codex-tools\bin\blindassist-python.cmd `
  scripts/run_dual_loop_d0_egomotion_error_attribution_r1.py `
  validate-implementation --implementation-lock <implementation_lock.json>

E:\codex-tools\bin\blindassist-python.cmd `
  scripts/run_dual_loop_d0_egomotion_error_attribution_r1.py `
  activate --repository-root E:\linnan\linnan `
  --implementation-lock <implementation_lock.json> `
  --implementation-review <implementation_review.json> --output <activation.json>

E:\codex-tools\bin\blindassist-python.cmd `
  scripts/run_dual_loop_d0_egomotion_error_attribution_r1.py `
  produce --activation <activation.json> --implementation-lock <implementation_lock.json>

E:\codex-tools\bin\blindassist-python.cmd `
  scripts/run_dual_loop_d0_egomotion_error_attribution_r1.py `
  validate-execution --run-root <run-r1> --protocol <protocol.json> `
  --dependency-receipt <dependency_receipt.json> --write-results
```

implementation review 必须是 canonical JSON receipt，精确绑定当前 protocol、
implementation lock、`HEAD == origin/master`，所有独立检查均为 `passed=true`，且
receipt 自身不授予 formal authority。`produce` 在 implementation lock、activation、
review、clean `HEAD == origin/master`、
冻结输入、前序 seal 与正式 namespace 缺失任一不满足时，必须在 formal marker 前
失败。当前尚无 activation，不得运行正式命令。

## 输出

dependency preflight 只写
`artifacts.local/evidence/dual-loop/d0-egomotion-error-attribution-r1/input-freeze/dependency_receipt.json`。
正式 one-shot 在 activation 后只写冻结的 `run-r1/` namespace：
`formal_start.json`、`event_table.jsonl`、`analysis.json`、
`producer_receipt.json`、`execution_validation.json`、
`execution_receipt.json` 与 `progress.json`；post-start failure 只写 consumed
`failure_receipt.json`，不得重跑。

## 安全边界

dependency preflight 只读取冻结的 `natural_events.jsonl`。正式 producer 只读协议
列出的 burned REveL/R2/Vicon 输入；production A/B 只作 eligibility hash binding，
trace 内容、旧 F-1B decision、Confirmation 与任何新数据均禁止读取。独立 validator
不得导入 producer 或 analysis，并须从原始冻结输入重算 Vicon、ROI/flow、469-event
table、missingness、Cliff、路由与唯一出口。

## 停止条件

输入哈希、1660/469 分母、唯一 event ID、单 capture、target 集、时间区间、
`159` 个跨 target overlap pairs、`0` 个同 target overlap pairs、`310` 个
components、component size distribution 或六个 60 秒块任一漂移即失败且不发布
receipt。implementation lock、activation、Git parity、source/BBOX closure、
canonical serialization、独立 exact comparison 任一失败即 fail closed。输出已存在
时拒绝覆盖；formal marker 后的任何失败均为 `CONSUMED / NO_RERUN`。

## 假设与规则质疑

parent event 仍只是同一 capture 内的 observational analysis unit，不是 469 个独立
样本。overlap component 处理局部跨 target 依赖，60 秒块只做固定会话非平稳性
敏感性；二者都不能把单 capture 升级为总体推断。

## 失败资产复用

失败 receipt 只能定位冻结事件身份或依赖结构漂移，不得转写为几何机制、算法效果或
后继优先级证据。
