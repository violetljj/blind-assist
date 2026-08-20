# GOAL-COPILOT-2 observability and reality audit result

Terminal status: `COMPLETE_ZERO_MODEL_CONSUMED_DIAGNOSTIC`.

Decision: `A_STOP_SYNTHETIC_MODERATE_AS_OPTIMIZATION_TARGET_AND_MOVE_TO_REAL_RGB_EVIDENCE`.

本审计只重放已消费的 GC2 development scenarios、冻结 simulator 和 GC2-B 锁定的公开
development winner。模型调用为 `0`；GC2 held-out 未读取、未解密、未评价。冻结 evaluator、
noise contract、winner 和既有结果均未修改。

## Failure autopsy

GC2-B winner 在 `COMBINED_MODERATE` 的 `12/12` episode 仍全部 timeout，completion `0/12`，
三个 family 均 `0/4`。逐 episode 的首个错误 action 分类为：

| 首次偏离机制 | Episodes |
|---|---:|
| stale evidence 指向错误 hidden step | `4/12` |
| tracking collapse 强制 target loss | `3/12` |
| target dropout 强制 target loss | `2/12` |
| bearing jitter 绕过 alignment | `2/12` |
| false target 导致反向搜索 | `1/12` |

失败没有集中到一个 confidence、completion 或 recovery gate。它在第一次动作或第二次动作就已覆盖
多种机制，因此不能从 `0/12` 反推“缺一个 uncertainty-aware representation”或“只需修一个 filter”。

## Counterfactual replay

在完全相同的 simulator 和 winner 上逐项关闭一个 moderate corruption，结果为：

| Cell | Completion | Wrong-way actions |
|---|---:|---:|
| all moderate | `0/12` | `65` |
| without bearing jitter | `1/12` | `59` |
| without delayed evidence | `0/12` | `64` |
| without false target | `0/12` | `63` |
| without nearness error | `0/12` | `65` |
| without target dropout | `0/12` | `62` |
| without tracking collapse | `0/12` | `65` |

没有单项 ablation 使结果恢复到“明显可解”。`COMBINED_MODERATE` 的断崖来自固定顺序组合后的
simulator dynamics；本结果不支持把一个 corruption 指定为真实视觉主因，也不支持围绕其中一个机制
继续开 policy rescue。

## Observability upper bounds

| Bound | Completion upper bound | 严格解释 |
|---|---:|---|
| A：hidden-state oracle | `12/12` | scenario/runtime 本身可完成 |
| B：完整 noisy history lookup | `12/12` | 45 个 consumed history state 无 exact action collision |
| C：当前六函数 surface lookup | `12/12` | 2,010 AST nodes，通过冻结 candidate contract |

B/C 是有限、确定性、已消费场景上的 lookup oracle。C 在第一帧识别 scenario trajectory，随后用 scalar
belief 编码精确动作序列，是刻意的 simulator memorization；它只证明当前语法不是这 12 条轨迹的绝对
表达上限。它不建立可迁移的 history identifiability、belief module、robust policy 或 search signal，不能
作为 GC2-C、representation ladder 或新 Sky search 的正证据。

## Real-RGB grounding

当前能复用的最接近证据是 `4,422` 帧已消费真实世界 RGB 经 Android production detector 路径的
device replay：`4,411` 帧有至少一个 detection，最长“无任何 detection”连续段为 `8` 帧，detection
confidence median `0.5791`，detector total latency median/max 为 `12/53 ms`。

这不是 real-phone camera capture，也没有 goal-target identity truth、calibrated bearing、tracking-ID
loss/reacquisition truth、relative-nearness truth 或 capture→Goal-Copilot observation timestamp。因而：

`REAL_PHONE_RGB_NOISE_GROUNDING_NOT_EVALUABLE`。

尤其 GC2 的 delayed evidence 以“前一个 symbolic action/hidden step”计量，而真实 device evidence 以
milliseconds 和 latest-frame scheduling 计量；两者没有可验证的映射。不能声称 GC2 moderate 落在真实
手机 RGB 的合理范围内。

## Decision and boundary

审计选择路线 A：停止把 synthetic `COMBINED_MODERATE` 当作优化目标，Goal Copilot policy search
继续关闭；下一条有意义的路线是独立 real-phone RGB target-evidence capture/audit，而不是 policy、
representation 或 search successor。

该真实证据工作尚未获得执行权限。执行前必须另行冻结 source、goal-target identity/truth、逐帧
detector/tracker 字段、camera/timing 映射、privacy、bounded clip roster 和 diagnostic-only claim ceiling。

GC2-C、held-out opening、追加 Sky/模型调用、扩预算、consumed representation ladder、产品与安全
主张均不授权。Claim ceiling 为
`consumed_symbolic_diagnostic_plus_non_phone_public_rgb_device_proxy_only`。

机器可读结果位于
`artifacts.local/evidence/goal-copilot/GOAL-COPILOT-2-OBSERVABILITY-AUDIT/result.json`，SHA-256 为
`39038e604a85947d2cfff8a5017cca307a7d301e95c50cc283b454aa927fc64a`。
