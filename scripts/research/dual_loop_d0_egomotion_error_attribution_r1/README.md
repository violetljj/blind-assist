# dual_loop_d0_egomotion_error_attribution_r1

状态：design review pass / dependency preflight valid / implementation authorized / formal not authorized

## 研究问题与版本

`D0_EGOMOTION_ERROR_ATTRIBUTION_R1` 只在已经烧毁的单个 REveL Dynamic capture
内，为下一次受控实验给出 operational priority。它不识别 dominant causal
mechanism，不产生有效性、泛化、产品或安全证据。

## 稳定 Interface

当前只开放 dependency preflight：

```powershell
E:\codex-tools\bin\blindassist-python.cmd `
  scripts/research/dual_loop_d0_egomotion_error_attribution_r1/freeze_dependency_receipt.py `
  --natural-events artifacts.local/evidence/dual-loop/target-track-causal-radial-geometry-lite-r0/input-freeze/natural_events.jsonl `
  --output artifacts.local/evidence/dual-loop/d0-egomotion-error-attribution-r1/input-freeze/dependency_receipt.json
```

正式 producer、analysis、validator、implementation lock、activation 与 runner
尚不存在。设计复核只授权实现与测试，不授权 activation 或正式执行。

## 输出

只写
`artifacts.local/evidence/dual-loop/d0-egomotion-error-attribution-r1/input-freeze/dependency_receipt.json`。
receipt 绑定 469 个 primary parent events、闭区间跨 target overlap pairs、
transitive overlap components，以及从最早事件起点固定的 60 秒 time blocks。

## 安全边界

preflight 只读取冻结的 `natural_events.jsonl`。它不读取 R2 candidate rows、Vicon
bag、production A/B trace、旧 F-1B decision、Confirmation 或任何新数据，也不运行
候选算法或生成科学出口。

## 停止条件

输入哈希、1660/469 分母、唯一 event ID、单 capture、target 集、时间区间、
`159` 个跨 target overlap pairs、`0` 个同 target overlap pairs、`310` 个
components、component size distribution 或六个 60 秒块任一漂移即失败且不发布
receipt。输出已存在时拒绝覆盖。

## 假设与规则质疑

parent event 仍只是同一 capture 内的 observational analysis unit，不是 469 个独立
样本。overlap component 处理局部跨 target 依赖，60 秒块只做固定会话非平稳性
敏感性；二者都不能把单 capture 升级为总体推断。

## 失败资产复用

失败 receipt 只能定位冻结事件身份或依赖结构漂移，不得转写为几何机制、算法效果或
后继优先级证据。
