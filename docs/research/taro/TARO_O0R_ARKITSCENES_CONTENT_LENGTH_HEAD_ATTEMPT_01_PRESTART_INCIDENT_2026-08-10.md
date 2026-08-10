# TARO O0R Content-Length HEAD Attempt 01 Pre-start Incident

状态：`PRESTART_FAILED_UNCONSUMED_SUPERSEDED`

Attempt 01 在任何 HEAD 调用之前停止。路径守卫把仓库授权的
`E:/linnan/linnan/artifacts.local` junction 解析到 `F:/ba-data/blindassist-artifacts-20260805` 后，
误判为 `PATH_ESCAPE`。失败点位于 HEAD output root 解析阶段；root 尚未创建。

因此本次 `HEAD/GET/body/source/truth/model-output = 0/0/0/0/0/0`，HEAD 与 truth one-shot 均未消费。
Attempt 01 不得原地重跑；其锁和失败保留在版本历史中。

唯一允许的修正是：仅对受信任的 `artifacts.local` namespace 做 junction-aware containment，receipt
继续保留仓库内 lexical path；增加回归测试、刷新 implementation lock，再提交独立 Attempt 02 锁。
