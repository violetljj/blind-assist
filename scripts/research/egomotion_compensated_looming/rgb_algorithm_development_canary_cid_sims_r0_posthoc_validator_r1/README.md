# CID-SIMS RGB canary posthoc validator R1

状态：development / frozen

## 研究问题与版本

协议 `RCLE_RGB_ALGORITHM_DEVELOPMENT_CANARY_R0_POSTHOC_VALIDATOR_R1` 只问 immutable
R0 output 是否满足 cache、ledger、pair identity 与 aggregate 一致性。它不重新
执行 RGB algorithm，也不重新裁决 R0 evidence validity。

## 稳定 Interface

从仓库根目录以模块方式调用 `run.py`，显式提供 repo、contract、implementation
lock、activation 和唯一 output。任何绑定、authority 或 self-lock 漂移均 fail
closed。

## 输出

只写入
`artifacts.local/evidence/rcle_rgb_algorithm_development_canary_r0_posthoc_validator_r1/`。

## 安全边界

最大权限是 R0 identity/cache/ledger/aggregate posthoc audit；禁止算法重跑、阈值
调整、独立确认、性能资格和产品/安全结论。

## 停止条件与失败复用

validation 非 `VALID` 即停止本版。R0 的 invalid validation 和本版复算均保留为
诊断、回归与审计资产，不得回写原终态。
