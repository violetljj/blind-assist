# SVRF-O0

状态：`ACTIVE_PREOUTCOME_PROTOCOL / RGB_ONLY / FRESH_PARENT_SOURCE_LOCK_NOT_ACTIVATED / REAL_O0_NOT_RUN / NO_TRAINING / DEFAULT_APP_UNCHANGED`

## 稳定 Interface

- candidate 只读取有序 RGB 与由这些 RGB 确定性派生的 relative depth、visual warp、flow、
  image-space channel 和 causal history；
- ToF、IMU、ARCore pose、已知高度、metric ground、source depth/pose 与未来帧均禁止进入 candidate；
- [`core.py`](core.py) 只实现 robust background scale-shift alignment、aligned relative-depth
  approach rate、rotation-compensated local expansion 与冻结无训练融合规则；
- [`evaluation.py`](evaluation.py) 强制 A0–A3/N0–N3 exact ledger、parent/source macro、matched
  coverage、负控退化和 UNKNOWN 非增益。
- [`validate_source_lock.py`](validate_source_lock.py) 校验 A2D2/Spring 官方 metadata hash、许可、
  archive checksum、Spring deterministic 5-parent selection、pre-lock prior-use 与 3+5 denominator；
  它不读取 RGB/depth/LiDAR 或 candidate outcome。

## 输出

输出状态只有 `NO_HIGH_RISK_EVIDENCE / APPROACHING / PATH_INTRUSION /
HIGH_RELATIVE_RISK / UNKNOWN`。`NO_HIGH_RISK_EVIDENCE` 不是 `CLEAR`，不输出米制 clearance 或
身体可通行结论。真实输入、truth、outputs 和 receipts 只能放在
`artifacts.local/evidence/svrf-o0/`。

## 安全边界

本模块的 synthetic tests 只证明公式、fail-closed 状态与 evaluator mechanics。旧 RCLE rotation、
warp residual、ADVIO/Bonn/ARKit/TUM outcome 均不成为 SVRF 证据；source-native depth/pose 只允许
在 evaluator firewall 后生成标签。O0 不训练、不接 Android、不产生提醒，也不证明真实 TTC、
身体距离、clearance、产品或安全。

## 停止条件

A3 必须在 matched coverage 上降低 best single-arm false-clear、控制 false-block、在至少 6 个
parent 上同向、保持覆盖，并让 N0–N3 明显退化。任一门失败即
`FAIL / CLOSE_SVRF_O0 / NO_TRAINING`；PASS 也只允许另立 learned/deployment successor，不自动执行。
