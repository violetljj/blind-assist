# RCLE temporal-structure diagnostic R1

状态：`development / executed`

## 研究问题与版本

`RCLE_TEMPORAL_STRUCTURE_DIAGNOSTIC_R1` 在已开放的 ADVIO sequence13、14、15、17
固定 601-pair 窗上比较：unchanged R3 的高响应更符合可跟踪的周期性 pose/flow
结构，还是 blur、feature collapse 与 forward-backward failure。独立描述单位是
capture session，pair 是重复纵向测量。允许 claim 仅为多 session 描述性时间结构
证据与 Development 优先级。

冻结合同：
`docs/research/rcle/RCLE_TEMPORAL_STRUCTURE_DIAGNOSTIC_R1_CONTRACT_2026-07-28.json`。

## 稳定 Interface

- `extract.py --session --source-root --output-dir --contract`：Stage 1
  response-blind signed pose 与 flow/radial direction 提取；输入、runtime、session 和
  contract 不匹配即失败，输出目录已存在即失败。
- `analyze.py --direction-root --proxy-root --r3-runs-root --contract --output`：
  Stage 2 固定连接并产生逐 session 结果。
- `validate.py --direction-root --proxy-root --r3-runs-root --contract --analysis
  --receipt`：不导入 production summarizer 的独立复算。

sequence16 和非 13/14/15/17 session 均拒绝。

## 输出

只写入
`artifacts.local/evidence/rcle_temporal_structure_diagnostic_r1/`，包括四份
direction ledger/summary、`session_analysis_r1.json` 与 validation receipt。

## 安全边界

不修改 R3、strict `>0.01/s`、三-pair、窗口或 PairState；不访问 sequence16、
风险/障碍/人工 gait 标签；不运行 Android。不能识别 normal gait、false alert、
障碍/风险状态、因果、泛化、产品或安全性。

## 停止条件

- 输入/hash/pair identity/firewall mismatch：`INVALID`；
- 少于三个 session 同时有至少四个周期和 70% direction coverage：
  `NOT_EVALUABLE`；
- 其余执行冻结的 motion / quality / mixed 三路终态，不得按输出调频带、阈值或
  换窗。

## 假设与规则质疑

候选差异是“track-supported 周期结构”对“measurement failure 时间聚集”；预期信息
增益是停止把低 feature count 当统一解释。falsifier 是 flow 不与 pose 主频同步，
或高响应不与 failure 共现。代价为四个已开放 10 秒窗的一次 flow-direction 提取。
正式输出前的合成径向场检查证明全局 median flow 会错误排斥 radial expansion，
因此冻结协议同时保留全局方向与 radial direction；正式输出后不再改定义。

## 失败资产复用

无论终态为何，direction ledger、周期/collapse 合成测试和独立 validator 可作为
Development negative evidence、回归 fixture 与 source characterization；不得包装为
unseen confirmation、风险性能或产品证据。
