# SVRF current

状态：`current / PAUSED_BY_D_ORACLE_1_UNIQUE_P0 / WILD_LAB / RGB_ONLY / A2D2_SPRING_SOURCE_LOCK_VALID / ARCHIVE_ACCESS_PARTIAL_PASS / STREAM_INDEX_AUTHORIZED_BUT_NOT_ACTIVE / REAL_O0_NOT_RUN / NO_TRAINING / DEFAULT_APP_UNCHANGED`

Failure Synthesis 已把 D-ORACLE-1 设为唯一 P0，要求先定位 downstream target-policy stack 与
representation 的损失。SVRF 的协议、source lock、archive capability 与 stream-index lock保持有效，
但当前不构成执行权限；这不是 SVRF 科学失败，也不消费其 roster。

## 主张

SVRF-R0 不恢复 metric ground、camera height 或身体 clearance。它只问：纯 RGB 派生的
background-aligned relative-depth dynamics、rotation-compensated local expansion 与 image-space
path intrusion，是否能稳定识别“相对逼近且侵入前向视觉通道”的区域。

输出 `NO_HIGH_RISK_EVIDENCE` 而不是 `CLEAR`；不声称米制距离、物理 TTC 或一定可通行。

## O0

- [Frozen protocol](SVRF_O0_SCALE_FREE_VISUAL_RISK_FIELD_PROTOCOL_2026-08-15.json) 固定 A0–A3、
  N0–N3、parent/source macro、matched coverage、UNKNOWN 和 kill gate；
- [Executable mechanics](../../../scripts/research/svrf_o0/README.md) 实现 scale-shift alignment、
  aligned depth change、local expansion、无训练融合与 evaluator；
- source-native depth/pose/geometry 只在 evaluator firewall 后构造 approach/ranking truth，不能进入
  candidate；
- [Source lock](SVRF_O0_A2D2_SPRING_SOURCE_LOCK_2026-08-15.json) 已冻结 A2D2 三条真实连续
  drive 与 Spring v2 五条 synthetic stress sequence；pre-lock tracked prior-use 与总账命中均为 0；
- [Archive capability lock](SVRF_O0_ARCHIVE_ACCESS_CAPABILITY_LOCK_2026-08-15.json) 已证明 Spring
  ZIP64 可用 Range 建 central index 并做 CRC-valid member 抽取；A2D2 官方桶只有 TAR、无独立
  PNG/NPZ 对象，必须先实现一次性流式 member index，不能假装随机抽取已就绪；
- [Stream-index execution lock](SVRF_O0_STREAM_INDEX_EXECUTION_LOCK_2026-08-15.json) 已按用户明确
  授权开放 outcome-blind A2D2 index 与 Spring manifest；selected payload、truth writer、candidate 和
  outcome access 仍关闭；
- 当前除 5 个 bounded ZIP-member CRC 样本外未物化 payload，且没有媒体/flow/disparity/label
  语义解码；causal windows 与 truth writer 未冻结，故 outcome access 继续关闭，`REAL_O0_NOT_RUN`；
- A2D2 是车载域且 5/8 parent 为 synthetic；未来 PASS 也不能写成助盲生态 confirmation。
  零 prior-use 只证明 BlindAssist 项目内 fresh，不能排除 DepthART/上游 foundation pretraining exposure；
  A2D2 payload 与派生帧不进入公共仓库，只允许本地分析和聚合指标。
- truth 与 candidate 的 `UNKNOWN` 均须保留 exact identity；winner coverage 固定为“truth 有效且
  candidate 有效 / 全部锁定 identity”。核心指标缺少任一 parent 支撑时为 `NOT_EVALUABLE`，禁止
  静默跳过该 parent；candidate 内参政策固定为 `FIXED_CANONICAL_OR_RGB_DERIVED`，source-native
  camera intrinsics 仅可位于 evaluator firewall 后。

## 既有资产边界

DepthART 只复用 relative shape/depth；旧 height、scale 和 D3R6 claim 不进入 SVRF。RCLE 只复用
Sparse-LK/local-affine/negative-control 思路；其 standalone rotation 与 warp-residual negative terminals
不改变。SATOM/TARO 只复用 UNKNOWN、matched-coverage、negative-control 和 no-rescue 纪律。

ADVIO 13/14/15/17 已用于 RCLE Development；sequence16 仍是 RCLE-reserved `SEALED_UNSEEN`，不分配
给 SVRF。Bonn、ARKitScenes、TUM、OpenLORIS 及所有已打开相关 outputs 也不能包装成 fresh O0。

## 唯一 successor

无。状态为 `NONE / PAUSED_BY_D_ORACLE_1_UNIQUE_P0`。只有 D-ORACLE-1 证明 representation layer
具有主要可恢复 headroom 且用户明确恢复 SVRF 后，才可另行授权既有 bus canary/member-index successor。
当前不运行 bus canary、archive index、payload materialization、truth writer或candidate。

## 禁止动作

- 不调 Bonn ground plane，不恢复 SATOM/RCLE，不运行 VITG/GA-SATOM；
- 不以既有 stream-index authorization 绕过 D-ORACLE-1 P0 暂停；
- 不用 IMU、ToF、ARCore pose、已知高度、source pose/depth 帮 candidate；
- 不通过增加 UNKNOWN 购买 false-clear 改善；
- 不训练、不接 Android、不输出 `CLEAR`、米制 clearance、物理 TTC 或安全结论。

默认 App 影响：`否`。
