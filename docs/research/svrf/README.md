# SVRF current

状态：`current / WILD_LAB / SVRF_O0_PREOUTCOME_PROTOCOL_FROZEN / RGB_ONLY / A2D2_SPRING_TWO_SOURCE_EIGHT_PARENT_METADATA_LOCK_VALID / BLOCKED_ON_SELECTIVE_MATERIALIZATION_AND_TRUTH_WRITER_LOCK / REAL_O0_NOT_RUN / NO_TRAINING / DEFAULT_APP_UNCHANGED`

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
- 当前只取得官方 checksums/license/API 与 Spring camera metadata，未下载或解码 RGB/depth/LiDAR，
  未冻结 causal windows 或 truth writer，故 outcome access 继续关闭，`REAL_O0_NOT_RUN`；
- A2D2 是车载域且 5/8 parent 为 synthetic；未来 PASS 也不能写成助盲生态 confirmation。
  零 prior-use 只证明 BlindAssist 项目内 fresh，不能排除 DepthART/上游 foundation pretraining exposure；
  A2D2 payload 与派生帧不进入公共仓库，只允许本地分析和聚合指标。

## 既有资产边界

DepthART 只复用 relative shape/depth；旧 height、scale 和 D3R6 claim 不进入 SVRF。RCLE 只复用
Sparse-LK/local-affine/negative-control 思路；其 standalone rotation 与 warp-residual negative terminals
不改变。SATOM/TARO 只复用 UNKNOWN、matched-coverage、negative-control 和 no-rescue 纪律。

ADVIO 13/14/15/17 已用于 RCLE Development；sequence16 仍是 RCLE-reserved `SEALED_UNSEEN`，不分配
给 SVRF。Bonn、ARKitScenes、TUM、OpenLORIS 及所有已打开相关 outputs 也不能包装成 fresh O0。

## 唯一 successor

`SVRF_O0_LOCKED_PARENT_SELECTIVE_MATERIALIZATION_PREFLIGHT`：只从已锁 A2D2/Spring parent
物化 outcome-blind causal windows、RGB 与 evaluator truth 必需的最小文件，绑定逐文件 hash、时间戳、
缺失分母和独立 truth/candidate writer；在 activation 之前不运行 candidate 或读取 outcome。

## 禁止动作

- 不调 Bonn ground plane，不恢复 SATOM/RCLE，不运行 VITG/GA-SATOM；
- 不用 IMU、ToF、ARCore pose、已知高度、source pose/depth 帮 candidate；
- 不通过增加 UNKNOWN 购买 false-clear 改善；
- 不训练、不接 Android、不输出 `CLEAR`、米制 clearance、物理 TTC 或安全结论。

默认 App 影响：`否`。
