# 双环真实几何 shadow cycle R0 结果

日期：2026-07-30（Asia/Hong_Kong）

## 结论

本轮第一次用真实录制场景的 LiDAR 支持几何走通现有
`DualLoopShadowAdmitter -> AssistDecisionKernel`：

```text
ENGINEERING_SHADOW_CYCLE_VALID / DIAGNOSTIC_ONLY / NO_EFFECT_CLAIM
```

但两个面向实时 RGB 的后继算法均未获得进入后续独立信息 screen 的资格：

```text
DEPTH_TEMPORAL_SOURCE:
  TEMPORAL_SOURCE_NOT_SUPPORTED_IN_THIS_DISCOVERY

GMC_FLOW_SOURCE:
  SOURCE_READINESS_NOT_MET /
  INDEPENDENT_INFORMATION_NOT_EVALUATED /
  DEVELOPMENT_ONLY

LIVE_ANDROID_SOURCE:
  NOT_IMPLEMENTED / NOT_AUTHORIZED
```

因此，工程环已经从“空接缝”推进到真实源回放闭环；科学/算法环仍未获得可进入
CrowdBot 独立信息 screen 或 Android live shadow 的 source。不能据此声称提醒改善、
双环有效、产品落地或安全提升。

## 1. JRDB 标注条件化 LiDAR 工程环

冻结 source identity：

`JRDB_ANNOTATION_CONDITIONED_LIDAR_CENTROID_REPLAY_V1`

源使用 JRDB 标注确定 target identity 和 2D region，并在对应 3D box 内使用真实
LiDAR 点质心。正接近率固定为：

```text
(range_previous_m - range_current_m) / dt_seconds
```

availability 使用当前 RGB、lower LiDAR、upper LiDAR 三者时间戳最大值；时钟域是
`REPLAY_TIMELINE`，TTL 100 ms。它不是独立 person truth，也不能伪装成 live
object detector 证据。为实际走生产 object-detector 行为链，回放 detection 保持
`DetectionSource.OBJECT_DETECTOR`；标注条件化另由
`DualLoopTargetProvenance.REPLAY_ANNOTATION` 留痕。稳定 adapter 只接受 replay
时钟域，生产 allowlist 保持为空。

Producer 结果：

- 4 个真实场景，480 帧；
- 10,786 个 exact 3D-to-2D joined target rows；
- 8,836 个两端均为 `sensor-supported`、且两端都存在 exact 2D join 的几何行；
- 151 个 stitched panorama 边界框被确定性 clamp，并逐 sequence 计数；
- protected alert outcome 未读取。

真实 JVM kernel replay 结果：

- 476 个实际决策帧，baseline 每帧选择一个 target；
- 474/476 个选中 target 实际得到 `ADMITTED_SHADOW`；
- 2/476 为 `EVIDENCE_ABSENT`；
- adapter abstention 0；
- JVM runner 在读入 replay row 前先校验 producer receipt identity、implementation
  SHA、输入 SHA 和 protected outcome 未打开，再从实际 TSV 重算 4 个 sequence、
  476 个 decision frames、10,786 行与 8,836 个 eligible rows；
- raw risk、stable risk、event、feedback、explanation、session summary 与 gateway
  call count 的逐帧差异全部为 0；
- baseline/shadow feedback gateway 均调用 476 次；
- event/feedback mutation 始终为 false。

证据：

- `artifacts.local/evidence/dual-loop/jrdb-shadow-replay-r0/producer_receipt.json`
  SHA-256
  `8b9ba38c22bc6f1975956e86c848677c813739ebe0a419a3cc7b6c20375a6240`
- `artifacts.local/evidence/dual-loop/jrdb-shadow-replay-r0/kernel_receipt.json`
  SHA-256
  `d25fa5aa5b39aae0c1627f8121b267e06db284233fc030f4b6ab06cb5cfd5fcf`

## 2. Depth Anything V2 target-depth Discovery

在既有 burned REveL 512-frame / 770 GT ROI 固定子集上，producer 只读 RGB 和
oracle ROI；Vicon truth 由 separate post-producer evaluator 后开。模型固定为
`depth-anything/Depth-Anything-V2-Small-hf` revision
`5426e4f0f36572d16453bbda7a8389317b1bef99`，未做阈值搜索。

结果：

- target depth 与物理 range 的 Spearman 为 `-0.7498`（n=488），说明静态距离
  排序含信息；
- frame-normalized target depth 与 range 的 Spearman 为 `-0.7533`（n=488）；
- 但 temporal rate 与 truth approach rate 的 Spearman 只有 `0.1883`（n=408）；
- 零 deadband 方向正确 `0.4902`、wrong-signed `0.2941`（n=408）。

这说明单帧相对深度可作距离层候选，但在本单一 burned capture/oracle ROI
Discovery 中，直接帧差没有支持 signed motion source。该 Discovery 没有执行
469-event readiness，也没有进入独立信息 screen。
证据
`artifacts.local/evidence/dual-loop/target-depth-geometry-discovery-r0/evaluation.json`
SHA-256 为
`e08d1cf7f5a837630f751b10b60707b0c9cdf6f7147cc9f748de4a5985196b91`。

## 3. 全局运动补偿 target-flow Discovery

针对 LITE R2 的无补偿 ROI flow，本轮固定一个机制后继：

1. 在所有 oracle target ROI 外检测背景特征；
2. LK 双向跟踪后用 RANSAC homography 表示相机/背景运动；
3. 先把 target tracks 映射到 homography 预测位置；
4. 只对实际 target endpoint 相对预测位置拟合 residual similarity scale；
5. `log(scale)/dt` 作为 signed approach rate，quality <0.50 直接 abstain。

Producer truth-blind 完成 13,014/13,014 行，11,381 行可用，运行 177.86 秒；随后
separate post-producer evaluator 使用原 469 个 primary natural events、0.02/s
deadband 和原 readiness floor。

| 指标 | LITE R2 原 ROI flow | 补偿 flow R0 |
|---|---:|---:|
| 正确事件 | 188/469 (40.1%) | 233/469 (49.7%) |
| wrong-signed | 161/469 (34.3%) | 91/469 (19.4%) |
| evaluable | 未作为本轮比较门 | 452/469 (96.4%) |

补偿在同一冻结事件集上描述性减少反号，但没有达到正确率 `>=60%`，且
truth-state 仍不均衡：

- approaching：100/200 正确，61/200 wrong-signed；
- quasi-static：23/92 正确；
- receding：110/177 正确，30/177 wrong-signed。

终点：

```text
SOURCE_READINESS_NOT_MET /
INDEPENDENT_INFORMATION_NOT_EVALUATED /
DEVELOPMENT_ONLY
```

这 469 个事件含 159 个跨 target 重叠 pair、310 个 overlap component，且全部来自
同一 capture。本轮没有依赖感知区间或显著性检验；同一 REveL truth 也已被前序工作
打开，所以这里只能称 retrospective Development，不能称 outcome-naive one-shot、
统计显著改善或独立信息。

Producer receipt SHA-256：
`670306e50abdc8d6da1c6a416eb023bbb383cb94c6679138b4b51fc9e54f01bd`

Evaluation SHA-256：
`96741d2fb7cbd986d99f31ad82933c4334cf220fa01ecbfc4ba3d72dd93a853c`

## 4. 主线决定

当前不把 bbox、原 ROI flow、直接 depth derivative 或本轮 compensated flow 加入
Android source allowlist，也不做事后 deadband 搜索。最有根据的新方向是：

1. 用陀螺仪/相机时钟显式消除旋转，再以背景 homography 处理剩余像面运动；
2. 将单帧相对深度只用于静态 near/far 层，不直接差分；
3. signed motion 仅在多帧同号、质量和 target continuity 同时满足时发出证据，
   其余明确 abstain；
4. 先在独立 source/session 复核 source-level direction，再进入冻结的
   `REAL_SHADOW_CYCLE_R0` 独立信息 screen；
5. screen 仍要求 negative correction >=2、两个 session 各>=1、positive harm=0，
   且不改变旧 F-1B、生产 A/B 或 D0 的 consumed 终态。

在出现满足这些条件的新 source 前，双环主线的诚实状态是：

```text
ENGINEERING_RING: LANDED_IN_REAL_SCENE_REPLAY_SHADOW
ALGORITHM_RING: IMPROVED_BUT_NOT_READY
ACTIVE_EFFECT: NOT_AUTHORIZED
PRODUCT_OR_SAFETY_CLAIM: NONE
```
