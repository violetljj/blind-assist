# GRAIL M1 Frozen-Visual Interaction-Pose Protocol

日期：2026-08-25（Asia/Hong_Kong）

状态：`FROZEN_BEFORE_VISUAL_COLLECTION_OR_OUTCOME / BUILDING_DISJOINT / INSTANCE_DISJOINT / B0_B1_B2_GRAIL / ONE_SHOT_TEST / STOP_BEFORE_M2_ON_FAIL`

## 问题与边界

M0 V2 已建立 synthetic native-teacher 上界，故 M1 只问：在冻结视觉 encoder、未见 house 与未见 instance 上，factorized set-valued GRAIL 是否明显超过简单 waypoint baseline，同时不增加 wrong-target 与 absence false commit。

本轮使用 ProcTHOR synthetic RGB、AI2-THOR native interaction-pose truth 和 simulator oracle instance masks。mask 只提供所有可见 stationary actionable candidate proposals；模型仍需用独立参考视角选目标并预测 pose。结论不得外推到 proposal、文本目标、自然场景或真实设备。

## 冻结 split 与模型

- train：ProcTHOR val 24 houses；dev：val 6 houses；test：test 12 houses；house 与 object instance 完全分开；
- test 排除 Development index 0、M0 V1 与 M0 V2 全部已消费 test houses；
- train indices：`194,871,930,688,671,749,640,949,775,876,892,544,694,325,987,553,814,463,404,381,945,337,606,684`；
- dev indices：`681,458,165,371,725,860`；
- test indices：`652,310,516,344,791,439,144,41,187,38,184,278`；
- frozen visual encoder：DINOv2-S weights SHA-256 `ae1e99fcefd534ed978cdeb8326f08030c96e28b7a81ffcbc98a857c84d14be1`；
- B1-only frozen relative depth：Depth-Anything-V2-S weights SHA-256 `3152477ce0d8d6978d76b995120de97cb5b928701fd0f817769f59e249a16b70`；
- encoder 全程冻结；只有轻量 heads 可训练。dev 只选择阈值/epoch，test outcome 不参与选择。

完整 house hashes 与 source identity 见 [`grail_m1_manifest_v1.json`](../../../scripts/research/grail/grail_m1_manifest_v1.json)，SHA-256 为 `012d8ae7b962206c812af36ac92a4cc4546109b0512b2ddd47d9965491348a9e`。

## 自动样本

每个 positive 包含 320x240 query RGB、同一 instance 的独立 native-pose reference crop、query 中所有可见 actionable candidate masks，以及全部 truth poses 在 query camera frame 下的 `(x,z,yaw)`。query 起点固定为目标 1.75--4.0 m 内按 sample hash 排序后首个目标可见 reachable position。absence 将 reference 与 different-house query 配对，因此 exact instance 必然不存在。wrong-target 分母要求 query 中至少两个与目标同类型的 candidate。

## 对照

- B0：DINO crop/reference cosine 选 candidate，bbox bearing + fixed 1 m waypoint；
- B1：同一 referent 选择，mask + frozen monocular relative depth（只在 train 标定单一尺度）得到 target range，再给最近前方 waypoint；
- B2：query/reference frozen feature 直接回归单一 waypoint + presence；
- GRAIL：每个 candidate 独立 referent head、validity head 与 K-set `(x,z,yaw)` head；candidate set 只用 permutation-invariant max/abstain 汇合。

主指标为任一预测 pose 在 truth set 的 0.5 m 与 20° 容差内。另分别报告 wrong-target pose rate、absence false commit 和候选排列一致性。

## 正式门

- test positives >=96，wrong-target cases >=24，absence cases >=96；
- GRAIL Interaction Pose Success 比 B0/B1 中最强简单 baseline 至少高 10 个百分点；
- GRAIL wrong-target rate 不得比该最强简单 baseline 高超过 2 个百分点；
- GRAIL absence false commit 不得高于该 baseline；
- candidate permutation consistency=100%。

任一门失败即 `STOP_BEFORE_M2`；不在 test roster 调 threshold、loss、K、head、采样或 reference 规则后重跑。通过才允许 M2 短时 belief、遮挡重捕获与 Android 三个未见现实环境。

Claim ceiling：synthetic ProcTHOR RGB + oracle candidate masks + simulator-native interaction-pose truth，reference-goal mode；无自然场景、proposal、text-goal、Android、用户、产品或安全结论。
