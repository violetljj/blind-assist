# Goal-Driven Visual Copilot

状态：`current / PRODUCT_MAINLINE=GOAL_DRIVEN_VISUAL_COPILOT / ALGORITHM_MAINLINE=GRAIL_R / R1C_O_OWNER_LOCAL_CEILING_ESTABLISHED / R1C_P_PAIRED_RGB_ORIENTATION_FAILED / R1C_L_TASK_TRAINED_PAIRWISE_OWNER_COORDINATE_AUTHORIZED / FINAL_TEST_UNOPENED / STOP_BEFORE_DEPTH_GEOMETRY_AND_M2 / DEFAULT_APP_UNCHANGED`

本页只维护当前问题、执行边界和少量证据入口。日期化 protocol/result 是不可变证据；旧 handoff、
archive 和历史 successor 不产生当前权限。系统蓝图见
[`V2 路线图`](BLINDASSIST_GOAL_DRIVEN_VISUAL_COPILOT_V2_ROADMAP_2026-08-21.md)。

## 当前问题

BlindAssist 当前只推进 GRAIL-R：从目标条件和视觉观测预测目标一致、可达、可交互的
set-valued terminal pose。`referent / affordance / reachability / visibility / arrival` 保持分离。

Privileged [`R1C-O owner-local ceiling`](GRAIL_R1C_OWNER_LOCAL_CANONICAL_COORDINATE_RESULT_2026-08-25.md)
在冻结 78-case 上达到 referent=`75/78`、complete=`58/78`，说明 owner-local coordinate 能恢复
R0 relational uplift；它不证明 RGB/mask 可获得该坐标。

确定性 [`R1C-V`](GRAIL_R1C_V_VISUAL_OWNER_ORIENTATION_RESULT_2026-08-25.md) 与 fresh
[`R1C-P`](GRAIL_R1C_P_SYMMETRY_AWARE_PAIRED_ORIENTATION_RESULT_2026-08-25.md) 均未建立可靠视觉
orientation；不得在其 consumed cohort 调 matcher、crop、symmetry、threshold、selector 或 pose head。

## 当前唯一执行

唯一已授权 successor 是
[`R1C-L task-trained pairwise owner coordinate`](GRAIL_R1C_L_TASK_TRAINED_PAIRWISE_OWNER_COORDINATE_PROTOCOL_2026-08-25.md)：

- 160/20 个 house-disjoint ProcTHOR houses 用于 train/validation；12-house final 保持未打开；
- 输入为 RGB+mask，不加入 depth；只训练一个 DINOv2-S/cross-attention 架构，最多两个 seeds；
- validation 相对 frozen OA-V2 slot uplift `<+8` 时停止，不访问 final；
- validation 过门后才允许一次 final，之后仍停止在 M2、depth geometry 和 Android/App 之前。

输出只进入 `artifacts.local/evidence/grail/`；训练或 final 结果不得改写 R1C-O、R1C-V、R1C-P 的历史终态。

## 已关闭或历史化的证据族

| 证据族 | 保留结论 | 入口 |
|---|---|---|
| GRAIL M0/M1/R0/R1A/R1B | synthetic task ceiling、frozen-encoder gap、relational oracle 与 ownership/canonical-coordinate failure attribution | [M0](GRAIL_M0_ORACLE_INTERACTION_POSE_RESULT_2026-08-25.md) / [M1 V2b](GRAIL_M1_V2B_DEVELOPMENT_RESULT_2026-08-25.md) / [R0](GRAIL_R0_PRIVILEGED_RELATIONAL_ORACLE_RESULT_2026-08-25.md) / [R1B](GRAIL_R1B_BILATERAL_GROUPING_PROBE_RESULT_2026-08-25.md) |
| Public-real / exact-instance identity | public-goal truth、C2 small roster、passive appearance/layout 的 failure evidence；不恢复 P1 或 App | [public-real](BLINDASSIST_PUBLIC_REAL_EPISODE_MINING_V0_RESULT_2026-08-23.md) / [C2](BLINDASSIST_PUBLIC_IDENTIFIABLE_REFERENT_C2_SMALL_ROSTER_RESULT_2026-08-24.md) / [layout closure](SPATIAL_LAYOUT_IDENTITY_VERIFICATION_V0_RESULT_2026-08-24.md) |
| Semantic anchor / SAGE-R | controlled exact-anchor demo 可保留；natural authority graph 已关闭 | [semantic anchor V1](SEMANTIC_DISTINCTIVE_ANCHOR_V1_RESULT_2026-08-24.md) / [SAGE-R V3-C](SAGE_R_V3_C_AUTHORITY_TYPED_NATURAL_RESULT_2026-08-24.md) |
| SAGE-LM last-mile geometry | controlled ceiling存在，但冻结 real-RGB boundary/portal 路线未建立可晋级 student | [V1-F closure](SAGE_LM_V1_F_ANCHOR_CONDITIONED_PORTAL_INTERIOR_FIELD_RESULT_2026-08-25.md) |
| L10M / Goal Copilot 2 packages | 历史代码原位保留，仅用于复现；不再算 current Module | [archive manifest](../../../scripts/research/archive_modules.json) |

完整历史数值留在上述 dated result、Git 历史和 `DEVELOPMENT_LOG.md`，不再复制回本 current README。

## 禁止动作

- 不在已消费 cohort 上调参、换臂、重抽、融合或把 Development 改写成 fresh/Confirmation；
- 不从旧 L10M、P1、SAGE、SAGE-LM、D-ORACLE 或其他 paused/closed 文档恢复 successor；
- 不提前访问 R1C-L final，不扩 architecture/backbone/crop/bin/loss/ensemble sweep；
- 不进入 depth geometry、M2、Android/default-App、导航、安全或用户有效性声明。

## 默认 App 与声明边界

默认 App 保持当前 YOLO/risk 正式路径不变。GRAIL-R 是 synthetic/controlled research integration；
任何正结果都不自动授权 Android 接线、模型晋级、产品或安全结论。
