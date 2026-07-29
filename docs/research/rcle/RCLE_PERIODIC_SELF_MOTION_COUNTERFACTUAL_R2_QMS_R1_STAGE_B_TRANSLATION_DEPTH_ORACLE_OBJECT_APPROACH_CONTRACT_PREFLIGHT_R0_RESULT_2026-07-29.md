# RCLE Stage B translation-depth oracle + object-approach contract preflight R0

日期：2026-07-29（Asia/Hong_Kong）

终态：

```text
SCIENTIFIC STATUS: NOT_RUN / NO STAGE B RESPONSE
PROTOCOL STATUS: CONTRACT_PREFLIGHT_PASS / VALID
EXECUTION AUTHORITY: EXECUTION_NOT_ACTIVATED
STAGE B RESPONSE READ/WRITE: 0 / 0
STAGE B WORKLOAD CALLS: 0
FORMAL 480+16: NOT_RUN / NOT_CONSUMED
```

## 一句话结论

Stage B 的 translation-depth oracle estimand、同 support 的 R3 baseline、
object-approach positive control、8 个独立 cluster identities、坐标与单位、
warp/visibility/valid-mask/local-fit 规则、rotation leakage 必过门、cluster-level
停止与 routing 已冻结并通过 `12/12` 独立合同预飞；这只说明合同可复算，不说明
oracle 有效。执行决定保持
`HOLD_STAGE_B_EXECUTION_PENDING_SEPARATE_ACTIVATION`。

## 冻结问题与 baseline

主要问题是：在 source-known metric depth、camera pose 和 object motion 下，
translation-depth oracle 是否能收缩静态场景的 camera-translation residual，
同时保留独立目标接近的正 signed response。

两个 analysis channel 在相同 track 和 common-cell support 上配对：

1. `R3_ROTATION_ONLY_BASELINE`：既有 R3 current-to-previous rotation warp、
   sparse tracking、common-cell 与 local affine fit 全部不变；
2. `R3_PLUS_SOURCE_KNOWN_TRANSLATION_DEPTH_ORACLE`：只在 R3 对齐坐标中减去
   source-known camera-translation endpoint displacement，再用相同参数重新拟合
   local affine。

oracle 不是候选算法，也不是可部署的单目 depth 方法。禁止用新增 support、
拟合后 scalar subtraction 或复用 baseline affine coefficients 制造收益。

## 几何、角色和独立 identities

冻结 4 个 pose blocks × 2 个新 scene/object ordinals，共：

```text
8 clusters
5 arms per cluster
40 sequence identities
602 frames / 601 pairs per sequence
```

五臂为：

| Arm | 角色 |
| --- | --- |
| `STATIC_SCENE` | 零运动负对照 |
| `EGO_ROTATION_STATIC_SCENE` | 必过 rotation-leakage boundary |
| `EGO_TRANSLATION_STATIC_SCENE` | translation-depth oracle estimand |
| `OBJECT_APPROACH_STATIC_CAMERA` | object-approach positive control |
| `OBJECT_APPROACH_PLUS_EGO_6DOF` | ego subtraction 不得吞掉目标运动的 combined boundary |

目标为 `object_id=1001` 的 fronto-parallel rectangle mesh，初始中心
`[0.2, 0.1, 6.0] m`、宽 `1.2 m`、高 `1.6 m`，沿 world `-z` 单调移动
`2.0 m`，终深度 `4.0 m`。endpoint radial scale 为 `1.5`，目标 material
point 用固定 barycentric identity，不逐帧重选。目标只承担几何 positive-control
角色，不是 person、danger、collision 或 alarm label。

新 identities 与 Stage A 及其冻结 exclusion authority 中的 formal、DEV、CAL、
PREFLIGHT identities 比较：独立 validator 扫描 `1,756` 个 identity values，
collision 为 `0`。当前只冻结 token、seed、cluster、target 和 sequence identity；
scene/target geometry 尚未物化，因此全部保持 `FROZEN_NOT_EXECUTABLE`。

## 坐标、warp、遮挡和 local-fit

坐标冻结为：

```text
pose: world_from_camera (R_wc, t_wc)
camera axes: +x right, +y down, +z optical forward
depth: previous/source-frame optical Z, metre
pixel: u right, v down
time: second
expansion: s^-1
```

对 previous pixel `p`：

```text
X_p = Z_p K^-1 p
R_cp = R_wc,current^T R_wc,previous
t_cp = R_wc,current^T (t_wc,previous - t_wc,current)
X_c,rigid = R_cp X_p + t_cp
H_rot = K R_cp K^-1
q_r3,rigid = pi(H_rot^-1 pi(K X_c,rigid))
u_T = q_r3,rigid - p
q_oracle = q_observed,r3 - u_T
```

R3 继续使用原有 previous-to-current homography、current-to-previous warp、
bilinear image、nearest mask 与 constant-zero border。oracle 不改变 R3。

有效点必须同时满足 previous depth、实际 current material point、object identity、
z-buffer visibility、图像边界、generator masks、R3 warp mask 和 sparse track。
静态点还必须与 current z-buffer depth 闭合；moving target 在实际 object-transformed
endpoint 上检查可见性，camera-only counterfactual endpoint 不要求可见。遮挡、
新显露、object switch、depth discontinuity、behind-camera 或越界均 abstain；
不补零、不插值、不 carry-forward，也不从固定分母删除。

oracle endpoint 必须重新执行 unchanged local affine fit。cell membership 仍由
previous feature coordinate 决定；baseline/oracle 只比较共同 track 与共同 cell。

## estimand、限界与分析单位

- translation signed suppression：
  `baseline signed P90 - oracle signed P90`；
- translation absolute leakage suppression：
  `(baseline absolute P90 - oracle absolute P90) / max(baseline, 1e-12)`；
- object approach retention：
  target-mask `oracle signed P90 / baseline signed P90`；
- rotation boundary：
  pure-rotation static scene 的 oracle translation displacement 必须为零，
  absolute P90 必须 `<=0.01/s`，固定 601-pair 分母的三-pair trigger density
  必须为 `0`，且 8/8 clusters 全部通过。

signed response 与 absolute leakage 分开报告，不比较数值大小。cluster 是分析
单位；pair、frame、track、cell、cycle 和 arm 都只是重复测量。不做 pair pooling、
p value、bootstrap、CI、max-t 或 formal classification。

rotation boundary 任一 cluster 失败，直接终止为
`B_ORACLE_NOT_EVALUABLE / ROTATION_LEAKAGE_BOUNDARY_FAIL`，不得继续解释
translation oracle 或用调 R3、阈值、三-pair、abstention 回救。

## 独立验证

独立 validator 不导入 RCLE evaluation、local-fit 或 Stage B producer，完成：

| Gate | 结果 |
| --- | --- |
| authority、R3、transport 与 source hashes | PASS |
| identity/role/count/disjointness | PASS，8 clusters / 40 sequences / 0 collision |
| `T=0` translation oracle | PASS，max `5.68e-14 px` |
| constant-depth plane analytic solution | PASS，max `8.88e-14 px` |
| translation direction reversal | PASS |
| scaled intrinsic normalized equivalence | PASS，max `0 px` |
| visibility/valid-mask rejection cases | PASS，8/8 |
| moving object anti-swallow | PASS，max `5.68e-14 px` |
| oracle endpoint local refit | PASS |
| frozen object geometry | PASS，endpoint scale `1.5` |
| rotation leakage limit present | PASS |
| response/formal firewall | PASS |

专项 mutation tests 为 `5/5 PASS`，覆盖 seed 篡改、rotation gate 降低、
response authority 篡改和 zero-translation rotation alignment。

证据：

- [frozen contract](RCLE_PERIODIC_SELF_MOTION_COUNTERFACTUAL_R2_QMS_R1_STAGE_B_TRANSLATION_DEPTH_ORACLE_OBJECT_APPROACH_CONTRACT_PREFLIGHT_R0_CONTRACT_2026-07-29.json)
- [identity lock](RCLE_PERIODIC_SELF_MOTION_COUNTERFACTUAL_R2_QMS_R1_STAGE_B_TRANSLATION_DEPTH_ORACLE_OBJECT_APPROACH_CONTRACT_PREFLIGHT_R0_IDENTITY_LOCK_2026-07-29.json)
- [independent receipt](RCLE_PERIODIC_SELF_MOTION_COUNTERFACTUAL_R2_QMS_R1_STAGE_B_TRANSLATION_DEPTH_ORACLE_OBJECT_APPROACH_CONTRACT_PREFLIGHT_R0_INDEPENDENT_RECEIPT_2026-07-29.json)
- [execution activation decision](RCLE_PERIODIC_SELF_MOTION_COUNTERFACTUAL_R2_QMS_R1_STAGE_B_TRANSLATION_DEPTH_ORACLE_OBJECT_APPROACH_CONTRACT_PREFLIGHT_R0_EXECUTION_ACTIVATION_DECISION_2026-07-29.json)

## 当前权限与停止点

当前不是 Stage B execution activation。以后若要运行，仍需另行满足：

1. 明确的 Stage B 单独执行授权；
2. exact frozen identities 的 scene/target geometry hashes；
3. 绑定这些 hashes 的独立 geometry-only receipt；
4. write-once Stage B response root 与未改变的 formal firewall。

本轮没有生成或读取 Stage B response，没有运行 renderer、tracker、R3 pair-core
或 Stage B local-fit workload；没有修改 R3、`>0.01/s`、三-pair、PairState 或
abstention；没有进入 C/D，也没有运行或消费正式 `480+16`。

因此科学状态仍为 `NOT_RUN`，协议状态为 `VALID`，执行权限为
`EXECUTION_NOT_ACTIVATED`。
