# GRAIL-R1C-P Symmetry-Aware Paired Orientation Protocol

日期：2026-08-25（Asia/Hong_Kong）

状态：`FROZEN_BEFORE_FRESH_RGB_COLLECTION_AND_MODEL_OUTCOME / SOURCE_DISJOINT_DEVELOPMENT / FIXED_ZERO_SHOT_OA_V2 / PAIRED_RELATIVE_FINAL_ONLY / FORMAL_TEST_UNOPENED / STOP_BEFORE_M2 / DEFAULT_APP_UNCHANGED`

## 研究问题

R1C-O 已证明 native owner-local coordinate 能恢复 relational ceiling；R1C-V 只关闭了 `proposal PCA axis + RGB gradient sign`。R1C-P 只问：固定的 symmetry-aware learned paired rotation 能否在 fresh house-disjoint ProcTHOR cohort 上，把同一 reference owner frame 传播到 query，并恢复冻结的 3×3 relation selector。

本轮改变的唯一信息源是 Orient Anything V2（OA-V2）的 learned absolute-orientation distribution 与 paired relative rotation。ownership grouping、proposal、slot、selector fields、appearance tiebreak、pose head、threshold=`0.9353410602`、negative pairing 与 evaluator不变。

## 数据与身份冻结

- 数据仍是 ProcTHOR val revision `439193522244720b86d8c81cde2e51e3a4d150cf`，SHA-256=`d808540514e26b6726cd2790490e669b572eeb94febb5188a2f403591dd21721`。
- 排除 M1 V2b 已用的 24 train + 6 dev houses。用 salt `BLINDASSIST_GRAIL_R1C_P_FRESH_VAL_V1` hash-rank 12 个新 house；结果由 `freeze_grail_paired_orientation_r1cp.py` 固化。
- 采集保持 V2b 的 query distance、position/yaw hash order、首个 visible position；额外持久化同一时刻 query/reference full RGB 与所有 visible actionable proposal masks，不读取模型结果。
- cohort admission 在模型调用前冻结：从带同类 distractor 的 rows 按 salt `BLINDASSIST_GRAIL_R1C_P_ADMISSION_V1` 取 43 个，并从不带同类 distractor 的 rows 取 35 个，合计 78。任一 quota 不足即 `NOT_EVALUABLE`，不增 house、不改 salt、不重采。

OA-V2 固定为官方代码 commit `73b11c9dc83e84daeb563d0c766831f2c66b0a18`，checkpoint `demo_ckpts/rotmod_realrotaug_best.pt`，SHA-256=`7b6b7f258d32b95123b9d023005ecca357d8ab944fb83476f532d3cf7a2295eb`，5,048,116,892 bytes。只允许一次 bfloat16 CUDA zero-shot inference；不训练、不校准、不扫 crop、mask、layer、angle bin、mode threshold 或融合权重。

## 固定 prediction contract

每个视角先复用冻结的 `predict_groups()`。每个 predicted owner group 的 union mask 取成员 proposal masks 的并集；union bbox 四边各 pad 10%，裁到图像边界，mask 外像素置白，随后使用 OA-V2 官方 `pad` preprocessing 到 518。禁止 native owner、object ID suffix、pose/depth/truth 或 evaluator label 进入 crop、grouping 或模型。

reference goal proposal/group 是 reference-goal 模式的公开输入。对每个 query candidate group，final arm 以同一个 reference owner crop 和该 candidate owner crop做 paired inference。官方 900-d head 的 reference azimuth `0:360` logits 经 sigmoid 得到 absolute distribution；paired 第二帧同一区间是 relative-azimuth distribution。reference symmetry 只使用官方 `val_fit_alpha()` 的固定 `0/1/2/4` 输出：`0` 为 `UNKNOWN`；否则从 argmax 产生 `alpha` 个等距 reference modes。relative rotation 的 azimuth/elevation/roll 使用 paired head 固定 argmax，并把每个 reference mode通过同一 `ΔR` 传播到 query；不得让两侧独立选择最有利 mode。

每个 propagated mode 将 OA-V2 canonical local-right 与 local-up 投影到各自图像，再复用冻结 3×3 `rank_bin`。若所有 reference modes 对 selector 给出同一 referent，保留该 referent；若 mode 间 referent 不同或任一必要 frame 不可投影，输出 `UNKNOWN/abstain`。不得用 native yaw 挑 mode。

## Arms、指标与裁决

| Arm | 用途 |
|---|---|
| `OA_V2_INDEPENDENT_ABSOLUTE_DIAGNOSTIC` | 两图各自 absolute argmax；只诊断独立估计误差累积 |
| **`OA_V2_PAIRED_RELATIVE_FINAL`** | reference distribution + paired `ΔR` 的 mode-coupled propagation；唯一裁决 arm |

final 必须同时满足：slot `>=70/78`、referent `>=70/78`、complete `>=50/78`、wrong-target `<=1/43`、absence `<=1/78`、permutation=`156/156`、selector collateral=`0`、complete collateral=`0`。另报告 symmetry alpha 分布、mode-consensus/UNKNOWN、Drawer/其他分层、相对 R1C-O 的恢复率，以及 independent diagnostic；diagnostic 不得替代 final。

若 final 全部过门，终态为 `GRAIL_R1C_P_PAIRED_VISUAL_OWNER_ORIENTATION_ESTABLISHED_FORMAL_TEST_ONLY`。否则终态为 `GRAIL_R1C_P_PAIRED_RGB_OWNER_ORIENTATION_NOT_ESTABLISHED_STOP_BEFORE_DEPTH_GEOMETRY`，并关闭 single/pair RGB owner-orientation；不得在结果后扫 crop、prompt、layer、bins 或 fusion。失败后仅允许另立 depth/normal geometry 协议，本轮不混合。

## Claim ceiling

最高只建立“一次固定 OA-V2 checkpoint 在一个 fresh house-disjoint synthetic ProcTHOR cohort 上，paired learned rotation 是否恢复 privileged coordinate mechanism”的窄结论。不建立 formal generalization、自然 RGB、真实相机、Android、用户、产品或安全证据。
