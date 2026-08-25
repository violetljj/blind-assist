# GRAIL-R1C-O Owner-Local Canonical Coordinate Result

日期：2026-08-25（Asia/Hong_Kong）

状态：`DEVELOPMENT / PRIVILEGED_NATIVE_OWNER_FRAME / REFERENT_75_OF_78 / COMPLETE_58_OF_78 / WRONG_TARGET_1_OF_43 / ABSENCE_0_OF_78 / R1C_O_CEILING_ESTABLISHED / R1C_V_PROTOCOL_ONLY / FORMAL_TEST_UNOPENED / STOP_BEFORE_M2 / DEFAULT_APP_UNCHANGED`

## 问题与冻结边界

R1C-O 按[结果前冻结协议](GRAIL_R1C_OWNER_LOCAL_CANONICAL_COORDINATE_PROTOCOL_2026-08-25.md)，只把 R1B 的 image-space sibling ordinal 换成 AI2-THOR native owner-local coordinate。78-case cohort、candidate set、M1 V2b checkpoint、appearance collision tiebreak、pose head、threshold=`0.9353410602`、negative pairing 与 evaluator 均未改变；selector 使用 R1 已确定的最小字段 `semantic type + sibling ordinal + nearest stable type`。

40 个 Drawer 的 runtime part position 只存在于 AI2-THOR native object metadata，不存在于 ProcTHOR house JSON。因此本轮复用 R1B 的冻结 Docker image 和 official release cache，对 6 个 Development house 做一次 metadata-only scene reset。6/6 houses 的 753 个 runtime objects 均有 native owner frame；78/78 targets 可评估。owner 来源为 native component prefix=`40`、`parentReceptacles`=`37`、standalone self=`1`。

## Coordinate contract

owner native yaw 的逆变换给出 `(right, up, front)`；slot 在同一 native owner、同一 semantic type 的全部 runtime siblings 上计算，而非当前视角可见集合。horizontal 使用 local-right 的 `LEFT/CENTER/RIGHT`，vertical 使用 negative-local-up 的 `TOP/MIDDLE/BOTTOM`；front 仅审计，不进入 selector。camera pose、RGB、bbox、mask 与 outcome 均不参与 frame 或 label。

canonical label agreement=`78/78` 是该 source-native object coordinate contract 的确定性结果，不是两个视觉估计器的独立一致率，也不建立 R1C-V obtainability。

## 结果

| Arm | Referent | Complete | Wrong-target | Absence false commit | R0 uplift recovery（referent / complete） |
|---|---:|---:|---:|---:|---:|
| Appearance-only M1 | 44/78 | 22/78 | 16/43 | 3/78 | 0% / 0% |
| R1B image-space bbox | 47/78 | 35/78 | 11/43 | 29/78 | 9.7% / 37.1% |
| Privileged R0 | 75/78 | 57/78 | 0/43 | 0/78 | 100% / 100% |
| **R1C-O owner-local canonical** | **75/78** | **58/78** | **1/43** | **0/78** | **100% / 102.9%** |

其他冻结指标：referent+target-pose=`61/78`、candidate permutation=`156/156`、selector collateral=`0`、complete collateral=`0`。R1B 的 23 个 cross-view-coordinate referent failures 中，R1C-O 救回 `20/23`，全部为 Drawer。

剩余 3 个 referent failure 仍是 Drawer，均进入 `RELATION_COLLISION_APPEARANCE_TIEBREAK`；其中 1 个形成 wrong-target。`58/78` 比 R0 多 1 个 complete 只说明本 consumed Development 上 canonical slot 消除了一个 R0 image-space collision，不构成优于 privileged R0 的一般性结论，也不授权调 3×3 bin、front axis 或 tiebreak。

## 裁决

```text
GRAIL_R1C_O_CANONICAL_COORDINATE_CEILING_ESTABLISHED_R1C_V_PROTOCOL_ONLY
```

R1C-O 证明当前 23-case dominant loss 确实来自 coordinate definition：把 owner-local part coordinate 设为 source-native、camera-independent label 后，R0 referent uplift 全部恢复，complete 也恢复到同一量级。ownership × coordinate system 的分解在该 synthetic Development 机制实验内成立。

唯一 successor 是另立 R1C-V 协议，检验 RGB/mask 可获得的 owner orientation 能恢复 R1C-O 的多少。不得在本 artifact 调 matcher、threshold、mask encoding、binning、front-axis fusion 或 pose head；formal test、M2、Android/default-App 保持关闭。

## Evidence identity 与 claim ceiling

- result artifact SHA-256：`cc8ca72294bd0a2a9d6c7e56dcf26651ce41bcfb26d1784ff3895ea96a5cadc6`
- native-coordinate artifact SHA-256：`acae38bdbe2802c8d443c0a8cd6bec385d413d9c5e91850d5eed9f4f196dd117`
- protocol/code frozen commit：`dc64b0ad`
- schema：`blindassist_grail_r1c_o_owner_local_canonical_coordinate_probe_v1`

本结果仅为 `PROJECT_CONSUMED_DEVELOPMENT_PRIVILEGED_COORDINATE_CEILING` synthetic ProcTHOR/AI2-THOR mechanism evidence，不建立 RGB orientation、自然场景、学习、formal generalization、Android、产品或安全 authority。

