# RCLE periodic self-motion counterfactual R2 quality manipulation successor R1

日期：2026-07-29（Asia/Hong_Kong）

## 结论

```text
PREDECESSOR P4:
  INTERVENTION_NOT_EVALUABLE / VALID / COMPLETE_PRE_R3_TERMINAL

QUALITY MANIPULATION SUCCESSOR R1:
  MATERIAL_OPERATOR_CAL_QUALIFIED / VALID

EXECUTION AUTHORITY:
  SUCCESSOR_FORMAL_NOT_ACTIVATED
```

旧 R2 P4 终态保持不变；它没有正式 R3 outcome，不能解释为 RCLE 算法失败。
本次只修复 low-texture 操纵与 response-blind qualification 的 estimand，
未修改或运行 R3，未读取 outcome，未建立或访问新 formal identities。

## 失败机制

旧 `alpha=0.15` 操纵只收缩每种材质内部的 checker residual，但正式 gate 使用
全帧 clean-q75 梯度密度。物体轮廓、遮挡边界和材质均值边界不受 alpha 控制，
因此该 gate 混入了 scene/view composition，不能稳定表示操纵剂量。

旧 160 条 sequence diagnostic 中，full-frame ratio 为
`0.196323–0.530632`；18 条低于冻结下界，且失败集中在 motion/block
subgroup。这个结果是门控与处理对象失配，不是 R3 response。

## 现成候选

先实现固定 linear-RGB bilateral candidate `QMS-R0`，在旧 80 scenes 上做
response-blind diagnostic。其 material-interior gradient ratio 约为 `1.0`，
八个 block×motion subgroup 均为 `0/20`，说明该固定参数在实际高对比 checker
材质上几乎没有形成目标剂量。该候选已淘汰，未为通过 gate 调参，也未进入新 CAL。

## QMS-R1 冻结实现

QMS-R1 使用一次共享 raycast，在 prequantization linear RGB 域固定：

```text
clean = 0.65 + 0.35 * checker
material_mean = 0.825
low = 0.825 + 0.15 * (clean - 0.825)
```

- clean/low 共享 geometry、valid mask、object id、pose 与相机参数；
- `PSF_NONE=true`，不引入 blur；
- 每帧 prequantization residual relation 最大绝对误差必须严格为 `0.0`；
- hard gate 是 9×9 erosion 后 material-interior residual-std ratio，
  sequence median 必须在 `[0.10, 0.20]`；
- full-frame gradient ratio 和 immediate-boundary contrast 仅作描述项；
- 禁止按 scene 反解 alpha、扩 grid、读取 R3 response 或以 overall 回救 subgroup。

## Response-blind qualification

| cohort | sequences | frames | residual ratio min / median / max | subgroup |
|---|---:|---:|---:|---:|
| consumed predecessor development | 160 | 2560 | `0.144624 / 0.149928 / 0.154132` | 8/8 均 `20/20` |
| new disjoint CAL | 32 | 512 | `0.145955 / 0.149543 / 0.157247` | 8/8 均 `4/4` |

新 CAL identities 使用独立 token 与 uint64 SHA-256 seed 域，不复用旧 80 formal
seeds。所有 512 个 frame state 的 exact prequantization error 均为 `0.0`。

独立 validator 只使用 Python 标准库，不导入 qualification producer、operator、
P4 或 R3；它独立复算 token/seed、cardinality、frame order、hash bindings、
sequence median、八个 subgroup、firewall 与 terminal。独立结果为：

```text
VALID / QMS_R1_INDEPENDENT_VALIDATION_VALID
32 sequences / 512 frame states / 8 subgroups
```

独立 receipt：
[QMS-R1 independent receipt](RCLE_PERIODIC_SELF_MOTION_COUNTERFACTUAL_R2_QUALITY_MANIPULATION_SUCCESSOR_R1_INDEPENDENT_RECEIPT_2026-07-29.json)

## 实现与验证

- `material_residual_contraction_r1.py`：
  `5e66d270c1267d36e927cf47808337e6c1c0da68566e039c9a6ad35eb7c7e8c6`
- `qms_r1_qualification.py`：
  `388af330396b4eeb982f2e56b93617951aa642e61e9fbb7a4f6aa692fe56e6e4`
- `validate_qms_r1_independent.py`：
  `245c92fd1954eef7fa188752dc029e75bd0d84aeb3bff27e31cdcd1d305bb51e`
- operator/qualification/candidate tests：`22/22 PASS`
- independent validator mutation tests：`11/11 PASS`
- independent evidence receipt SHA-256：
  `aa26bd8ae3dc65c9dca71986630754b249f7a49e06f2511aa8e660fc9efaaa95`

本结果只签署 quality manipulation qualification。它不授权新 480+16、
formal seed、科学 outcome 解读、R3/阈值/三-pair 修改、sequence16、Android、
实时集成或 P4 activation。若继续，必须另立全新 formal identity/activation
边界，旧 P4 evidence 不得进入新 estimand。
