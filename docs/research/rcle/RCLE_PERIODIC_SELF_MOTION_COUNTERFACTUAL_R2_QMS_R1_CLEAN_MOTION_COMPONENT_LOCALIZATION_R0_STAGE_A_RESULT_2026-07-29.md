# RCLE clean motion-component localization R0：Stage A 收口

日期：2026-07-29（Asia/Hong_Kong）

终态：

```text
STAGE 1: COMPLETE / VALID / 16 OF 16
STAGE 2: COMPLETE / VALID / 16 OF 16
STAGE A CLOSEOUT: VALID / COMPLETE
NEXT DECISION: ACTIVATE_STAGE_B_CONTRACT_PREPARATION_ONLY
FORMAL 480+16: AUTHORIZED_ONE_SHOT / NOT_CONSUMED / NOT_RUN
```

## 一句话结论

在两批完全预冻结且互不替换的四-block scene clusters 中，rotation-only 相对
static 的 absolute P90 leakage 与 translation-only 相对 rotation-only 的
signed P90 response 均连续两批 `4/4` 正方向；full-6DoF 相对较大 single arm
只有 `2/4 → 3/4`，跨批方向与中位数不稳定。Stage A 因而完成运动分量定位，只
支持进入“translation-depth oracle + object-approach positive control”的
Stage B 合同准备，不授权 Stage B 执行、算法修改、C/D 或正式 `480+16`。

## 冻结设计和统计单位

- 四臂：`STATIC / ROTATION_ONLY / TRANSLATION_ONLY / FULL_6DOF`；
- 四个 block：`ADVIO_13 / 14 / 15 / 17`；
- 每批 `4 clusters / 16 sequences / 9,632 frames / 9,616 pairs`；
- Stage 1 使用 ordinal `0`，Stage 2 使用执行前已冻结的 ordinal `1`；
- block × scene-seed cluster 是方向判断单位；pair 只是 longitudinal repeat；
- signed expansion 与 absolute leakage 分开，三项 contrast 的量纲语义不同，
  不比较数值大小，也不据此宣称某运动成分“主导”；
- 不做 pair pooling、p value、置信区间、bootstrap、max-t 或 formal
  classification。

## 两批独立结果

| Contrast | Stage 1 正方向 | Stage 1 中位数 | Stage 2 正方向 | Stage 2 中位数 | 收口 |
|---|---:|---:|---:|---:|---|
| `ROTATION_MINUS_STATIC`，absolute P90 | 4/4 | 0.097896 | 4/4 | 0.109383 | 稳定复现，作为边界条件 |
| `TRANSLATION_MINUS_ROTATION`，signed P90 | 4/4 | 0.108742 | 4/4 | 0.074172 | 稳定复现，购买 Stage B 合同 |
| `FULL_MINUS_MAX_SINGLE`，signed P90 | 2/4 | -0.024464 | 3/4 | 0.015238 | 不稳定，不激活 interaction 分支 |

这里的“稳定复现”是明确标注的 descriptive routing convention：同一 contrast
在 Stage 1 和 Stage 2 都至少 `3/4` block 为正。它不是预注册的确认性总体推断。

Stage 1 每个 block 的 contrast：

| Contrast | ADVIO 13 | ADVIO 14 | ADVIO 15 | ADVIO 17 |
|---|---:|---:|---:|---:|
| rotation absolute | 0.091956 | 0.103836 | 0.169853 | 0.087018 |
| translation signed | 0.003462 | 0.072374 | 0.145109 | 0.205583 |
| full interaction signed | 0.038792 | 0.006952 | -0.055879 | -0.113457 |

Stage 2 每个 block 的 contrast：

| Contrast | ADVIO 13 | ADVIO 14 | ADVIO 15 | ADVIO 17 |
|---|---:|---:|---:|---:|
| rotation absolute | 0.104600 | 0.114166 | 0.150132 | 0.092590 |
| translation signed | 0.074039 | 0.031682 | 0.157636 | 0.074305 |
| full interaction signed | 0.013211 | 0.017265 | -0.036835 | 0.053813 |

## 执行与独立验证

| Stage | Wall time | Launch available RAM | Minimum available RAM | Swap in/out | Heartbeat max | Residual workers |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 2,315.08 s | 8,825,257,984 B | 7,623,782,400 B | 0 / 0 | 20.03 s | 0 |
| 2 | 2,196.32 s | 9,734,844,416 B | 9,037,230,080 B | 0 / 0 | 20.03 s | 0 |

每批均由独立 validator 从 observation-only cell primitives 重建 pair scalar，
再按 cluster 聚合。Stage A closeout 使用第三个、与 producer 和逐批 validator
分离的 validator，重新绑定所有输入 hash、重算三项 contrast、核对两批 analysis
controls 和 formal firewall。

- [Stage A independent closeout receipt](RCLE_PERIODIC_SELF_MOTION_COUNTERFACTUAL_R2_QMS_R1_CLEAN_MOTION_COMPONENT_LOCALIZATION_R0_STAGE_A_INDEPENDENT_CLOSEOUT_RECEIPT_2026-07-29.json)
- [Stage B activation decision](RCLE_PERIODIC_SELF_MOTION_COUNTERFACTUAL_R2_QMS_R1_CLEAN_MOTION_COMPONENT_LOCALIZATION_R0_STAGE_B_ACTIVATION_DECISION_2026-07-29.json)
- [Stage 1 result](RCLE_PERIODIC_SELF_MOTION_COUNTERFACTUAL_R2_QMS_R1_CLEAN_MOTION_COMPONENT_LOCALIZATION_R0_STAGE_1_RESULT_2026-07-29.md)

## 下一边界

Stage B 目前只允许准备和冻结以下合同内容：

1. translation-depth oracle 的 estimand、输入真值和停止规则；
2. object-approach positive control 的独立 identity、角色与几何验收；
3. rotation compensation / warp / valid-mask / local-fit 的限界审计条件；
4. signed response 与 absolute leakage 的分离输出；
5. cluster-level 独立 validator、receipt 和 `go / freeze / stop /
   not-evaluable` 决策。

Stage B response read 和执行尚未授权。rotation leakage 的两批 `8/8` 复现是必须
携带的机制边界，不是立即修改 rotation algorithm 的权限。full interaction
分支不激活。

## 不形成的结论

Stage A 没有测试 object approach、深度补偿收益、真实危险、报警准确率、自然
session 泛化、Android、产品或安全效果。它只完成受控 generator 内部的运动分量
响应定位。

两批 formal firewall 的 before/after 完全一致：

```text
formal_sequences_run = 0
formal_r3_pair_core_calls = 0
formal_authority_consumed = false
successor_formal_path_absent = true
```

因此已有 successor formal 一次性授权仍是
`AUTHORIZED_ONE_SHOT / NOT_CONSUMED`，本路线没有替代或消费正式 `480+16`。
