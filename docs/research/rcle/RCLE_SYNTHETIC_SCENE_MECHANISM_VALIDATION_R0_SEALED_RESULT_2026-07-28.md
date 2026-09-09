# RCLE Synthetic Scene Mechanism Validation R0 sealed result

终态：
`SYNTHETIC_MECHANISM_SEALED_REVISE / VALID / CLAIM_CONSUMED / NO_RETRY`。

## 一次性执行

- claim：
  `RCLE_SYNTHETIC_SCENE_R0_SEALED_20260728_ONCE`
- activation SHA-256：
  `29dba7f5d1c32dacb4fa0f95ca223b748a9249e69f14839993ac573f29d618b1`
- terminal SHA-256：
  `8a9d0073f77edab401b146b4bd570d8f7fad30b2653637b47ddad0f6ebfb039f`
- dataset receipt SHA-256：
  `579fb8e1975b5056edc0cec88ca0712e457edfecad675bfb159f6973bafd0736`
- scientific summary SHA-256：
  `ed78ac26bc607055467d3bd0fa3f977a15f87e2ba436f64810fecb588a53e57b`
- validation SHA-256：
  `d5e87995cea6053d5de592356487f1134080450bf3723fe7ed9f8e821a6e476b`

正式入口在一个 consumed claim 内完成 generation、初始 validator、algorithm/truth
evaluation、最终 validator 和 exclusive terminal。执行产生 12 个序列、1212 帧、
1200 pair；validator 为 `PASS / errors=[]`。terminal 完成后，直接 evaluator 和第二次
execute 均被合同禁止。

## 冻结门结果

11/12 个 sequence-local gate 通过。唯一失败为：

- scene：`offset_corridor_sealed`
- motion：`positive_rotation_approach`
- failed check：`positive_r1_trigger_retention`
- old trigger：87/100
- R1 trigger：75/100
- R1 coverage：0.75
- retention：`75/87 = 0.8620689655 < 0.90`
- first-trigger delay：0.20 s，PASS
- algorithm evaluable：99/100，PASS
- truth median expansion：0.095637/s，PASS
- algorithm median expansion：0.087930/s，PASS
- algorithm/truth error：PASS

同一场景的 rotation+approach 与 approach 差为 0.003831/s，PASS。因此终态不是
“接近信号消失”，而是三连续 pair 规则在一个未见旋转接近场景中的 trigger retention
低于冻结下限。禁止降低 0.90 gate、换 scene/seed、修改本输出或重跑 claim。

20.79% RGB 时间轴 QA（252/1212 帧）未见缺帧、黑帧或运动路径异常；QA 不改变
scientific terminal。

## 后续决定

决定：`FREEZE_STRESS_R1_DIAGNOSTIC`。

理由：sealed R0 暴露的是未见场景下的连续触发稳健性缺口。下一阶段不作为 R0
回救，也不调算法或门槛；使用全新 scene identities，以 one-factor-at-a-time 方式冻结
光照、低纹理、运动模糊、局部遮挡、depth noise 和 pose perturbation，保留 matched
clean control，报告每个 scene × motion × corruption 的固定分母结果。

最大结论仍为
`CONTROLLED_SYNTHETIC_CAUSAL_MECHANISM_EVIDENCE_ONLY`。本终态不是现实外部确认，
不授权 Android、产品、人因或安全结论。
