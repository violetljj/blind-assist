# GRAIL-R0 Privileged Relational Oracle Result

日期：2026-08-25（Asia/Hong_Kong）

状态：`DEVELOPMENT / EVALUABLE / RELATIONAL_INFORMATION_BREAKS_REFERENT_BOTTLENECK / R1_RELATION_ACQUISITION_AUTHORIZED / FORMAL_TEST_UNOPENED / STOP_BEFORE_M2 / DEFAULT_APP_UNCHANGED`

## 问题与唯一变量

R0 复用已经消费的 ProcTHOR synthetic Development 78-case cohort，只回答一个机制问题：如果给 referent selector 一份明确独立于 appearance 的关系信息，M1 的 `44/78` referent ceiling 能否被击穿。

以下全部冻结不动：78 个 positive、43 个同类干扰 case、78 个 absence pair、candidate set、M1 V2b checkpoint、K=3 pose head、GRAIL threshold=`0.9353410602` 与原 evaluator。未训练或微调任何网络，未读取或采集 formal test。

唯一新增量是从同 revision ProcTHOR native house metadata 构造的 privileged target/candidate relation signature：

```text
semantic type
+ parent/receptacle or structure relation
+ room type
+ coarse height
+ sibling left/center/right and top/middle/bottom
+ three nearest stable object types with relative direction/distance/height
```

selector 要求 coarse signature 精确匹配。唯一匹配时关系 oracle 建立 referent confidence；多个匹配时只用冻结 M1 appearance score 打破碰撞；无精确匹配时拒绝。Object ID 只用于把 runtime candidate 映射回 native metadata，不进入 signature 或排序。

## 结果

| 指标 | M1 reference-only | GRAIL-R0 | 变化 |
|---|---:|---:|---:|
| Referent top-1 | 44/78 | **75/78** | **+31** |
| Referent correct 且 frozen target pose head 可成功 | 34/78 | **61/78** | **+27** |
| Complete interaction pose | 22/78 | **57/78** | **+35** |
| 最强简单基线 B1 | 23/78 | 23/78（冻结） | — |
| Wrong-target commit | 16/43 | **0/43** | -16 |
| Absence false commit | 3/78 | **0/78** | -3 |
| Candidate permutation | 78/78 | **156/156**（positive + absence） | 全通过 |

关系 signature 在 69/78 个 positive 上唯一；9/78 存在关系碰撞并回退到冻结 appearance。三次 top-1 未救回都来自碰撞，且均被原 threshold 拒绝，因此没有新增 wrong-target commit。冻结 target-pose oracle 重新得到 `64/78`；R0 选对且 pose-capable 为 `61/78`，其中 4 个仍被原 threshold 拒绝，最终为 `57/78`。

相对 M1，selector fail-case rescue=`31`、selector collateral=`0`；complete fail-case rescue=`35`、complete collateral=`0`。

## 裁决

```text
GRAIL_R0_RELATIONAL_INFORMATION_CAN_BREAK_REFERENT_BOTTLENECK
```

R0 把 referent top-1 从 `44/78` 提升到 `75/78`，并让同一个 frozen pose head 的 complete success 从 `22/78` 提升到 `57/78`，明显超过 B1 `23/78`。因此当前数据支持一个窄而清晰的因果判断：GRAIL M1 的主要缺口确实是 reference-only evidence 缺少可区分 referent 的信息；interaction-pose architecture 不是本轮主要瓶颈。

唯一 successor 是 GRAIL-R1：保持 task、cohort role、pose interface 与 evaluator 不变，把 privileged relation 换成可由 RGB/语义模型、用户目标文本或可信环境图实际获得的关系表示，并分别报告 oracle-to-obtainable gap、referent rescue、complete pose 与 collateral。R0 不授权重新训练 pose head、reference-only matcher/backbone/loss/threshold sweep、formal M1 test、M2 temporal/active、Android 或默认 App 集成。

## Claim ceiling 与证据 identity

本结果只是一项 `PROJECT_CONSUMED_DEVELOPMENT` 的 synthetic ProcTHOR privileged-metadata mechanism probe。target relation 来自 evaluator/native target metadata，并未由视觉系统、自然语言或用户提供；oracle candidate masks、native object/room/pose truth 仍存在。它不证明视觉关系抽取、自然场景、开放世界、formal generalization、Android、产品或安全能力。

- ProcTHOR val SHA-256：`d808540514e26b6726cd2790490e669b572eeb94febb5188a2f403591dd21721`
- V2b dev collection：`5a7478f4ccd871f684f318f278ae56e34ce1163f58b81ddc245072b1a13f0037`
- frozen features：`caae36d9fa9d7af1dd684166d097faa6336ab554cac51afa6acd2d479998de0d`
- frozen checkpoint：`d838e8c1f648a771a41a32df7cbc0146b6bcebe98715fcd7f7c6c24ed7988b18`
- frozen M1 result：`9657dd5b9306b1400a2dc1ef3e5fa7e0db26b2bcce7f81b82aa5b539e9a83fcb`
- R0 result：`7e4a31735234cf3489b3059cd4f8b6cff704899ccad011fc562e1e5c24f39b48`

运行入口：

```powershell
E:\codex-tools\bin\blindassist-python.cmd scripts/research/grail/run_grail_relational_r0.py --dataset artifacts.local/datasets/grail-r0/val.jsonl.gz --collection artifacts.local/evidence/grail-m1/dataset-v2b/dev/collection.json --features artifacts.local/evidence/grail-m1/run-v2b/features-dev.pt --checkpoint artifacts.local/evidence/grail-m1/run-v2b/checkpoint.pt --development-result artifacts.local/evidence/grail-m1/run-v2b/development-result.json --output artifacts.local/evidence/grail-r0/relational-oracle-result.json
```
