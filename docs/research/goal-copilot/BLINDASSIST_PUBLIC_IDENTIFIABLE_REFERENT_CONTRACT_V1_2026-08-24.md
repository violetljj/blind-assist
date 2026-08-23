# Public Identifiable Referent Contract V1 (2026-08-24)

状态：`C0_C1_CONTRACT_MECHANICS_READY / REFERENCE_IMAGE_ANCHORED_UNIQUE / PUBLIC_PRIVATE_FIREWALL / C2_NOT_AUTHORIZED / NO_COHORT / NO_BASELINE / NO_ALGORITHM / NO_P1`

## 结论

`PUBLIC_IDENTIFIABLE_REFERENT_CONTRACT_V1` 已实现。V1 先在 episode pixels、candidate、provider output 和 outcome
之前冻结用户公开可见的 referent，再把 provider-public contract 与 evaluator-private physical identity lock 分离。
它修复的是 SUN3D 暴露的任务定义漏洞，不实现新的识别或导航算法。

第一科研桥梁固定为：

```text
public reference image + public target selector + optional language
                              ↓
                    opaque reference anchor
                              ↓
      evaluator-private one-to-one physical instance lock
                              ↓
        later current-frame visibility + region truth
```

`REFERENCE_IMAGE_INSTANCE` 必须是 `UNIQUE`。参考图必须满足二选一：

- `FULL_FRAME_SINGLE_INSTANCE`：全帧只承担一个目标实例；
- `PUBLIC_TARGET_REGION`：公开给 provider 的 reference image 同时公开 normalized target region。

因此“图片 A 对应实例 A”不是 evaluator 事后断言；它由 pre-observation private identity lock、admissible binding
authority、source record SHA-256、与 public reference image 相等的 private binding SHA 及 public/private body hashes
共同固定。可选语言只作为 recognition evidence，不能单独成为 identity authority。

## 合同与防火墙

| 层 | 字段/责任 | provider 可见 |
|---|---|---:|
| Public goal | goal text、modality、`UNIQUE/SET_VALUED/AMBIGUOUS`、opaque anchor | 是 |
| Reference evidence | image SHA/dimensions、全帧单实例或公开 target region、可选描述 | 是 |
| Private identity lock | legal physical instance IDs、world anchors、binding authority/source hash | 否 |
| Later truth | frame SHA、`VISIBLE/NOT_VISIBLE/UNKNOWN`、合法 target regions | 否 |
| Audit | public/private/truth hash binding、primary-evaluable disposition | 否 |

冻结前必须同时满足：episode observation pixels=`NOT_CAPTURED`、candidate/provider output=`NOT_CREATED`、outcome
access=`NONE`。Identity binding 只接受 `SOURCE_NATIVE_REFERENCE_LINK / NATIVE_INSTANCE_ID /
INDEPENDENT_PREOBSERVATION_REVIEW`；teacher/model consensus 明确不能创建 physical identity authority。

Cardinality 语义保持：

- `UNIQUE`：恰好一个合法 physical instance 和一个 world anchor；可见帧恰好绑定该实例的 region；
- `SET_VALUED`：至少两个合同上等价合法的 instances，每个都有 world anchor；当前帧只可返回合法集合的子集；
- `AMBIGUOUS`：不携带 legal target/world anchor，current-frame truth 只能是 unscored `UNKNOWN`，不得伪造 target；
- `NOT_VISIBLE`：合法目标在当前帧无 region；`UNKNOWN` 不得偷算 negative。

## 文献映射

- [IEVE](https://arxiv.org/abs/2402.17587) 提供 instance image goal 的任务抽象：goal image 指定具体 object instance，
  并要求跨视角排除相似 distractor。V1 只借任务定义，不实现 Exploration/Verification/Exploitation 或 RL。
- [GOAT-Bench](https://arxiv.org/abs/2404.06609) 区分 category、language description 和 image goal。V1 同样不把
  category goal 的集合语义冒充 instance identity。
- [REVERIE](https://arxiv.org/abs/1904.10151) 提供 public referring expression → remote physical object → target bbox
  的 evaluator 结构模板。V1 不迁移其数据集、导航器或 benchmark claim。

## 实现与验证

- machine-readable schema：
  `scripts/research/goal_copilot_bridge/public_identifiable_referent_contract_v1/public_identifiable_referent_contract_v1.schema.json`；
- freeze/firewall/truth validator：同目录 `contract.py`；
- 14 个 tests 覆盖 reference-image UNIQUE、public/private split、timing gates、exact reference-image SHA binding、
  teacher rejection、cardinality、normalized region、VISIBLE/NOT_VISIBLE、SET_VALUED、AMBIGUOUS、wrong-instance 与
  hash tampering。

所有输出继续固定：`cohort_freeze_authorized=false / passive_baseline_authorized=false /
algorithm_authorized=false`。本轮没有下载数据、冻结 episode、调用 detector/provider/teacher、实现 Active Referent
Search、接入 control 或修改默认 App。

唯一下一边界是另立 C2 small-roster protocol：只能寻找 5--8 个 independently public-identifiable、source-disjoint
episodes，并在任何 current-frame pixels/provider output 前冻结 reference image、identity lock、roster 与预算。本文件不
授权该协议或执行。

Claim ceiling：
`PUBLIC_REFERENT_CONTRACT_AND_FIREWALL_MECHANICS_ONLY_NO_COHORT_BASELINE_IDENTITY_ALGORITHM_ACTIVE_SEARCH_CONTROL_SAFETY_OR_PRODUCT_CLAIM`。
