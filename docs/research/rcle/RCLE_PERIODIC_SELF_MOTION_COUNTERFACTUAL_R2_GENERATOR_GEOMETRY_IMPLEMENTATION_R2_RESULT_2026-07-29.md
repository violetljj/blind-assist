# RCLE periodic self-motion counterfactual R2 generator geometry R2 result

终态：`INTERVENTION_NOT_EVALUABLE / HOLD_P1 / EXECUTION_NOT_AUTHORIZED`

分轴阅读：

- 科学状态：`OBSERVED_GEOMETRY_GATES_PASS / CLAIM_NOT_SIGNABLE`
  （G01–G14 为 14/14）；
- 协议状态：`INVALID_KEYSET`；
- 当时执行权限：`HOLD_P1`。

这里的 `INVALID` 只表示 R2 receipt 不能签署，不表示几何实现失败。后续隔离
keyset repair 已在不改科学证据字节的条件下关闭协议错误；不得用新结果覆盖本文件
记录的历史回执。

哈希冻结后的唯一 R2 全门运行没有获得可签署 PASS：

```text
status: INVALID
terminal: INTERVENTION_NOT_EVALUABLE
state: HOLD_P1
geometry gates: 14/14 PASS
validation errors: R0_RECEIPT_EVIDENCE_HASH_KEYSET
```

R2 receipt SHA-256：
`75899978919b67be260bbba1161d69ea09b42384f1730ec866a243f6d0f41a32`

R2 implementation lock SHA-256：
`2a1921201c8215efbc0f05f5007908674e247325885d7640a592e731537d719f`

## 已通过的几何事实

- G01–G14 全部返回 `PASS`；
- G01/G02 对 88 个 reference scene 做完整 `360×640` 独立 raycast 和 hash 重算；
- G03 对 40,000 个项目样本及 48 个闭式 pinhole fixture 样本通过；
- G08 覆盖 160 个 motion identity、865,440 个对应点，visibility mismatch 为 0，
  analytic disocclusion point 为 1,053；
- G11/G12 对六-arm 配对、允许字段差异和 source-known geometry identity 实算通过；
- G13 的 16 条 guard arm 均为 602/602 帧目标可见，source-component
  translation/rotation-matrix 与 approach monotonicity 全通过；
- G14 从 `first`/`second` 实算 base replay 和 8 个 GUARD 双构建 replay，
  mismatch 均为 0。

这些通过项在科学轴上仍然成立，但当时不能越过 receipt 的 `INVALID` 协议状态取得
后继执行权限。

## 阻断原因

R2 validator 要求 R0 receipt 的 evidence key 集包含
`generator_receipt.json`，而冻结 R0 receipt 的真实键名是
`producer_receipt.json`。这属于 predecessor hash/keyset validator
实现错误。它发生在 lock 冻结并写出正式 R2 receipt 后，因此不能原地修改、
覆盖回执或重跑 R2。

R2 producer/validator 还没有“目标 receipt 已存在即拒绝写入”的文件级写保护；
默认命令可能覆盖 consumed 路径。因此现有 R2 命令只可读审计，禁止再次调用
producer 或向该 receipt 路径写入。

## 冻结与范围

- R2 的 88 条 all-seed scene record 与 R1 逐字节一致；
- 80 条 MAIN record 与 R0 逐字节一致；
- numeric seed 替换数为 0，trajectory change 数为 0；
- package schema 明确冻结 MAIN scene v1、GUARD scene v2 和不变的外层 JSONL
  record envelope；
- R0、R1、R2 的失败回执全部保留，不互相覆盖。

未读取或运行 RCLE output；未执行 P2、P3、P4；未修改 R3、阈值或三-pair；
未进入 sequence16、CoTracker、Android 或 realtime。P2 不获授权。若未来继续，
必须由用户另行授权新的 P1 validator 版本；不得重跑或修补 R2。
