# HFTF Stage C SANPO F0.1 teacher-opportunity result

日期：2026-08-01

workflow：`DEVELOPMENT_STANDARD`

终态：`F0_1_SANPO_TEACHER_OPPORTUNITY_READY_FOR_CORPUS`

下一步：`TRAIN_CANDIDATE_CORPUS_AND_DEV_REFERENCE_TARGET_MATERIALIZATION`

## 结论

冻结的 cross-split SANPO body/head teacher opportunity 门通过。12/12 个 parent
session 均通过 source authority、media/pose/hash binding、current/future known
coverage、future positive/negative 与 UNKNOWN 防火墙；完整运行在独立进程中复跑
字节一致。

这只授权下一步物化：

- 6 个 train source 的 stride-8/offset-4 candidate teacher corpus；
- 3 个 dev source 的 stride-4/offset-2 reference evaluation targets。

official-test heldout 的 training corpus 永远不授权；heldout reference cell targets
继续保持未物化，必须等 train/dev corpus 验证、所有 seed 的 dev checkpoint 冻结后
才进入 ordered evaluation。

## 冻结输入与封口

- source lock：12 sources，`6 train / 3 dev / 3 official-test heldout`；
- acquisition audit：300 RGB、300 masks、300 metric depths，12/12 通过；
- source authority：12/12 为
  `HFTF_H0_2_SANPO_CANONICAL_PROXY_REPLICATED`；
- authority cohort：
  `F0_1_SANPO_SOURCE_AUTHORITY_COHORT_READY`；
- teacher execution contract SHA-256：
  `29e449f729942bfea8919d93bb404360829c42681cdc7d8f8fc86559c99b79b6`；
- teacher report SHA-256：
  `9db97892ae93267856e1388bccf808deb8947311e25cc5b39a1c362b4bb348b5`。

两次独立进程输出的完整报告 SHA-256 相同。每个进程内部还执行两遍完整科学
payload，并以 canonical UTF-8 JSON 比较，结果为 byte exact。

## 机会门结果

角色 gate view 按 outcome 前冻结的执行合同：

- train：candidate，stride 8 / offset 4；
- dev：reference，stride 4 / offset 2；
- heldout：reference，stride 4 / offset 2。

两个 pixel lattices 不相交。5 FPS 来源使用 19 个 anchors；10 FPS 目标时间线使用
13 个 anchors。所有 source 的 usable anchors 只由 `[-0.8,-0.6,-0.4,-0.2,0]s`
历史与精确 `+0.4s` availability 决定。

| 指标 | 冻结门 | 观察到的最弱值 | 结果 |
|---|---:|---:|---|
| 每 source × height × horizon known coverage | `>=0.10` | `约 0.23` | PASS |
| 每 source × height future positive-known cells | `>=5` | `6` | PASS |
| 每 source × height future negative-known cells | `>=20` | `182` | PASS |
| 每 role × height 有 positive future 的 sources | `>=2` | train `6`、dev `3`、heldout `3` | PASS |
| UNKNOWN→SAFE violations | `0` | `0` | PASS |
| second-run payload | byte exact | byte exact | PASS |

最接近边界的是 official-test source `9bee9c83…` 的 head future positives=`6`。
该值仍超过冻结门，但应作为后续最差 source 监控点，不得据此改阈值或换来源。

## 因果与标签边界

- future origin 只用 anchor 与精确 `-0.4s` pose 的地面切向速度推进；
- future pose 只用于把 future observation 变换到 world 并检查同一 world probe；
- grid orientation、anchor、source 或 sample 不读取 future；
- future obstacle support 对两帧逐 cell 取最大值，避免跨时重复点把阈值 2
  人为抬高；
- future known 对同一 9 个 probes 做 observation OR，再应用 `>=5/9`；
- UNKNOWN 始终 masked，不生成 SAFE target；
- foot layer 未计算，EgoWalk foot-ground 路线仍为 `NOT_EVALUABLE`。

## 证据上限

SANPO body/head 标签仍是 source-specific synthetic geometry proxy，不是人体校准、
事件真值、碰撞真值或安全证据。teacher opportunity 通过不证明 student 可学习，
更不证明 history 比 single-frame 好。

本结果没有训练学生、没有打开 heldout student output，没有改变 HFTF 候选支线地位、
研究主线、默认 App、Android、生产或安全权限。只有后续 train/dev corpus 与训练链
完整通过，才允许冻结 checkpoint 后一次性打开 official-test heldout student 结果。
