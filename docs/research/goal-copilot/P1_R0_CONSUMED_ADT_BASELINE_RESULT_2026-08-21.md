# P1-R0 consumed ADT baseline result — 2026-08-21

终态：`REAL_RGB_PERSISTENCE_HEADROOM_ESTABLISHED_ON_CONSUMED_ADT`

Claim ceiling：`CONSUMED_ADT_INDOOR_OBJECT_DEVELOPMENT_BASELINE_ONLY`

## 执行完整性

本轮按 [`P1-R0 consumed ADT baseline protocol`](P1_R0_CONSUMED_ADT_BASELINE_PROTOCOL_V1.md) 消费已关闭的
P1-D0 15-episode cohort。公开输入只含 RGB path/hash、timestamps、video frame index、P0 handoff 和每个 episode
一次性的 frame-0 oracle bbox。Tracker receipt 为：

```text
episodes                         15
oracle_initializations           15
post_initialization_gt_reads      0
detector/model/Sky calls          0
```

`track` 阶段只打开 public input 与 RGB；完成并封存 `prediction.json` 后，`evaluate` 才读取 private ADT
truth。冻结 P1 evaluator 与 P0 handoff synthetic mechanics 均未修改或重跑为真实画面主张。

首次 v1 adapter audit 发现原始 D0 `episode_id` 字符串嵌有 source UID；tracker 没有使用 identifier 内容，但
public surface 仍违反 no-object_uid 合同，因此 v1 被标记
`INVALID_PUBLIC_IDENTIFIER_LEAK / NO_SCIENTIFIC_VERDICT`，未覆盖。有效 v2 使用 `p1-r0-consumed-NNN` 等
无语义序号 alias；公开文件经 `object_uid / physical_target / visibility / distractor / temporal_mode / GT path /
10+ digit identifier` 扫描为零命中。v2 保持相同 tracker 实现与全部冻结阈值，结果与 invalid v1 数值一致。

## 冻结 evaluator 结果

不生成加权总分。Aggregate：

| 面 | 结果 |
|---|---:|
| correct-identity coverage | `87/777 = 11.20%` |
| wrong-instance asserted frames | `1,221/1,308 = 93.35%` asserted frames |
| identity switches | `59` |
| false reacquisition | `0`；但没有任何 reacquisition event，不能解释为安全成功 |
| max wrong-lock | `255 frames / 8,498 ms` |
| temporary-occlusion recovery | `0/3` |
| out-of-view/long-loss reacquisition recall | `0/6` |
| reacquisition precision | `null`（零次 reacquisition event） |
| false-loss | `192/777 = 24.71%` observable frames |

按预冻结的穷尽规则，coverage 略高于 `0.10` research floor，且 identity-safety failures 非零，因此机械终态为
`REAL_RGB_PERSISTENCE_HEADROOM_ESTABLISHED_ON_CONSUMED_ADT`。这不是“baseline 可用”；它只说明真实 RGB 上已经
同时出现少量正确 persistence 与大量可研究的 unsafe failure。

## Post-outcome 描述性 autopsy

该分解不改变冻结终态：

```text
correct target asserted                         87
wrong background / no visible ADT instance   1,094
wrong other ADT physical instance              127
visible frames without assertion                203
wrong-background share of wrong assertions     89.60%
```

因此第一 bottleneck 不是 distractor ReID，也不是 policy：朴素 translation-only flow 在相机运动、尺度/形变和
小目标条件下长期漂离任何可见实例，却仍持续断言 TRACKING。模板重捕获没有产生一次确认事件，所以当前也没有
reacquisition precision 样本；不能把 `false_reacquisition=0` 写成保守重捕获已经解决。

Safety-first 下只选择一个 successor：

```text
P1-A1_CONSERVATIVE_LOCAL_TRACK_VALIDITY
```

它只研究 local track 的 RGB contradiction / drift rejection，使已经漂离目标的 bbox 尽快输出 null；candidate
generator、固定模板 reacquisition、P1 state machine、evaluator、P1-D0 episodes 与 truth firewall 全部冻结。第一目标是
降低 wrong-background asserted frames / max wrong-lock，不以 coverage 换来的表面提升作为成功。完成这个单变量
实验前，不加入 ReID、强 tracker、SAM、CoTracker、Sky、fresh cohort 或 Android。

## 本地 evidence identity

有效 ignored evidence root：`artifacts.local/evidence/p1_r0_consumed_adt_baseline_v2/`

```text
public_input.json       F821660F32CE82EFE6E1427C976575C52A7F57A2BA4A5CF64726E338E074C6B9
private_eval_input.json 59B29B41BD1FA95D3E6BD72D910D4AC7ABC542F2E607D77DB254D46508749E9B
prediction.json         5F8FE5C256E55AB706219242A03CC92D9CC1AFC8B86A2971BF97F26225FA90C3
result.json             5F9969CC5399920DE9A0735F814570F31F3CAD5F4B9A7DB1D516E4F849E1B4DC
```

这些文件是 consumed Development evidence，不得移动成公开 benchmark、fresh receipt、产品或 safety 证据。
