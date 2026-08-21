# P1-R0 consumed ADT baseline adapter protocol V1

状态：`CONSUMED / TERMINAL_RECORDED / CONSUMED_DEVELOPMENT_ONLY / DEFAULT_APP_UNCHANGED`

结果：[`REAL_RGB_PERSISTENCE_HEADROOM_ESTABLISHED_ON_CONSUMED_ADT`](P1_R0_CONSUMED_ADT_BASELINE_RESULT_2026-08-21.md)。

## 问题

本轮只回答：在 P0 已经一次性给出正确 physical referent 后，一个故意朴素、只读 RGB 的 tracker 在
P1-D0 的 15 个 consumed ADT episode 上如何失败。它不是 tracker 选型、模型搜索、fresh confirmation、
产品或安全验证。

## Truth firewall

执行固定为三个文件边界：

```text
P1-D0 episode truth + ADT source identity
  -> prepare-public
  -> public_input.json（RGB path、timestamps、首帧 oracle bbox；无 future truth）
  -> track
  -> prediction.json（bbox/null、P1 state/event；无 GT）
  -------------------- truth firewall --------------------
  -> evaluate（重新读取 P1-D0/ADT GT）
  -> metrics.json
```

`track` 子命令只接受 `public_input.json`，并拒绝额外字段。每个 episode 只在 frame 0 使用一次
`initial_target_bbox_xyxy`；frame 1..N 不读取 target UID、future bbox、visibility、temporal tag、loss/reappearance
时间、distractor identity 或 evaluator phase。`NO_REFERENT -> UNBOUND` 继续由已冻结 synthetic mechanics 专项
测试覆盖，不为真实 ADT 画面重复构造 P0 handoff claim。

## 固定 baseline

- OpenCV sparse Lucas-Kanade flow：从首帧 oracle bbox 内取点，只做局部平移连续性；弱 flow 时输出 null。
- 固定首帧 RGB crop 的灰度模板：进入 LOST 后每 3 帧做一次全帧、`0.8/1.0/1.2` 三尺度
  normalized-correlation 搜索；不更新模板。
- 重捕获候选需 `correlation >= 0.72`、top1-top2 margin `>= 0.05`，并在最近 3 次 search 中有 2 次
  spatial-compatible hit；确认前输出 null。
- 复用冻结 P1-R0 baseline state machine：无 candidate 的前 2 帧为 `TEMP_UNOBSERVABLE`，随后 `LOST`；
  只有 tracker 自己确认的 RGB candidate 才能触发 `REACQUIRED`。
- 无 detector、SAM、CoTracker、ReID、Sky、GT reset、future-truth initialization 或 oracle recovery。

Evaluator adapter 对每个 asserted bbox 与同 timestamp 的 ADT visible boxes 做 IoU 匹配；`IoU >= 0.30`
绑定最佳 source `object_uid`，否则绑定 episode-local background identity。这个映射只发生在 truth firewall
之后。冻结 P1 evaluator 本身不修改。

## 固定报告面

不生成总分。报告：wrong-instance asserted frames、identity switches、false reacquisition、最大 wrong-lock、
correct-identity coverage、temporary-occlusion recovery、reacquisition precision/recall、false-loss，以及按 episode
和 temporal tag 的分解。

## 三种穷尽终态

按以下顺序机械判定：

1. `correct_identity_coverage < 0.10`：
   `P1_R0_BASELINE_BELOW_PERSISTENCE_RESEARCH_FLOOR_OBSERVATION_FIRST`。
2. coverage 至少 0.10，且 wrong-instance、identity switch 或 false reacquisition 任一非零：
   `REAL_RGB_PERSISTENCE_HEADROOM_ESTABLISHED_ON_CONSUMED_ADT`。
3. 其余（没有 identity-safety failure）：
   `NO_MATERIAL_PERSISTENCE_HEADROOM_ON_CURRENT_DEVELOPMENT_COHORT`。

第三种终态不等于产品足够强；若 coverage/recovery 仍低，只能记录 utility 缺口，不能制造 identity-safety
研究主张。任何终态的 claim ceiling 都固定为
`CONSUMED_ADT_INDOOR_OBJECT_DEVELOPMENT_BASELINE_ONLY`。
