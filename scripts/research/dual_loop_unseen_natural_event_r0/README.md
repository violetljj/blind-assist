# dual_loop_unseen_natural_event_r0

状态：development

## 研究问题与版本

`DUAL_LOOP_R1_UNSEEN_NATURAL_EVENT_R0` 在未参与 R1 设计或调参的连续自然步行视频上，
检验冻结提交 `039757b2da41c051373f8ee3189c4b06028f5295` 是否能减少完整误提醒窗口，
同时保持同一批 baseline 已命中正例及其逐事件时延。首次来源只产生 event-level
canary，不作总体外推、Confirmation、产品或安全结论。

## 稳定 Interface

来源选择必须先于 payload 与 baseline/candidate 输出访问：

```powershell
python -m scripts.research.dual_loop_unseen_natural_event_r0.select_source `
  --output-dir artifacts.local/evidence/dual-loop-r1-unseen-natural-event-r0/source-selection
```

selector 只读取 Wikimedia Commons category/API metadata，按协议固定的 eligibility、
精确已使用标题排除和 Unicode title 升序输出 registry 与选择 receipt。已存在输出会
拒绝覆盖。后继 truth ledger、baseline adequacy、candidate replay 和 evaluator 必须
绑定该 receipt、source bytes SHA-256 与各自实现哈希。

payload 下载并核对 SHA-256 后，使用下列两个独立入口形成盲审 bundle 和固定 10Hz
replay 输入。审阅 bundle 需要 Pillow；两个入口均拒绝覆盖已有目录或复用残留 `.tmp`
目录：

```powershell
python -m scripts.research.dual_loop_unseen_natural_event_r0.prepare_review_bundle `
  --video artifacts.local/downloads/dual-loop-r1-unseen-natural-event-r0/shanghai-shopping-street-480p.webm `
  --output artifacts.local/evidence/dual-loop-r1-unseen-natural-event-r0/review-bundle-r1 `
  --ffmpeg E:\codex-tools\media\ffmpeg\ffmpeg-8.1.2-essentials_build\bin\ffmpeg.exe

python -m scripts.research.dual_loop_unseen_natural_event_r0.prepare_input `
  --video artifacts.local/downloads/dual-loop-r1-unseen-natural-event-r0/shanghai-shopping-street-480p.webm `
  --output artifacts.local/evidence/dual-loop-r1-unseen-natural-event-r0/input-10hz-r1 `
  --ffmpeg E:\codex-tools\media\ffmpeg\ffmpeg-8.1.2-essentials_build\bin\ffmpeg.exe
```

## 输出

只写入显式 `artifacts.local/` 目录：

- `source_registry.json`：完整 metadata snapshot；
- `source_selection_receipt.json`：冻结规则、eligible 顺序、rank-1 source 与 derivative；
- 后继输入、truth、trace、评价与 learning record 各使用独立子目录。

## 安全边界

选择阶段禁止读取视频 payload、baseline、R1 candidate 或 truth outcome。事件真值必须在
baseline/candidate 前从冻结 RGB 形成，并标记为 model-reviewed evidence。active 仍只在
隔离设备回放中否决 simulated feedback-controller acceptance；不改 raw/stable risk、
目标选择、事件 identity/lifecycle 规则，也不声称物理播放或用户感知。

## 停止条件

单来源依次终止为：`FIRST_UNSEEN_SOURCE_NOT_EVALUABLE`、
`FIRST_UNSEEN_SOURCE_GUARDRAIL_FAILED`、
`FIRST_UNSEEN_SOURCE_NO_EVENT_LEVEL_EFFECT` 或
`FIRST_UNSEEN_SOURCE_EVENT_SIGNAL / SECOND_INDEPENDENT_SESSION_REQUIRED`。
窗口不得围绕输出裁切；candidate 打开后不得换来源、阈值、延迟容差或分母。若 rank-1
不可评价，只能完整披露该终点后按已冻结顺序启动新的独立 source instance。

## 假设与规则质疑

causal difference 是多框共同缩小对当前反馈的保守反证；expected information gain 是
完整负窗是否消失而非少数 row 被抑制；falsifier 是同 ID 正例丢失、超时或新增负窗；
成本是一次公开视频下载、模型复核和固定设备回放。该路线明确质疑“必须恢复完整运动
机制才能纠错”，但不降低证据隔离或把弱信号包装成效果。

## 失败资产复用

不可评价来源保留为 source-characterization；guardrail 失败保留为 counterexample；
无事件效果 trace 保留为 regression fixture。它们均不得重新包装成 unseen
Confirmation，也不得用于回调 R1。

“已使用”只限制同一候选、同一独立确认主张，不是对数据集的全局封存。旧 session
仍可用于机制开发、失败归因、回归测试或另一项预先冻结的问题；只需明确其
Development 身份。数据集缺少原生提醒事件标签也不是排除条件：允许在任何算法输出
打开前，从冻结 RGB 由两路隔离大模型复核并在分歧时裁决，形成 model-reviewed event
truth。不得把帧、滑窗或同一 capture 中的重复事件伪装成独立 session。

## 当前 rank-1 终点

上海 rank-1 的两路 canonical-prompt RGB 复核均为 0 个正例，已在 baseline 前以
`FIRST_UNSEEN_SOURCE_NOT_EVALUABLE / VALID` 关闭。`finalize_rank1_truth.py`
复核两份 AI receipt 的输入、prompt、身份、隔离与候选不可见性，发布 6 个一致负窗
及 terminal receipt。完成公开披露后，后继只可按原 registry 固定顺序启动 rank-2。

## 当前 rank-2 真值与执行入口

Shiraz rank-2 已在 baseline/candidate 均未打开时完成两路独立 RGB 复核与第三路
分歧裁决，冻结 7 个正例事件和 6 个负窗，状态为
`TRUTH_FROZEN_ADEQUATE`。模块内 `run_device.ps1` 将设备执行拆为
baseline-only 与 candidate-only；candidate 只有在 host evaluator 确认 baseline
至少命中 1 个正例且误触发 1 个负窗后，才接受哈希绑定的授权。candidate 重放
baseline trace 的同一 detections/metrics，并逐帧硬校验 raw/stable risk 不变。

```powershell
pwsh -NoProfile -File scripts/research/dual_loop_unseen_natural_event_r0/run_device.ps1 -Action Build
pwsh -NoProfile -File scripts/research/dual_loop_unseen_natural_event_r0/run_device.ps1 -Action Install
pwsh -NoProfile -File scripts/research/dual_loop_unseen_natural_event_r0/run_device.ps1 -Action PrepareInput
pwsh -NoProfile -File scripts/research/dual_loop_unseen_natural_event_r0/run_device.ps1 -Action RunBaseline
pwsh -NoProfile -File scripts/research/dual_loop_unseen_natural_event_r0/run_device.ps1 -Action CollectBaseline
pwsh -NoProfile -File scripts/research/dual_loop_unseen_natural_event_r0/run_device.ps1 -Action EvaluateBaseline
```

只有 `baseline-evaluation-r1/candidate_authorization.json` 存在且有效，才继续
`StageAuthorization / RunCandidate / CollectCandidate / EvaluateCandidate`。

## rank-2 终点

SM-S9280 上的固定 baseline/candidate 已完成。7/7 正例均保留，三项新增时延为
100 ms、其余为 0；5 个 baseline-false 负窗全部仍为 false，`corrected=0`。
全序列 accepted-feedback rows 为 `508 -> 494`，只构成密度信号。正式终点：

`FIRST_UNSEEN_SOURCE_NO_EVENT_LEVEL_EFFECT / DENSITY_SIGNAL_ONLY`

因此 active R1 关闭并保持默认 off；不在 Shiraz 上调参或增加 hold/latch。详见
`DUAL_LOOP_R1_UNSEEN_NATURAL_EVENT_R0_RANK2_EFFECT_RESULT_2026-07-31.md`。
