# HFTF Stage C D36：THOR-MAGNI production track-veto event replay

日期：2026-08-03

证据角色：Development / production decision-kernel event bridge

研究主线：不变

默认 App：不变

## 问题

D33 已证明 detector-track 七帧状态可预测约一秒后的同一行人 range direction，
D34 已证明 production Kotlin state 与 Python source rule 零漂移。D36 不等待
D35 的设备性能结论，独立回答：

> 在已有真实连续事件 proxy 上，把 production
> `CausalTrackTristateGeometryProducer` 的 `CONTRADICT_APPROACH` 只用于否决当前
> feedback opportunity，是否能相对 production `AssistDecisionKernel` OFF 减少
> negative false-active，同时不损失 positive event recall？

D35 继续负责 Android parity/runtime；D36 负责离线事件层科学效用。两者并行，
任何一方的工程状态都不阻塞另一方。

## 固定 cohort

复用 D12/D24 的 THOR-MAGNI proximity-eligible Development cohort：

- 19 个真实记录 source sessions；
- 530 anchors；
- 157 positive onset anchors、373 negative anchors；
- 同 source 内连续 positive anchors 按 D24 的 `45 scene frames` gap 合并为
  107 positive events；
- positive 定义仍为 current `>1.25 m`、未来窗口内首次进入 `<=1.25 m`；
- 不修改 fold、event grouping、anchor identity 或 truth。

该 cohort 已 outcome-open，只能建立 decision bridge Development 证据；不能作为
未来主线晋级所需的独立 generalization。

## source replay

- 视频、SHA、anchor 与 source identity 原样继承 D31；
- detector 原样继承 D31：
  YOLO11n、Ultralytics `8.4.102`、person only、`imgsz=640`、
  confidence `0.10`、NMS IoU `0.50`、`max_det=30`；
- 每个 anchor 独立取以 anchor 结尾的七帧 causal window；
- 以视频声明 FPS 选择最接近 15 Hz 的固定 frame step，使七帧时间跨度接近 D33；
- source producer 不读取 onset label、future range、crossing time 或 fold outcome；
- anchor-frame detector count 与 D31 cache 必须逐 sample 一致；D31 top-8 selected
  normalized boxes 的最大绝对误差必须 `<=1e-5`。

## production Kotlin replay

每个 anchor window 独立 reset：

1. baseline arm：production `AssistDecisionKernel`，mode `OFF`；
2. candidate arm：同一 production kernel、同一 detections/timestamps；
3. production `CausalTrackTristateGeometryProducer` 只消费 baseline raw-risk 当前
   selected target；
4. evidence 以 production admission contract 注入 candidate arm，
   mode `ACTIVE_CONTRADICT_ONLY`；
5. 只有 admitted `CONTRADICT_APPROACH` 可以把当帧 feedback reason 改为
   `DUAL_LOOP_CONTRADICTED`；
6. candidate 不得修改 raw risk、stable risk、target selection 或创建新提醒；
7. 每个 window 记录任意 frame 是否触发 feedback、首次触发位置、contradiction
   admission 与路径 parity。

这不是把已有 scene-scale active route 当作 track 结果；D36 显式注入并核验
`CAUSAL_TRACK_TRISTATE_R0` production evidence。

## evaluability gates

1. 530/530 anchors、19/19 sessions 完整回放；
2. detector anchor-frame 与 D31 count/selected-box parity 完整；
3. baseline/candidate raw-risk 与 stable-risk mismatch 均为 0；
4. baseline positive alerted anchors 至少 20；
5. baseline negative alerted anchors 至少 20；
6. admitted `CONTRADICT_APPROACH` 至少 10 anchors、覆盖至少 5 sessions。

任一不足：

`D36_THOR_MAGNI_PRODUCTION_TRACK_VETO_EVENT_NOT_EVALUABLE`

## support gates

在可判定前提下全部满足：

1. candidate positive-event losses = 0；
2. candidate positive-anchor losses <= 1，且 recall delta >= `-0.01`；
3. negative alerted anchors 至少减少 10；
4. negative alerted anchors relative reduction >= `0.20`；
5. 至少 3/5 folds 的 negative alerted anchors 减少；
6. candidate-only triggered frames = 0。

通过：

`D36_THOR_MAGNI_PRODUCTION_TRACK_VETO_EVENT_SUPPORTED`

可判定但未通过：

`D36_THOR_MAGNI_PRODUCTION_TRACK_VETO_EVENT_NOT_SUPPORTED`

无论终态如何，都保留 baseline/candidate positive anchor recall、positive event
recall、negative false-active、fold/source 分解、admission coverage 与 suppression
原因。

## 工程失败

路径、视频 decode、CUDA、dependency、Gradle、TSV、serialization、落盘或中断失败
都是 repairable engineering failure。修复后按同一冻结输入与规则重跑；不烧毁
cohort，也不产生科学负终态。中间输入和报告使用临时文件加原子替换。

## 主张边界

D36 通过只建立：在 outcome-open THOR-MAGNI event proxy 上，production
track-state contradict-only feedback veto 相对 production OFF 形成 paired event
utility increment。它不建立 Android device runtime、CameraX continuity、独立数据
generalization、默认 App、研究主线替换、产品效果或 human safety。

只有 D35 device canary 与后续独立 event cohort 都通过，才允许把该 track-veto
路线列为传统主线的正式晋级候选。
