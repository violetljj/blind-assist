# USTRF 跨相机 Codex 代理评测 R0（2026-07-21）

状态：首个真实 Android 对比闭环完成；仅 provisional proxy，不是人工真值或生产授权。

## 结论

当前下一方向不是先微调模型，而是先建立“相机域适配 + 外部路线投影 + 多轮 Codex 弃权/共识”的跨设备评测层。R0 已在公开 POV 视频上把当前 Android `detector_bbox_explicit_route` 与 Codex 教师/因果基线放进同一 500ms 因果窗口。首个可评分负样本中，Codex 六轮一致认为右侧人行道无路线内事件；Android 臂在首帧对道路车辆产生一次 `HIGH` 路线告警。设备回执显示该车框到路线最短距离为 `48.4636px`，而冻结走廊半宽为 `51.2px`，以 `2.7364px` 的边界余量被保留。优先变量因此是跨相机路线投影和走廊几何，不是 detector AP 或大模型微调。

## 数据与合同

- MuSoHu `frontal_approach_1.MP4`：George Mason University Dataverse datafile 806，CC0 1.0，16,994,976 bytes；下载 MD5 `90a3ffc4188c99b049a65c0894124b66` 与官方元数据一致。原始 360° 全景被误当透视输入时，三轮 full-context Codex 对风险分别给出 critical/none/caution，按 2/3 门判为 `NO_TWO_OF_THREE_CONSENSUS`。随后 70° 透视投影又暴露 forward yaw 无可信设备收据和近场手臂遮挡，故该样本停止，不进入 USTRF 排名。
- Pexels item 3874684：Joe Valdes 的连续行人 POV，Pexels License，SHA-256 `c9567af0...df36cc`。R0 使用视频前 6 秒、250ms 教师帧、500ms 因果帧；人工近似路线沿右侧人行道，所有设备/米制/U0/训练/生产权限均为 false。
- Codex 角色：`full_context_teacher` 可见 24 帧，只生成 provisional silver；`causal_codex_baseline` 只见 12 个 500ms 帧。每个角色三轮隔离评审，输入清单、prompt、schema、输出和 bundle 均 SHA-256 绑定。
- Android 角色：同一原视频、同一 12 帧和同一路线进入现有 `run_ustrf_sc_u0_android_bbox_route_adapter.py`；SM-S9280 真实运行 shipped YOLO11n TFLite、bbox route gate 与 shared Kotlin kernel。没有用 host 伪造风险 trace。

## 首轮数字

| 候选 | Codex provisional reference | 事件 | 代理结果 |
| --- | --- | ---: | --- |
| causal Codex | 0 事件 | 0 | false alerts/min `0.0` |
| Android detector+bbox+route | 0 事件 | 1 | false alerts/min `10.0`（仅 6 秒诊断窗） |

Android 逐帧关键证据：`t=0ms` 检出 2 辆车，route gate 保留 1 辆，raw/stable=`HIGH/HIGH`、feedback=`TRIGGERED`；`t=500ms` gate 保留 0 辆，但 stable 仍为 `HIGH`；`t=1000ms` 恢复 `NONE`。12 帧 PTS 误差为 `0..5500us`，均低于 20ms 设备门；单帧 TFLite inference 约 `42–44ms`，total detect `53–105ms`。

`10.0/min` 是把 1 次事件除以 0.1 分钟的短窗换算，只用于暴露首帧边界假阳性；它不能外推到真实长期假告警率。reference 也是 Codex 共识，不是人类事件真值，因此报告只允许比较和发现失败模式。

## 产物

- MuSoHu 失败闭环：`artifacts.local/evidence/ustrf-crosscam-codex/musohu-frontal-approach-1-r0/`
- Pexels 可评分闭环：`artifacts.local/evidence/ustrf-crosscam-codex/pexels-3874684-negative-r0/`
- 关键文件：`teacher_consensus.json`、`android_bbox_route_output.json`、`ustrf_android_candidate.json`、`proxy_report.json`。
- 可复用入口：`scripts/research/ustrf_crosscam_codex/`；生成物继续只写 ignored `artifacts.local/`。

## 下一轮冻结方向

1. 先增加至少 `3 source × 正/负各 2` 的小规模公开 pinhole/head-mounted 集，按 source 分开报告；不要在同一 Pexels 负样本上调窄 `0.08` 走廊回救。
2. 每个 source 在看风险答案前冻结 projection mode、forward axis、路线 polygon 和一组宽/中/窄几何敏感性；报告 worst geometry，而非只报最好宽度。
3. Codex full-context 若三轮无 2/3 共识，样本直接记为 teacher abstention，不生成事件分数；它证明“可以直接用 Codex”，也证明不能把单次回答当真值。
4. 正样本加入后再比较 event recall、critical miss、onset/clearance；当前只有负样本，不能评价召回。
5. 细调或训练必须等待公开代理结果在 source-held-out 上稳定，并继续与正式 120/60 自动多模型事件参考、设备米制几何、App/生产门完全隔离。

## R1 后续

路线投影收据、polygon 走廊和边界不确定性已在 [路线投影与走廊几何 R1](USTRF_CROSSCAM_ROUTE_PROJECTION_CORRIDOR_R1_2026-07-21.md) 实现并完成同设备测试。R0 的原始结果保持不变；R1 是新的研究审计层，不回写或重解释 R0 指标。
