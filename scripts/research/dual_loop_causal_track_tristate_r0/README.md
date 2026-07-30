# Dual-loop causal track tri-state R0

状态：`confirmation_passed / android_shadow_integrated`

## 研究问题与版本

`DUAL_LOOP_CAUSAL_TRACK_TRISTATE_R0` 检验一个选择性纠错源：同一稳定 track
连续 7 帧框高的 log-OLS 趋势只有在 6 次相邻变化严格同号且
`|slope| >= 0.2/s` 时才输出 `CONFIRM_APPROACH` 或
`CONTRADICT_APPROACH`，其他情况一律 `ABSTAIN`。选择原因是 8 个 burned
Development 会话中该固定机制跨会话复现了高精度，同时避免把低可信连续值当成
逐帧风险加分。

Confirmation evidence instance 固定 3 个 metadata-only 哈希选择的 JRDB
会话。先冻结会话/窗口，只下载 2D 标签并封存 source 输出，之后才下载 3D
标签评价。允许 claim 仅为 annotation-track mechanism confirmation。

## 稳定 Interface

稳定入口为 `scripts/run_dual_loop_causal_track_tristate_r0.py`：

```text
freeze -> acquire --role source_2d -> produce
       -> acquire --role truth_3d -> evaluate
```

`produce` 在 truth 目录已存在时 fail closed。所有 source 行绑定
`frame_detection_id + track_id + track_epoch + immutable_roi_id`。

## 输出

只写
`artifacts.local/evidence/dual-loop/causal-track-tristate-r0/confirmation/`；
freeze、两阶段 acquisition receipt、producer JSONL/receipt 和 result 均拒绝覆盖。

## 安全边界

输入 box 与 track identity 仍来自 JRDB annotation，不是 live detector；
3D annotation range 只作离线 truth。Confirmation 通过后允许把同一冻结数学规则
移植到隔离 Android shadow；Android 使用当前 production-selected detection 和
轻量 IoU/中心距离连续性，而不是 JRDB annotation track，因此仍需 live evidence。
它不授权 active fusion、提醒逻辑变更、产品、人因或安全效果。普通
`DualLoopShadowAdmitter()` 的 allowlist 仍为空；只有 kernel 的显式 shadow 默认值
准入 `CAUSAL_TRACK_TRISTATE_R0`，且无法触达 event/feedback seam。

## 停止条件

任一 input/hash/阶段顺序漂移立即停止。三会话中的任一会话若 evidence <20、
coverage <0.5% 或 precision <90%，或 pooled confirm/contradict precision
任一 <90%，则 terminal 为 `ANNOTATION_TRACK_SOURCE_CONFIRMATION_NOT_MET`。
失败只关闭本固定选择性 source，不关闭纠错式双环问题。

## 假设与规则质疑

该候选质疑“source 必须高覆盖才有价值”：纠错环的首要损失是不可靠反证造成的
正例伤害，因此优先冻结高精度、可弃权的稀疏证据。其 falsifier 是独立会话精度
或最低 coverage 不过门；执行成本仅为 metadata 与标签 JSON，不需要 RGB、
模型或 GPU。

## 失败资产复用

失败会话可作为 track discontinuity、box jitter、遮挡与 panorama stitching 的
counterexample/regression fixture；不得重新包装成 unseen confirmation。
