# RCLE ecological response discovery R0

状态：`discovery`

当前归类：`CAPABILITY_DISCOVERY / OUTPUT_INSPECTED`。上位方法与访问规则以
[RCLE current](../../../../docs/research/rcle/README.md)和
[数据能力驱动主线 R2](../../../../docs/research/rcle/RCLE_DATA_DRIVEN_RESEARCH_MAINLINE_R2_2026-07-28.md)
为准。

## 研究问题与版本

`RCLE_ECOLOGICAL_RESPONSE_DISCOVERY_R0` 观察冻结 RCLE RGB pair core、raw local
expansion、source-pose rotation-compensated local expansion、三连续 pair 状态机和
简单 global image-scale proxy 在自然连续相机运动中的响应、支持率与失败模式。

首个 evidence instance 使用已经本地化的 ADVIO office03 sequence 15。它是一个已看过
内容、曾用于非 RCLE 机制探针的单一 capture session，只能产生 Discovery 证据。
该 session 已查看 RCLE 输出，不能再作为同一命题的 sealed evaluation。

## 稳定 Interface

从仓库根目录运行：

```powershell
& E:\codex-tools\projects\blindassist\toolchain\venv-corridor-causal-py311\Scripts\python.exe `
  -m scripts.research.egomotion_compensated_looming.ecological_response_discovery_r0.runner `
  --source-root artifacts.local/evidence/public-advio-r792-turn-intent-20260719/extracted/advio-15 `
  --output-dir artifacts.local/evidence/rcle_ecological_response_discovery_r0/advio15_native60_r0 `
  --resize-scale 0.5
```

输入要求为 ADVIO `iphone/frames.mov`、`iphone/frames.csv` 与
`ground-truth/pose.csv`。runner 校验视频帧数、时间戳、pose 四元数、单调性和本地文件
SHA-256；拒绝覆盖既有输出。`--max-pairs` 只用于 runtime pilot。
`--start-frame` 可把同一预声明连续区间切成不重叠的有序运行块，以适配宿主单调用
时限；相邻块应让后一块从前一块的末帧开始，避免遗漏边界 pair。使用
`aggregate_chunks.py` 按帧索引校验并合并时，必须披露每个块开头的 RCLE track
state 重置；聚合器只跨块重算三连续触发状态，不声称恢复了未分块的底层信号。

`--resize-scale 0.5` 只缩小空间分辨率，并按相同尺度共轭变换旋转单应矩阵；
时间戳、原生约 60 Hz pair 顺序、阈值和三连续 pair 规则保持不变。它是低成本
Discovery 配置，不与全分辨率结果宣称数值等价。

## 输出

只写入指定的 `artifacts.local/evidence/rcle_ecological_response_discovery_r0/`
子目录：`pair_ledger.jsonl`、`segment_summary.jsonl`、`summary.json`、
`response_curves.png` 和 `progress.json`。

## 安全边界

本轮不使用风险/告警真值，不计算 AUROC、F1、性能或泛化，不训练、不选阈值、不修改
Android。`bbox_growth` 因无冻结目标框而为 `NOT_EVALUABLE`；image-scale proxy 不是
bbox baseline。旧 `RGB_SEGMENT_CONFIRMATION_R1_NOT_EVALUABLE` 终态保持不变。

只看 RGB 内容本身不会自动烧毁 future evaluation；本 sequence 被限制为
Discovery/Development，是因为已经查看了 RCLE 输出。未来 evaluation 必须在任何
算法调试前按 person/capture session/route/sequence 预留，不能从本长视频随机切
clip 伪造独立性。

## 停止条件

视频、时间戳、pose、标定或核心接口无法闭合时只停止本 evidence instance。代表性
pilot 若预计全序列超过 3 分钟，先保留 pilot，并使用 guarded host runner 或缩到预先
声明的连续 Discovery 片段；不得把 runtime 失败写成算法失败。

## 假设与规则质疑

预先观察 raw/compensated/image-scale 的响应分布、支持率、触发密度、角速度关联和
失败片段，不预设 RCLE 胜出。source-pose compensation 使用 ADVIO 官方 sequence
13–17 iPhone 标定；不从输出反推相机参数。

## 失败资产复用

本数据和输出可继续用于 development、failure analysis、regression、stress 和 demo，
不得重新包装成 sealed unseen evaluation 或外部确认。
