# RCLE 退化归因与 flow-quality diagnostic R0 结果

日期：2026-07-28

终态：`HOLD_FLOW_QUALITY_GATE / VALID`

## 决策

当前固定的 response-blind flow-quality gate **不值得优先进入候选实现**，但也不恢复
standalone rotation 路线。

在 ADVIO sequence13、14、15、17 的相同 601-pair 连续片段上，高 absolute
compensated response 最一致地集中在 pose-derived 步态振荡代理：`3/4` 个 session
的风险比 `>=1.5`。模糊代理和低纹理代理各为 `2/4`。固定 flow-quality gate 只有
sequence14 的高响应风险比 `>=1.5`，其余 session 为无富集、不可计算或反向富集。

因此当前证据更支持：

1. 下一轮若继续做机制研究，应先区分“高频头部/步态运动造成的真实、可追踪 flow”
   与 blur/texture 造成的测量退化；
2. 不应把低 feature count 直接当成高响应的统一替代指标；
3. 不继续叠加 rotation-only 模型；
4. 不进入 reference-track、bearing、Android 或风险性能评价。

这里的“风险比”是 session 内 descriptive prevalence ratio；pair 是纵向测量，不是
独立样本，也没有风险/障碍真值。

## 冻结设计与防火墙

[R0 合同](RCLE_DEGRADATION_FLOW_QUALITY_DIAGNOSTIC_R0_CONTRACT_2026-07-28.json)
在新代理提取前固定，SHA-256 为
`4efedd84acf45dac98500f66e8f70909034098fbb351ac29037c7eaa0edae67a`。

实验分两阶段：

- Stage 1 只读取已开放的 RGB、timestamp 和 source pose，禁止读取 R3 response、
  trigger、风险/障碍/人工事件标签；
- Stage 2 才把冻结 proxy ledger 与既有 R3 ledger 按 `session + pair_index`
  精确连接；
- sequence16 继续 `SEALED_UNSEEN`，无下载、提取、内容或算法访问；
- R3、strict `>0.01/s` 和三连续 evaluable pair 不变；
- gate 拒绝只新增 abstention/reset，不修改 accepted pair 的 response；
- 每个 session 固定 601-pair 分母，不 pooled 为 `n=2404`，不计算 AUROC/F1。

代理定义：

- 模糊：有效像素 Laplacian variance 的 session 内最低 20%；
- 低纹理：Shi-Tomasi feature density 的 session 内最低 20%；
- 步态振荡：translation/angular speed 相对 31-pair 居中 median 的高频残差组合，
  取 session 内最高 20%；它只是 head-motion proxy，不是人工 gait phase；
- 高响应：R3-evaluable pair 的 absolute compensated response 最高 20%。

固定 flow gate 要求 feature `>=60`、forward-backward consistent track `>=60`、
consistent fraction `>=0.5`、median round-trip error `<=0.75 px`、3×3 occupied
cell `>=5`。阈值没有根据 response 或标签调节。

## 退化归因

表中 RR 为高 absolute response 在代理退化层相对其余 evaluable pair 的 prevalence
ratio；capture 为该层捕获的高响应比例。

| session | blur RR / capture | low-texture RR / capture | gait proxy RR / capture | low-flow RR / capture |
| --- | ---: | ---: | ---: | ---: |
| 13 | 0.885 / 17.6% | 0.434 / 9.2% | **1.519 / 26.9%** | 0.000 / 0.0% |
| 14 | 1.130 / 21.0% | **2.327 / 36.1%** | **2.841 / 40.3%** | **2.557 / 30.3%** |
| 15 | **2.017 / 33.3%** | **2.689 / 40.0%** | 0.759 / 15.8% | `NOT_EVALUABLE` / 0.0% |
| 17 | **2.291 / 35.8%** | 0.336 / 7.5% | **1.901 / 31.7%** | 0.393 / 9.2% |

结果不是单一退化机制：

- 步态振荡代理在 13/14/17 一致富集；
- 模糊代理在 15/17 富集；
- 低纹理代理在 14/15 富集；
- fixed flow gate 只在 14 富集。

blur 与 texture proxy 仍可能互相混杂；gait proxy 也没有人工步态标签，所以只能形成
下一轮可证伪机制问题，不能称为已确认原因。

## Flow gate 结果

| session | 全 601 pair 拒绝率 | 高响应 RR | 原始 / gated 三-pair 密度 | 相对下降 |
| --- | ---: | ---: | ---: | ---: |
| 13 | 1.16% | 0.000 | 0.5291 / 0.5291 | 0.0% |
| 14 | 15.47% | 2.557 | 0.5225 / 0.4426 | 15.29% |
| 15 | 0.17% | `NOT_EVALUABLE` | 0.4043 / 0.4043 | 0.0% |
| 17 | 20.97% | 0.393 | 0.8985 / 0.7621 | 15.19% |

预冻结推进门要求四项各至少 `3/4` session 通过。实际为：

| decision check | 通过 session 数 |
| --- | ---: |
| 拒绝率在 2%–30% | 2/4 |
| gate-rejected 高响应 RR `>=1.5` | 1/4 |
| 三-pair 密度相对下降 `>=20%` | 0/4 |
| 至少一个命名代理在 gate rejection 中 RR `>=1.5` | 3/4 |

唯一满足跨 session 条件的是 gate 与某个退化代理存在重叠；它没有进一步稳定定位高
response，也没有达到预冻结的 trigger-density 降幅，因此终态为
`HOLD_FLOW_QUALITY_GATE`。

拒绝几乎完全由 feature/track support 主导：

- sequence14：93 个拒绝中 93 个 `FB_TRACKS_LT_60`，85 个 `FEATURES_LT_60`；
- sequence17：126/126 个 `FB_TRACKS_LT_60`，124 个 `FEATURES_LT_60`；
- sequence13、15 分别只有 7 和 1 个拒绝。

将 median forward-backward error 门从 `0.75 px` 改为敏感性边界 `0.5` 或
`1.0 px`，四个 session 的拒绝率和 gated trigger density 均不变；说明当前结果不是
round-trip error 阈值造成，而是低 feature support gate 主导。

## 判读边界

本实验没有风险、障碍或接近标签，因此不能回答：

- 被 gate 移除的 trigger 是否是假警；
- gait/blur/texture 高响应是否对应真实接近；
- gate 是否保留正响应、改善 recall/precision 或提高安全性；
- 新 session、其他数据来源或 Android 上是否成立。

合理的后继不是调低/调高当前 gate 直到通过，而是另立 Development R1：在不看风险
标签的前提下，预先冻结一个能把“高频但 track-consistent 的 head motion”与
“feature collapse / blur”分开的时间结构诊断。只有该诊断在多个 session 稳定分离，
才值得冻结候选；否则保留 `HOLD`。

## 复现与验证

- Python `3.11.9`
- NumPy `2.1.3`
- OpenCV `4.10.0`
- 5 项 focused tests：`PASS`
- 分析：
  `artifacts.local/evidence/rcle_degradation_flow_quality_diagnostic_r0/session_analysis_r0.json`
  SHA-256
  `07ee6482f04ecb773ca0167a57ea6878b2a84abdd17c3e2d8a51ec300b30ea04`
- 独立 validator：`VALID / failures=[]`
- validation receipt SHA-256：
  `41c5f0a8b1fd66368586d5c8dc00766b8155f39d7751e0e31b56e9bb423109a6`

四份 proxy ledger 的 SHA-256 为：

- sequence13：
  `636d8e02ac8700196f1895f7b0764c1945b61f9b76fc6c1e6aa3dd7eaffd48f8`
- sequence14：
  `be9df679892d27c141c4407eabf2c8416bf9f651f663b26a09ba98b316f02499`
- sequence15：
  `b90b23a457c956594344b677aa7a75e9a179810cafcd6dda7307fda5eef9ffbe`
- sequence17：
  `4a4470da1b7a2c876bb390dac53d675aa2b0ec398f6f97026e4900563cd299ca`
