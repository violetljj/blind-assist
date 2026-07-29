# RCLE periodic self-motion R2 QMS-R1 four-block DEV diagnostic R0

日期：2026-07-29（Asia/Hong_Kong）

## 终态

```text
SCIENTIFIC STATUS: PERIODIC_SELF_MOTION_SENSITIVITY_OBSERVED_IN_CONTROLLED_DEV
PROTOCOL STATUS: VALID / DEV_DIAGNOSTIC_COMPLETE
CLAIM CEILING: CONTROLLED_GENERATOR_INTERNAL_DEVELOPMENT_DIAGNOSTIC_ONLY
FORMAL AUTHORITY: UNCHANGED_NOT_CONSUMED
SUCCESSOR FORMAL 480+16: NOT_RUN
```

本任务完成了一个不替代正式推断的轻量 RCLE 测试。它使用四个 motion
blocks、每 block 两个全新 scene seeds、每个 scene 六个配对 arms，共
`8 clusters / 48 sequences / 28,896 frames / 28,848 pairs`。每条 sequence
保留完整 `602 frames / 601 ordered pairs`，因此三连 trigger 与 abstention
reset 的时间结构没有被抽帧破坏。

## 身份与执行隔离

新的 48 identities 对旧 formal、QMS-R1 DEV、QMS-R1 CAL、旧 PREFLIGHT、
successor formal 和 QMS-R1 activation PREFLIGHT 六个域，在
`numeric_seed_uint64`、`token`、`token_sha256`、`cluster_id`、
`sequence_id` 和 `scene_geometry_sha256` 六类字段上重叠均为零。

W8 以一个 worker 对应一个 scene cluster 的方式运行；每个 cluster 精确包含
六臂、`603` 次 QMS `render_pair`（static `1`、static reuse `601`、
periodic `602`）。8/8 clusters 原子完成，运行约 `37.8 min`；启动可用内存
`8.12 GiB`，最低 `6.53 GiB`，swap in/out 均为零，最大 heartbeat 间隔
`20.03 s`，无残留 worker。

## 描述结果

以下均以 scene cluster 为观察单位，仅为描述统计：

| contrast | cluster mean | median | range | sign |
|---|---:|---:|---:|---:|
| `MOTION_CLEAN` | 0.25 | 0.25 | 0.18–0.29 | 8 positive / 0 negative |
| `BLUR_STATIC` | 0.00 | 0.00 | 0.00–0.00 | 8 zero |
| `LOW_TEXTURE_STATIC` | 0.00 | 0.00 | 0.00–0.00 | 8 zero |
| `MOTION_X_BLUR` | 0.02 | 0.02 | -0.02–0.05 | 5 positive / 3 negative |
| `MOTION_X_LOW_TEXTURE` | 0.00 | 0.01 | -0.05–0.06 | 5 positive / 3 negative |

static clean、blur 和 low-texture 的平均 trigger density 均为 `0.00`；
periodic clean 为 `0.25`，periodic blur 为 `0.27`，periodic low-texture
为 `0.26`。因此，在这 8 个全新受控 scene clusters 中，周期自运动相对静态
产生了一致的正 trigger-density 差异；这说明 QMS-R1 下的 RCLE 仍对该周期
自运动构造敏感。

blur interaction 与 low-texture interaction 均混合，不支持跨 block 的统一
质量交互方向。尤其 low-texture 的 tracking quality 较弱：clean-trackable
为 static `6/8`、periodic `5/8`，平均 quality-failure density 分别约
`0.25` 和 `0.14`；因此 low-texture interaction 不应被解释为稳定算法效应。

## 独立验证与边界

独立 validator 未导入 DEV producer、QMS operator、R3 transport 或正式
analysis。它独立派生 seed/token/scene/trajectory 和六臂 grid，并从全部
`601-row` ledgers 重算严格 `>0.01/s`、连续三 pair、abstention reset、
tracking quality、arm summaries 与五个 cluster contrasts。终态为：

```text
VALID / DEV_DIAGNOSTIC_COMPLETE
```

本任务没有 bootstrap、置信区间、p 值、max-t 或 formal classification。
每个 block 只有两个 seeds，不能宣称总体效应、自然视频有效性、产品性能或
安全性，也不能替代冻结的 480+16 confirmatory design。

它能回答的开发问题是：RCLE 在全新受控 scenes、完整时间序列和冻结 QMS-R1
条件下是否仍会表现出周期自运动敏感性。答案是“会，且 8/8 clusters 同向”。
因此若当前目的只是判断是否存在值得修复的 RCLE 自运动问题，不必为这个问题
再支付 11 小时正式运行成本。

successor formal 的一次性授权没有被消费、撤销或改写；正式输出目录仍不存在，
正式 sequence 与正式 R3 pair-core call 均为零。
