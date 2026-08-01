# HFTF Stage C foot-ground student canary E0.1

日期：2026-08-01

状态：`FROZEN_BEFORE_FRESH_EVALUATION_RGB_DEPTH_OR_LABEL_OUTCOME`

## 1. 为什么是新 successor

E0 原样关闭为 `E0_FRESH_TEACHER_MECHANICS_NOT_EVALUABLE`：`.8 s` known coverage
仅 2/6 source 通过，而 `.4 s` 为 6/6。E0.1 不降低 `.70` 门，也不删除 E0 失败
source；它关闭 `.8 s` formulation，只保留已跨六 source 支持的 `.4 s`。

原 E0 dev/heldout 已 burned，不能继续评价新 formulation。E0.1 只复用原四条 train
作为 consumed training data，不获得 fresh evidence credit；dev/heldout 完全换新。

## 2. Fresh evaluation sources

从原 95 条 healthy inventory 排除 D0/D1 与 E0 共八条 consumed trajectories，仍按
总字节、trajectory ID 升序并要求不同 recording date，锁定：

| role | trajectory | date | rows | total bytes |
| --- | --- | --- | ---: | ---: |
| dev | `2024_12_01__15_29_33` | 2024-12-01 | 840 | 186,308,939 |
| heldout | `2024_07_10__11_01_46` | 2024-07-10 | 868 | 186,698,749 |

冻结时两条 RGB/depth 与 geometry-label outcome 均未读取。机器合同绑定每个
pose/RGB/depth 的 path、size 与 SHA-256；任何门失败不得换样。

## 3. Teacher 与学生合同

输出固定为 5 个方向 × `[current,.4 s]` 的 known/risk-proxy 两头。teacher 严格复用
D0 current 与 D1 `.4 s` causal-origin mechanics；学生只读
`anchor-2,anchor-1,anchor` RGB。

三臂共用 frozen MobileNetV3-Small、三 slot 和相同参数 head：

- `SF_FUTURE_0_4`：current RGB 复制三次，监督 current/.4；
- `HIST_CURRENT`：三帧 history，只监督 current，future 复制 current 概率；
- `HIST_FUTURE_0_4`：三帧 history，监督 current/.4。

训练 seed、优化器、阈值网格与 success margins 均在新评价媒体前冻结。

## 4. 顺序门与成功

新 dev/heldout 必须分别通过 exact transport、plane known `>=.95`、speed eligible
`>=.95`、`.4 s` candidate known `>=.70`、known loss/UNKNOWN→SAFE 为 0，以及各自
至少 2 个 risk cells/2 anchors 和 100 个 known no-risk cells。失败停止，不训练。

若允许训练，heldout co-primary 为 `.4 s` risk macro-F1。`HIST_FUTURE_0_4`
三 seed 中位数必须同时超过两个基线至少 `.03`，每 seed 增量为正，current-risk 与
known macro-F1 不劣于最佳基线超过 `.02`。

成功上限仍只是单一新 heldout trajectory 上的 `.4 s` foot-ground geometry-proxy
agreement canary；不支持 body/head、完整 HFTF、事件效果、研究主线、App 或安全。

机器可读真源：
[HFTF_STAGE_C_FOOT_GROUND_STUDENT_CANARY_E0_1_2026-08-01.json](HFTF_STAGE_C_FOOT_GROUND_STUDENT_CANARY_E0_1_2026-08-01.json)
