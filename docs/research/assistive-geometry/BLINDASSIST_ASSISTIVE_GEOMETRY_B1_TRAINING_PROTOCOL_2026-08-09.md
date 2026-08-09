# BlindAssist Assistive Geometry B1 training protocol

状态：`B1_PROTOCOL_FROZEN / IMPLEMENTATION_NOT_AUTHORIZED / FORMAL_TRAINING_NOT_AUTHORIZED`

机器合同：[JSON](BLINDASSIST_ASSISTIVE_GEOMETRY_B1_TRAINING_PROTOCOL_2026-08-09.json)

## 决策

B1 的 target、模型接口、A0–A4 additive arms、loss 数值、confidence calibration、optimizer、
seed、数据角色和停止条件已经一次性冻结。当前只授权 TRAIN target materialization、模型实现和
shape/backward/resume smoke；在 implementation lock 通过前仍不得启动正式 student training，
也不得打开 DEVELOPMENT/CONFIRMATION outcome。

## 模型最小形态

```text
RGB 1×3×608×448 + dynamic K
              ↓
DepthART-S metric encoder + camera adapters
              ↓
shared DPT feature: 48 channels, stride 4
       ┌──────────────┼────────────────────┐
       ↓              ↓                    ↓
metric depth      ground logit      fixed-third band pooling
                                           ↓
                                  shared 48→32 MLP
                                ┌──────────┼──────────┐
                                ↓          ↓          ↓
                           clearance   occupancy   confidence
                              [3]        [3,3]        [3]
```

新增 head 只使用 Conv、ReLU、Resize、ReduceMean、MatMul、逐元素算子、Sigmoid 和 Softplus，
保留后续 HTP-friendly co-design，但 B1 不因此获得部署权威。

## 三个语义冲突的关闭

1. B0 interface 写的是 `confidence[band,horizon]`，correctness 却要求一个 band 的三个 horizon
   全部正确。B1 将 primary confidence 明确定义为 `[3]` band-level correctness，并在
   GeometryState interface 上把每个 band 值重复到三个 horizon；不再训练九个语义冲突的值。
2. 无 intrusion 的 clear frame 没有有限 clearance 数值。它被定义为 censored-clear：不进入
   clearance regression，但只要 forward support 足够，仍进入 occupancy 与 confidence target；
   UNKNOWN 不因此变成 clear negative。
3. B0 gravity-ground reader 依赖 `up_camera`，而学习图输入保持 `RGB+K`。A0 评价使用 source
   pose 提供的 exogenous `up_camera`，所有 arm 共享且不作为 label；未来 runtime 缺 IMU/pose
   时必须输出 `UNKNOWN_GROUND_GEOMETRY`，不能假设相机水平。

## 数据角色

- TRAIN：16 videos / 4,800 frames，只用于 target materialization 与 fitting；
- DEVELOPMENT_CALIBRATION：4 videos，仅拟合 confidence temperature/threshold；
- DEVELOPMENT_SELECTION：4 videos，仅做 checkpoint/arm selection；
- CONFIRMATION：8 videos，继续 sealed。

原 8 个 DEVELOPMENT identity 先按 `visit_id, video_id` 排序，再固定前四 calibration、后四
selection；该划分发生在 outcome 打开前。已消费 120-frame cohort 与 DepthART R2 roster 继续
禁止复用。

## Loss 与近场权重

| Loss | lambda |
|---|---:|
| masked log-depth | 1.0 |
| valid-neighbor log-gradient | 0.5 |
| ground BCE | 0.5 |
| ground-plane depth residual | 0.25 |
| clearance Huber, delta 0.25 m | 1.0 |
| occupancy BCE | 1.0 |
| false-clear extra | 2.0 |
| confidence BCE | 0.5 |

`occupancy BCE + false-clear extra` 使 truth-occupied 的总正例权重精确为 3。dense metric loss
按 `0.25–2 m / 2–5 m / 5–6 m = 3/2/1` 加权，优先购买近场几何，而不是只刷全局 AbsRel。

## Confidence 冻结规则

confidence target 是 detached task correctness。对有 clearance truth 的 band：误差不超过
0.25 m 且三个有效 horizon 决策全部正确才为 1；对 censored-clear：三个 horizon 均有支持、
truth-clear 且预测均低于 0.5 才为 1。deterministic invalid 永远 mask，不是 0。

只在 DEVELOPMENT_CALIBRATION 上：

1. 在 `T=0.5..3.0, step 0.05` 中以 band-level NLL 拟合 temperature；
2. 在 threshold `0.50..0.95, step 0.05` 中找满足 false-clear、false-block、ECE 和
   valid-to-UNKNOWN 绝对门的候选；
3. 最大化 known coverage，平局取更小 threshold；无可行 threshold 直接 FAIL。

DEVELOPMENT_SELECTION 不得反向修改 temperature、threshold grid、loss 或 optimizer。

## 训练锁

- seeds：`17 / 29 / 43`；
- 20 epochs；micro-batch 4，gradient accumulation 4，effective batch 16；
- AdamW；encoder LR `2e-5`，heads LR `1e-4`，weight decay `0.01`；
- warmup 300 optimizer steps，cosine 到 `0.05×`，gradient clip 1.0；
- checkpoints：epoch `5/10/15/20`；
- teacher models 在 B1/B2 禁用。

## 唯一 successor

`BLINDASSIST_ASSISTIVE_GEOMETRY_B1_TARGET_MATERIALIZATION_AND_MODEL_IMPLEMENTATION_LOCK`

该 successor 只允许生成 TRAIN target cache、实现共享 decoder/head/loss，并运行 synthetic、
forward/backward、flip/K、resume smoke。implementation lock 未 PASS，不得启动正式训练或读取
DEVELOPMENT/CONFIRMATION outcome。
