# dual_loop_segmentation_residual_aware_ddrnet

状态：development；`COMPLETE / VALID / FP_WEIGHTED_SAMPLING_NOT_SUPPORTED /
SINGLE_SUCCESSOR_STOP / THREE_PAIRED_SEEDS / CONSUMED_DEVELOPMENT_ONLY`

## 研究问题与版本

本 Module 是条件门精确三臂家族关闭后的训练后继。R0 只实现
`FP_WEIGHTED_UNGUIDED_FULL_FRAME`：保持 DDRNet、四类真值、loss、70% hazard-guided
crop、三 seed 和训练预算不变，只改变 30% unguided full-frame branch 在已选 session
内的 frame probability。

虽然目录名覆盖 residual-aware / FP-aware 训练方向，本轮协议没有 train YOLO union，
因此结果必须称为 `FP_AWARE`，不得冒称 residual-aware。

## 稳定 Interface

```powershell
python -m scripts.research.dual_loop_segmentation_residual_aware_ddrnet.train ...
python -m scripts.research.dual_loop_segmentation_residual_aware_ddrnet.evaluate ...
python -m scripts.research.dual_loop_segmentation_residual_aware_ddrnet.validate ...
```

`train --preflight-only` 只核验输入并从 train role 构造 same-seed FP weights，不训练候选、
不访问 candidate outcome。正式训练、320-frame consumed evaluation 与 validation 使用
互不覆盖的输出。validator 会重新装载六个 same-seed checkpoints 并在 320 帧上重新
推理，只有 prediction masks、checkpoint hashes、逐帧账本和聚合指标全部一致才可写出
`VALID`；其余执行或合同异常统一 fail closed。

## 输出

本地输出固定在：

```text
artifacts.local/evidence/dual-loop-segmentation-fp-aware-ddrnet-r0/
```

不得写入历史 R1/R2-P0、Android asset 或默认模型位置。

## 安全边界

- `train` 只用于 weight 构造和训练；
- `dev` 只用于原 checkpoint rule；
- terminal 只用已 consumed 的 320 帧；
- 不访问 fresh/Confirmation，不运行 INT8/runtime/device；
- 不接 Android、risk/feedback、TTS、振动、提醒或默认 App。

## 停止条件

三个 same-seed pair 必须各自通过冻结的 relative 五门与 absolute 四门。任何少数 seed
正信号都不能救援终态；R0 后不在同一 consumed outcome 上继续调 sampler、loss 或 target。

完整协议见
[FP-aware DDRNet R0 protocol](../../../docs/research/dual-loop/DUAL_LOOP_SEGMENTATION_FP_AWARE_DDRNET_R0_PROTOCOL_2026-08-01.md)。

正式结果见
[FP-aware DDRNet R0 result](../../../docs/research/dual-loop/DUAL_LOOP_SEGMENTATION_FP_AWARE_DDRNET_R0_RESULT_2026-08-01.md)。
三个 same-seed pair 均未通过全部九门；正式终态为
`FP_WEIGHTED_SAMPLING_NOT_SUPPORTED`。不得在相同 consumed outcome 上选择 seed、
改 sampler/crop、加 loss 或调 target 救援。
