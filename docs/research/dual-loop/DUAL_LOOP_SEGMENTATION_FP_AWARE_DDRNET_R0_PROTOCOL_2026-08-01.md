# FP-aware DDRNet R0 Development Protocol

状态：`DESIGN_FROZEN / IMPLEMENTATION_FROZEN /
PREFLIGHT_VALID_NOT_EXECUTED /
SINGLE_SUCCESSOR / THREE_PAIRED_SEEDS /
CONSUMED_DEVELOPMENT_ONLY / FINAL_CONFIRMATION_NOT_ACTIVATED /
DEFAULT_APP_UNCHANGED`

日期：2026-08-01（Asia/Hong_Kong）

## 研究问题

条件门 R0/R0.1 已表明，三个固定阈值静态手工门只能交换 FP 与 recall，不能同时保护
最弱 session 和两个 hazard classes。下一主边界不再调整 gate，而是问：

> 在同一 DDRNet-23-Slim、同一四类真值、同一 loss 和预算下，仅将原 30% unguided
> full-frame 分支改为 train-only baseline-FP 加权 full-frame 抽样，能否在三个
> same-seed 配对中都产生稳健的 pixel/component utility increment？

协议 ID：

```text
DUAL_LOOP_SEGMENTATION_FP_AWARE_DDRNET_R0
```

唯一候选：

```text
FP_WEIGHTED_UNGUIDED_FULL_FRAME
```

本轮诚实命名为 `FP_AWARE`，不是 `RESIDUAL_AWARE`。hard-negative 定义没有引入
train split 的冻结 YOLO box union，因此不能声称它专门学习 YOLO residual。它仍属于
用户授权的 residual-aware / FP-aware DDRNet 后继边界。

## 唯一变量

每个 seed 都使用同 seed 的历史 R1 DDRNet checkpoint，在 `train` 400 帧上生成：

```text
FP_pixel =
    baseline argmax in {boundary_step_curb, obstacle}
    AND canonical truth in {walkable, unknown_nonwalkable}
```

这里的 `walkable/unknown_nonwalkable` 只是 canonical 四类合同中的非候选类，不能解释
为现实安全背景。

原 R1 和本候选都先 uniform 选择 train session。70% branch 原样保留 hazard-guided
crop；其中 boundary/obstacle 为 `0.65/0.35`。只有剩余 30% branch 改变：

```text
R1 baseline:
    session 内 uniform frame draw
    -> unchanged full frame

FP-aware candidate:
    P(frame | session) =
        frame_same_seed_FP_pixel_count / session_same_seed_FP_pixel_count
    -> unchanged full frame
```

因此不把 full frame 改成 crop，不同时改变尺度/上下文；若任一 seed 的任一 train
session FP 权重总和为零，则在训练前终止为 `NOT_EVALUABLE`，不搜索替代规则。

逐项保持不变：

- 官方 DDRNet-23-Slim 架构与 ImageNet 初始化、四类新 head；
- 400 train / 200 dev membership；
- 70% positive-guided crop、`0.65/0.35` 类别比例、crop/flip；
- weighted CE `0.5` + Dice `0.4` + focal `0.1`，以及 train-only class weights；
- Adam、100-step head warmup、其后 backbone fine-tune；
- 每 seed 1200 optimizer steps，seed `20260711/12/13`；
- 每 seed 仍以 dev mIoU 与 boundary IoU 的调和平均选择 checkpoint。

R0 不加入 residual relabel、YOLO-conditioned train weight、ignore/uncertainty target、
新增 FP loss、component balancing、hard-negative crop、confidence threshold 或新模型。

## 数据角色

| role | rows / sessions | 本轮用途 | 禁止 |
|---|---:|---|---|
| `train` | 400 / 8 | 同 seed FP weight 构造、candidate training | terminal evaluation |
| `dev` | 200 / 4 | 原样 checkpoint selection | terminal、policy tuning |
| `consumed_old_blind` | 120 / 2 | 结果前冻结的 Development stress | training、checkpoint selection |
| `r1_consumed_fresh` | 200 / 4 | 结果前冻结的 Development stress | 恢复 fresh 身份、training、selection |

终态只用后两项 320 帧、6 session；Atlas 中的 dev 200 不进入 terminal。全部数据都已经
consumed，本轮不访问或创建 fresh/Confirmation outcome。

## 三 seed 配对与门

每个 candidate seed 只与同 seed 的冻结 R1 baseline checkpoint 比较。不得选最好
candidate seed；三个 seed 都必须通过全部九门。

Relative 五门复用 conditional-gating 的阈值：

```text
FP pixel reduction >= 0.30
overall recall retention >= 0.90
minimum-session recall retention >= 0.80
boundary recall retention >= 0.80
obstacle recall retention >= 0.80
```

Absolute utility 四门复用 R1：

```text
C-A delta recall >= 0.05
C-A delta false-positive area fraction <= 0.05
candidate component recall >= 0.50
false activation components/frame <= 3.0
```

candidate mask 仍是 `predicted hazard AND outside frozen YOLO box union`；truth 是
canonical hazard outside 同一 box union；connectivity 为 8，component hit 为任意正
交集。session 是最低独立单位，pixel/frame 是重复观察。

## 终态预冻结

```text
全部输入、训练、评价与验证合同有效，
且三个 same-seed pair 各自通过 relative 5 + absolute 4：
    FP_WEIGHTED_SAMPLING_SUPPORTED_DEVELOPMENT_ONLY

任一输入、训练、membership、identity、metric 或 validation 合同失效：
    FP_WEIGHTED_SAMPLING_NOT_EVALUABLE

其余全部有效结果：
    FP_WEIGHTED_SAMPLING_NOT_SUPPORTED
```

`1/3` 或 `2/3` seed 有信号只能作诊断，不能救援终态、选择 seed、改变权重、改成 crop
或在同一 consumed outcome 上继续堆 loss/target。

即使 supported，也只授权另立 INT8 fidelity/runtime successor；不选择默认模型，不产生
Android、QNN/A568、risk/event、feedback、TTS、振动、提醒、产品或安全权限。

## 可执行合同

Config：
[configs/dual_loop_segmentation_fp_aware_ddrnet_r0/default.json](../../../configs/dual_loop_segmentation_fp_aware_ddrnet_r0/default.json)

Module：
[scripts/research/dual_loop_segmentation_residual_aware_ddrnet/](../../../scripts/research/dual_loop_segmentation_residual_aware_ddrnet/)

预期本地输出：

```text
artifacts.local/evidence/dual-loop-segmentation-fp-aware-ddrnet-r0/
  training/
  evaluation/
  validation.json
```

执行顺序：

```powershell
$python = 'E:\codex-tools\tools\venvs\blindassist-venv-export312\Scripts\python.exe'

& $python -m scripts.research.dual_loop_segmentation_residual_aware_ddrnet.train `
  --config configs/dual_loop_segmentation_fp_aware_ddrnet_r0/default.json `
  --device cuda `
  --preflight-only

& $python -m scripts.research.dual_loop_segmentation_residual_aware_ddrnet.evaluate `
  --config configs/dual_loop_segmentation_fp_aware_ddrnet_r0/default.json `
  --preflight-only

& $python -m scripts.research.dual_loop_segmentation_residual_aware_ddrnet.train `
  --config configs/dual_loop_segmentation_fp_aware_ddrnet_r0/default.json `
  --device cuda

& $python -m scripts.research.dual_loop_segmentation_residual_aware_ddrnet.evaluate `
  --config configs/dual_loop_segmentation_fp_aware_ddrnet_r0/default.json `
  --device cuda

& $python -m scripts.research.dual_loop_segmentation_residual_aware_ddrnet.validate `
  --config configs/dual_loop_segmentation_fp_aware_ddrnet_r0/default.json `
  --result artifacts.local/evidence/dual-loop-segmentation-fp-aware-ddrnet-r0/evaluation/result.json `
  --output artifacts.local/evidence/dual-loop-segmentation-fp-aware-ddrnet-r0/validation.json `
  --device cuda
```

所有 runner 拒绝覆盖非空 output。训练过程写 machine-readable progress；FP32-only
评价不创建 TFLite、runtime 或 production asset。validator 除了从预测账本独立复算
逐帧和汇总指标，还会重新装载三组 baseline/candidate checkpoints，在同一 320 帧上
重新推理，并要求每个压缩 prediction mask 逐像素一致；任何解析、identity、hash、
inference 或 aggregation 异常都 fail closed 为 `FP_WEIGHTED_SAMPLING_NOT_EVALUABLE`。

## Outcome-blind preflight

候选训练前的只读 preflight 已完成。三个 same-seed R1 baseline 均在 400/400 train
frames、全部 8 train sessions 上产生正 FP 权重，因而没有触发 zero-session
`NOT_EVALUABLE`：

| seed | train FP pixels | covered frames | covered sessions |
|---:|---:|---:|---:|
| 20260711 | 818,645 | 400 | 8 |
| 20260712 | 1,088,041 | 400 | 8 |
| 20260713 | 2,089,096 | 400 | 8 |

这些是读取 canonical train truth 后产生的 train-only sampler metadata，不是 candidate
result，也不影响门或终态。独立 evaluation membership preflight 也已验证 consumed
320 rows、6 sessions 与 320 条冻结 YOLO trace 一一对应。两个 preflight 都没有训练
candidate，也没有读取 320 帧 canonical truth pixels 或 terminal outcome。
