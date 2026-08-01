# dual_loop_dg_srf_structural_complementarity_f0

状态：Development；`COMPLETE / VALID /
STRUCTURAL_SIGNAL_NOT_SUPPORTED_STOP`

协议：
[`DG_SRF_IMAGE_SPACE_STRUCTURAL_COMPLEMENTARITY_F0_PROTOCOL_2026-08-01.md`](../../../docs/research/dual-loop/DG_SRF_IMAGE_SPACE_STRUCTURAL_COMPLEMENTARITY_F0_PROTOCOL_2026-08-01.md)

正式结果：
[`DG_SRF_IMAGE_SPACE_STRUCTURAL_COMPLEMENTARITY_F0_RESULT_2026-08-01.md`](../../../docs/research/dual-loop/DG_SRF_IMAGE_SPACE_STRUCTURAL_COMPLEMENTARITY_F0_RESULT_2026-08-01.md)

520-frame 执行与 29,031 项独立复算已完成。D1-D4 均未形成跨组 stable signal，D4
只在 1/10 组优于最佳单信号，九门只过 4/9；当前精确定义的 F0 已关闭。下文保留冻结
接口与可复算入口。

## 研究问题

本 Module 只回答：

> 固定的 Depth Anything V2 Small 单帧相对逆深度中，是否包含能在实际 YOLO
> 覆盖区之外，以低于冻结 raw DDRNet residual 的假激活代价，对
> `boundary_step_curb / obstacle` canonical hazard pixels 提供跨
> source-session 稳定互补的图像空间结构信号？

它不检测或标注“未知障碍”，不恢复米制距离、地面高度、真实路线或事件风险。输入
520 帧和 outcome 全部已消费，结论上限为 consumed Development。

## 稳定 Interface

使用已安装 PyTorch GPU 环境：

```powershell
$py = "E:\codex-tools\tools\venvs\blindassist-torch-gpu\Scripts\python.exe"
$root = "artifacts.local/evidence/dg-srf-image-space-structural-complementarity-f0"

& $py -m scripts.research.dual_loop_dg_srf_structural_complementarity_f0.prepare `
  --repo-root . `
  --config configs/dg_srf_image_space_structural_complementarity_f0/default.json `
  --output-root "$root/prepared-v1"

& $py -m scripts.research.dual_loop_dg_srf_structural_complementarity_f0.produce `
  --repo-root . `
  --config configs/dg_srf_image_space_structural_complementarity_f0/default.json `
  --prepared-root "$root/prepared-v1" `
  --output-root "$root/producer-v1" `
  --mode full --device cuda

& $py -m scripts.research.dual_loop_dg_srf_structural_complementarity_f0.evaluate `
  --repo-root . `
  --config configs/dg_srf_image_space_structural_complementarity_f0/default.json `
  --prepared-root "$root/prepared-v1" `
  --producer-root "$root/producer-v1" `
  --output-root "$root/evaluation-v1"

& $py -m scripts.research.dual_loop_dg_srf_structural_complementarity_f0.validate `
  --repo-root . `
  --config configs/dg_srf_image_space_structural_complementarity_f0/default.json `
  --prepared-root "$root/prepared-v1" `
  --producer-root "$root/producer-v1" `
  --evaluation-root "$root/evaluation-v1" `
  --output "$root/validation-v1.json"
```

Pilot 只允许在正式 inference 前验证模型、方向、尺寸、输出健康与序列化：

```powershell
& $py -m scripts.research.dual_loop_dg_srf_structural_complementarity_f0.produce `
  --repo-root . `
  --config configs/dg_srf_image_space_structural_complementarity_f0/default.json `
  --prepared-root "$root/prepared-v1" `
  --output-root "$root/pilot-v1" `
  --mode pilot --pilot-count 8 --device cuda
```

Pilot 不访问 canonical truth、A/B packed masks 或终态指标。

## 固定计算

官方语义把输出冻结为 affine-invariant inverse depth，方向固定为
`RAW_LARGER_IS_NEARER`。synthetic canary 只能验收或失败，不能选择 sign。raw 数值只在
单帧内作稳健归一化，不允许跨帧比较或解释为米。

四个子信号 `N / E / R_plus / R_minus` 分别固定到 `[0,1]`。D4 为严格
`1:1:1:1` 等权；D3 固定同时保留正、负 surface-trend residual，不允许 outcome 后挑
符号。先生成结构场，再乘 `1 - A` 做 YOLO residualization。D5 只是固定
`lambda=.25` 的 lower-center image-space proxy 消融，不能叫路线或走廊。

LOSO 的 held-out session 不参与阈值选择。19 点阈值网格固定最大化九门的最小规范化
margin；没有全过点时仍选择 least-shortfall 点并写
`NO_ALL_GATE_OPERATING_POINT`。

## 输出与独立验证

全部运行结果写入 ignored 的 `artifacts.local/`：

- `prepared-v1/inference_manifest.jsonl`：只含身份、角色与 RGB 路径/hash；
- `producer-v1/depth_maps.npy`、`depth_index.jsonl`、canary 与 runtime receipt；
- `evaluation-v1/`：group AUPRC、LOSO 阈值、逐帧 operating metrics 与 provisional
  terminal；
- `validation-v1.json`：不 import producer、operators 或 evaluator 的 reference
  arithmetic，从 raw depth、truth、A/B mask 重新计算并给出最终 terminal。

validator 的 `VALID` 只证明保存的 depth output 之后的结构算子、指标和终态可独立
复算；checkpoint 到 depth map 由固定 source/checkpoint/preprocess hash 和 canary
约束，不伪装成第二次 520 帧推理。

## 安全边界

10 个组全部来自 SANPO-Real，缺 participant、route、parent-capture identity；两套
YOLO detector 又与 source role 完全混杂。grouped consistency 不是独立人群、路线或
跨数据集泛化证据。

## 停止条件

唯一科学终态：

```text
STRUCTURAL_SIGNAL_SUPPORTED_FOR_F1_DESIGN
SIGNAL_PRESENT_BUT_COMPOSITE_NOT_READY
STRUCTURAL_SIGNAL_NOT_SUPPORTED_STOP
NOT_EVALUABLE
```

正终态最多授权另立 F1 设计，不自动授权 F1 执行。其他终态下不得在相同 520 帧上改
权重、梯度尺度、trend、形态学、lambda，或引入 Video Depth/时序 latch 救援。无论
终态如何，均不接 Android、QNN/A568、提醒、TTS、振动、risk/feedback 或默认 App。

## 假设、规则质疑与失败资产复用

新因果变量是预训练相对深度的结构表征，不是继续挽救已关闭的 segmentation gate。
falsifier 是 D1-D4 没有跨组稳定增量，或 D4 无法同时约束九项 utility、类别分裂和最弱
组。负结果仍可保留 raw depth、health failure 和 per-session AP 作为论文反例、测试
fixture 与 visual-only diagnostic，但不能成为产品检测能力。
