# SANPO GPU 利用率与训练隔离报告

- 日期：2026-07-13
- 设备：NVIDIA GeForce RTX 5060 Laptop GPU，8151 MiB
- 训练栈：Windows 原生 PyTorch 2.11.0+cu130、Keras 3.15、mixed float16
- 当前结论：GPU 路线可用且吞吐已优化；正式训练仍因来源 inventory 与程序化原始证据未完全闭合而关闭。

> 训练协议更新：来源闭环后的首轮真实重跑证明 batch 64/`val_loss` 早停不适合当前小数据。后续正式比较改用 [SANPO 分割训练协议 v2](SANPO_TRAINING_PROTOCOL.md)：默认 batch 12、固定 optimizer-step 预算、session-balanced rare-class crop、CE+Dice+Focal、mIoU/boundary IoU 联合 checkpoint 和三 seed 稳定性报告。本文下方 batch 64 结论仅是历史吞吐/首轮重跑记录，不再是优化默认值。

## 根因与修正

原共享模型按空间尺寸选择特征时，把 LR-ASPP 高层语义分支接到了 MobileNetV3 的 1/16 浅层，导致后半段主干不在 Keras functional graph 中。模型实际只有 197,212 个参数，工作负载过小；当前数据量也只有数百帧，每轮 batch 数很少，因此 GPU 利用率低。

现已把高层分支固定到排除 squeeze/excitation 1x1 张量后的最深 8x8 特征，低层分支保持 32x32，输出保持 256x256x4。修正后参数量为 670,588。GPU 训练入口同时增加：

- 默认 batch 64；
- mixed float16 和高精度 TensorFloat-32 matmul 策略；
- 可选 Keras/Torch JIT；
- fit 总耗时、images/s 和 CUDA 峰值显存记录；
- dev 数组预载复用，避免训练后再次串行读取图片。

## 独立合成吞吐实测

测速由 `scripts/benchmark_sanpo_gpu_throughput.py` 完成，仅使用设备端合成张量，不读取 train/dev/blind 数据。

| Batch | 吞吐 images/s | 单步耗时 ms | 峰值 CUDA 显存 |
|---:|---:|---:|---:|
| 16 | 101.31 | 157.93 | 0.74 GB |
| 32 | 190.76 | 167.75 | 1.46 GB |
| 64 | 357.93 | 178.80 | 2.88 GB |
| 96 | 430.34 | 223.08 | 4.32 GB |
| 112 | 449.27 | 249.29 | 5.04 GB |
| 128 | 480.85 | 266.20 | 5.74 GB |

batch 128 的纯吞吐最高，但对 300 帧训练集每轮仅约 3 个 optimizer step，会明显改变优化行为。默认选择 batch 64，在约 1.88 倍于 batch 32 的吞吐下保留更多更新步数和约 5 GB 显存余量；batch 96/128 仅用于明确控制学习率、epoch 与优化协议的实验。

## Blind 隔离状态

门禁生成与训练授权消费已拆开：

1. 独立门禁进程可以读取两个 `benchmark_only` blind session，校验 300+120、四类掩码、来源、隐私、session 隔离和所有 SHA256，并生成报告及 sidecar。
2. GPU 与 TensorFlow 训练入口只消费预生成报告，校验固定 `training_manifest.jsonl` 和单独列出的 train/dev 资产哈希。
3. 训练入口不再调用 `run_gate()`；即使 `blind_holdout` 被移走或不可访问，授权消费仍可完成。

SHA256 sidecar 能检测误改，但不是抵抗本机恶意伪造的密钥签名。当前威胁模型以不可绕过的固定入口、报告绑定和自动哈希校验为准。

## 跨后端数值等价门

GPU 权重不能再直接交给 TensorFlow 导出。`scripts/sanpo_backend_equivalence.py` 会在两个独立解释器中，用共享的 `sanpo_segmentation_model.py` 图加载同一份 `.weights.h5`，对 4 张由固定 seed `20260713` 生成的 256×256 RGB 合成输入分别运行 Keras Torch 与 TensorFlow 推理。该过程不读取 dataset root、train/dev manifest 或 blind holdout。

等价报告 schema 已升级为 `blindassist_sanpo_backend_equivalence_v2`。`backbone_alpha`、`decoder_channels` 与 `input_size` 不再是调用方默认猜测：orchestrator 会把三者显式传给两个 worker，将完整 `model_config` 及 canonical JSON SHA256 写入带 sidecar 的报告；导出与质量门 consumer 必须提供相同预期配置，任何字段缺失、alpha/decoder/输入尺寸错配或配置哈希变化都会在加载权重前拒绝。输入尺寸只允许 256、384、512；固定合成输入的 shape 和 SHA256 随分辨率变化，数值阈值仍保持 `max_abs <= 1e-4`、`argmax agreement >= 0.9998`，不会为 512 消融放宽。

阈值在正式重训前固定为 `max_abs <= 1e-4` 且逐像素 `argmax agreement >= 0.9998`。一次未训练随机权重的集成 smoke 得到 `max_abs=7.15e-7`、`argmax agreement=0.999851`，因此阈值能容纳已观察到的跨框架浮点运算次序差异，同时最多允许约万分之二的像素类别分歧。报告同时绑定权重 SHA256、共享模型定义 SHA256、等价检查工具 SHA256、固定输入 SHA256、解释器路径和阈值，并生成报告 sidecar。

```powershell
& .\.venv-export312\Scripts\python.exe scripts\sanpo_backend_equivalence.py `
  --weights test-artifacts.local\segmentation-candidate\<run>\mobilenetv3_lraspp.weights.h5 `
  --torch-python E:\codex-tools\tools\venvs\blindassist-torch-gpu\Scripts\python.exe `
  --tensorflow-python .\.venv-export312\Scripts\python.exe `
  --backbone-alpha 0.75 `
  --decoder-channels 96 `
  --input-size 512 `
  --report test-artifacts.local\segmentation-candidate\<run>\backend_equivalence.json
```

只有 `status=green` 且 `export_authorized=true` 的报告可传给导出入口：

```powershell
& .\.venv-export312\Scripts\python.exe scripts\train_export_sanpo_segmentation.py `
  --dataset-root test-artifacts.local\datasets\<canonical> `
  --training-gate-report qa\training_gate_report.json `
  --import-weights test-artifacts.local\segmentation-candidate\<run>\mobilenetv3_lraspp.weights.h5 `
  --backend-equivalence-report test-artifacts.local\segmentation-candidate\<run>\backend_equivalence.json `
  --backbone-alpha 0.75 `
  --decoder-channels 96 `
  --input-size 512 `
  --export-only
```

缺失报告、sidecar 不匹配、阈值被放宽、权重或模型定义发生变化、固定输入契约不一致、`max_abs`/`argmax agreement` 任一失败时，导出入口会在导入 TensorFlow和写入 TFLite 前拒绝执行。该绿灯只授权 benchmark-only 导出，不构成模型晋级；设备端连续序列、独立 blind 事件指标和生产晋级门仍须单独通过。

## 验证与限制

- evidence-v4 最终 canonical 门禁已全绿，最终根报告 SHA256 为 `32968a7afa081f122cee463e6578feba6efea65172f81a9a0d4341dbf7af23d4`；旧 canonical/模型结果仍是 audit-only。
- 修正模型的 GPU 重跑使用 batch 64、seed `20260711`、`no-jit`，早停于 8 epoch；fit 约 25.80 秒，训练吞吐约 62.02 images/s，峰值 CUDA 显存 2,253,943,296 字节。dev mIoU `0.1711`，boundary/curb IoU `0.0000144`，不满足候选质量要求。
- 正式权重首轮 Torch↔TensorFlow 等价结果为 `max_abs=0.0875044`、`argmax agreement=0.990448`。逐层比较确认权重、BN、SE 与 resize 不是根因，漂移来自 Torch GPU worker 默认 CuDNN TF32；工具现强制 TF32-off 精确 float32 契约并由 consumer 校验。原阈值不变，正式复跑为 `max_abs=0.0000634193`、`argmax agreement=1.0`，报告 SHA256 `11646481868c209aa768f67c7a6ca6238c67f5efbd6fd867a40583a45372e3db`。由于 dev 质量门仍失败，本轮不导出 TFLite。
- `test_sanpo_v3_dataset_controls.py`：9 tests passed；其中覆盖 blind 目录移除后的授权消费与 manifest 篡改拒绝。
- `test_train_export_sanpo_segmentation.py`：10 tests passed，1 skipped。
- `test_sanpo_backend_equivalence.py`：覆盖固定输入确定性、指标计算、报告/sidecar/权重/阈值绑定和失败拒绝。
- 相关 Python 文件 `py_compile` 通过，`git diff --check` 通过。
- 此次已在新证据门禁全绿后重新训练，但没有导出 TFLite。正式权重未通过跨后端等价门且 dev 质量较差，不得作为正式候选或复制到 App assets。

## 复现命令

```powershell
& 'E:\codex-tools\tools\venvs\blindassist-torch-gpu\Scripts\python.exe' scripts\benchmark_sanpo_gpu_throughput.py --batches 16,32,64,96 --steps 8 --warmup 2
& 'E:\codex-tools\tools\venvs\blindassist-torch-gpu\Scripts\python.exe' scripts\benchmark_sanpo_gpu_throughput.py --batches 96,112,128 --steps 8 --warmup 2
```
