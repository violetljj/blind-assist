# SANPO GPU 利用率与训练隔离报告

- 日期：2026-07-13
- 设备：NVIDIA GeForce RTX 5060 Laptop GPU，8151 MiB
- 训练栈：Windows 原生 PyTorch 2.11.0+cu130、Keras 3.15、mixed float16
- 当前结论：GPU 路线可用且吞吐已优化；正式训练仍因来源 inventory 与程序化原始证据未完全闭合而关闭。

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

## 验证与限制

- `test_sanpo_v3_dataset_controls.py`：9 tests passed；其中覆盖 blind 目录移除后的授权消费与 manifest 篡改拒绝。
- `test_train_export_sanpo_segmentation.py`：10 tests passed，1 skipped。
- 相关 Python 文件 `py_compile` 通过，`git diff --check` 通过。
- 此次没有重新训练或导出 TFLite。此前 canonical green、dev 指标与 TFLite 结果已降级为 audit-only；source receipt inventory 逐资产绑定、Guide 原图/YOLO polygon 的程序化来源证明及跨后端数值等价验证完成前，不得作为正式候选或复制到 App assets。

## 复现命令

```powershell
& 'E:\codex-tools\tools\venvs\blindassist-torch-gpu\Scripts\python.exe' scripts\benchmark_sanpo_gpu_throughput.py --batches 16,32,64,96 --steps 8 --warmup 2
& 'E:\codex-tools\tools\venvs\blindassist-torch-gpu\Scripts\python.exe' scripts\benchmark_sanpo_gpu_throughput.py --batches 96,112,128 --steps 8 --warmup 2
```
