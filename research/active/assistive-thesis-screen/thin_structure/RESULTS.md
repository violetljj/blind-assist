# 细结构与计算分配：50 帧小规模验证

2026-10-15，EXPLORE / 已消费 Development。**不支持把“低分辨率＋挑局部放大”直接作为已有效的毕业论文方法。** 两个实际运行的预训练模型给出的结论不同；在更贴近街景的 Cityscapes 模型上，最佳低成本基线仍是整图 256。更高分辨率覆盖了更多标签组，却降低了总体像素 IoU，适合继续研究“对象覆盖与细节完整度”的取舍，尚不能称避障收益。

## 实际做了什么

固定复用已存在的 SANPO 官方 train、人工注释 50 帧 / 10 段（2 段头戴、8 段胸前），全部读取，45 帧含 pole；未扩选、未读保护集。原生 RGB/mask 为 2208×1242，评价正类为 SANPO class24，正类共 **1,356,188 像素**，有效标注像素共 **134,785,833**。class0 unlabeled 排除，不把 UNKNOWN 当背景。

pole 有 **82 个标签组帧**：62 个正 ID 组帧来自 23 个 `session/camera/class/instance` 组合键，另 20 个 instance0 组帧身份未知；不能称为 82 个独立杆。标签包括较粗灯柱、附属部分和边界截断，不能把整组面积当杆身宽度。5 帧/段高度相关。

已实际运行本地 SegFormer-B0 ADE20K，以及新下载的 SegFormer-B0 Cityscapes。后者固定 revision `21b3847fae21ddee674abd31129307b6a1235bd9`，5 个文件含元数据合计 **14,965,088 B**。两者无训练；ADE 预测取 pole/streetlight，Cityscapes 取 pole，类别对应不完全相同。

同模型比较：整图 256、整图 512；低分辨率预测平均熵最高的两个象限，或 RGB 竖向细节残差最高的两个象限，各以 256 输入再次推理并替换局部输出。两个选择器都不读取 mask，均覆盖图像 50% 面积，输入像素数为整图 512 的 75%。Cityscapes 另跑模型标称输入 **整图 1024**。裁块未经专门训练，硬替换是本轮简洁实现，不等于 SparseRefine。

## 结果：以 Cityscapes 为主要参考

以下是排除 unlabeled 后的指标。像素精度/召回的分母分别为预测正像素/人工 pole 像素；组帧覆盖定义为对应标签组至少 20% 像素被预测为 pole，20% 是本轮预先写入脚本的诊断门槛，并非安全检出门槛。

| 方法 | pole IoU | pole 像素召回 | 正 ID 组帧覆盖 | 所有标签组帧覆盖 | forward 中位 |
|---|---:|---:|---:|---:|---:|
| 整图 256 | **23.73%** | **29.44%** | 12/62 | 17/82 | **11.12 ms** |
| 整图 512 | 18.68% | 23.60% | 19/62 | 23/82 | 14.63 ms |
| 整图 1024 | 18.17% | 23.70% | **28/62** | **36/82** | 70.41 ms |
| 256＋熵选两块 | 20.19% | 25.63% | 18/62 | 24/82 | 32.21 ms |
| 256＋RGB细节选两块 | 19.95% | 25.66% | 15/62 | 23/82 | 32.65 ms |

与最佳低成本整图 256 相比，RGB 局部方案 IoU **−3.79 个百分点**、召回 **−3.78 个百分点**，forward 约 **2.94 倍**；82 组帧在 20% 门槛下救回 13、丢失 7，净 +6。与整图 512 相比，RGB 局部方案同门槛救回 5、丢失 5，净 0。输入像素减少没有换来本机的推理加速，三次串行模型调用有额外成本。

整图 1024 的 any-hit 为 58/82，整图 256 为 37/82；其中正 ID 分别为 39/62、21/62。这个指标只要求一个重叠像素，不能当完整目标检出。较大分辨率增加标签组覆盖，但大面积目标、分类和背景误报仍会支配像素 IoU，两个指标不能互换。

ADE 初轮整图 256/512、熵块、RGB块 IoU 为 5.75/7.38/8.41/8.67%，RGB块虽相对改善，绝对召回仍只有 9.26%。换成街景预训练后收益方向改变，故不依据较弱 ADE 基线宣布局部方法有效。

RGB 选择器用 50% 面积覆盖 88.70% pole 标签像素，Cityscapes 熵选择为 81.93%；**选中目标区域不等于修复目标预测**。这支持把下一疑点放在特征/语义保真与上下文融合，而不是继续为选择器调阈值。该覆盖率是像素加权，可能偏向大柱，不证明对每根细杆都选得好。

## 表示诊断与下一方向

人工 mask 单独缩为 256 后按占比≥0.5 二值化再放回原图，pole 像素召回 **93.85%**、IoU **88.64%**，82/82 标签组仍≥20%支持；512 对应 96.97%、93.96%。这是 **oracle 表示诊断**，输入就是人工真值，绝不是模型收益，也不能证明 RGB 降采样不丢信息。它只说明本样本的标签几何没有在 256 层面整体消失，当前跨域识别错误明显大于该特定表示操作的损失。

可形成两个研究方法点，但均需真正胜过基线后才能确定：

1. **语义漏检感知的预算分配**：让 RGB 原生细线/边缘证据与低分辨率预测的分歧触发局部计算，特别覆盖“模型很自信但漏掉了杆”的区域；与纯熵选择、固定区域和整图分辨率比较。收益假设是粗模型无法给消失目标提供足够不确定性；最弱假设是图像细节能区分真实杆与栏杆/纹理，本轮仅证明像素覆盖，未证明检出收益。
2. **保留上下文的细结构更新**：共享整图特征，在细化时保留低分辨率类别/拓扑上下文，并围绕正 ID 组的支持完整度分配更新；后续才考虑带失效机制的因果时序记忆。收益假设是硬裁块/硬替换破坏上下文，导致“选对了仍预测错”；最弱假设是共享特征或时序能减少损失且不留下过期假目标，本轮未训练、未验证时序。

**推荐顺序：先固定 Cityscapes/SANPO 任务适配的强基线，再试共享特征的上下文细化。** 不建议现在把粗到细或 RGB 边缘本身作为创新点；[SparseRefine（ECCV 2024）](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/08392.pdf)已经采用低分辨率预测、熵选择、稀疏高分辨率细化与门控融合。本轮没有复现 SparseRefine，也未跑 SANPO 训练模型或当前 SOTA；**强任务基线 NOT_RUN**，实际跑过的是两个轻量跨域预训练参考及高分辨率参考。

## 核查、失败与成本

同图 512、4 CPU 线程、3 次热态 forward：ADE CPU/GPU 中位 143.99/6.15 ms，Cityscapes 124.09/14.82 ms，采用 RTX 5060 Laptop GPU。两轮脚本主体分别 29.11/28.91 s；峰值 CUDA allocated 约 1.71/0.66 GB。进程包含库导入等的 GPU 总占用保守≤120/300 s，CPU command-wall 含环境探查、下载、评分保守≤180/600 s。forward 不是端到端、手机或无竞争负载时延；两个不同模型的时间只供本次执行说明。

两轮均 50/50 完成、推理失败 0。Cityscapes 第一次 urllib 下载出现 TLS EOF，保留已成功元数据后以 requests 取得同 revision，未关闭证书校验。搜索中的一个 CVF 直链失效，改核 SANPO 官方仓库。原始评价误将 class0 unlabeled 算负例，已修评分器并从保存的预测独立重建 **650 个候选mask的混淆计数**（含100+100 oracle表示候选）：TP/FN 完全一致，FP 按有效标注区域减少；旧值原样保留，报告只用 `evaluated/`。没有因此重新训练、调参或重跑推理。

实际查看两模型各10段首帧的 contact sheet：语义目标与图像大体对应，明显漏杆与背景误识同时存在；此为缩略图检查，不重新裁定人工边界。脚本语法和范围内 diff 检查完成。任务模型进程已结束，无需保留服务。报告是小样本 Development，不是独立确认、真实距离或导航/安全结论。

## 产物与复现

- `run_pilot.py`：模型与oracle表示实验；默认ADE。输入/权重/样本SHA、固定参数见各轮 manifest。后续脚本已排除class0，旧轮从保存预测重评分。
- `fetch_cityscapes.py`：无凭据公开下载，50 MB脚本上限；revision和SHA见下载收据。
- `rescore_and_audit.py`：独立读取保存预测、修正unlabeled、配对组帧计数和逐段汇总。
- payload：`artifacts.local/work/assistive-thesis-screen-20261015/thin_structure/`；各轮 `evaluated/summary.json`、`evaluated/frame_metrics.csv` 是当前数字；原 `summary.json` 和原 `frame_metrics.csv` 为未排除void的历史，不用于最终报告。`instance_metrics.csv`、`selectors.json`、`predictions/`、`contact_sheet.jpg` 保留完整候选证据。
- payload 中的 `RESULTS.json` 为紧凑结果，`evaluation_audit.json` 为重算收据，`runtime_receipt.json` 记录本机解释器。下面使用已配置的 `$thinPython`。

```powershell
& $thinPython -X utf8 research/active/assistive-thesis-screen/thin_structure/run_pilot.py --out artifacts.local/work/assistive-thesis-screen-20261015/thin_structure/replay_cityscapes --model-dir artifacts.local/work/assistive-thesis-screen-20261015/thin_structure/cityscapes_b0 --target-ids 5 --strong1024 --max-wall 150
```

数据来源合同沿用[已核50帧人工来源](../../hardware-bringup/DATA_OPPORTUNITY_CATALOG_20261013.md)，本轮直接核读[SANPO官方仓库](https://github.com/google-research-datasets/sanpo_dataset)、[Cityscapes模型入口](https://huggingface.co/nvidia/segformer-b0-finetuned-cityscapes-1024-1024)及上述SparseRefine原文。未下载数据全集。
