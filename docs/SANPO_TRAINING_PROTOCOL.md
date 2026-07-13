# SANPO 分割训练协议 v2（benchmark-only）

本协议解决旧训练中“200 帧 + batch 64 = 每轮仅 4 次更新”、类别极不平衡、只按 `val_loss` 选模以及单 seed 偶然性的问题。它只改变候选训练与审计方法，不读取 benchmark-only blind，不授权导出或替换 App 模型。

## 固定比较单位

- 训练预算按 **optimizer step** 计算，不按 epoch；默认每个 seed 最多 `1200` step，至少 `300` step 后才允许早停。
- 默认优化 batch 为 `12`。GPU 吞吐 batch 与优化 batch 是两个独立问题；batch 64/96/128 只用于吞吐测量，不再作为小数据训练默认值。
- 每 `50` step 在 dev 上评估一次。checkpoint/早停监控值为 dev mIoU 与 `boundary_step_curb` IoU 的调和平均，因此任一项坍塌为 0 时评分也为 0。
- 默认运行三个预注册 seed：`20260711,20260712,20260713`。报告保存均值、标准差、最差 seed、每个 seed 的完整混淆矩阵和 checkpoint 轨迹；正式结论不能只挑最好 seed。

## 两阶段稳定训练

真实三 seed 首轮显示 alpha 1.0 的最佳 checkpoint 多出现在 50–100 step，随后 boundary IoU 明显坍塌。默认训练因此改为两阶段，但保留 `--no-two-stage` 作为审计对照：

1. `head_warmup`：默认 100 step，冻结 MobileNetV3，只训练 LR-ASPP/semantic logits，学习率 `3e-4`；阶段内独立保存最佳 checkpoint。
2. 阶段切换前恢复 head 最佳 checkpoint。
3. `backbone_finetune`：解冻 backbone，但默认继续冻结其 BatchNorm 统计；使用 `5e-5` 起始学习率并 cosine decay 到 10%。该阶段也有独立 checkpoint，同时只有超过全局 mIoU/boundary 调和评分时才替换最终权重。

报告逐次记录 stage、stage/global step、当时学习率、trainable/frozen layer 数、阶段 checkpoint 与全局 checkpoint。该机制不改变质量门槛，只减少小数据继续训练造成的灾难性遗忘。

## 数据采样与损失

训练 sampler 先均匀选择 source session，再在该 session 内选帧，避免 50 张相邻帧在统计上压过较短 session。每批默认 70% 尝试 rare-class guided crop，其中 boundary/curb 与 obstacle 的目标概率为 65%/35%；裁剪范围为原帧边长的 55%–85%，之后用双线性/最近邻分别恢复图像与 mask。其余样本保留完整视野，所有样本只使用语义安全的水平翻转。

损失为：

```text
0.50 * capped inverse-sqrt weighted CE
+ 0.40 * class-weighted soft Dice
+ 0.10 * class-weighted focal(gamma=2)
```

逆频率权重上限固定为 4.0，防止极少 boundary 像素通过无限放大权重诱发“全图 boundary”。训练报告必须保存实际 class pixel count、loss weight、session draw 和 guided-crop 命中次数，便于审计采样是否按预期执行。

## 模型消融

共享 backend-neutral 图现显式记录：

- `backbone_alpha`：只允许 `0.75` 或 `1.0`；默认 `0.75`。
- `decoder_channels`：默认 `96`。

先用完全相同的数据、step、batch、loss、seed 和评估频率比较 alpha 0.75/1.0；只有 alpha 1.0 在多 seed 均值与最差 seed 上都有实质改善，才考虑增加模型容量。不要在数据 session 数不足时同时搜索大量结构和训练超参。

输入分辨率允许 `256/384/512` 并写入模型合同。boundary 在 256 下可能经 1/32 backbone 下采样后丢失，建议先做 384/512 单 seed probe，再决定是否承担三 seed 成本；384 建议 batch 6–8，512 建议 batch 4。

```powershell
$python = 'E:\codex-tools\tools\venvs\blindassist-torch-gpu\Scripts\python.exe'
$dataset = 'test-artifacts.local\datasets\sanpo-v3-canonical-evidence-v4-20260713'

& $python scripts\train_sanpo_segmentation_keras_torch.py `
  --dataset-root $dataset `
  --input-size 384 --batch-size 6 `
  --optimizer-steps 600 --minimum-optimizer-steps 250 --eval-every-steps 25 `
  --head-warmup-steps 100 --learning-rate 3e-4 `
  --finetune-learning-rate 5e-5 --finetune-final-lr-ratio 0.1 `
  --seed 20260711 --backbone-alpha 1.0 --decoder-channels 96 `
  --weights test-artifacts.local\segmentation-candidate\protocol-v2-384-probe\candidate.weights.h5 `
  --report test-artifacts.local\segmentation-candidate\protocol-v2-384-probe\training_report.json

& $python scripts\train_sanpo_segmentation_keras_torch.py `
  --dataset-root $dataset `
  --input-size 512 --batch-size 4 `
  --optimizer-steps 600 --minimum-optimizer-steps 250 --eval-every-steps 25 `
  --head-warmup-steps 100 --learning-rate 3e-4 `
  --finetune-learning-rate 5e-5 --finetune-final-lr-ratio 0.1 `
  --seed 20260711 --backbone-alpha 1.0 --decoder-channels 96 `
  --weights test-artifacts.local\segmentation-candidate\protocol-v2-512-probe\candidate.weights.h5 `
  --report test-artifacts.local\segmentation-candidate\protocol-v2-512-probe\training_report.json
```

## 结果边界

- trainer 仍只消费 SHA256 绑定的 green 授权和 canonical `training_manifest.jsonl`；报告固定写入 `blind_holdout_access=not_accessed_by_trainer`。
- 三 seed dev 改善只代表优化协议更稳定，不是 blind 结果，更不是生产晋级证据。
- 训练后仍需跨后端等价、INT8 量化保真、独立 blind 事件指标和同机连续场景门。跨后端、导出与质量门命令必须原样传递本轮 `--backbone-alpha` / `--decoder-channels` / `--input-size`；v2 等价报告会哈希绑定配置与固定输入尺寸并拒绝错配。官方 SANPO 512 路线只能作为同协议消融，384/512 不改变既有数值或质量阈值。任何一项失败都保持 `do_not_replace_default_model`。
- 2026-07-13 已在新的 real-only canonical v4 上完成预注册三 seed 审计。384×384、alpha 1.0、decoder 96 的 seed `20260711` 达到 dev mIoU `0.4344`、boundary IoU `0.4506`，但 seed `20260712/20260713` 只有 `0.1804/0.1734` 与 `0.2498/0.1548`（mIoU/boundary IoU）。因此协议成功暴露了旧单 seed 报告隐藏的高方差；正式结论仍是 `do_not_replace_default_model`，下一步必须提升跨 session 泛化和 seed 最差值，而不是继续挑最好 checkpoint。

## 快速验证

```powershell
.\.venv-export312\Scripts\python.exe -m py_compile `
  scripts\sanpo_segmentation_model.py `
  scripts\train_sanpo_segmentation_keras_torch.py `
  scripts\test_train_sanpo_segmentation_protocol.py

.\.venv-export312\Scripts\python.exe -m unittest `
  scripts.test_train_sanpo_segmentation_protocol `
  scripts.test_train_export_sanpo_segmentation
```
