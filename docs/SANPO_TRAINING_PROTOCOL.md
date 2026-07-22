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

## P0 seed 因子审计

`--head-only` 会在整个短跑中冻结 backbone；`--seed-pairs` 使用 `model_seed:sampler_seed` 显式拆分模型随机状态和 sampler RNG。2026-07-13 的五组 OFAT 审计显示：固定 sampler 时 selection score 跨度为 `0.2685`，固定 model seed 时仅为 `0.0112`，前者约为后者 `24.1×`。当前 seed 高方差主因因此指向 head 初始化/模型随机状态，而不是 sampler 顺序。完整矩阵、macro-session、worst-scene 和限制见 [SANPO_P0_SEED_FACTOR_AUDIT_2026-07-13.md](SANPO_P0_SEED_FACTOR_AUDIT_2026-07-13.md)。

该结果把下一步锁定为 P1 LR-ASPP head 对齐；不授权读取 blind、导出候选或替换 App 模型。

## P1 LR-ASPP 结构对齐结果

2026-07-13 完成 sigmoid/no pooled-BN、OS4 detail、dilated OS16 semantic 的分阶段五组 seed 对照。sigmoid/no pooled-BN 将最佳 mIoU/boundary 提升至 `0.4642/0.5235`，并通过 Torch→TensorFlow 等价门，但固定 sampler 的 model-seed selection range 仍为 `0.2951`，未解决初始化稳定性。OS4/OS32、OS4/OS16、OS8/OS16 均出现 boundary 下限坍塌，因此默认合同保留 OS8/OS32，只采用 sigmoid/no pooled-BN 修正。完整证据见 [SANPO_P1_LRASPP_ALIGNMENT_2026-07-13.md](SANPO_P1_LRASPP_ALIGNMENT_2026-07-13.md)。

下一步进入 P2 确定性 quota sampler。P1 结果不授权导出、设备测试或 App 模型替换。

## P2 确定性 quota sampler 结果

P2 实现 boundary、obstacle、zero-boundary hard negative、per-session unknown-rich full-frame 各 25% 的确定性周期。batch 6、100 step 的 600 draw 严格闭合为 `150×4`；checkpoint 自动对齐完整 quota cycle。候选资格使用 canonical 原分辨率 mask，避免 3 个在 384 resize 后丢失细 boundary 的帧被误归为 hard negative；unknown-rich 使用每 session 内 q75，覆盖全部 8 个 train session。

五组对照显示 sampler-seed range 从 `0.0072` 降至 `0.0024`，但 model-seed 最差 selection 从 `0.1970` 降至 `0.1700`，range 扩至 `0.3097`，最佳 mIoU 也降至 `0.4484`。因此 quota sampler 仅保留为显式审计选项，不替换默认 sampler。完整证据见 [SANPO_P2_DETERMINISTIC_QUOTA_AUDIT_2026-07-13.md](SANPO_P2_DETERMINISTIC_QUOTA_AUDIT_2026-07-13.md)。

下一步进入 P3：按原始类别像素分布重构 split，并增加每场景独立 session 数。P2 不授权导出或 App 模型替换。

## P3 session split 重构门

P3 不允许在当前每场景 `2 train + 1 dev` 的 12 个 official-train session 上仅做重排后宣称完成。`plan_sanpo_p3_session_split.py` 只读取 official-train 候选的原分辨率 native mask，并在 session 原子性和底层 raw-mask SHA 隔离下搜索 split。固定门为每场景 train `4–6`、dev `2–3`；四类 train/dev pixel-share ratio `<=2×`；dev boundary 至少 3 个贡献 session、单 session 贡献 `<=50%`；其他 split/class 单 session 贡献 `<=60%`，每类两边至少 2 个贡献 session。无可行组合时不写 plan。

当前 canonical 的 train/dev boundary 占比为 `0.857%/16.976%`，且本地没有额外完整 official-train session；因此至少还需每场景新增 3 个、合计 12 个独立 session。候选必须依次通过 official-train 稀疏 mask 发现、连续 50 帧几何门、隔离的 GPT/Codex RGB/scene 共识门和 P3 分布规划门，之后才能重建 canonical 或启动训练；分歧交由全新上下文的第三模型仲裁。阶段证据见 [SANPO_P3_SPLIT_RECONSTRUCTION_2026-07-13.md](SANPO_P3_SPLIT_RECONSTRUCTION_2026-07-13.md)。

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
