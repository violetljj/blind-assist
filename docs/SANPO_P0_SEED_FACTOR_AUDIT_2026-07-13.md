# SANPO P0 模型初始化 / sampler seed 因子审计

## 结论

本轮 5 个 head-only 短跑表明，当前 384×384 MobileNetV3Small + LR-ASPP 候选的 seed 高方差主要来自**模型初始化及与该 seed 绑定的 Torch 随机状态**，不是现有 sampler 随机序列。

- 固定 sampler seed `20260711`、改变三个 model seed 时，selection score 范围为 `0.1739–0.4424`，跨度 `0.2685`。
- 固定 model seed `20260711`、改变三个 sampler seed 时，selection score 范围为 `0.4312–0.4424`，跨度 `0.0112`。
- 两个描述性跨度相差约 `24.1×`。mIoU、boundary IoU、unknown IoU 也呈相同方向。

因此 P1 应优先修正 LR-ASPP head 的结构与小 batch 数值稳定性；P2 quota sampler 仍有必要解决数据覆盖问题，但它不是本轮 seed 高方差的主因。该结论不等于因果显著性检验，也没有估计 model seed 与 sampler seed 的交互项。

## 预注册矩阵

固定配置：real-only canonical r3、384×384、batch 6、alpha 1.0、decoder 96、head-only、100 optimizer steps、每 25 step 评估、LR `3e-4`。只读取 train/dev；blind 未被 trainer 访问。

| Model seed | Sampler seed | Selection | mIoU | Boundary IoU | Unknown IoU | Macro-session mIoU | Worst-scene mIoU |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 20260711 | 20260711 | 0.4424 | 0.4344 | 0.4506 | 0.4407 | 0.3283 | 0.2680 |
| 20260712 | 20260711 | 0.1739 | 0.1646 | 0.1842 | 0.2511 | 0.1378 | 0.1198 |
| 20260713 | 20260711 | 0.1982 | 0.2870 | 0.1514 | 0.2938 | 0.2487 | 0.1756 |
| 20260711 | 20260712 | 0.4388 | 0.4332 | 0.4446 | 0.4558 | 0.3277 | 0.2642 |
| 20260711 | 20260713 | 0.4312 | 0.4227 | 0.4401 | 0.4492 | 0.3239 | 0.2697 |

五组的 worst scene 均为 `step_curb`。这与下一次大跨越诊断中的 scene/session 偏斜一致：幸运 seed 并没有消除最弱场景，只是把全局与 boundary 指标推高。

## 实现与证据

训练器新增：

- `--head-only`：整个短跑冻结 backbone，只训练 `lraspp_*` 与 `semantic_logits`。
- `--seed-pairs model_seed:sampler_seed`：分别控制建模/训练随机状态与 sampler RNG，并在权重文件名和 JSON 中独立记录。
- 最终 checkpoint 报告 global、macro-session、各 session、各 scene 与 worst group 指标。
- `p0_factor_variation` 保存固定一因子时另一因子的均值、标准差、范围和极值。

本地产物（不提交）：

- `test-artifacts.local/segmentation-candidate/p0-head-only-seed-factor-20260713/training_report.json`
- 报告 SHA256：`0c10c4ed3d2c1fb3707c86bf99b64df2f9441c6e58711a33e53a8c616bee8f38`
- `blind_holdout_access=not_accessed_by_trainer`
- `promotion=do_not_replace_default_model`

## 证据边界与下一步门

- `model_seed` 同时控制初始化和可能的 Torch 随机算子；当前没有把它收窄成数学意义上的“只改变初始权重”。P1 对照必须保持相同执行合同。
- 5-run OFAT 设计不估计交互。只有 P1 后固定 sampler 的三 model seed 仍高方差，才追加 `2×2` 交互审计。
- 当前 taxonomy 把 stairs 与 curb 合并，P0 无法诚实输出二者独立指标；应在 P5 控制实验中处理。
- HUMAN/MACHINE annotation quality 分项属于 P4，本轮未伪造代理指标。
- P1 进入条件：先把 LR-ASPP 改为 sigmoid gate 并移除 pooled-BN，再以同一 5-run 矩阵复跑；随后才比较 OS4 detail / OS16 semantic。继续扫 LR、512、decoder 160 或 boundary 权重不再是当前主线。
