# HFTF Stage C D5 TartanGround Development pilot

## 结论

TartanGround 已从“目录看起来足够大”推进到三个可执行结果：

1. 官方 Hugging Face revision
   `388faf9c800568cfc6828fa47e063f8369397eb3` 完整覆盖锁定 catalog 的
   198 个 differential-drive parents 和 7,722 个 archive paths；
2. 三个 outcome-open 探索窗口证明 RGB/depth/seg/pose 共同时间轴和 metric
   geometry 可用，并观察到少量但明确的 future-label 非冗余。
3. 六个 train environments、两个 dev environments 的 student Development
   证明 RGB 标签可学习；直接从随机初始化联合训练 history 失败，而从 single
   checkpoint 分阶段微调 history 出现小幅、随机性可重复但环境不稳健的增量。

当前终态为：

`STAGED_HISTORY_SIGNAL_OBSERVED_BUT_ENVIRONMENT_ROBUST_INCREMENT_NOT_ESTABLISHED`

这足以保留 HFTF 作为候选支线，并优先扩大环境覆盖和诊断最差环境；不需要先完成
197-parent 产品级 census。它不证明 HFTF 超过主线或系统具有安全效用。

## Provider 与映射

- official dataset：`theairlabcmu/TartanGround`
- pinned revision：`388faf9c800568cfc6828fa47e063f8369397eb3`
- provider files：34,673
- catalog parents：198
- catalog archive paths：7,722
- revision 中缺失路径：0
- 全量 URL map SHA-256：
  `c3961c4c32f16af040745681e0a8ced4b9dca37bf96bfb11f1cb71a6fa2ee957`

稳定地址由
`https://huggingface.co/datasets/theairlabcmu/TartanGround/resolve/{revision}/{archive_path}`
机械生成。解析工具允许在网络或文件故障后修复重跑，不是 one-shot，也不烧毁 source。

## 三个探索 sentinel

- `AbandonedCable/Data_diff/P1000`
- `MiddleEast/Data_diff/P1002`
- `WaterMillNight/Data_diff/P1002`

每个 `metadata.zip` 都包含：

- finite positive `robot_height` 与 `time_step=0.1 s`；
- 12 个 camera pose streams，行数分别为 609、1,194、681，并与 `num_poses`
  完全一致；
- RGB/depth/seg 的 zero-based frame ID 集合分别与 pose 行
  `0..num_poses-1` 完全一致；
- 动态 `lcam_front` 6-DoF pose；
- 约 0.25 m 的固定左右 stereo baseline。

官方文档给出的 640×640 pinhole、`fx=fy=320`、10 Hz、同步采样和 NED pose
约定，与实际 payload 一致。

## Future-label pilot

每个 parent 取一个 outcome-open 25-raw-frame span，在 9 个 5 Hz anchors 上比较
`.4/.8 s`：

- field：`6 direction × 6 distance × foot/body/head`
- 标准探索速度：1.0 m/s
- baseline：current depth 对未来 candidate-envelope 位置的几何标签
- oracle：对应 future depth/pose 对同一位置的几何标签
- 三个窗口只读取各 13 张所需 depth frame；读取失败可修复重跑

汇总结果：

| 指标 | 结果 |
|---|---:|
| 双方 known 的 future cell-observations | 2,555 |
| 风险状态变化 | 54 |
| risk onset | 30 |
| risk clearance | 24 |
| future-only newly-known | 43 |
| 状态变化占 common-known | 2.11% |

按 parent 的风险状态变化为 `15 / 30 / 9`，三条均非零。future oracle 的
height-specific risk cell-observations 合计为 foot 284、body 522、head 221；这只说明
三个高度层在探索窗口中都有非退化输出，不是风险 prevalence。

坐标链通过 source payload 自身交叉核验：把 current depth 点用 pose 投到 future
camera 后，`.4/.8 s` 的 pair-median relative depth error 在三个 parent 上为
`.00068–.00144`，落在 5% 内的点比例中位数为 `.871–.988`。因此观测到的 future
差异不能用明显的 NED/OpenCV 坐标接错解释。

## Environment-clustered student Development

### Corpus

六个 train environments：

- `AbandonedCable`
- `CoalMine`
- `Gascola`
- `OldScandinavia`
- `Rome`
- `SeasonalForestWinterNight`

两个 dev environments：

- `MiddleEast`
- `WaterMillNight`

每个 source 取一个 centered 81-raw-frame span，形成 33 个 5 Hz anchors；每个
anchor 使用 `[-.8,-.6,-.4,-.2,0] s` 五帧 RGB，并具有
`current/.4/.8 s × foot/body/head × 6×6` teacher field。总计：

| 项目 | 结果 |
|---|---:|
| train samples | 198 |
| dev samples | 66 |
| RGB/depth PNG | 592 |
| manifest SHA-256 | `f9832d5b91d530fe70bfb65d55d639d5f040a7efa02ddb8abf40bf6d61981b76` |
| samples JSONL SHA-256 | `649d8ffc1e550b209ed64fcc87de20858da707089a5c31b7c00fabc14591ec75` |

592 个 PNG 均通过解码检查，264 条样本的 RGB 路径全部存在。网络中断、局部文件和
输出写入错误均属于可修复工程故障，不关闭 source。

### 比较设计

- backbone：ImageNet MobileNetV3-Small feature encoder；
- input：`5×RGB @ 128×224`；
- temporal fusion：同一个 depthwise `5×1×1` Conv3D；
- 输出：risk + known，三个 horizon、三个 height、`6×6`；
- 两臂参数量完全相同：1,087,464；
- checkpoint selection：dev 上 near/far × body/head 四组 F1 的宏平均；
- foot 仍报告，但不用于 selection，避免类别比例主导选择；
- 固定空间先验由 train 中每个 field cell 的均值构造。

### 绝对学习性与直接 history 结果

| 模型 | future body/head macro F1 | future body/head micro F1 | 相对 single macro |
|---|---:|---:|---:|
| train cell-prior | 0.2874 | 0.3261 | -0.2562 |
| single，seed 17 | 0.5435 | 0.6057 | 0 |
| history from scratch，seed 17 | 0.4996 | 0.5521 | -0.0439 |

single 明显超过固定空间先验，支持 RGB future-field learnability。history 从随机
初始化直接联合训练则降低整体 F1 和 recall，不能作为正结果。

交叉输入诊断显示，single checkpoint 在不重训时把 repeated-current 输入换成真实
history，macro F1 从 `0.5435` 升至 `0.5509`；history checkpoint 换回 single 输入
仍只有 `0.4980`。这把失败定位为训练/优化问题，而不是证明历史帧没有信息。

### 分阶段 history fine-tune

固定同一个 seed-17 single checkpoint，使用较小学习率对真实 history 微调 5
epochs；微调前的真实-history evaluation 作为 epoch 0。三个微调随机种子结果：

| fine-tune seed | selected epoch | macro F1 | macro delta vs single | micro F1 |
|---:|---:|---:|---:|---:|
| 17 | 5 | 0.5549 | +0.0114 | 0.6270 |
| 29 | 3 | 0.5565 | +0.0130 | 0.6304 |
| 43 | 4 | 0.5512 | +0.0077 | 0.6259 |

macro delta 平均 `+0.0107`，范围 `+0.0077..+0.0130`；micro F1 平均增量
`+0.0221`。三次 aggregate delta 同号，说明“先学静态场景、再学 history”的短程
优化信号对微调随机性可重复。

但 environment-level 结果不支持稳健增量：

| environment | single macro F1 | fine-tune macro F1 range | delta range |
|---|---:|---:|---:|
| MiddleEast | 0.5903 | 0.6130–0.6249 | +0.0227..+0.0346 |
| WaterMillNight | 0.4111 | 0.4035–0.4082 | -0.0076..-0.0028 |

全部 aggregate 提升都来自 `MiddleEast`；`WaterMillNight` 三次均轻微下降。相对
single，三次 fine-tune 的整体 body/head FPR 从 `0.4232` 降至
`0.4098–0.4123`，recall 从 `0.7535` 升至 `0.7810–0.7877`，但这些 aggregate
改善不能覆盖最差环境的负 delta。

## 边界与下一实验

当前正结果只支持：

`teacher feasible + RGB learnable + staged-history optimization signal`

尚未支持：

- history 对独立环境具有稳定增量；
- 当前小增量能在完整 pipeline 随机种子上复现；
- synthetic proxy 能迁移到真实视障步行；
- 事件级 critical-hazard recall、false alerts 或 warning lead time 改善；
- HFTF 超过当前主线或进入 App。

下一步不是立即重复更多完整训练，也不是打开 held-out。先增加 outcome-open
Development environments，并针对 `WaterMillNight` 检查 scene appearance、
teacher prevalence、known coverage 与过预测来源。只有 staged-history 在更多环境
上同号，才进行完整 pipeline 多种子和时间顺序/残差消融；随后才保留未用于迭代的
held-out environments 做偏差敏感评价。

## 复现

```powershell
E:\codex-tools\tools\venvs\blindassist-venv-export312\Scripts\python.exe `
  scripts/research/hftf/resolve_stage_c_d5_s0b_p0c_tartanground_provider.py `
  --revision 388faf9c800568cfc6828fa47e063f8369397eb3

E:\codex-tools\tools\venvs\blindassist-venv-export312\Scripts\python.exe `
  scripts/research/hftf/run_stage_c_d5_tartanground_development_pilot.py

E:\codex-tools\tools\venvs\blindassist-venv-export312\Scripts\python.exe `
  scripts/research/hftf/materialize_stage_c_d5_tartanground_development_corpus.py

E:\codex-tools\tools\venvs\blindassist-venv-export312\Scripts\python.exe `
  scripts/research/hftf/train_stage_c_d5_tartanground_development_student.py `
  --samples artifacts.local/evidence/hftf/stage-c-d5-tartanground-development-corpus-v0/samples.jsonl `
  --pretrained artifacts.local/models/hftf/torch/hub/checkpoints/mobilenet_v3_small-047dcff4.pth `
  --output-root artifacts.local/evidence/hftf/stage-c-d5-tartanground-development-student-v0/single-seed17 `
  --arm single --seed 17 --epochs 20
```

网络读取完成后可用 `--skip-fetch` 重算 geometry result。生成数据位于 ignored
`artifacts.local/evidence/hftf/stage-c-d5-s0b-p0c-provider-resolution-20260802/`。
