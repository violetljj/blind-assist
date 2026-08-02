# HFTF Stage C D8：THOR-MAGNI 局部路线监督与 RGB-history 筛查

日期：2026-08-02

证据角色：Development / source-native geometric proxy

研究主线：不变

默认 App：不变

## 结论

THOR-MAGNI 已经提供了足够的跨会话局部动作性监督，不需要先等待人工逐事件裁决。
19 个 Pupil 会话共产生 1,078 个 1 Hz 历史窗口；近距和走廊侵入的正负例分布在全部
或几乎全部会话与五个 source-session-held-out 折中。

冻结 MobileNetV3-small 表征的首次低成本筛查给出一个分层结果：

- 2,304-D history 表征相对 576-D current-only 对**近距**和**未来走廊侵入**
  有稳定排序信号；
- history 对完整 `2 × 6 × 4 = 48` cell 占用场和连续最小距离排序没有稳定增量。

随后完成的等容量 temporal-head 对照没有复制该粗粒度增量。两臂使用相同 5×576
接口、相同 4,610 参数 head、相同训练预算和三个 seed；近距 AUROC/AP 仅 2/5 折
为正，走廊 AP 也仅 2/5 折为正。

因此本次不是统一的“成功”或“失败”。保留三个不同层级的终态：

- `D8_HIGH_DIMENSIONAL_COARSE_ACTIONABILITY_SEPARABILITY_SIGNAL_OBSERVED`
- `D8_EQUAL_CAPACITY_TEMPORAL_ACTIONABILITY_INCREMENT_NOT_STABLE`
- `D8_TEMPORAL_SPATIAL_CORRIDOR_SIGNAL_WEAK_NOT_ACTIONABLE`
- `D8_EQUAL_CAPACITY_TEMPORAL_SPATIAL_ACTIONABILITY_INCREMENT_NOT_STABLE`
- `D8_FULL_LOCAL_FIELD_HISTORY_INCREMENT_NOT_SUPPORTED_ON_FROZEN_REPRESENTATION`

较高维 screen 是真实观察，但不能排除容量混杂，不能升级为 history 独立增量。
保留空间 layout 后只出现幅度很小的 corridor-specific signal，预定的双目标总门仍
失败。当前 THOR frozen-backbone 路线关闭；不调 epoch、seed、head 或 target
救援，也不再增加数据治理层。下一科学变量必须是独立来源复现，而不是同源模型搜索。

## 监督物化

输入是 D7 已本地化的 19 个 THOR-MAGNI Pupil 视频及其 QTM scenario CSV。物化器逐
manifest 校验视频与 CSV SHA，但不建立新的 closed set、root authority 或 one-shot
执行制度。工程中断可以在未产生完整输出时修复并重跑。

每个样本使用：

- RGB history scene-frame offsets：`[-24, -18, -12, -6, 0]`；
- anchor stride：30 scene frames，约 1 Hz；
- wearer speed floor：`0.25 m/s`；
- 未来跨度：2 秒，每 0.1 秒采样；
- wearer-motion-relative field：2 个 horizon、6 个方向、4 个 0–4 m 距离环；
- 连续目标：未来 2 秒内与任一其他 tracked body 的最小同步距离；
- 描述性代理：`distance <= 1.25 m` 与前向 0–4 m、横向 `±0.90 m` 的走廊侵入。

物化结果：

| 项目 | 结果 |
|---|---:|
| source sessions | 19 |
| samples | 1,078 |
| fold sample counts | 308 / 255 / 129 / 103 / 283 |
| 近距正例 | 705（65.40%） |
| 走廊侵入正例 | 610（56.59%） |
| 最小距离 median | 1.016 m |
| 最小距离 p10 / p90 | 0.623 / 2.354 m |

近距正例覆盖 19/19 会话，负例覆盖 18/19；走廊侵入正例和负例都覆盖 19/19。五个
held-out 折都同时包含两类粗粒度目标的正负例。

## RGB-history 筛查

同一官方 torchvision ImageNet MobileNetV3-small 权重对 5,390 个去重视频帧抽取
576-D pooled feature。比较臂为：

- `current`：当前帧 576-D；
- `history`：当前帧、current-minus-earliest、current-minus-previous、五帧标准差，
  共 2,304-D；
- 两臂只在各自 train folds 上标准化，使用固定 `L2 ridge(alpha=10)` 多输出线性
  排序读出；
- split 固定为 `SHA-256(source_session_id) mod 5`，train/test session 无交集。

成功门要求近距、走廊、48-cell micro AUROC/AP 和最小距离 Spearman 的 history
增量都满足 median `> 0` 且至少 3/5 折为正。该总门没有通过。

### 跨折增量

| 指标（history - current） | mean | median | 正折 |
|---|---:|---:|---:|
| 近距 AUROC | +0.0559 | +0.0358 | 5/5 |
| 近距 AP | +0.0205 | +0.0322 | 4/5 |
| 走廊侵入 AUROC | +0.0511 | +0.0473 | 5/5 |
| 走廊侵入 AP | +0.0269 | +0.0378 | 4/5 |
| 48-cell micro AUROC | -0.0103 | -0.0130 | 1/5 |
| 48-cell micro AP | -0.0074 | -0.0091 | 0/5 |
| 最小距离 Spearman | +0.0236 | -0.0101 | 2/5 |
| 最小距离 MAE 改善 | +0.0145 m | +0.0029 m | 3/5 |

粗粒度 AUROC 的逐折结果没有方向冲突：

| fold | 近距 current → history | 走廊 current → history |
|---:|---:|---:|
| 0 | 0.50 → 0.54 | 0.51 → 0.57 |
| 1 | 0.56 → 0.59 | 0.61 → 0.66 |
| 2 | 0.61 → 0.63 | 0.59 → 0.61 |
| 3 | 0.55 → 0.64 | 0.49 → 0.57 |
| 4 | 0.50 → 0.60 | 0.58 → 0.62 |

这建立了较高维 history 表征的 coarse separability signal，但当前 absolute AUROC
仍仅约 0.54–0.66，且 history arm 维度更高。它只授权等容量 temporal-head 验证，
不能直接声称 history 独立增量、完整未来场可预测或系统效用成立。

## 等容量 temporal-head 结果

两臂都实例化相同模型：

- 输入接口固定为 `5 × 576`；current arm 将当前 feature 重复五次；
- current identity path 加 learned per-time/per-channel bounded residual fusion；
- `LayerNorm + Linear(576,2)`，总参数量 4,610；
- 120 个固定 epochs、AdamW、source-balanced BCE、seeds `17/23/41`；
- held-out fold 不参与标准化、训练或模型选择，直接读取 final epoch。

三个 seed 的 fold-mean history-minus-current：

| 指标 | mean | median | 正折 |
|---|---:|---:|---:|
| 近距 AUROC | -0.0039 | -0.0113 | 2/5 |
| 近距 AP | -0.0080 | -0.0013 | 2/5 |
| 走廊侵入 AUROC | +0.0071 | +0.0108 | 3/5 |
| 走廊侵入 AP | +0.0013 | -0.0009 | 2/5 |

逐折 AUROC 的 seed mean：

| fold | 近距 current → history | 走廊 current → history |
|---:|---:|---:|
| 0 | 0.47 → 0.45 | 0.61 → 0.60 |
| 1 | 0.60 → 0.61 | 0.64 → 0.66 |
| 2 | 0.57 → 0.56 | 0.59 → 0.59 |
| 3 | 0.60 → 0.59 | 0.45 → 0.46 |
| 4 | 0.50 → 0.52 | 0.61 → 0.62 |

这没有通过四项指标都要求 median `>0` 且至少 3/5 折为正的门。由于该对照专门
排除了输入维度和 head 参数量差异，它 supersede 先前较高维 screen 对“history
独立增量”的解释，但不删除先前观察值。

## 等容量 temporal-spatial 结果

pooled head 可能过早丢失画面位置，因此执行最后一个同源表示对照。输入改为 frozen
MobileNet 的 `5 × 576 × 4 × 7` feature map；两臂共享相同 13,586 参数：
per-time/per-channel residual fusion、GroupNorm、`Conv1x1(576,16)` 与
`Linear(16×4×7,2)`。其余 split、seeds、epochs、loss 与 final-epoch evaluation
保持不变。

三个 seed 的 fold-mean history-minus-current：

| 指标 | mean | median | 正折 | 正 unit |
|---|---:|---:|---:|---:|
| 近距 AUROC | -0.0016 | -0.0008 | 2/5 | 5/15 |
| 近距 AP | -0.0006 | -0.0001 | 1/5 | 7/15 |
| 走廊侵入 AUROC | +0.0040 | +0.0027 | 5/5 | 13/15 |
| 走廊侵入 AP | +0.0038 | +0.0027 | 5/5 | 9/15 |

空间 layout 的保留确实只改变了空间定义的走廊目标，近距目标没有改善。这是一个
机制一致但幅度很小的 corridor-specific weak signal。由于 AP 只有 9/15 个
fold×seed units 为正，且预定门要求近距与走廊四项指标同时成立，本结果不授权
end-to-end fine-tuning、field 恢复或主线比较。

## 可复现证据

监督输出：

```text
artifacts.local/evidence/hftf/
  stage-c-d8-thor-magni-local-route-supervision-v0/
```

- `samples.jsonl` SHA-256：
  `c2c63251f727fe5f89241a060f0dcc3ec5a851bb7b878f18b8e0745e25d5363a`
- `report.json` SHA-256：
  `0fb1636e06f78640a554627e728460c0d19dfbfe32b32263c3c113f3121f1747`

筛查输出：

```text
artifacts.local/evidence/hftf/
  stage-c-d8-thor-magni-rgb-history-screen-v0/
```

- `report.json` SHA-256：
  `67555819ac229b65ca248dd6ae7e2dab9e2e2574a74dc54b2c2a266f639772cd`

等容量 temporal-head 输出：

```text
artifacts.local/evidence/hftf/
  stage-c-d8-thor-magni-equal-capacity-temporal-head-v0/
```

- `report.json` SHA-256：
  `c7e5d6d957ecb2a6a2fe8a068e9ea55190d5489b50698d09549fadb368a086ce`

spatial feature 与 temporal-spatial head 输出：

```text
artifacts.local/evidence/hftf/
  stage-c-d8-thor-magni-spatial-features-v0/
  stage-c-d8-thor-magni-equal-capacity-temporal-spatial-head-v0/
```

- `features.npz` SHA-256：
  `9a80d6ca6f3b36aee3efed91f89802ebd7e5f9a972cca226175644bd55135838`
- temporal-spatial `report.json` SHA-256：
  `22b6dabe6bd47e404ac29639bd85106cc140bd7bf62bed7c2a1527a38ffe38ff`

复现命令：

```powershell
E:\codex-tools\tools\venvs\blindassist-torch-gpu\Scripts\python.exe `
  scripts/run_research_tool.py hftf `
  materialize_stage_c_d8_thor_magni_local_route_supervision.py `
  --output-root artifacts.local/evidence/hftf/<new-supervision-run>

E:\codex-tools\tools\venvs\blindassist-torch-gpu\Scripts\python.exe `
  scripts/run_research_tool.py hftf `
  evaluate_stage_c_d8_thor_magni_rgb_history_screen.py `
  --samples artifacts.local/evidence/hftf/<new-supervision-run>/samples.jsonl `
  --output-root artifacts.local/evidence/hftf/<new-screen-run>

E:\codex-tools\tools\venvs\blindassist-torch-gpu\Scripts\python.exe `
  scripts/run_research_tool.py hftf `
  run_stage_c_d8_thor_magni_equal_capacity_temporal_head.py `
  --samples artifacts.local/evidence/hftf/<new-supervision-run>/samples.jsonl `
  --features artifacts.local/evidence/hftf/<new-screen-run>/features.npz `
  --output artifacts.local/evidence/hftf/<new-head-run>/report.json

E:\codex-tools\tools\venvs\blindassist-torch-gpu\Scripts\python.exe `
  scripts/run_research_tool.py hftf `
  extract_stage_c_d8_thor_magni_spatial_features.py `
  --samples artifacts.local/evidence/hftf/<new-supervision-run>/samples.jsonl `
  --output artifacts.local/evidence/hftf/<new-spatial-run>/features.npz

E:\codex-tools\tools\venvs\blindassist-torch-gpu\Scripts\python.exe `
  scripts/run_research_tool.py hftf `
  run_stage_c_d8_thor_magni_equal_capacity_temporal_head.py `
  --samples artifacts.local/evidence/hftf/<new-supervision-run>/samples.jsonl `
  --features artifacts.local/evidence/hftf/<new-spatial-run>/features.npz `
  --output artifacts.local/evidence/hftf/<new-spatial-head-run>/report.json
```

## 主张边界

QTM body centroid 提供 source-native 几何，不是人类提醒真值。`future_proximity` 与
`future_corridor_intrusion` 是局部路线代理，不证明碰撞、用户意图、提醒正确性、
安全性或 App 效果。本结果不改变当前 YOLO 主线，也不触发 HFTF 晋级。
