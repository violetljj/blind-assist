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

- history 相对 current-only 对**近距**和**未来走廊侵入**有稳定排序增量；
- history 对完整 `2 × 6 × 4 = 48` cell 占用场和连续最小距离排序没有稳定增量。

因此本次不是统一的“成功”或“失败”。保留两个不同层级的终态：

- `D8_COARSE_ACTIONABILITY_HISTORY_INCREMENT_SUPPORTED_DEVELOPMENT_ONLY`
- `D8_FULL_LOCAL_FIELD_HISTORY_INCREMENT_NOT_SUPPORTED_ON_FROZEN_REPRESENTATION`

下一实验只验证等容量 compact temporal actionability head，不微调 backbone、不搜索
48-cell field 表示，也不再增加数据治理层。

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

这支持“history 含有 current-only 没有的粗粒度局部动作性信息”，但当前 absolute
AUROC 仍仅约 0.54–0.66，且 history arm 维度更高。本结果因此只授权下一次等容量
temporal-head 验证，不能直接声称完整未来场可预测或系统效用成立。

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
```

## 主张边界

QTM body centroid 提供 source-native 几何，不是人类提醒真值。`future_proximity` 与
`future_corridor_intrusion` 是局部路线代理，不证明碰撞、用户意图、提醒正确性、
安全性或 App 效果。本结果不改变当前 YOLO 主线，也不触发 HFTF 晋级。
