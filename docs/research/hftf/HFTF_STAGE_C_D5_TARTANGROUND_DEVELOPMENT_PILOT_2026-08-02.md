# HFTF Stage C D5 TartanGround Development pilot

## 结论

TartanGround 已从“目录看起来足够大”推进到两个可执行正结果：

1. 官方 Hugging Face revision
   `388faf9c800568cfc6828fa47e063f8369397eb3` 完整覆盖锁定 catalog 的
   198 个 differential-drive parents 和 7,722 个 archive paths；
2. 三个 outcome-open 探索窗口证明 RGB/depth/seg/pose 共同时间轴和 metric
   geometry 可用，并观察到少量但明确的 future-label 非冗余。

这足以进入较大的 Development corpus 与 student baseline，不需要先完成 197-parent
产品级 census。它不证明 student 更好、HFTF 超过主线或系统具有安全效用。

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

## 边界与下一实验

当前正结果只支持：

`aligned geometry teacher feasible + future label opportunity exists`

尚未支持：

- RGB student 能学习这些标签；
- history 比 single frame 有增量；
- synthetic proxy 能迁移到真实视障步行；
- 事件级 critical-hazard recall、false alerts 或 warning lead time 改善；
- HFTF 超过当前主线或进入 App。

下一步直接形成 environment-clustered Development corpus，先比较同参数
single-frame future 与 history-RGB future 两臂，并报告绝对 learnability、future
delta、worst-environment 与计算成本。只有该信号稳定后，才保留一批未用于迭代的
held-out environments 做偏差敏感评价。

## 复现

```powershell
E:\codex-tools\tools\venvs\blindassist-venv-export312\Scripts\python.exe `
  scripts/research/hftf/resolve_stage_c_d5_s0b_p0c_tartanground_provider.py `
  --revision 388faf9c800568cfc6828fa47e063f8369397eb3

E:\codex-tools\tools\venvs\blindassist-venv-export312\Scripts\python.exe `
  scripts/research/hftf/run_stage_c_d5_tartanground_development_pilot.py
```

网络读取完成后可用 `--skip-fetch` 重算 geometry result。生成数据位于 ignored
`artifacts.local/evidence/hftf/stage-c-d5-s0b-p0c-provider-resolution-20260802/`。
