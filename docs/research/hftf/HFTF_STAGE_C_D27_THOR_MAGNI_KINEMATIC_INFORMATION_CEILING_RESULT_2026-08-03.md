# HFTF Stage C D27：THOR-MAGNI kinematic information-ceiling result

日期：2026-08-03

证据角色：Development / source-native information ceiling

研究主线：不变

默认 App：不变

## 结论

D27 的冻结 gate 11/11 全部通过：

`D27_THOR_MAGNI_HISTORY_KINEMATIC_INFORMATION_CEILING_SUPPORTED`

history-kinematic 相对 current-static：

| metric | mean delta | 正折 |
|---|---:|---:|
| source-macro direction×horizon AUROC | +0.10833 | 5/5 |
| source-macro direction×horizon AP | +0.17781 | 5/5 |
| source-macro safest-direction accuracy | +0.13955 | 5/5 |
| pooled direction×horizon AUROC | +0.09163 | 5/5 |
| pooled direction×horizon AP | +0.24982 | 5/5 |

safest-direction source-macro accuracy 从 current-static 平均 `0.69080` 提升到
history-kinematic `0.83035`。

## 三方向一致性

| direction | source-macro horizon-macro AUROC delta | 正折 | AP delta | 正折 |
|---|---:|---:|---:|---:|
| left | +0.11345 | 5/5 | +0.16539 | 5/5 |
| center | +0.08086 | 5/5 | +0.17820 | 5/5 |
| right | +0.13069 | 5/5 | +0.18985 | 5/5 |

D26 的 right-direction 局部信号在 source-native information ceiling 中不是唯一方向；
三方向都存在强历史运动增量。D26 center 失败因此更像 RGB representation/training
问题，而不是 center target 没有 temporal information。

## D27 实际证明了什么

两个 oracle 使用完全相同的：

- 530 samples、19 source sessions、五 folds；
- wearer 当前位置、速度与 `-30°/0°/+30°` 候选路径；
- 0.10 s 采样、`.5/1/1.5/2 s` horizons；
- 其他人体 anchor 当前位置；
- source-recorded future trajectory truth。

唯一差异是：

- current-static 把其他人体冻结在 anchor 位置；
- history-kinematic 只从 anchor 前 0.4 s 到 anchor 的世界位置估计速度并恒速外推。

prediction 侧没有读取 anchor 后位置。2,927 个 current-body observations 中，
2,787 个具有有效历史速度，coverage `95.22%`；其余自动退化为静止。

因此 D27 建立的是：

> D26 action-conditioned tracked-body target 中存在强、跨折、跨方向的
> history-motion information ceiling。

它把失败定位为：当前 whole-frame RGB dense-flow student 没有把 object-centric
motion 稳定恢复出来。它不是“target 没有历史信息”。

## 与 D26 的层级关系

D27 不撤销 D26：

- D26 仍是 RGB history student 相对 RGB current student 的整体
  `NOT_SUPPORTED`；
- D26 right-direction representation signal 仍保留；
- D27 是 source-native geometry oracle，不是 RGB student。

正确结论不是“D27 证明 D26 模型有效”，而是：

1. target 有充足 temporal information；
2. 简单人体速度已能显著提高未来冲突 ranking 与方向选择；
3. 下一学生应显式蒸馏 object-motion residual，而不是继续让 whole-frame flow 网络
   自己在 full truth loss 中发现它。

## 可复现证据

```text
artifacts.local/evidence/hftf/
  stage-c-d27-thor-magni-kinematic-information-ceiling-v0/
    report.json
    report.json.sha256
    oracle_scores.npz
    oracle_scores.npz.sha256
```

- report SHA-256：
  `d280c965e0f7876ba01ef202e764898e5e43081ac84222c5e5bf8c76f70ba8f0`；
- oracle scores SHA-256：
  `44e0ff0e83e08a7dc6cff78e1daab48f3d516eaa2be5519eba523c7df09e9fb3`；
- report 约 281 KiB，score matrix 约 34 KiB；
- no training、no threshold、no seed、no window 或 smoother search；
- oracle risk 随 horizon 单调性 violation 为 0。

## 主张边界

D27 不建立：

- RGB history student learnability；
- 真实用户事件、提醒提前量或 false-alert 改善；
- 静态障碍、路缘、落差、foot/head traversability；
- 主线替换、App、生产或安全主张。

这些是后续独立层，不把 D27 的 information-ceiling 正结果改写为失败。

## 下一科学变量

下一学生不再直接用 full future truth 让大网络自行分离静态与运动。冻结一个明确的
teacher-distillation 问题：

- current arm：从 current RGB 预测 D27 current-static 三方向×四 horizon distance
  field；
- history arm：从 history RGB + flow 预测 D27 history-kinematic field；
- 相同 spatial field architecture 与容量；
- 先检验 teacher score regression 与真实 future-truth ranking，再决定是否接入
  decision layer。

这会把 D27 已证明的 object-motion residual 直接变成学习目标。如果仍失败，瓶颈才可
进一步定位为 RGB/flow 对 source-native body velocity 的可识别性或数据规模，而不是
继续调 D26 full-truth class loss。
