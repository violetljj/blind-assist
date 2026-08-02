# HFTF Stage C D28：THOR-MAGNI kinematic field distillation result

日期：2026-08-03

证据角色：Development / object-motion teacher-student canary

研究主线：不变

默认 App：不变

## 结论

D28 完整产生 5 folds × current/history = 10 个训练 runs。冻结 gate 只通过
2/12，整体终态为：

`D28_THOR_MAGNI_KINEMATIC_FIELD_DISTILLATION_INCREMENT_NOT_SUPPORTED`

history student 相对 current student：

| metric | mean delta | 正折 |
|---|---:|---:|
| source-macro direction×horizon AUROC | -0.02350 | 2/5 |
| source-macro direction×horizon AP | -0.02159 | 2/5 |
| source-macro safest-direction accuracy | +0.01347 | 2/5 |
| pooled direction×horizon AUROC | -0.01708 | 1/5 |
| pooled direction×horizon AP | -0.00856 | 2/5 |
| source-macro teacher MAE | +0.03538 m | 3/5 |

teacher MAE noninferiority 与结构单调性通过；所有真实未来 ranking、方向广度和
safest-choice gates 均未通过。不能把 D27 的强 oracle ceiling 改写成 D28 RGB
student 增量。

## 五折异质性

| fold | AUROC delta | AP delta | safest-choice delta | teacher MAE delta |
|---:|---:|---:|---:|---:|
| 0 | -0.04917 | -0.06047 | -0.11981 | +0.03311 m |
| 1 | -0.03661 | -0.05020 | -0.02279 | -0.07257 m |
| 2 | +0.01212 | +0.01289 | +0.14163 | -0.07582 m |
| 3 | -0.08766 | -0.02705 | -0.06667 | +0.15738 m |
| 4 | +0.04382 | +0.01688 | +0.13500 | +0.13482 m |

fold 2 和 fold 4 的三项真实未来指标同时为正，说明该机制并非在每个 source split
上都失效；但只有 2/5 正折，无法建立 source-general increment。

teacher MAE 也不是充分解释：

- fold 1 的 history teacher MAE 改善，但真实未来 AUROC/AP/choice 均下降；
- fold 4 的 history teacher MAE 变差，但三项真实未来指标均上升。

因此瓶颈不只是“能否回归 D27 distance field”，还包括 teacher approximation
error 的结构、body identity/motion correspondence 与真实 future ranking 的对齐。

## 三方向

| direction | source-macro horizon-macro AUROC delta | 正折 | AP delta | 正折 |
|---|---:|---:|---:|---:|
| left | -0.01464 | 2/5 | -0.01721 | 2/5 |
| center | -0.05397 | 2/5 | -0.03313 | 2/5 |
| right | -0.00189 | 3/5 | -0.01442 | 3/5 |

right 方向仍最接近非劣，但 mean AUROC/AP 都为负；D28 不产生新的方向级正终态。
D26 的 right-direction Development 信号仍按其原证据层保留，不由 D28 撤销。

## D27 与 D28 的层级关系

D27 仍建立：

- source-native 历史人体速度相对静态人体位置有强 direction×horizon
  information ceiling；
- 三方向 AUROC/AP 与 safest-choice 在 5/5 folds 同向改善；
- target 不是“没有 temporal information”。

D28 只关闭：

> 让现有 whole-frame RGB + dense-flow spatial network 直接回归 D27 两个
> distance fields，就能稳定恢复该 information increment。

它不关闭 object-centric track、instance correspondence、显式速度瓶颈或
track-conditioned field。下一候选必须改变可识别变量，而不是在同一 530 anchors
上搜索 loss、beta、epoch、seed、head 或 teacher mixture。

## 工程与复现

- 530 samples、19 source sessions、5 paired folds、10 training runs；
- 1,046,896 trainable parameters；
- fixed seed17、30 epochs、fixed final epoch；
- 10 个 checkpoints 全部落盘；
- prediction horizon monotonicity violation：0；
- stderr 为空，无 OOM、路径、parser、cache、serialization 或 fsync invalid；
- current/history 初始化逐 fold 一致；
- 未搜索 threshold、loss、beta、epoch、seed、距离上限或 teacher mix。

```text
artifacts.local/evidence/hftf/
  stage-c-d28-thor-magni-kinematic-field-distillation-v0/
    report.json
    report.json.sha256
    checkpoints/
    stdout.log
    stderr.log
```

report SHA-256：
`2f359f12b04a15fa9de7f109e87231bc7c738de2dac95fb134762f18e119e29c`。

## 下一科学变量

下一步进入 object-centric 可识别性诊断，而不是另一个 whole-frame head：

1. 用当前 RGB 检测到的人体实例建立 current geometry；
2. 用历史帧 instance correspondence 估计每个人体的 image/world motion；
3. 把这些显式 object slots 送入冻结的 kinematic field solver；
4. 分开测量 detection coverage、identity/velocity error、teacher-field MAE 与
   future-truth ranking。

先做 teacher-forced correspondence 与视觉 correspondence 的 paired ceiling。
若 teacher-forced slot 可恢复 D27、视觉 slot 不行，瓶颈是实例检测/匹配；若两者都
不能，瓶颈是从图像到 world motion 的几何标定；只有视觉 slot 在 source-heldout
上超过 current-static，才进入多 seed 或 decision-layer replication。
