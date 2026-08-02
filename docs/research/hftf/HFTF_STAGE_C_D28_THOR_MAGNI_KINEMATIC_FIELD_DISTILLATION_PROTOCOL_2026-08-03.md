# HFTF Stage C D28：THOR-MAGNI kinematic field distillation protocol

日期：2026-08-03

证据角色：Development / object-motion teacher-student canary

研究主线：不变

默认 App：不变

## 假设

D27 已以 11/11 gate 建立 source-native history-kinematic information ceiling；D26
则表明 whole-frame RGB dense-flow student 直接学习 full future truth 时没有整体增量。
D28 的新假设是：

> 把静态几何与历史运动显式拆成两个 teacher distance fields，能否让等容量 RGB
> students 学到 D27 已证明的 object-motion increment？

这不是再换 backbone，也不是调 D26 threshold。它改变训练目标的因果分解方式。

## 冻结 teacher 与 student

复用 D27 的 530 samples、19 source sessions、五 folds 与
`3 directions × 4 horizons` oracle score：

- `current` student：
  - 输入：重复当前 RGB、零 flow；
  - teacher：D27 current-static negative minimum-distance field；
- `history` student：
  - 输入：真实五帧 RGB 与 dense flow；
  - teacher：D27 history-kinematic negative minimum-distance field。

两个 students 复用 D26 的 MobileNetV3、flow-aligned dense-dynamics 与完整
`128×4×7` spatial feature。最后 head 改为 12 个连续 distance scores，参数完全相同，
相同 seed17 初始化独立训练。

输出以 `-sigmoid` 限制在 `[-1,0]`，对应 `[-10 m,0]`；沿 horizon 做 cumulative max，
结构上保证 risk score 单调不下降。teacher score 除以 10 进入 loss。

训练固定：

- source-balanced Smooth L1；
- `beta=0.05`（归一化单位，即 0.5 m）；
- 30 epochs、D22 学习率/AdamW/batch；
- fixed final epoch；
- 水平翻转同步交换 left/right teacher fields；
- 不搜索 loss、beta、距离上限、teacher mixture、epoch、seed 或 head。

## 冻结评价

held-out 上分别披露：

1. 对各自 teacher field 的 source-macro / pooled MAE；
2. 用 student score 对 D26 source-recorded future truth 计算：
   - source-macro 与 pooled direction×horizon AUROC/AP；
   - 三方向 horizon-macro；
   - 287 个 exact-time nonredundant anchors 的 source-macro safest-choice accuracy；
3. 所有主效应差值为 `history student - current student`。

## 冻结 gate

D28 只有在以下条件全部满足时支持：

1. source-macro direction×horizon AUROC mean delta 至少 `+0.010`；
2. source-macro direction×horizon AP mean delta 至少 `+0.005`；
3. AUROC/AP 各至少 3/5 folds 为正；
4. AUROC/AP 各至少 2/3 directions 的五折 mean 为正；
5. source-macro safest-choice accuracy mean delta 至少 `+0.020`，且至少
   3/5 folds 为正；
6. pooled direction×horizon AUROC/AP mean delta 均不低于 `-0.005`；
7. history teacher MAE 相对 current teacher MAE 的 mean 增量不超过 `+0.25 m`；
8. prediction horizon monotonicity violation 为 0。

通过终态：

`D28_THOR_MAGNI_KINEMATIC_FIELD_DISTILLATION_INCREMENT_SUPPORTED`

失败终态：

`D28_THOR_MAGNI_KINEMATIC_FIELD_DISTILLATION_INCREMENT_NOT_SUPPORTED`

## 边界

D28 通过只建立 THOR-MAGNI source-heldout、teacher-distilled RGB representation 与
direction-choice Development increment。D27 oracle 不是人体事件真值，D28 也不覆盖
静态障碍、路缘、foot/head 或真实提醒效用。

通过后才允许多 seed 或独立 source replication；失败则保留 D27 information ceiling，
并把瓶颈进一步定位为现有 RGB/flow 对 body velocity 的可识别性或数据规模，不在同
cohort 搜索 beta、teacher mix 或更大 head。工程异常仍只修复重跑，不烧毁 cohort。
