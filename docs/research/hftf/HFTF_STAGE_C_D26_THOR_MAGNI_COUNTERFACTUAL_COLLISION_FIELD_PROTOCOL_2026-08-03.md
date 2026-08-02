# HFTF Stage C D26：THOR-MAGNI counterfactual collision field protocol

日期：2026-08-03

证据角色：Development / action-conditioned future-field canary

研究主线：不变

默认 App：不变

## 科学问题

D23 的 route-agnostic proximity 二分类有 representation 增量，但 D24 没有事件效用，
D25 的首次进入时间也没有跨 source 增量。共同缺口是：目标只问“任何人体会不会靠近”，
没有问“如果用户选择左/中/右哪条路径，哪条会与真实未来人体轨迹冲突”。

D26 回到 HFTF 的核心输出合同：

> 在同一时刻一次预测多个候选行动方向的短时未来身体冲突场。

它不再修改 D23/D25 score，也不更换 backbone。唯一新的科学变量是
action-conditioned counterfactual supervision geometry。

## Source-native counterfactual teacher

对 D12 中 530 个当前没有人体进入 1.25 m 的 anchors：

1. 用 QTM anchor 前后各 0.25 s 的 wearer centroid 估计当前平面速度与前向；
2. 从当前 wearer 世界坐标滚动三条恒速候选路径：
   `left=-30° / center=0° / right=+30°`；
3. 其他人体继续使用 scenario CSV 中真实记录的未来世界轨迹；
4. 每 0.10 s 计算候选 wearer 位置与所有其他人体的同步距离；
5. 记录每条候选路径首次进入 1.25 m 的时间：
   `(0,.5] / (.5,1] / (1,1.5] / (1.5,2] / >2 s`。

这是行人—行人的 source-native 几何冲突代理。它不包含墙、路缘或其他未进入 QTM
body roster 的静态障碍，也不是用户事件或安全真值。

## 冻结机会

协议前只做标签普查，不运行 D26 模型：

- samples：530；source sessions：19；五个 source-session-held-out folds；
- 三方向五类 counts：
  - left：`78/44/35/33/340`；
  - center：`60/41/34/32/363`；
  - right：`72/63/46/28/321`；
- 287/530 anchors 的三方向精确首次冲突时间不完全相同；
- 271/530 anchors 的三方向五类 time bin 不完全相同；
- 231/530 anchors 的三方向在“2 秒内冲突/不冲突”上存在分歧；
- 三方向 × 四累计 horizon 在每个 fold 都同时有正负。

因此可直接检验多方向 future-field representation，不需要为可评价性继续扩数据治理。

## 冻结学生与对照

复用 D22 的：

- MobileNetV3 编码器；
- 五帧 `128×224` RGB cache；
- current-to-history RAFT flow；
- flow-aligned 20-channel dense dynamics；
- 五个 source-session-held-out folds；
- 30 epochs、batch、学习率、AdamW 与 fixed-final-epoch evaluation。

为避免全局池化删除方向信息，D26 固定保留 `128×4×7` projected spatial map，直接用
一个线性 field head 输出 `3 directions × 5 time classes`。两个 arm 等容量并从相同
初始化独立训练：

1. `current`：重复当前帧、零 flow、精确零 dynamics；
2. `history`：真实五帧与 dense flow。

训练损失固定为 source-balanced、每方向五类 inverse-frequency-balanced cross
entropy，三个方向等权。水平翻转时同步交换 left/right labels。只运行 seed17，不搜索
方向角、阈值、class weight、head、loss、epoch 或 seed。

## 冻结指标

每方向分别从五类 softmax 构造结构单调的
`P(T≤.5/1/1.5/2 s)`，计算：

- source-session-macro 与 pooled AUROC/AP/Brier；
- 三方向 × 四 horizon 等权 macro；
- 每方向四 horizon macro；
- 对 287 个 exact-time direction-nonredundant anchors，计算 safest-direction
  accuracy：预测五类分布的期望时间最大方向是否属于真值精确首次冲突时间最晚方向
  集合，再做 source macro。

所有差值均为 `history - current`。

## 冻结 gate

D26 只有在以下条件全部满足时支持：

1. source-macro direction-horizon-macro AUROC mean delta 至少 `+0.010`；
2. source-macro direction-horizon-macro AP mean delta 至少 `+0.005`；
3. AUROC/AP 各至少 3/5 folds 为正；
4. left/center/right 的五折 mean AUROC delta 至少 2/3 为正；
5. left/center/right 的五折 mean AP delta 至少 2/3 为正；
6. source-macro safest-direction accuracy mean delta 至少 `+0.020` 且至少
   3/5 folds 为正；
7. pooled direction-horizon-macro AUROC/AP mean delta 均不低于 `-0.005`；
8. cumulative monotonicity violation 精确为 0。

通过终态：

`D26_THOR_MAGNI_COUNTERFACTUAL_COLLISION_FIELD_INCREMENT_SUPPORTED`

失败终态：

`D26_THOR_MAGNI_COUNTERFACTUAL_COLLISION_FIELD_INCREMENT_NOT_SUPPORTED`

## 工程故障与主张边界

复用 D25 修复后的逐 arm CPU checkpoint 与 GPU release，不同时保留两个训练模型。
路径、CSV、CUDA、缓存、checkpoint、序列化、落盘或中断异常仍是工程无效；只要完整
held-out metrics 尚未产生，就修复后按同协议从 fold0 重跑，不烧毁 cohort。

即使通过，D26 也只建立 THOR-MAGNI tracked-body proxy 上的 Development
action-conditioned future-field increment。它不包含完整 foot/body/head traversability，
不建立真实提醒、墙/路缘/落差效用、主线替换、App、生产或安全主张；通过后才允许进入
真实连续事件层或独立数据源复现。
