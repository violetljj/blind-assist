# HFTF Stage C D25：THOR-MAGNI time-to-entry ordinal protocol

日期：2026-08-03

证据角色：Development / continuous-time representation canary

研究主线：不变

默认 App：不变

## 新科学变量

D23 建立了 2 秒 proximity-onset representation increment；D24 表明该二分类器没有
把动态信息稳定转成事件排序和召回，但 lead-time credit 留下了弱时间信号。对现有分数
做的明确 post-hoc 标量诊断也显示，简单缩放或反转
`history_logit - zero_dynamics_logit` 无法恢复事件 AUROC。

因此 D25 不再修补 D23 score，也不更换 backbone。它把监督问题从：

> 未来 2 秒内是否首次进入 1.25 m？

改为：

> 首次进入 1.25 m 发生在 0–0.5、0.5–1.0、1.0–1.5、1.5–2.0 秒中的哪一段，
> 还是 2 秒内不进入？

模型输出五类 ordinal time-to-entry distribution；四个累计概率
`P(T≤0.5/1.0/1.5/2.0)` 由 softmax 累加得到，结构上保持单调。这改变的是预测对象，
不是把 YOLO 换成另一个主模型。

## 冻结数据与机会

- 继承 D12 的 1,078 THOR-MAGNI samples、19 source sessions 与五个
  source-session-held-out folds；
- 只训练 530 个 `proximity_eligible=true`、当前未进入 1.25 m 的 anchors；
- 用 D8 原始 scenario CSV、QTM anchor 和相同 `0.10 s / 2.0 s` future scan 重建
  首次进入时间；
- 157 个进入 anchors 与 373 个 2 秒内不进入 anchors；
- 五个 ordinal classes：
  - `(0,0.5]`：61；
  - `(0.5,1.0]`：32；
  - `(1.0,1.5]`：35；
  - `(1.5,2.0]`：29；
  - `>2.0 / censored`：373；
- 四个累计 horizon 的正例为 `61/93/128/157`，每个 horizon 在每个 fold 都有正负。

上述普查在协议冻结前完成，只确认 target opportunity，不读取任何 D25 模型结果。

## 冻结模型与对照

复用 D22 的 MobileNetV3、flow-aligned 20-channel dense-dynamics encoder、RGB cache、
RAFT flow、30 epochs、batch、优化器和 source-session folds，只把最后的二目标 head
替换为五类 ordinal head。

两个等容量 arm 从相同初始化独立训练：

1. `current`：重复当前帧、零 flow、精确零 dynamics；
2. `history`：真实五帧与 current-to-history dense flow。

训练目标固定为 source-balanced、class-balanced cross entropy。只执行 seed17 五折
canary，不做 epoch selection、threshold、class weight、horizon、seed 或模型搜索。
所有指标使用固定 final epoch。

## 冻结指标与 gate

对四个累计 horizon 分别计算 held-out：

- source-session-macro AUROC / average precision；
- pooled AUROC / average precision；
- Brier score 作为披露性校准指标，不进入 gate。

主要汇总是四个 horizon 的等权 macro。所有差值均为 `history - current`。

D25 只有在以下条件全部满足时支持：

1. source-macro horizon-macro AUROC mean delta 至少 `+0.010`；
2. source-macro horizon-macro AP mean delta 至少 `+0.005`；
3. AUROC/AP 各至少 3/5 folds 为正；
4. 四个 horizon 的五折 mean AUROC delta 至少 3/4 为正；
5. 四个 horizon 的五折 mean AP delta 至少 3/4 为正；
6. pooled horizon-macro AUROC/AP mean delta 均不低于 `-0.005`；
7. 累计输出 monotonicity violation 必须精确为 0。

通过终态：

`D25_THOR_MAGNI_TIME_TO_ENTRY_INCREMENT_SUPPORTED`

失败终态：

`D25_THOR_MAGNI_TIME_TO_ENTRY_INCREMENT_NOT_SUPPORTED`

失败只关闭当前 D22 encoder 上的 ordinal time-to-entry successor，不撤销 D23
representation 正结果，也不把 D24 改写为工程失败。

## 最小工程与主张边界

训练前只做现有路径、hash、shape、checkpoint 与 target count 的必要校验。路径、CSV、
CUDA、缓存、空 batch、序列化、落盘或中断异常属于可修复工程故障；在五折 held-out
metrics 全部产生前修复并从头重跑，不烧毁 source。

即使通过，D25 也只建立 source-held-out Development time-to-entry representation
increment；它不建立真实提醒效用、可部署阈值、主线替换、App、生产或安全主张。通过后
才允许把 ordinal cumulative score 接入一个冻结的真实序列 event test。
