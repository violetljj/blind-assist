# SATOM-R0 Bonn Real E0 execution result

状态：`TERMINAL / REAL_E0_NOT_EVALUABLE / DEPTHART_GROUND_HEIGHT_OBSERVABILITY_FAIL / NO_ARM_METRIC / NO_HEADROOM_CLAIM / CLOSED_NO_TUNING`

## 结论

预冻结的 Bonn Real E0 没有形成一次有效的 SATOM arm 比较。失败发生在 frozen DepthART
dense prior 到 task-height geometry 的物化边界：单目先验不能在首个冻结 parent 上稳定提供
满足预冻结守卫的重力约束地面高度。因此不能计算 `satom_round_robin` 相对四个 comparator
的 coverage、false-clear、false-block、MAE、ECE 或 matched-coverage，也不能签署正结果或
算法负结果。

这是一项 `NOT_EVALUABLE` 执行终态，不是 SATOM fusion 假设被反证。SATOM-R0 仍按停止
条件关闭：不再放宽高度门、不在已打开 Bonn/DepthART 输出上调参、不训练 adaptive policy，
也不进入真实 ToF E1。

## 冻结身份

- 原始执行锁：[Real E0 execution lock](SATOM_R0_BONN_REAL_E0_EXECUTION_LOCK_2026-08-15.json)，
  commit `f33b0c6d488fb7c7990faf1e0dab53cf015937f2`；
- 数据：元数据哈希排序前 24 个动态 Bonn candidates，冻结后 6 parents × 48 frames；
- prior：DepthART `depthart_metric_indoor_s_448`，source commit
  `0384521b3bcb4c64adf03eeb5d55ebdb1cbdd84c`，checkpoint SHA-256
  `597631AC7AEAB8346F4DB013C3C65EF3203DF373E21C7265D7A147093C667E65`；
- PRIMARY：`satom_round_robin`；四个 required comparators、matched-coverage points 和
  winner rule 均在像素/候选输出前冻结，未因失败改变。

## 三次执行有效性守卫

1. 原始 raw projected-depth `q98` 在首帧返回候选高度 `2.858844397500748 m`，超出
   `[0.5, 2.5] m`；0 输出文件、0 arm metrics。该方法会把远墙当地面，物化无效。
2. [A1](SATOM_R0_BONN_REAL_E0_EXECUTION_VALIDITY_AMENDMENT_2026-08-15.json) 只替换为仓库
   既有的重力约束直方图支撑估计；首个 parent 第 19 帧 truth 缺少足够单帧支撑；
   0 输出文件、0 arm metrics。
3. [A2](SATOM_R0_BONN_REAL_E0_PARENT_HEIGHT_AMENDMENT_2026-08-15.json) 在候选与 truth
   严格分离的前提下，冻结 parent median、至少 `24/48` 有效帧及 height MAD
   `<=0.25 m`。首个 parent 48 帧完成推理后，candidate height MAD 超门；
   0 输出文件、0 arm metrics。

同一高度 observability 边界连续三次触发后停止。`artifacts.local/evidence/satom-r0/real-e0-r0/`
中没有 manifest、bundle、evaluation result 或 assessment；不得把部分推理解释为 Real E0。

## 机制判断

SATOM-R0 当前 task clearance 需要相机离地高度。Bonn pose 的世界原点不是可靠地面零点，
而冻结 DepthART prior 的地面支撑高度在动态片段内又不稳定；truth 高度不能回流给 candidate。
因此这批输入无法在不引入 truth scale/height leakage 或事后放宽门的情况下回答核心比较。

未来若重开，必须是新的 pre-outcome 协议，并在读取候选输出前提供以下任一 materially
different 条件：source-native 相机高度/地面平面；不依赖绝对地面高度的任务表示；或独立
metric sensor 对地面高度的物理观测。当前没有 active successor。

## Claim ceiling

本轮只证明预冻结 Real E0 的输入几何不可评估。它不证明 SATOM 有效或无效，不证明真实
ToF、Android、部署、产品、安全或论文创新；默认 App 未改变。
