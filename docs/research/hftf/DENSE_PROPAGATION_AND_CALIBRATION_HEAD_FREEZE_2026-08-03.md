# Dense propagation and calibration-head freeze

日期：2026-08-03

状态：`TWO_SUCCESSORS_FROZEN_BEFORE_OUTPUT_READ`

方案一的 consumed R1 已判负，不能在同一 outcome 上调参。这里一次性冻结两个实质不同
的后继：方案二传播 Metric3D 相对 DA 的稠密残差；方案三让冻结 DA 特征预测全局
`a,b`，手机运行时不再调用 Metric3D。

方案二每五帧请求一个 Metric3D keyframe，按 `142.33 ms` 完成时刻因果可见。RAFT-small
同时计算 current→anchor 与 anchor→current flow，只有 forward/backward error
`<=1.5 px` 的区域传播 residual。当前 DA 负责随时间变化和新区域；若 flow-consistent
coverage `<50%`、dense affine 不可识别、anchor 未完成或 source-age `>1 s`，整帧
`UNKNOWN`。非 keyframe Metric3D depth 只能做 full-rate comparator 或方案三 teacher，
代码必须阻止其进入方案二候选。

方案三使用 DA V2 Metric Hypersim ViT-S 第 11 层 384 维 CLS token，固定一个 770 参数
双输出 ridge head。每帧 teacher label 是 DA dense depth 到 Metric3D dense depth 的稳健
全局 affine `a,b`。四个 TUM sequence 做 LOSO；每折只用另外三段的 teacher label 和
feature normalization，held-out Metric3D 不得进入拟合/预测。RGB-D sensor depth 只在
四折预测全部物化后进入 clearance evaluator。

两者都只使用已消费 TUM Development 数据。任何结果都不得升级为 fresh、端侧温度/
内存、最终相机、告警、安全、ToF 替代或默认 App 证据。完整常数、输入 hash、门与停止
规则见两份 JSON 合同。

控制面补记：首次 dense-cache 物化在 frame 0 因 A0 summary report 不含 intrinsics
而停止，没有形成可读 cache 或候选结果。协议随后只增加原始四份 pose-torso manifest
及其 SHA 以绑定内参；算法、模型、拟合、flow、门和数据角色均未改变，失败输出根保留。
第二次物化已完成 DA cache，但在任何 Metric3D inference 前因现有 vendored
`mmengine/mmcv` 根未加入 `sys.path` 而停止；修复仅接入仓库既有依赖根并改用新的输出
目录，仍不改变算法或门，第二个 partial 根同样保留。

方案三首个输出后的 truth-firewall 复核发现：实现逐折预测后立即评价，虽然 truth 从未
进入训练、归一化或选择，仍弱于冻结文字要求的“四折预测全部物化后再开 truth”。实现
随后只调整执行顺序，并在新输出根确定性复核；folds、aggregates、increment、终态和
all-consumed 模型 SHA 均逐项一致。首个输出保留，不覆盖。
