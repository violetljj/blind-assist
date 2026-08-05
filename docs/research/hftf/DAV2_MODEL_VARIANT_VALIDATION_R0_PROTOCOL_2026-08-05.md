# DA V2 模型变体准确率与 false-clear 门 R0

日期：2026-08-05（在启动任何分辨率、精度、student、token 或 attention 变体前冻结）

## 结论

P1 冻结为一个 120 帧、四个连续 30 帧窗口的工程回归门。它同时重算像素深度相对误差、
地面恢复、三带 clearance、VALID/UNKNOWN、false-clear、时序与几何状态变化，并锁定 12 个
典型失败帧。任何 P2 候选必须先物化完整 `120x480x640` optical-z 深度缓存并一次性过门，
再允许消耗真机 profile 预算。

这批 TUM 数据已经被既有 Development 工作消费，因此只能判断“相对 canonical FP16 是否
非劣、性能收益是否值得”，不能建立产品或安全 authority。最终相机仍需新的
session/parent-disjoint RGB-D 或量距真值集。

## 两层门

第一层是 canonical 非劣化门，回答“95 ms 降到 40 ms 是否以不可接受的质量损失换来”。
候选必须同时满足：

- 每帧注册深度的 raw metric AbsRel 中位数，较 canonical 增量不超过 `0.02`；
- 只用于相对结构诊断的一次乘性尺度对齐 AbsRel，增量不超过 `0.01`；
- 地面恢复成功率下降不超过 `1 pp`；
- clearance MAE 增量不超过 `0.025 m`；
- collision agreement 下降不超过 `2 pp`；
- false-clear 增量不超过 `1 pp`；
- temporal clearance-delta MAE 增量不超过 `0.015 m`；
- VALID/UNKNOWN exact agreement 至少 `98%`；
- 3 band × 3 horizon 几何状态 exact agreement 至少 `95%`；
- 连续帧几何“变化/不变”一致率至少 `95%`。

所有门为 AND，不做总分补偿。一次 per-frame median scale 只隔离相对结构，不是部署尺度，
禁止把它用于 clearance 或产品路径。

第二层沿用既有绝对 task 门：paired-valid `>=90%`、clearance MAE `<=0.25 m`、collision
agreement `>=90%`、false-clear `<=5%`、temporal delta MAE `<=0.15 m`。即使全过，因数据
已消费也只到 Development task evidence；当前 canonical DA V2 本身不满足这一层。

## 性能放行

质量缓存先完成、哈希先锁定，再做同设备固定 APK 的端到端 P95。候选至少需要 `1.15x`
P95 speedup 才值得引入新的模型维护成本。App latency、QNN accelerator time、teacher cadence
和 YOLO cadence 分开报告，不以单个 HTP inference 数字替代端到端频率。

## 数据与失败场景

- TUM Freiburg 3 `walking_static` 三个窗口与 `walking_xyz` 一个窗口，各 30 帧；
- roster 对每一张 RGB 和注册深度记录 SHA-256；
- canonical aligned-depth cache SHA-256：
  `9A7FC55DB6B3E7C467B5BAFE68D3603F4B463C498558B7236D603569595D3A34`；
- 12 个固定失败帧覆盖 canonical false-clear cluster、最大 clearance/depth 误差、
  sensor `UNKNOWN_GROUND` / canonical `VALID` 不一致，以及 walking_xyz 尾部。

机器可读权威为同名 JSON 和
`DAV2_MODEL_VARIANT_VALIDATION_R0_ROSTER_2026-08-05.json`。执行器为
`scripts/research/hftf/evaluate_dav2_model_variant_gate_r0.py`。

## P2 规则

首批只允许预先登记的少量 arms；不能在本 cohort 上搜索分辨率、量化层、student checkpoint、
seed、teacher cadence 或阈值后挑最好者。推荐顺序仍是：固定低分辨率蒸馏 canary、固定轻量
student、选择性 mixed precision、固定 token/attention 降本，最后才评估多速率 teacher/student。
任一 arm 未过第一层即停止其真机深 profile；过第一层但未过第二层，只能作为高频相对结构或
disagreement 候选，不能独立驱动米制 clearance。
