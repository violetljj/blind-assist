# HFTF Stage C：current signed-clearance D1 fresh 一次性执行合同

## 结论

本合同在三条预声明 fresh SANPO-Synthetic parent session 的任何本地媒体、几何 teacher 结果或 student prediction 被打开前，冻结 D1 的一次性 fresh 评估。

它只回答一个窄问题：在相同 current RGB、编码器、known head、推理预算、三 seed 与固定 checkpoint 下，`SIGNED_CLEARANCE_CURRENT` 是否比 `DIRECT_RISK_CURRENT` 更容易跨来源学习风险。

即使所有 gate 通过，权限也只到“可另行冻结 causal-transport 实验合同”；不会修改传统主线、App、默认模型，也不产生安全或生产证据。

## 为什么 source 合同不预填本地文件哈希

三条 fresh source 在冻结时仍保持未打开，因此此刻不存在可诚实引用的本地 manifest、dataset spec、pose、media 或 authority 哈希。

冻结选择权威是已经 hash-bound 的 G0 source plan：精确 session、顺序、`0,2,...,48` 源帧、20→10 FPS 以及远端 description/pose object receipts。合同还冻结 acquisition 与 authority-verifier 实现。打开后产生的本地哈希只作为传输与权威收据封存，不得反过来改变来源、阈值、checkpoint、gate 或执行顺序。

## 一次性防火墙

1. 先提交并推送本合同和全部实现。
2. 只获取三条固定 fresh source；不得打开 reserved heldout。
3. package materializer 在首次读取本地 fresh package source/media 前先耐久写入
   execution receipt；任何后续失败都消费本次 materialization，禁止重跑或换源。
4. package validator 同时发布：
   - 完整 `validation.json`：含 truth hash、teacher receipt hash 与 opportunity 统计，只供 evaluator/validator；
   - 独立 `prediction_authorization.json`：严格白名单且不含 truth、teacher 或 opportunity 信息，只供 predictor。
5. 六个 checkpoint 对 75 条 current RGB 形成精确 450 条 prediction。全局 completion receipt 落盘后才允许 primary prediction-to-truth join。
6. evaluator 必须先耐久写入消费 receipt，再单次读取 truth bytes，同时核验 hash 与解析内容；此后禁止第二次 forward、重跑、换源或救援调参。package materializer/validator 在预测前的 truth 访问只用于隔离的 label/opportunity 物化与复核，predictor 不可见；primary join 后的独立 validator 可只读重算，但不构成第二次 primary join，也绝不授权第二次模型 forward。

## Opportunity 与结果 gate

每条 source、每个 `body/head` 高度必须同时满足：

- known coverage ≥ 0.10；
- known positive ≥ 5；
- known negative ≥ 20；
- UNKNOWN→SAFE 违规为 0。

任一 opportunity cell 不足，直接 `NOT_EVALUABLE`，不得预测或换源。

效果 gate 完全继承 D1 冻结设计。风险效果按三 seed 中位数；四项 clearance MAE 先对六个 source×height cell 等权 macro，再要求每一个 seed 都通过，即用三 seed 最大 MAE 与阈值比较。这一解释在 fresh outcome 前冻结，因为原设计只给出了 MAE 阈值，并未授权用 median seed 放松回归质量。

## 终态

- 全部 gate 通过：`SIGNED_CLEARANCE_CURRENT_BRIDGE_SUPPORTED_FOR_CAUSAL_TRANSPORT_CONTRACT_ONLY`
- 任一效果 gate 失败：`SIGNED_CLEARANCE_CURRENT_CROSS_SOURCE_LEARNABILITY_NOT_SUPPORTED_STOP`
- opportunity 不足或一次性执行链在消费后失败：`NOT_EVALUABLE`，不得同 cohort 重跑、换源或救援。

完整机器可读合同、父证据哈希、实现哈希、checkpoint 哈希、路径和 gate 数值见同名 JSON。
