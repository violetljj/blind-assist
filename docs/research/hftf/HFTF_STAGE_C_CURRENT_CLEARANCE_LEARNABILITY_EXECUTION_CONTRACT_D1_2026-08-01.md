# HFTF Stage C G0-D1 current-clearance learnability execution contract

日期：2026-08-01

状态：`FROZEN_BEFORE_D1_DEVELOPMENT_CORPUS_OR_STUDENT_OUTCOME`

## 1. 本合同授权什么

本合同只授权把 G0 source plan 中已 outcome-open 的 9 个 Development 来源物化成
current-only 语料，并训练同一 F0.1 student 的两个输出机制：

- `DIRECT_RISK_CURRENT`：直接输出 binary-risk logit；
- `SIGNED_CLEARANCE_CURRENT`：输出不经 activation/clamp 的线性 meter clearance，
  推理时严格用 `< 0 m` 导出 risk。

它不授权获取或打开三条 one-shot fresh 来源、三条 reserved official-test 来源，
也不授权 future/temporal、Android、App、生产、安全主张或研究主线晋级。

## 2. 语料与防泄漏

语料必须精确来自 source plan 的前 6 个 prior-train 与后 3 个 prior-dev 来源，每源
25 个 source-plan 冻结 target-timeline current frame（7 源为 10 FPS、2 源为
5 FPS），共 `150 train + 75 model-selection`。这个执行语义由独立的 timeline
amendment 在任何 corpus outcome 前纠正并绑定。student JSONL
只能含当前 RGB、identity 与 `known/risk/clearance` targets；UNKNOWN 的 risk 与
clearance 必须同时为 null，绝不当作 SAFE。depth、mask、pose、teacher receipt、
future 字段和 fresh/reserved session 一律不得进入 loader。

materializer 与独立 validator 分离。validator 要重新绑定 source plan、逐字节校验
真实 manifest/authority/current RGB/depth/mask/pose receipts，并从这些 authority
inputs 独立重推全部 labels；同时验证 exact 6/3 source partition、每源冻结 frame
set、risk 与 `clearance < 0` 等价、UNKNOWN firewall，以及逐
`source × height` 的正负类、近边界和 clearance 分布非退化。只有
`G0_D1_DEVELOPMENT_CORPUS_VALIDATED` 才允许优化。

## 3. 两阶段训练

两个 arms 均使用 seeds `17/29/43`、同一 F0.1 模型/输入/参数量、30 epochs 与冻结
loss。Phase A 只用 6 train sources 拟合，3 model-selection sources 只按预声明的
source-macro F1、worst-source F1、micro F1、clearance arm 的 source-macro MAE
严格平分规则和最早 epoch 选点。

Phase B 对每个 arm/seed 重置模型、optimizer 与 RNG，用全部 9 个 Development
来源完成 30 epochs，只冻结对应 Phase A 已选 epoch。Phase B 开始前必须重新计算
Phase A 的 30-epoch selection、校验 canonical report/checkpoint hash 和 identity；
不能只相信 report 内的 epoch 字段。

冻结 runtime 是 `blindassist-python.cmd`、Torch `2.11.0+cu128`、Torchvision
`0.26.0+cu128`、CUDA、float32 no AMP、deterministic algorithms、workers 0。
版本、CUDA、预训练 checkpoint 或 implementation SHA 漂移即在 optimization 前停止。

冻结实现 SHA-256：

- corpus materializer：`da0523fe7a01064540b788d9e92f889c0a7e331ae6e71ba5683023c96a70c153`
- corpus validator：`bdfb8eb15cee7232d681e96c30e4b3186331ddec4e68d5226f2b311ca743e39c`
- current student trainer：`d0d668b509015f5c18e6e40f5cd4ccac17f1523ac8744c5f6c78e60c287ec716`
- training validator：`68713284875550ee7c31d335ae6025333b21571d4092937bcd62b0b2da4749b5`

## 4. fresh 前门禁

独立 training validator 必须看到完整 `2 phases × 3 seeds × 2 arms` 目录，重算
Phase A selection、核对 Phase B epoch、严格加载全部 checkpoints、确认相同 seed
跨 arm/phase 的初始 state 与 loss 参数一致，并冻结 6 个最终 Phase B checkpoint
hash。唯一允许继续设计 fresh 执行合同的终态是：

`G0_D1_SIX_FINAL_CHECKPOINTS_FROZEN`

即使到达该终态，也只表示 Development checkpoint 已冻结，不表示 fresh learnability
成立。fresh acquisition、prediction-first 和 truth join 仍需另行冻结一次性执行合同。
