# Last-10m completion nearness S0/S1 result — 2026-08-22

状态：`S1_COMPLETE_AND_SEALED / DEPTH_REDUCES_FALSE_COMPLETION_BUT_LOSES_TRUE_COMPLETION / NOT_READY_FOR_COMPLETION_CONTROL`

## 问题

S0v11 已否决用 detector bbox height 代替真实接近程度。本阶段只检验一个更窄的问题：在 provider 只看到当前
RGB 和预先存在的 public goal semantics 时，独立 metric-depth 是否能比 bbox-only baseline 更可靠地约束
door completion。它不恢复 P1、identity memory、tracking 或 AMRM，也不声称 door 可穿行。

## 合同与数据边界

- `NYUv2 S0`：`LEFTMOST_VISIBLE_DOOR / UNIQUE`，24 near + 24 far；合同、阈值、模型和 path-hash roster
  规则都在读取 private label/depth 前冻结。
- `SUNRGBD S1`：fresh `VISIBLE_DOOR / SET_VALUED`，只纳入预声明的 B3DO、SUN3D 和 Kinect2 来源，排除所有
  NYUv2 ancestry。冻结后在 1,382 个合法样本（480 near、902 far）中按 sample-path SHA-256 取 24 + 24；
  48 帧包含 74 个 exact-`door` legal target。
- provider 固定为 YOLOE-26n-seg text prompt `door`、Top-10 proposal 和 Depth Anything V2 metric
  Hypersim ViT-S ONNX；交互阈值 2.0 m，target hit 为 IoU >= 0.30。S1 selection 预注册为 provider-score Top-1。
- provider 从不读取 segmentation、bbox、mask、depth、near/far stratum。两个 formal run 均一次完成、无 replay。
- ONNX Runtime 的 CUDA EP 因本机缺少 `cublasLt64_13.dll` 未加载，按同一冻结 ONNX 自动回退 CPU；YOLOE
  仍在 CUDA。receipt 记录实际 providers。

## 结果

| cohort | candidate availability | selected target | bbox-only TP/FP/FN/TN | depth-gated TP/FP/FN/TN | terminal |
|---|---:|---:|---:|---:|---|
| NYUv2 S0 | 45/48 | 12/48 | 0/1/8/39 | 0/0/8/40 | false commit removed, zero end-to-end TP |
| SUNRGBD S1 | 41/48 | 24/48 | 2/8/8/30 | 1/1/9/37 | false completion reduction trades off TP |

S1 的 depth gate 将 false completion 从 8 降到 1，但 true completion 从 2 降到 1；因此它不能被提升为
completion controller。条件于 provider 已选中合法 door，metric-depth 的近/远分类为 TP=7、FP=0、FN=3、TN=14，
MAE 0.622 m、median absolute error 0.358 m。这说明独立 depth 提供了真实但不充分的 nearness observability：
它显著抑制过早完成，同时存在 near-door 低估覆盖和 proposal selection 两个瓶颈。

原始 S1 evaluator 曾把“帧内存在任一 near door”错误绑定到被选 candidate；SET_VALUED contract 要求绑定
matched target 的 depth。模型未重跑，原始 evaluation 保留，权威结果写入 `evaluation_corrected.json`。

## 收口与下一算法门

本阶段建立的是：`goal-semantic proposal -> selected region -> independent metric nearness` 可以运行且比 bbox
heuristic 更少误完成，但尚未达到可用控制覆盖。下一算法工作必须同时处理：

1. 在 bounded candidate pool 内对每个 candidate 独立估深，按 goal relevance 与可交互近距联合选择，而不是先
   Top-1 再估深；
2. completion 仍需 doorway/functional-region visibility 与稳定的 near evidence，不能把 `door detected + depth <= 2m`
   当作可穿行或到达；
3. 新算法只能在 development data 上形成，最终要用新的、truth-before-use 合法 cohort 确认。

Claim ceiling：静态室内 RGB-D door proposal/nearness 与 premature-completion engineering evidence only；不证明
building entrance、真实物理接近、穿行性、导航、产品 readiness 或用户安全。
