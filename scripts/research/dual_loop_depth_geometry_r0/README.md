# Dual-loop target-depth geometry Discovery R0

状态：`DISCOVERY / DEVELOPMENT_INPUT_ONLY`

## 问题

在已经烧毁、仅可作 Development 的 REveL 512-frame 固定子集上，官方
Depth Anything V2 Small 的 target-ROI 相对深度是否包含比已拒绝的即时框面积和
稀疏径向流更有希望的径向方向信息。

本轮不是效果 A/B，也不选择提醒或融合策略。REveL Vicon truth 只由 separate
post-producer evaluator 打开；producer 只读取固定 RGB 与既有 GT ROI。

## 稳定 Interface

```text
produce.py
  --details <fixed RGB/ROI ledger>
  --image-root <REveL image directory>
  --output <artifacts.local JSONL>
  --receipt <artifacts.local JSON>
  --batch-size 1

evaluate.py
  --features <producer JSONL>
  --producer-receipt <producer JSON>
  --radial-ledger <existing Vicon radial ledger>
  --output <artifacts.local JSON>
```

## 输出

全部输出写入：

`artifacts.local/evidence/dual-loop/target-depth-geometry-discovery-r0/`

Producer 只保存 target ROI 深度摘要、帧内归一化摘要、身份与哈希；不保存 RGB、
深度图、风险、提醒或个人信息。

## 安全边界

- 输入是已访问的单一 REveL capture，只能作 Development/diagnostic。
- GT ROI 是 oracle-conditioned，producer receipt 会显式记录
  `oracle_roi_opened=true` 与 `vicon_truth_opened=false`；它不是运行时检测器、
  独立感知源或 Android 可用性证据。
- 固定逐帧 batch，避免对异构纵横比图像做会改变几何的 padding。
- Depth Anything 是模型生成的相对深度，不是传感器真值。
- 不读取旧 F-1B sealed decision output 或 CrowdBot 风险/提醒结果。
- 不做阈值搜索；方向读出固定为“模型输出增大代表更近”。
- 不将 pooled correlation 或同源方向一致写成效果、泛化、产品或安全结论。

## 停止条件

输入哈希漂移、GT ROI 数不为 770、模型加载失败、输出缺行、非有限深度或 producer
receipt 不闭合时返回 invalid。若方向一致性或 target-depth/range 单调关系不优于随机
水平，只关闭该模型候选，不回调阈值或挑子组。
