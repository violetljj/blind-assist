# DEPTHART_ADMISSION_R1：False-Block Resolution & Deployment Audit

状态：`FROZEN / DEVELOPMENT / R0_TERMINAL_IMMUTABLE / RESEARCH_MAINLINE_DEPTHART`

R0 的 `FAIL` 永久保留。本轮不修改 R0 阈值，也不把 R0 结果重标为 PASS；R1 只回答四个
新问题：false-block 的来源、非对称代价下是否可形成 Pareto、relative 与 metric 的
责任边界、以及 ONNX/QNN 是否值得继续投入。

## 四个固定工作包

- A0：官方 `lower_bound_resize` 内参变换审计。TUM 640×480→640×480；无 crop/padding，
  `fx'=sx fx, fy'=sy fy, cx'=sx cx, cy'=sy cy`，平移项为零。用合成 1280×720 与
  640×480→448×448 机械覆盖非等比例情况。
- A1：false-block 按 band、horizon、truth clearance range、sequence/frame 拆分，
  生成 cases 与 contact sheet。ground boundary、thin structure、textureless、confidence
  若 roster 没有标签，必须标为 `NOT_AVAILABLE`。
- A2：DepthART-S relative 224 的 truth-aligned diagnostic control。对 relative 输出
  只允许每帧 affine disparity 对齐到 sensor truth；该 scale 使用 truth，明确是诊断，
  不能作为部署尺度或 R1 admission authority。
- A3：官方 exporter 的 static metric graph inventory：输入必须包含 `image,K`，输出
  `depth`，记录 custom `com.depthart::SelectiveScan` 数量、节点域和 converter 可用性。
  没有 QNN SDK 不得把 graph inventory 写成 HTP 支持。

## 非对称决策规则（只适用于新 R1 holdout）

R0 的严格 AND gate 不动。R1 若进入新的、预注册的 session/parent-disjoint holdout，
采用成本分层：false-clear 是危险漏放，权重高于 false-block；但 false-block 必须有
上限，防止系统退化为“全部 UNKNOWN/封路”。R1 不在 R0 的 120 帧上搜索这个上限、阈值
或后处理。

建议性决策形状（尚未在 R0 结果上套用）：`false_clear <= 8%`、`false_block <= 2%`、
clearance/temporal 通过，并完成 ONNX parity 与真机 HTP feasibility。最终阈值必须在
R1 新 holdout 开始前再次冻结。

## 权威边界

DepthART 现在是 `preferred experimental backbone / research mainline`，但仍不是正式
主干、Android 默认模型、产品或安全证据。DA2 仍是 frozen baseline、teacher、regression
reference 与 fallback。FRESH-TF 和已打开 successors 继续保持暂停。
