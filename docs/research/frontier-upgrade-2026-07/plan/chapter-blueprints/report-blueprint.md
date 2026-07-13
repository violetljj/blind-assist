# 报告段落蓝图

## P1 执行结论
- Role: conclusion first
- Main claim: 下一次跨越来自多层组合，不来自单一更大模型。
- Evidence IDs: LOCAL-01, LOCAL-02, PID, MRFP, UPC, VALUES, DTERN, AIGD, VISASSIST
- Contrast: 单 seed/global mIoU 与事件级安全之间的差异。
- Forbidden content: 把建议写成已经验证的事实。

## P2-P4 当前问题链
- Role: evidence-backed diagnosis
- Main claim: 边界丢失、模型随机状态、标签与域偏移、置信度和逐帧决策形成级联。
- Evidence IDs: LOCAL-01..LOCAL-04
- Contrast: 已闭合工程门与尚未闭合模型质量门。
- Forbidden content: 重复项目历史。

## P5-P12 方法路线综合
- Role: thematic literature synthesis
- Main claim: 每条路线解决不同层次问题，存在明确依赖顺序。
- Evidence IDs: all paper IDs grouped by theme
- Contrast: 可立即迁移、需受控验证、仅长期参考。
- Forbidden content: 一篇一段的机械摘要。

## P13-P18 建议架构与实验
- Role: constructive proposal
- Main claim: Mobile-PID + training-only robustness + calibrated abstention + causal temporal event layer 是可证伪路线。
- Evidence IDs: paper synthesis + current repo contracts
- Contrast: teacher/offline/benchmark-only/production candidate boundaries.
- Forbidden content: 未定义指标的“大幅提升”。

## P19-P21 风险与结论
- Role: limitation and decision rules
- Main claim: 只有 worst-seed、blind event gate 和端侧 budget 同时过门才允许晋级。
- Evidence IDs: LOCAL-03, LOCAL-04, VALUES, VISASSIST
- Contrast: 研究价值与产品安全证据。
- Forbidden content: 上线承诺。
