# 证据覆盖审查

结论：**PASS**。核心路线判断均有本地实验或审计证据支撑，论文证据只用于机制迁移与候选设计，没有替代项目内晋级门禁。

| 核心判断 | 项目内证据 | 论文证据 | 覆盖强度 | 边界 |
|---|---|---|---|---|
| P1 提高了最好结果，但没有关闭 seed 方差，近期先执行 P2 | LOCAL-03、LOCAL-06 | PIDNet、Mobile-Seed | 强 | Mobile-PID-lite 仅为重入候选，不是已验证改进 |
| 数据质量应进入采样、损失和评测闭环 | LOCAL-07 | UPC、SWSEG | 强 | HUMAN/MACHINE 来源统计是一轮派生审计，不是 canonical schema |
| 单点 mIoU 不足以支持上线，应补 risk-coverage 与校准 | LOCAL-04 | ValUES、Kandinsky | 强 | 论文指标需按 BlindAssist 风险语义重新定义 |
| 事件级评测必须区分实际重复播报与被抑制的重复尝试 | LOCAL-05 | DTERN、BOFP、Escalator Problem | 强 | 当前 `repeatedAlertCount` 不能直接解释为用户实际收到的重复提醒 |
| 助盲系统需要从像素分割上升到风险、路线与交互层 | LOCAL-04、LOCAL-05 | VisAssist、CLIP-BLV、AI Guide Dog | 中强 | 候选能力尚未通过 BlindAssist blind/INT8/device 门禁 |
| 正式晋级必须同时通过 worst-seed、blind event、INT8 与设备预算 | LOCAL-03、LOCAL-04、LOCAL-06 | 论文仅作机制先验 | 强 | 报告没有宣称任何新模型已替换默认模型 |

检查结果：

- `refs/evidence-map.md` 已覆盖报告全部核心主张，并标注证据强弱与适用边界。
- 14 篇论文均有来源、哈希、页数和精读笔记；报告中的迁移建议可追溯到对应论文或本地证据。
- 未发现把 oracle、head-only、派生审计或论文结果误写为生产晋级证据的情况。
- 剩余不确定性均以待执行实验或条件候选表达，不构成无证据结论。
