# 当前研究决定

更新：2026-09-27（用户与两位执行者共识，由用户确认）。主线：盲杖互补的前视障碍感知。

Status: `L10_R0_PAUSED / DTR_R2_DYNAMIC_RETAINED`（历史保留）；ToF 阶段：`TEMPORARILY_CLOSED`。

## 当前决定

- **用户决定：ToF 阶段暂时封口，不再启动新的 ToF 实验。** 后续优先级仅为公开真实直方图数据检索；不把检索扩展为新实验。相机线仍未决定，硬件恢复时间未知。
- **接受 v4 执行偏差并披露。** 六个主检验及 A1/A2 保留为“正式结果，带已披露偏差”；不写成完全符合冻结流程，不事后修改协议。接受理由及恢复时点见[v4结果追加决定](../research/active/dtr-r0/nearfield/CNH_TRACK_A_SCALE_V4_RESULTS_20260926.md)。
- v1/v2 原失败保留；修复 v2、位置/质量/软先验诊断均为已消费 Development，不追认为正式证据。相机必要性、ToF物理上限与真实效果均未建立。
- 手机保留 A 基线及原首页→手动开始 A+LOCAL→结束返回首页；UNKNOWN 不等于无障碍。City、保护test、UE/RGB和硬件第二阶段仍暂停。

## 保留证据

|证据|关键数字|边界|
|---|---|---|
|v3 / v4 主检验|63 / 64 audit单位，六项均成立；S2−B1-R HEAD/BODY：+0.099/+0.079、+0.092/+0.078|受控仿真相对增益|
|v4 A1/A2|BODY近事件1990；没报减少5.1–5.4pp。HEAD/BODY近事件1936/1990；单帧及时优势7.4–9.2 / 4.8–7.4pp|相同calib预算，audit实际假警不等|
|候选与软先验|修复v2：32calib/63audit；中等组合AUC .9保留HEAD/BODY增益37.1%/17.0%|消费Development；非通用候选规格|

完整主张、分母、来源提交与禁用措辞见[论文主张台账](../research/active/dtr-r0/THESIS_CLAIMS_20260927.md)。

## 未决与入口

相机线、产品报警工作点仍未决定；1.8秒“序列×查询盒”假警率不能换算实际提醒负担。真实计数/串扰/安装标定待硬件。确认畅通距离仅作≥10cm、ρ≥0.5、最坏摆放的附录辅助地图。

[公开数据检索](../artifacts.local/work/tof-real-histogram-search-20260927/REPORT.md)：已核实LCSPCData（TMF8820）真实直方图与部分真值的文件目录；THDR3K（L8CH）入口仍未核验。未下载数据；后续文件审计/验证另行决定，不直接迁移为L8CH标定。

[路线当前页](../research/active/dtr-r0/CURRENT.md) · [v3结果](../research/active/dtr-r0/nearfield/CNH_TRACK_A_SCALE_V3_RESULTS_20260926.md) · [软先验诊断](../research/active/dtr-r0/nearfield/CNH_SOFT_PRIOR_DEV_20260927.md) · [项目入口](PROJECT_STATE.md)

[封口前全文](operations/snapshots/CURRENT_DECISION_20260927_PRE_TOF_CLOSE.md)逐字节保存；旧快照中的待决状态不覆盖本页。本次不制作章节大纲或图表清单。
