# 前视障碍感知：当前状态

更新：2026-09-27（用户与两位执行者共识，由用户确认）。Track A 程序化 ToF 仿真阶段暂时封口。

Status: `DTR_R2_DYNAMIC_RETAINED`（仅为历史保留）；ToF 阶段：`TEMPORARILY_CLOSED`。

## 当前决定

- **不再新增 ToF 实验。** 用户指定下一优先级仅为公开真实直方图数据检索。相机线未决定；硬件恢复时间未知。历史诊断里的“下一步”建议不构成执行授权。
- **v4 偏差已由用户接受，必须披露。** 六个主检验与 A1/A2 为“正式结果，带已披露偏差”。在 GPU 宏 AP 已打印后补做未执行的数值比较，再续跑冻结命令；不得抹去偏差或追改协议，见[v4结果](nearfield/CNH_TRACK_A_SCALE_V4_RESULTS_20260926.md)。
- v3/v4 只支持同一生成器分布下的相对读出效果；v1/v2 原失败不变。修复 v2 和后续诊断是已消费 Development，不能升级成正式复现。
- City/test、UE/RGB、硬件第二阶段继续暂停。手机 A/A+LOCAL 与首页语义保留，研究结论不自动进入实机。

## 关键证据

|证据|数字与分母|解释|
|---|---|---|
|v3/v4|63/64 audit单位；S1、S2、S3六个预设比较均成立|v4接受偏差，非无偏差执行|
|v4 A1|BODY近事件1990；没报减少5.1–5.4pp|主要转为报晚|
|v4 A2|HEAD/BODY近事件1936/1990；及时优势7.4–9.2/4.8–7.4pp|同calib预算，实际假警不等|
|候选质量与软先验|32calib/63audit；小块近事件432/404；中等组合AUC≤.9未保留一半理想角度增益|只限当前机制和误差分布|

[论文主张台账](THESIS_CLAIMS_20260927.md)列全量来源、提交、分母、范围与禁止措辞。不得从这些结果推出物理上限、任何ToF无效、相机必要/无用或普适候选要求。

## 未决与证据入口

相机线与产品报警工作点未决定；模拟假警单位不能替代真实负担。SNR6只经ZJUL5粗锚定；真实直方图与硬件标定不足。头戴查询是“重力水平、偏航跟头”；带噪自运动是假设，非仅IMU已实现。

[公开数据检索](../../../artifacts.local/work/tof-real-histogram-search-20260927/REPORT.md)找到TMF8820的LCSPCData真实直方图/部分真值目录；L8CH THDR3K入口仍未核验。仅查网页，未下载；不可将跨型号数据直接当L8CH标定。

[v3结果](nearfield/CNH_TRACK_A_SCALE_V3_RESULTS_20260926.md) · [v4协议](nearfield/CNH_TRACK_A_SCALE_V4_PROTOCOL_20260926.md) · [软先验](nearfield/CNH_SOFT_PRIOR_DEV_20260927.md) · [ZJUL5粗锚点](nearfield/CNH_ZJUL5_SNR_ANCHOR_20260926.md)

[总决定](../../../docs/CURRENT_DECISION.md)拥有跨路线优先级；[封口前全文](CURRENT_HISTORY_20260927_PRE_TOF_CLOSE.md)逐字节归档。本次仅收口证据与决定，不制作章节大纲或图表清单。
