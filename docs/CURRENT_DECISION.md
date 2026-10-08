# 当前研究决定

更新：2026-10-08。主线是盲杖互补的前视障碍感知，冻结M3和手机A+LOCAL保留。历史`L10_R0_PAUSED / DTR_R2_DYNAMIC_RETAINED`不表示恢复动态研究。

## 当前：按用户关心的障碍形状检验能力

用户最新授权在对齐直行、理想位姿、不实测条件下，优先横杆、竖杆、柜体突出物，标牌边缘作补充。[形状能力图](../research/active/dtr-r0/nearfield/CNH_ALIGNED_SHAPES_DEV_20261008.md)已完成492场景×K4；316物理接触、344高度query接触、88pass/88clear，全格可评价。固定原−10°、M3与原阈值，不训练。

对应高度及时HEAD/BODY：横杆119/224、85/224；竖杆217/224、177/224；柜体107/128、94/128；标牌74/112、50/112。整体clear46/4576报警时间格、25段、24clip，不是实际提醒频率。GPU墙钟187.188/600s，独立分类、原平滑、分组/账本/结果核验通过。有限方截面/板状模拟代理，不是实机效果。

[横杆缓存诊断](../research/active/dtr-r0/nearfield/CNH_BAR_CACHED_DIAGNOSTIC_DEV_20261008.md)：暗4cm BODY原1/56，同θ去平滑8/56，但匹配全批46/4576 clear格后仍1/56。全批HEAD/BODY净+35/+3伴随损1/10，clear段25→34、pass clip74→86，不升级政策。全部172 BODY接触在f13/.97m仍有对应高度首命中；加长88对均增加目标贡献，不能单归因于退出或无额外回波。理想径向对齐增加条件分离度，但丢角度后的值远低于原角距参考，不是报警增益。下一优先A/B逐级表示损失、负背景及侧别保留，角距联合累积仍开放；完整无杆投影/M3对照NOT_RUN。不自动训练。此前[−19°俯角](../research/active/dtr-r0/nearfield/CNH_ALIGNED_BOUNDARY_DEV_20261008.md)仍不推进，不继续扫角度；方向/Nymeria位移接口与设备暂缓，旧方向瓶颈未解决。

## 停止与保留

[方向扰动续训](../research/active/dtr-r0/nearfield/CNH_QUERY_PERTURB_TRAIN_DEV_20261008.md)未通过：47接触、同56/2240报警格，E1单Aug/Control/M3为35/40/42，双40/44/45；exact各配对损7，超过2件护栏。本配方停止，不加seed/epoch、不换扰动分布；实质不同机制另行审查。本批方向失配导致的损失各仅1件，不足判断补偿能力。

方位标签微调及稀疏空间/关联L3暂停；[空间诊断](../research/active/dtr-r0/nearfield/CNH_BEARING_SUPPORT_CROSS_DEV_20261008.md)、原L2净增益及逐事件损失、去平滑负结果、步态/EMA整套输入对照保留，不是用户收益证明。盲杖互补、UNKNOWN不保证安全，三态/query覆盖仍待验证。

[480新模拟unit确认](../research/active/dtr-r0/nearfield/CNH_REAL_HEAD_CONFIRM_RESULTS_20261007.md)保留原身份，E1较exact少39/51件及时（/1002），EMA收回25/34；不与后续已消费Development及本轮有限网格混同。HEADS-UP来源窄，EMA事后选择；BlindWays已用于开发回放，不能重称独立确认。

[Nymeria样本与IMU](../research/active/dtr-r0/nearfield/CNH_NYMERIA_SAMPLE_AUDIT_DEV_20261008.md)：时钟异常按原段保留，注册只用所选合法段整体拟合、仅作离线参考；右首60秒原始IMU→CPF姿态前缀检查通过，姿态精度未评价，位置/PDR未实现，E1整链NOT_EVALUABLE。未来骨盆/closed-loop不进估计器，不自动重开配方。

恢复设备先核对M3真实输入与逐query真值；101/101和53ms属A+LOCAL，M3/CNH实机效果尚未建立。City、保护test、新UE及硬件第二阶段暂停。

[路线当前页](../research/active/dtr-r0/CURRENT.md) · [设备状态](PROJECT_STATE.md) · [RUNS](../research/active/dtr-r0/RUNS.md)。本次更新前全文：Git `5f5554b3` 同路径。
