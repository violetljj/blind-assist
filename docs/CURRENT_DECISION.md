# 当前研究决定

更新：2026-10-08。主线是盲杖互补的前视障碍感知，冻结M3和手机A+LOCAL保留。历史`L10_R0_PAUSED / DTR_R2_DYNAMIC_RETAINED`不表示恢复动态研究。

## 当前：对齐直行下的感知边界

用户授权暂缓方向估计和实机工作，先固定直行、yaw对齐，补能力图并比较一项针对性改动。[边界与俯角报告](../research/active/dtr-r0/nearfield/CNH_ALIGNED_BOUNDARY_DEV_20261008.md)已完成：168有限立方体场景、K4光子重复，72接触/48擦身/48clear；原−10°对几何预选−19°，冻结M3、exact相对位姿oracle输入，不训练。

同28/2496报警时间格、残差0，HEAD31→29/144，救6损8；BODY31→52/144，救25损4。各高度配对损失超过预设≤2，且HEAD净负，不推进−19°，保留原安装角度。段数16→20，时间格相等不等于提醒次数相等。GPU墙钟133.594/1800s，独立合同及最终结果复算通过；有限模拟EXPLORE，不是实机效果。

补充小目标暴露新的局部弱点：所有接触cube截止前曾可见，但4cm目标及时仅7/144，两角度均如此。几何可见不保证信号充分，不能直接归因模型。下一优先复用本轮光子检查小目标可判别信号，再决定观测侧或读出侧方案；不继续扫角度，不自动训练。方向损失瓶颈未解决，Nymeria因果位移接口与设备验证暂缓。

## 停止与保留

[方向扰动续训](../research/active/dtr-r0/nearfield/CNH_QUERY_PERTURB_TRAIN_DEV_20261008.md)未通过：47接触、同56/2240报警格，E1单Aug/Control/M3为35/40/42，双40/44/45；exact各配对损7，超过2件护栏。本配方停止，不加seed/epoch、不换扰动分布；实质不同机制另行审查。本批方向失配导致的损失各仅1件，不足判断补偿能力。

方位标签微调及稀疏空间/关联L3暂停；[空间诊断](../research/active/dtr-r0/nearfield/CNH_BEARING_SUPPORT_CROSS_DEV_20261008.md)、原L2净增益及逐事件损失、去平滑负结果、步态/EMA整套输入对照保留，不是用户收益证明。盲杖互补、UNKNOWN不保证安全，三态/query覆盖仍待验证。

[480新模拟unit确认](../research/active/dtr-r0/nearfield/CNH_REAL_HEAD_CONFIRM_RESULTS_20261007.md)保留原身份，E1较exact少39/51件及时（/1002），EMA收回25/34；不与后续已消费Development及本轮有限网格混同。HEADS-UP来源窄，EMA事后选择；BlindWays已用于开发回放，不能重称独立确认。

[Nymeria样本与IMU](../research/active/dtr-r0/nearfield/CNH_NYMERIA_SAMPLE_AUDIT_DEV_20261008.md)：时钟异常按原段保留，注册只用所选合法段整体拟合、仅作离线参考；右首60秒原始IMU→CPF姿态前缀检查通过，姿态精度未评价，位置/PDR未实现，E1整链NOT_EVALUABLE。未来骨盆/closed-loop不进估计器，不自动重开配方。

恢复设备先核对M3真实输入与逐query真值；101/101和53ms属A+LOCAL，M3/CNH实机效果尚未建立。City、保护test、新UE及硬件第二阶段暂停。

[路线当前页](../research/active/dtr-r0/CURRENT.md) · [设备状态](PROJECT_STATE.md) · [RUNS](../research/active/dtr-r0/RUNS.md)。本次压缩前全文：Git `fb8641cb` 同路径。
