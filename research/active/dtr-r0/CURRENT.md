ALGORITHM_EXPLORATION / M3_RETAINED / HARDWARE_DEFERRED

# 前视障碍感知：当前状态

更新：2026-10-08。本轮用户优先形状能力图完成，冻结M3不替换。
Status: `DTR_R2_DYNAMIC_RETAINED`（历史保留，不表示恢复动态研究）。

## 最新决定

最新用户优先对象为横杆/竖杆/柜体突出物，标牌补充；继续限定对齐直行、理想位姿、不实测。[形状报告](nearfield/CNH_ALIGNED_SHAPES_DEV_20261008.md)完成492 AABB代理×K4：316物理接触、344高度query接触（28长柱跨两高度）、88pass、88clear。固定−10°/M3/原θ，未训练，全部可评价。

HEAD/BODY及时：横杆119/224、85/224；竖杆217/224、177/224；柜体107/128、94/128；标牌74/112、50/112。任一高度物理及时849/1264；clear46/4576格、25段、24/352clip，pass74/352clip另列。相同原阈值不等于各组误报成本相同，时间格不是真实提醒次数。GPU墙钟187.188/600s，9代表CPU渲染paritymax2.84e−14、独立492分类/120分组/3936账本/48配对重算通过，资源释放。

暗4cm横杆30/90cm长：HEAD4/28、4/28，BODY0/28、1/28；全部316接触几何曾可见。下一先查这些横杆的有效信号与角距分布，选择观测或读出候选；不预设浅边是全部类型瓶颈。柜体大面配对HEAD55→52/64、BODY47→47/64，noise不同，仅描述；小立方体的特权信号分离度作补充，不能统领路线或证明可学。原始载荷、CSV/图与冻结PLAN位于`artifacts.local/work/cnh-aligned-shapes-dev-20261008/`。

此前[立方体/俯角对照](nearfield/CNH_ALIGNED_BOUNDARY_DEV_20261008.md)同成本HEAD31→29、BODY31→52/144，损失超护栏；−19°仍不推进，原−10°保留。不再扫角度，不自动训练或实测；方向/Nymeria位移接口本轮暂缓。

## 其余路线保留

[扰动训练](nearfield/CNH_QUERY_PERTURB_TRAIN_DEV_20261008.md)停止：同47事件、56/2240报警格，单Aug/Control/M3=35/40/42，双40/44/45，exact各损7，三条件失败；不加seed/epoch或换分布。本批仅各1件方向失配损失，不足判断补偿能力。方位微调和稀疏空间/关联L3暂停；原L2净增益与配对损失、去平滑负结果、[空间诊断](nearfield/CNH_BEARING_SUPPORT_CROSS_DEV_20261008.md)及步态/EMA整套输入对照保留。

[480新模拟确认](nearfield/CNH_REAL_HEAD_CONFIRM_RESULTS_20261007.md)身份不改：E1较exact少39/51及时（/1002），EMA收回25/34，同目标下实际误报不等；来源窄、EMA事后选，不是新人群/实机证据。后续回放与筛查属于已消费Development；BlindWays不能再称独立确认。读出HB净及时区间跨零，UNKNOWN/三态/query覆盖待验证。

[Nymeria样本/IMU](nearfield/CNH_NYMERIA_SAMPLE_AUDIT_DEV_20261008.md)合法段整体注册为离线参考，原时钟异常保留；原始右首60秒→CPF姿态七处/35项前缀及fixtures通过，姿态准确度未评，位置/PDR未实现、E1整链NOT_EVALUABLE。未来骨盆和闭环轨迹只作参考；因果位移接口/参与者覆盖工作本轮暂缓，不自动重跑配方。

设备恢复先核对M3真实处理与逐query真值；手机101/101及53ms是A+LOCAL，M3/CNH实机效果未建立。设备、City、保护test、新UE及硬件第二阶段暂停。

[RUNS](RUNS.md) · [总决定](../../../docs/CURRENT_DECISION.md)。压缩前全文：Git `fb8641cb` 同路径。
