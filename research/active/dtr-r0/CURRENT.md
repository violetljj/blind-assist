ALGORITHM_EXPLORATION / M3_RETAINED / HARDWARE_DEFERRED

# 前视障碍感知：当前状态

更新：2026-10-08。形状能力图及横杆缓存诊断完成，冻结M3不替换。
Status: `DTR_R2_DYNAMIC_RETAINED`（历史保留，不表示恢复动态研究）。

## 最新决定

最新用户优先对象为横杆/竖杆/柜体突出物，标牌补充；继续限定对齐直行、理想位姿、不实测。[形状报告](nearfield/CNH_ALIGNED_SHAPES_DEV_20261008.md)完成492 AABB代理×K4：316物理接触、344高度query接触（28长柱跨两高度）、88pass、88clear。固定−10°/M3/原θ，未训练，全部可评价。

HEAD/BODY及时：横杆119/224、85/224；竖杆217/224、177/224；柜体107/128、94/128；标牌74/112、50/112。任一高度物理及时849/1264；clear46/4576格、25段、24/352clip，pass74/352clip另列。相同原阈值不等于各组误报成本相同，时间格不是真实提醒次数。GPU墙钟187.188/600s，9代表CPU渲染paritymax2.84e−14、独立492分类/120分组/3936账本/48配对重算通过，资源释放。

[横杆诊断](nearfield/CNH_BAR_CACHED_DIAGNOSTIC_DEV_20261008.md)：暗4cm HEAD/BODY原8/1、同θ原始22/8、同clear格原始15/1（各56事件），7次BODY救回未保留。全批同46/4576格HEAD救36损1、BODY救13损10；clear段25→34、pass clip74→86，不推广去平滑。172 BODY接触在f13/.97m均有对应高度首命中，f14/f15降为102/52；88加长配对均增目标贡献。A/B完整前缀保留非零信息，B只检验6/9/17cm间隔。理想径向对齐提升条件分离度，但角度求和丢失较多信息，未评价及时收益。下一优先投影/读出的信息保留、负背景和侧别；角距联合累积仍开放，完整无杆投影/M3对照NOT_RUN。CPU缓存诊断无推理/训练，工件`artifacts.local/work/cnh-bar-cached-diagnostic-dev-20261008/`；旧能力图和小立方体补充保留。

此前[立方体/俯角对照](nearfield/CNH_ALIGNED_BOUNDARY_DEV_20261008.md)同成本HEAD31→29、BODY31→52/144，损失超护栏；−19°仍不推进，原−10°保留。不再扫角度，不自动训练或实测；方向/Nymeria位移接口本轮暂缓。

## 其余路线保留

[扰动训练](nearfield/CNH_QUERY_PERTURB_TRAIN_DEV_20261008.md)停止：同47事件、56/2240报警格，单Aug/Control/M3=35/40/42，双40/44/45，exact各损7，三条件失败；不加seed/epoch或换分布。本批仅各1件方向失配损失，不足判断补偿能力。方位微调和稀疏空间/关联L3暂停；原L2净增益与配对损失、去平滑负结果、[空间诊断](nearfield/CNH_BEARING_SUPPORT_CROSS_DEV_20261008.md)及步态/EMA整套输入对照保留。

[480新模拟确认](nearfield/CNH_REAL_HEAD_CONFIRM_RESULTS_20261007.md)身份不改：E1较exact少39/51及时（/1002），EMA收回25/34，同目标下实际误报不等；来源窄、EMA事后选，不是新人群/实机证据。后续回放与筛查属于已消费Development；BlindWays不能再称独立确认。读出HB净及时区间跨零，UNKNOWN/三态/query覆盖待验证。

[Nymeria样本/IMU](nearfield/CNH_NYMERIA_SAMPLE_AUDIT_DEV_20261008.md)合法段整体注册为离线参考，原时钟异常保留；原始右首60秒→CPF姿态七处/35项前缀及fixtures通过，姿态准确度未评，位置/PDR未实现、E1整链NOT_EVALUABLE。未来骨盆和闭环轨迹只作参考；因果位移接口/参与者覆盖工作本轮暂缓，不自动重跑配方。

设备恢复先核对M3真实处理与逐query真值；手机101/101及53ms是A+LOCAL，M3/CNH实机效果未建立。设备、City、保护test、新UE及硬件第二阶段暂停。

[RUNS](RUNS.md) · [总决定](../../../docs/CURRENT_DECISION.md)。更新前全文：Git `5f5554b3` 同路径。
