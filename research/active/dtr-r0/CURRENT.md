ALGORITHM_EXPLORATION / M3_RETAINED / HARDWARE_DEFERRED

# 前视障碍感知：当前状态

更新：2026-10-09。局部角距读出救回部分细杆，但全批损失较多，M3不替换。
Status: `DTR_R2_DYNAMIC_RETAINED`（历史保留，不表示恢复动态研究）。

## 最新决定

最新用户优先对象为横杆/竖杆/柜体突出物，标牌补充；继续限定对齐直行、理想位姿、不实测。[形状报告](nearfield/CNH_ALIGNED_SHAPES_DEV_20261008.md)完成492 AABB代理×K4：316物理接触、344高度query接触（28长柱跨两高度）、88pass、88clear。固定−10°/M3/原θ，未训练，全部可评价。

HEAD/BODY及时：横杆119/224、85/224；竖杆217/224、177/224；柜体107/128、94/128；标牌74/112、50/112。任一高度物理及时849/1264；clear46/4576格、25段、24/352clip，pass74/352clip另列。相同原阈值不等于各组误报成本相同，时间格不是真实提醒次数。GPU墙钟187.188/600s，9代表CPU渲染paritymax2.84e−14、独立492分类/120分组/3936账本/48配对重算通过，资源释放。

[横杆诊断](nearfield/CNH_BAR_CACHED_DIAGNOSTIC_DEV_20261008.md)：暗4cm HEAD/BODY原8/1、同θ原始22/8、同clear格原始15/1（各56），7次BODY救回未保留。全批同46/4576格HEAD救36损1、BODY救13损10，不推广去平滑。172 BODY接触在f13/.97m均仍有对应高度首命中，88加长配对均增目标贡献。径向对齐提升其条件分离度，但丢角度后远低于原角距参考，不是及时收益；旧证据保留。

[同窗投影/FP16](nearfield/CNH_BAR_REPRESENTATION_DEV_20261008.md)：28 A/24 B几何×f12/f13=104窗，B仍仅6/9/17cm间隔。联合条件d²下界全≥96.1847%、界宽≤3.9904e−8；迭代上限不称收敛。A远负差几乎未入取样范围。两处均值舍入误差中位0.0226%–0.0789%、max0.1396%；144原K4归一化逐值一致，A原hist缺失不回填，旧证据保留。

[冻结M3读出](nearfield/CNH_BAR_READOUT_DEV_20261008.md)：37端点×64新条件噪声，f12/f13及α=.25/1共9472ensemble输入；对应高度方差失配68/76、同draw66/76，d_J只作敏感性。BODY d_MC f12/f13中位A .180/.235、B .199/.107，不是及时或信息损失率；原logit/梯度核验保留。

[局部条带候选](nearfield/CNH_BAR_LOCAL_READOUT_DEV_20261009.md)完整492×K4，同46/4576 clear格：暗4cm BODY1→9/56（救8损0），HEAD8→16/56（救11损3）；全批HEAD517→530/688救53损40，BODY406→423/688救72损55，竖杆退步。clear段25→28、clip24→28/352，pass74→29/352。37×64保存draw完整窗BODY70→158/896救132损44，固定θ的clear均9/832（未再校准）。运行50.219/900s、原Projection/MC逐值parity和账本核验完成，无训练。保留细杆专项候选，不替换M3；下一局部证据补充方案须约束原检出损失并同clear成本检验，NOT_RUN。远负差取样仍开放，不自动扫patch/改俯角布局。载荷`artifacts.local/work/cnh-bar-local-readout-dev-20261009/`。

此前[立方体/俯角对照](nearfield/CNH_ALIGNED_BOUNDARY_DEV_20261008.md)同成本HEAD31→29、BODY31→52/144，损失超护栏；−19°仍不推进，原−10°保留。不再扫角度，不自动训练或实测；方向/Nymeria位移接口本轮暂缓。

## 其余路线保留

[扰动训练](nearfield/CNH_QUERY_PERTURB_TRAIN_DEV_20261008.md)停止：同47事件、56/2240报警格，单Aug/Control/M3=35/40/42，双40/44/45，exact各损7，三条件失败；不加seed/epoch或换分布。本批仅各1件方向失配损失，不足判断补偿能力。方位微调和稀疏空间/关联L3暂停；原L2净增益与配对损失、去平滑负结果、[空间诊断](nearfield/CNH_BEARING_SUPPORT_CROSS_DEV_20261008.md)及步态/EMA整套输入对照保留。

[480新模拟确认](nearfield/CNH_REAL_HEAD_CONFIRM_RESULTS_20261007.md)身份不改：E1较exact少39/51及时（/1002），EMA收回25/34，同目标下实际误报不等；来源窄、EMA事后选，不是新人群/实机证据。后续回放与筛查属于已消费Development；BlindWays不能再称独立确认。读出HB净及时区间跨零，UNKNOWN/三态/query覆盖待验证。

[Nymeria样本/IMU](nearfield/CNH_NYMERIA_SAMPLE_AUDIT_DEV_20261008.md)合法段整体注册为离线参考，原时钟异常保留；原始右首60秒→CPF姿态七处/35项前缀及fixtures通过，姿态准确度未评，位置/PDR未实现、E1整链NOT_EVALUABLE。未来骨盆和闭环轨迹只作参考；因果位移接口/参与者覆盖工作本轮暂缓，不自动重跑配方。

设备恢复先核对M3真实处理与逐query真值；手机101/101及53ms是A+LOCAL，M3/CNH实机效果未建立。设备、City、保护test、新UE及硬件第二阶段暂停。

[RUNS](RUNS.md) · [总决定](../../../docs/CURRENT_DECISION.md)。更新前全文：Git `9e7682b0` 同路径。
