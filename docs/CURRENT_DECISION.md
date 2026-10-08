# 当前研究决定

更新：2026-10-08。主线是盲杖互补的前视障碍感知，方向估计优先，冻结M3和手机A+LOCAL保留。

历史状态`L10_R0_PAUSED / DTR_R2_DYNAMIC_RETAINED`保留，不表示恢复动态研究。

## 方向扰动训练筛查未通过，停止配方

用户修订圆桌结论授权[单种子小筛查](../research/active/dtr-r0/nearfield/CNH_QUERY_PERTURB_TRAIN_DEV_20261008.md)。Aug/Control从同M3seed0权重续训，同27456原训练行、2epoch，标签固定在实际travel走廊；只扰动当前query，零值原输入逐值复现。冻结五种子M3与seed0另列，首轮不归因机制。

测速后、训练及新结果前冻结8cal/24eval，评价47接触/224clear。同实际56/2240报警格、16格残差0：E1单路Aug35、Control40、冻结M3为42；双路40/44/45。exact Aug对M3救回/损失单0/7、双1/7，超过损失≤2/net≥−2护栏；两配置三条件均未满足，停止这个配方，不追加seed/epoch或择扰动。GPU3087.703/3600秒，独立审计17.188秒PASS，科研CPU核验<90/300秒，计算结束。工作点为Development评价选点，非部署校准；一件为2.128pp，不宣称统计优势/非劣或否定全部方向鲁棒训练。

方位标签微调及稀疏空间支持/关联L3暂停，不作下一优先。设备恢复先核对真实输入处理和逐query标定真值；101/101样本及53ms属于A+LOCAL，不是M3实机效果。BlindWays已经用于开发回放，不能重称独立确认。

逐事件补核：本批冻结M3五种子exact及时/E1不及时单双路各1件、反向0，都是99017/37；其他E1未及时单4/双1在exact下也未及时。方向失配损失余量过小，不足以判断方向补偿能力；各query独立选同成本阈值，非固定阈值方向因果对照。本配方维持停止，实质不同机制需另行审查。

## 既有结果保留

[空间支持交叉诊断](../research/active/dtr-r0/nearfield/CNH_BEARING_SUPPORT_CROSS_DEV_20261008.md)全96unit字段完整、229接触/384clear，NOT_EVALUABLE0；单簇改选四挑战净差单+1/−1/0/0、双全0。集合扩张及双分歧即弃权损失大量正确支持；关联机会过线但竞争物体存在/整体更近在正确事件也全部成立，不证明可辨识性或关联收益。覆盖UNKNOWN不否决，粗簇非深度/实例，不改接触指标。

原[EMA门控L2](../research/active/dtr-r0/nearfield/CNH_EMA_GATED_BEARING_DEV_20261008.md)支持净增益及逐事件损失保留；[k1去平滑](../research/active/dtr-r0/nearfield/CNH_EMA_BEARING_DESMOOTH_DEV_20261008.md)未稳定改善，保留原L2，不追加平滑择参。[步态对EMA](../research/active/dtr-r0/nearfield/CNH_TORSO_EMA_COMPARE_DEV_20261008.md)单路降低优先级、双路混合保留，输入契约不等，只比较整套方案。均已消费Development，非方向定位、用户收益或实机证据。

## 证据与后续边界

[480新模拟unit确认](../research/active/dtr-r0/nearfield/CNH_REAL_HEAD_CONFIRM_RESULTS_20261007.md)原身份保留：E1相对exact少39/51件及时（/1002），EMA收回25/34；同校准目标，实际误报不等。头动约6分钟、来源窄，EMA事后选择，非实机/新人群确认。读出HB净及时区间跨零，覆盖/UNKNOWN/三态仍待验证，UNKNOWN不保证安全。

[Nymeria原版样本](../research/active/dtr-r0/nearfield/CNH_NYMERIA_SAMPLE_AUDIT_DEV_20261008.md)：149170骨架帧有7异常间隔/2倒序，按段保留；注册只用最大合法连续段内整体拟合，78878帧结果为离线参考。原始右/左IMU标称1000/800Hz，全流时间无倒序；右首60秒60129条全有效，厂参解析→初始化→gyro姿态→CPF水平前向七处/35项前缀比较通过，七类fixtures通过，科研CPU<100/180秒，无新下载/训练。姿态精度未评价，位置/PDR未实现，**E1整链NOT_EVALUABLE**；实际包缺open-loop，未来骨盆/closed-loop不进入估计器。下一先补因果位移接口与成本，再定行走参与者抽样；完整raw-head组清单约10.316GiB未下载。单样本不代表覆盖，三份头动数据统一标明上游因果限制，不自动重跑配方。行进意图仍开放；设备、City、保护test、新UE及硬件第二阶段暂停。

[路线当前页](../research/active/dtr-r0/CURRENT.md) · [设备状态](PROJECT_STATE.md) · [RUNS](../research/active/dtr-r0/RUNS.md)。此前全文：Git `401467ce` 同路径。
