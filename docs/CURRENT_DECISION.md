# 当前研究决定

更新：2026-10-08。唯一研究主线为盲杖互补的前视障碍感知。方向估计优先，读出线保留不上调；M3、手机A/A+LOCAL和既有论文证据保留不替换。

历史路线状态：`L10_R0_PAUSED / DTR_R2_DYNAMIC_RETAINED`。

## 最新：空间支持交叉诊断完成，保留L2；关联先查可辨识性

[空间支持×错误解剖](../research/active/dtr-r0/nearfield/CNH_BEARING_SUPPORT_CROSS_DEV_20261008.md)完成全96unit、229接触/384clear；hist/ambient/boxes完整，NOT_EVALUABLE0。先冻结观测及绝对救回/损失/弃权分流规则，再保存21019个全窗首提醒观测并接truth。唯一正簇改选四挑战clean/unique净差单+1/−1/0/0、双全0；直接集合扩张损失大量正确支持，双分歧即弃权去错10/4/22/21却损正确114/134/98/110。当前直接候选均未达投入条件，原L2保留。

原冻结关联机会单20/15/22/23、双21/6/26/23过线，结果保留；复核及本地核对发现竞争物体存在/整体XZ更近在正确事件中也全部成立，条件未证明可辨识性或关联收益，不能据过线提高方法优先级。关联保留为下一诊断问题：先比较正确/错误的簇型与支持并集，再拆query×支路×body/head赢家；形成完整候选后冻结适用门槛、评价最终输出，不自动继承本轮SWITCH门槛。不因竞争物体改原指标。边界×同窗yaw仅线索；覆盖UNKNOWN不能否决，名义证书成立后仍误弃正确方向。粗簇非深度/实例，当前失败不否定方法类别。

CPU字段核验12.906秒、分析102.516/900秒，8针对性测试及21019候选/2290事件独立汇总通过（0.359秒），无新推理，计算结束。该Development诊断未实现完整在线L3，离线成本非手机延迟，手机仍暂停。[k1去平滑负结果](../research/active/dtr-r0/nearfield/CNH_EMA_BEARING_DESMOOTH_DEV_20261008.md)保留，不追加本批k/阈值择参。

[原EMA门控方位检查](../research/active/dtr-r0/nearfield/CNH_EMA_GATED_BEARING_DEV_20261008.md)的L2−L1四挑战支持单+22/+39/+32/+33、双+21/+30/+29/+25及局部损失保留。原[三查询独立触发](../research/active/dtr-r0/nearfield/CNH_SECTOR_NOTICE_DEV_20261008.md)失败不混同补充标签；本次k1失败亦不否定所有时间/空间机制。均为已消费Development，支持范围非定位/物体归因/听觉收益，M3与手机不替换。

## 步态对EMA：按配置收敛

[推理级对照](../research/active/dtr-r0/nearfield/CNH_TORSO_EMA_COMPARE_DEV_20261008.md)完成96unit、229事件/384clear，各条件实际FA124/4992且残差0。四挑战gait−EMA单路净−2/−2/−8/−1，降步态候选优先级；双路+5/+1/−5/0，混合保留。均不自动追加冷启动/归附录，不八格求和，零界线不证明统计优势/非劣。

E1/躯干/gait理想60Hz fullclip与EMA带噪5Hz窗口重置输入不等；EMA保留相对旋转、适配理想骨盆原点，非确认零平移Q，只比较整套方案。人工yaw、future-conditioned模拟、源重叠/Pxx身份未知及评价选点边界保留。成本、配对和5测试/独立核验在报告，计算结束；[原物理yaw](../research/active/dtr-r0/nearfield/CNH_TORSO_HEAD_YAW_DEV_20261007.md)不改。

## 写作与证据身份

[480新模拟unit确认](../research/active/dtr-r0/nearfield/CNH_REAL_HEAD_CONFIRM_RESULTS_20261007.md)保留原确认身份，不与后续Development混同。E1比exact少39/51件及时（/1002），EMA收回25/34、约三分之二；双路及时优势未建立，比较用同校准目标、实际误报不等。并集判畅通把静默68→19/21 of1002，UNKNOWN增加15.64/14.23pp，及时不变。方向误差不能解释全部静默。

头动源约6分钟、可能同一人，EMA事后选择；非实机/新人群确认。[章节](../research/active/dtr-r0/thesis/CNH_READOUT_COVERAGE_TRISTATE_CHAPTER_20261006.md)已同步。不再开同类确认批。

[读出区间](../research/active/dtr-r0/nearfield/CNH_EVENT_LEDGER_UNCERTAINTY_DEV_20261007.md)未确认HB策略改善/非劣，启动及5%反例保留。读出、覆盖、三态仅阶段性结果；候选query完整覆盖/UNKNOWN/三态仍待验证，稀疏helper未接pipeline、不授权CLEAR。

行进意图估计仍开放；带头部朝向的真实行走数据到达后，先走已有replay。方向准确且覆盖充分的事件若仍集中漏报，再提升读出优先级。设备验证继续暂缓，City、保护test、新UE及硬件第二阶段暂停。UNKNOWN不保证安全。

[路线当前页](../research/active/dtr-r0/CURRENT.md) · [设备状态](PROJECT_STATE.md) · [RUNS](../research/active/dtr-r0/RUNS.md)。此前全文：Git `6ebb8338` 同路径。
