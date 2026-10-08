# 当前研究决定

更新：2026-10-08。唯一研究主线为盲杖互补的前视障碍感知。方向估计优先，读出线保留不上调；M3、手机A/A+LOCAL和既有论文证据保留不替换。

历史路线状态：`L10_R0_PAUSED / DTR_R2_DYNAMIC_RETAINED`。

## 最新：EMA固定提醒＋补充方位有缓存增益

[固定EMA方位缓存检查](../research/active/dtr-r0/nearfield/CNH_EMA_GATED_BEARING_DEV_20261008.md)完成96unit、229接触/384clear。保持EMA全部提示逐值不变，只比较首次主窗及时标签；L1用−ema_rel，L2用既有三查询分数argmax。四挑战L2−L1首条支持单路+22/+39/+32/+33、双路+21/+30/+29/+25；各格仍有配对损失，非统计非劣。L1剩余clean oracle余量单42/54/54/56、双43/36/55/48，不能当可实现收益。

保留“EMA触发＋查询分数补方位”为当前最强方位候选，不替换M3或手机。CPU0.922/180秒，4契约测试及96unit/30格/6870行独立重算通过。L2本次复用缓存，不表示未来额外推理免费。空间L3未实施：CNH非深度图，噪声支持/簇关联待定义；单exposure样本约0.078秒，全部窗口每格1741–2249首次需计成本。首条支持不等于唯一定位、物体归因或听觉收益；仍为已消费Development。

[原固定三查询独立触发首检](../research/active/dtr-r0/nearfield/CNH_SECTOR_NOTICE_DEV_20261008.md)负结果保留，本次不进入其去重/NAT或择角阈值；它不否定EMA门控后的补充标签。本轮计算结束，载荷与成本在报告。

## 步态对EMA：按配置收敛

[推理级对照](../research/active/dtr-r0/nearfield/CNH_TORSO_EMA_COMPARE_DEV_20261008.md)完成96unit、229事件/384clear，各条件实际FA124/4992且残差0。四挑战gait−EMA单路净−2/−2/−8/−1，降步态候选优先级；双路+5/+1/−5/0，混合保留。均不自动追加冷启动/归附录，不八格求和，零界线不证明统计优势/非劣。

E1/躯干/gait理想60Hz fullclip与EMA带噪5Hz窗口重置输入不等；EMA保留相对旋转、适配理想骨盆原点，非确认零平移Q，只比较整套方案。人工yaw、future-conditioned模拟、源重叠/Pxx身份未知及评价选点边界保留。成本、配对和5测试/独立核验在报告，计算结束；[原物理yaw](../research/active/dtr-r0/nearfield/CNH_TORSO_HEAD_YAW_DEV_20261007.md)不改。

## 写作与证据身份

[480新模拟unit确认](../research/active/dtr-r0/nearfield/CNH_REAL_HEAD_CONFIRM_RESULTS_20261007.md)保留原确认身份，不与后续Development混同。E1比exact少39/51件及时（/1002），EMA收回25/34、约三分之二；双路及时优势未建立，比较用同校准目标、实际误报不等。并集判畅通把静默68→19/21 of1002，UNKNOWN增加15.64/14.23pp，及时不变。方向误差不能解释全部静默。

头动源约6分钟、可能同一人，EMA事后选择；非实机/新人群确认。[章节](../research/active/dtr-r0/thesis/CNH_READOUT_COVERAGE_TRISTATE_CHAPTER_20261006.md)已同步。不再开同类确认批。

[读出区间](../research/active/dtr-r0/nearfield/CNH_EVENT_LEDGER_UNCERTAINTY_DEV_20261007.md)未确认HB策略改善/非劣，启动及5%反例保留。读出、覆盖、三态仅阶段性结果；候选query完整覆盖/UNKNOWN/三态仍待验证，稀疏helper未接pipeline、不授权CLEAR。

行进意图估计仍开放；带头部朝向的真实行走数据到达后，先走已有replay。方向准确且覆盖充分的事件若仍集中漏报，再提升读出优先级。设备验证继续暂缓，City、保护test、新UE及硬件第二阶段暂停。UNKNOWN不保证安全。

[路线当前页](../research/active/dtr-r0/CURRENT.md) · [设备状态](PROJECT_STATE.md) · [RUNS](../research/active/dtr-r0/RUNS.md)。此前全文：Git `6ebb8338` 同路径。
