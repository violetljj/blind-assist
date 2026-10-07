# 当前研究决定

更新：2026-10-08。唯一研究主线为盲杖互补的前视障碍感知。方向估计优先，读出线保留不上调；M3、手机A/A+LOCAL和既有论文证据保留不替换。

历史路线状态：`L10_R0_PAUSED / DTR_R2_DYNAMIC_RETAINED`。

## 步态对EMA：按配置收敛

[推理级对照](../research/active/dtr-r0/nearfield/CNH_TORSO_EMA_COMPARE_DEV_20261008.md)完成96已消费Development评价unit、229事件/384clear，复用四物理yaw挑战和零观测，没有重新渲染或调参。每条件新E1实际FA124/4992，各臂整tie匹配残差0；启动/主窗成本另报。

gait−EMA四挑战的单路救回/损失为8/10、9/11、6/14、11/12，净−2/−2/−8/−1；双路为11/6、13/12、13/18、18/18，净+5/+1/−5/0。依运行前规则，单路步态候选降优先级；双路取舍未决、保留记录、不自动归附录。两种配置均不自动追加冷启动。四组全零也只保留；不八格求和，零界线是投入规则而非统计优势/非劣证明。

输入无法完全对齐：E1理想60Hz头位置；躯干/gait理想60Hz肩线/骨盆代理、fullclip历史；EMA冻结τ0.5/0.25s、带噪5Hz、每窗口重置。EMA旋转保持确认的带噪sensor相对角，平移适配共同理想骨盆原点，区别于确认零平移Q。因此只比较整套方案在现有输入下的表现，不隔离归因。

prepare1.578/120秒、新M3推理215.578/1200秒、分析检查15.284/180秒；5测试和96unit/2400raw/307200query帧/50工作点核验PASS，计算已结束。人工yaw、future-conditioned模拟、源重叠及Pxx映射未知、理想原点、评价工作点非部署校准的边界保留，无CI/实机/新人群结论。[原物理yaw结果](../research/active/dtr-r0/nearfield/CNH_TORSO_HEAD_YAW_DEV_20261007.md)不改；单路202→148只与偏头限制方向一致，成因未验证。

## 写作与证据身份

[480新模拟unit确认](../research/active/dtr-r0/nearfield/CNH_REAL_HEAD_CONFIRM_RESULTS_20261007.md)保留原确认身份，不与后续Development混同。E1比exact少39/51件及时（/1002），EMA收回25/34、约三分之二；双路及时优势未建立，比较用同校准目标、实际误报不等。并集判畅通把静默68→19/21 of1002，UNKNOWN增加15.64/14.23pp，及时不变。方向误差不能解释全部静默。

头动源约6分钟、可能同一人，EMA事后选择；非实机/新人群确认。[章节](../research/active/dtr-r0/thesis/CNH_READOUT_COVERAGE_TRISTATE_CHAPTER_20261006.md)及答辩问答已同步。不再开同类确认批。

[读出区间](../research/active/dtr-r0/nearfield/CNH_EVENT_LEDGER_UNCERTAINTY_DEV_20261007.md)未确认HB策略改善/非劣，启动及5%反例保留。读出、覆盖、三态仅阶段性结果；候选query完整覆盖/UNKNOWN/三态仍待验证，稀疏helper未接pipeline、不授权CLEAR。

行进意图估计仍开放；带头部朝向的真实行走数据到达后，先走已有replay。方向准确且覆盖充分的事件若仍集中漏报，再提升读出优先级。设备验证继续暂缓，City、保护test、新UE及硬件第二阶段暂停。UNKNOWN不保证安全。

[路线当前页](../research/active/dtr-r0/CURRENT.md) · [设备状态](PROJECT_STATE.md) · [RUNS](../research/active/dtr-r0/RUNS.md)。此前全文：Git `6ebb8338` 同路径。
