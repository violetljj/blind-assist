# 当前研究决定

更新：2026-10-08。唯一研究主线为盲杖互补的前视障碍感知。用户选择**方向估计优先，读出线保留**；冻结M3、手机A/A+LOCAL与已有论文证据保留。

历史路线状态：`L10_R0_PAUSED / DTR_R2_DYNAMIC_RETAINED`。

## 物理头yaw模拟重渲染已完成

[物理yaw配对重渲染](../research/active/dtr-r0/nearfield/CNH_TORSO_HEAD_YAW_DEV_20261007.md)完成原96评价unit、229事件/384严格clear控制：常量±15°、native2–4s峰值±20°脉冲；新sensor/光子/noisy/全部query及M3分数重算，gait、位置、几何/截止冻结。每条件用新E1匹配实际全13误报124/4992，残差0；gait−raw单/双净及时依次+1/+8、+6/+7、+1/+1、+6/+5，zero原+8/+5。

gait及时零202/197，四挑战148/205、169/199、195/197、201/198；单路常量yaw绝对损失明显。gait−新E1四挑战0/+3、−2/+3、0/−2、+4/+1，未全面胜E1。脉冲实际受扰194/229、零δ35件全部保留；4件未更新、同score同θ报警翻转0。启动/主窗/原阈值成本独立列出，不能差分归因。

决定：保留受限候选，M3冻结；下一补1–2秒有实际截止机会的冷启动fixture。旧观测上7项候选query稀疏覆盖接口检查通过，但helper未接入pipeline，不代表完整三态、新yaw覆盖或CLEAR资格。局部转弯/报警未改善先查机制，不自动推翻方向优先；机制核对后重复损失再降候选优先级。

8物理测试、首unit零重渲染及192脉冲前缀PASS；全96unit/80阈值/20工作点/262损失独立重算通过。两组缓弯净0各1救回/1损失，检查两损失一件当前更新、另一件最后更新0.7秒前；共原躯干θ均未及时，不据此归因吸收转弯。prepare1.219/120s、run1445.360/2400s、分析含局部机制56.903/180s。人工物理yaw、future-conditioned模拟、理想位置/torso proxy及同源/重叠/Pxx映射未知边界保留，无CI或实测头动/人群/设备结论；评价选点非部署校准，固定短窗FA非会话成本。CPU/GPU已结束。[前轮仅输入压力](../research/active/dtr-r0/nearfield/CNH_TORSO_INPUT_SENSITIVITY_DEV_20261007.md)、[同轨迹结果](../research/active/dtr-r0/nearfield/CNH_TORSO_NATIVE_DEV_20261007.md)与[RUNS](../research/active/dtr-r0/RUNS.md)保留。

## 保留的读出与确认依据

[读出账本与区间](../research/active/dtr-r0/nearfield/CNH_EVENT_LEDGER_UNCERTAINTY_DEV_20261007.md)保留全部280事件/728控制：原高度排序有信号，HB校准策略及时区间均跨零，不足以确认改善或证明非劣；启动误报与5%反例保留。读出线是否追加检查按后续证据安排，探针/集成未形成必须执行方案。

[真实头动确认](../research/active/dtr-r0/nearfield/CNH_REAL_HEAD_CONFIRM_RESULTS_20261007.md)为480新模拟unit：NAT+E1静默68/1002，不能全部归因于方向误差，也不能跨不同事件集合排名；方向误差已有明确损失证据，用户已决定优先研究。它不是实机或新人群确认；本轮Development限制不覆盖其原确认身份。

UNKNOWN不保证安全，校准目标不等于实际误报。City、保护test、新UE采集及硬件第二阶段暂停，设备回放待会话。[路线当前页](../research/active/dtr-r0/CURRENT.md) · [设备状态](PROJECT_STATE.md)。此前全文：Git `5668620a` 同路径。
