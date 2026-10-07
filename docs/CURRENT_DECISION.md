# 当前研究决定

更新：2026-10-07。唯一研究主线为盲杖互补的前视障碍感知。用户选择**方向估计优先，读出线保留**；冻结M3、手机A/A+LOCAL与已有论文证据保留。

历史路线状态：`L10_R0_PAUSED / DTR_R2_DYNAMIC_RETAINED`。

## 朝向输入敏感性已完成

[输入压力重放](../research/active/dtr-r0/nearfield/CNH_TORSO_INPUT_SENSITIVITY_DEV_20261007.md)复用同轨迹96评价unit、229事件/384严格clear控制，固定gait/M3；只改躯干yaw输入，sensor/观测/骨盆/几何/截止不改，完整60Hz clip状态。四组固定挑战为恒定±5°及2–4s峰值±10°脉冲，同全13实际误报124/4992时，gait−raw单/双净及时依次+26/+25、+16/+12、+3/+4、+4/+6；零扰动恢复+8/+5。

收益在这些人工输入误差下保留，但gait绝对及时从零202/197降为168/170、184/186、181/181、185/188，不能把相对净差扩大称为原能力恢复。E1保持197/198且输入未扰动，不能据此宣称真实头yaw鲁棒或自动转选E1。原阈值成本另列；未更新4/4/4/5/4事件保留，同query/score且同阈值报警翻转0。

决定：保留受限候选，M3冻结；下一优先适配误差query下侧向覆盖/三态，真实sensor头yaw及1–2秒有截止机会的冷启动仍未测。旧query不变门不能直接继承。证据仍为已消费、future-conditioned模拟、torso-aligned sensor及理想骨盆原点，同源/重叠/Pxx映射未知，无CI、人群或设备主张；全13短窗非任意会话成本，评价误报匹配非部署校准。

4测试、96unit/40阈值/20工作点完整独立核验通过；独立阶段墙钟准备5.219/120s、推理含setup/零核对/保存344.969/600s、分析含失败审查及复核7.534/120s，进程已结束。[前轮同轨迹结果](../research/active/dtr-r0/nearfield/CNH_TORSO_NATIVE_DEV_20261007.md)与[原近似回放](../research/active/dtr-r0/nearfield/CNH_TORSO_BIAS_DEV_20261007.md)保留，不跨事件集相减归因。[RUNS](../research/active/dtr-r0/RUNS.md)。

## 保留的读出与确认依据

[读出账本与区间](../research/active/dtr-r0/nearfield/CNH_EVENT_LEDGER_UNCERTAINTY_DEV_20261007.md)保留全部280事件/728控制：原高度排序有信号，HB校准策略及时区间均跨零，不足以确认改善或证明非劣；启动误报与5%反例保留。读出线是否追加检查按后续证据安排，探针/集成未形成必须执行方案。

[真实头动确认](../research/active/dtr-r0/nearfield/CNH_REAL_HEAD_CONFIRM_RESULTS_20261007.md)为480新模拟unit：NAT+E1静默68/1002，不能全部归因于方向误差，也不能跨不同事件集合排名；方向误差已有明确损失证据，用户已决定优先研究。它不是实机或新人群确认；本轮Development限制不覆盖其原确认身份。

UNKNOWN不保证安全，校准目标不等于实际误报。City、保护test、新UE采集及硬件第二阶段暂停，设备回放待会话。[路线当前页](../research/active/dtr-r0/CURRENT.md) · [设备状态](PROJECT_STATE.md)。此前全文：Git `5668620a` 同路径。
