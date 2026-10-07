ALGORITHM_EXPLORATION / M3_RETAINED / HARDWARE_DEFERRED

# 前视障碍感知：当前状态

更新：2026-10-08。用户选择**方向估计优先，读出线保留**，持续目标“持续进行算法探索”。冻结M3不替换。
Status: `DTR_R2_DYNAMIC_RETAINED`（历史保留，不表示恢复动态研究）。

## 最新方向证据

[物理头yaw重渲染](nearfield/CNH_TORSO_HEAD_YAW_DEV_20261007.md)完成：96评价unit、229事件/384clear，常量±15°、native2–4s峰值±20°；实际模拟sensor/光子/noisy/query/M3分数重算，固定gait、位置、几何/截止。每条件新E1选择全13实际FA124/4992，所有臂残差0；gait−raw单/双净及时+1/+8、+6/+7、+1/+1、+6/+5，零原+8/+5。

gait绝对及时零202/197，四挑战148/205、169/199、195/197、201/198；单路常量yaw仍明显受损。gait−新E1四挑战0/+3、−2/+3、0/−2、+4/+1，未全面胜E1。脉冲实际受扰194/229、零δ35，全部样本保留；4未更新事件同query/score共θ翻转0。启动、主窗和旧阈值成本另列，不差分作贡献。

保留受限候选，不继续本批择参。下一补1–2秒有实际截止机会的冷启动fixture；完整candidate query覆盖/UNKNOWN/三态仍待验证。旧观测7项稀疏覆盖接口PASS，helper未接入pipeline、不授权CLEAR或证明新yaw覆盖。局部转弯/报警无改善先查机制，核对后重复损失再降候选优先，不自动推翻方向优先。[前轮仅输入扰动](nearfield/CNH_TORSO_INPUT_SENSITIVITY_DEV_20261007.md)、[同轨迹](nearfield/CNH_TORSO_NATIVE_DEV_20261007.md)保留。

人工物理yaw模拟、已消费Development、future-conditioned场景、理想位置/torso proxy、重叠窗口/clip/Pxx和同源映射未知仍在，非实测头动/眼镜/人群，无CI。固定短窗FA非会话成本，评价选点非部署校准。8物理测试、首unit零重渲染、192pulse前缀通过；全96unit/80threshold/20工作点/262损失独立核验PASS。两缓弯损失状态/方向/共θ检查保留：共原躯干θ均未及时，不能判定吸收turn。prepare1.219/120、run1445.360/2400、分析含局部机制56.903/180秒；CPU/GPU已结束。

## 保留的其他证据

[读出账本/区间](nearfield/CNH_EVENT_LEDGER_UNCERTAINTY_DEV_20261007.md)：280事件/728控制，原高度排序有条件信号，HB策略净及时区间均跨零，启动误报和5%损失保留；M3不替换，追加检查按后续证据，探针/集成未定为必须执行。

[真实头动确认](nearfield/CNH_REAL_HEAD_CONFIRM_RESULTS_20261007.md)：480新模拟unit，NAT+E1静默68/1002，不能全部归因于方向误差；并集判畅通降低静默并增加UNKNOWN负担。不是实机/新人群确认，本轮读出和方向Development限制不覆盖其原确认身份。

[章节](thesis/CNH_READOUT_COVERAGE_TRISTATE_CHAPTER_20261006.md)与[主张台账](THESIS_CLAIMS_20260927.md)保留。UNKNOWN不保证安全；City、保护test、新UE及硬件第二阶段暂停。旧停止范围不扩大，不恢复已停止路径延长。

[RUNS](RUNS.md) · [总决定](../../../docs/CURRENT_DECISION.md)。此前全文：Git `5668620a` 同路径。
