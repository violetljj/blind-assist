ALGORITHM_EXPLORATION / M3_RETAINED / HARDWARE_DEFERRED

# 前视障碍感知：当前状态

更新：2026-10-07。用户选择**方向估计优先，读出线保留**，持续目标“持续进行算法探索”。冻结M3不替换。
Status: `DTR_R2_DYNAMIC_RETAINED`（历史保留，不表示恢复动态研究）。

## 最新方向证据

[朝向输入敏感性](nearfield/CNH_TORSO_INPUT_SENSITIVITY_DEV_20261007.md)完成：复用原96评价unit、229事件/384clear控制，只扰动估计器躯干yaw，完整60Hz clip状态，sensor/光子/位置/几何/截止与M3冻结。恒定+5°/−5°、2–4s峰值+10°/−10°脉冲，同全13实际FA124/4992，gait−raw单/双净及时依次+26/+25、+16/+12、+3/+4、+4/+6；零扰动恢复原+8/+5。

gait绝对及时零202/197，四挑战168/170、184/186、181/181、185/188；相对收益保持不代表能力恢复。E1未扰动，不能据其197/198不变称真实头yaw鲁棒。原阈值成本差另列、不作可减性分解；未更新4/4/4/5/4事件保留，同query/score同阈值报警翻转0。

保留受限候选，不继续本批择参。下一优先适配当前误差query侧向覆盖/三态，不能继承旧query不变门；真实sensor头yaw与有1–2秒截止机会的冷启动仍未测。局部/报警无改善先查机制，不自动推翻方向优先。[前轮同轨迹](nearfield/CNH_TORSO_NATIVE_DEV_20261007.md)与[原近似搬运](nearfield/CNH_TORSO_BIAS_DEV_20261007.md)保留。

已消费Development、人工压力假设、future-conditioned场景、torso-aligned sensor/理想骨盆原点仍在，非实测头动/眼镜/覆盖或三态验证。源窗口/clip/Pxx重叠、同源模型与映射未知，无新unitCI/人群结论；固定短窗FA非会话成本，评价选点非部署校准。4测试及96unit/40threshold/20工作点完整核验通过；独立阶段墙钟prepare5.219/120、run344.969/600、analysis含失败与复核7.534/120秒，GPU已结束。

## 保留的其他证据

[读出账本/区间](nearfield/CNH_EVENT_LEDGER_UNCERTAINTY_DEV_20261007.md)：280事件/728控制，原高度排序有条件信号，HB策略净及时区间均跨零，启动误报和5%损失保留；M3不替换，追加检查按后续证据，探针/集成未定为必须执行。

[真实头动确认](nearfield/CNH_REAL_HEAD_CONFIRM_RESULTS_20261007.md)：480新模拟unit，NAT+E1静默68/1002，不能全部归因于方向误差；并集判畅通降低静默并增加UNKNOWN负担。不是实机/新人群确认，本轮读出和方向Development限制不覆盖其原确认身份。

[章节](thesis/CNH_READOUT_COVERAGE_TRISTATE_CHAPTER_20261006.md)与[主张台账](THESIS_CLAIMS_20260927.md)保留。UNKNOWN不保证安全；City、保护test、新UE及硬件第二阶段暂停。旧停止范围不扩大，不恢复已停止路径延长。

[RUNS](RUNS.md) · [总决定](../../../docs/CURRENT_DECISION.md)。此前全文：Git `5668620a` 同路径。
