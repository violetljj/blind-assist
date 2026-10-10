# 前视障碍感知：当前状态

更新：2026-10-10。盲杖互补前视感知；原M3、5格局部融合、L2与A+LOCAL保留。

RGB [尺度读出与新增近带覆盖](nearfield/RGB_NEAR_READOUT_SCALE_DEV_20261010.md)完成R0/R1/R2/R3×五主干臂；旧496预测不重跑，官方按序新增三Validation visit各16帧，近POS44/75/51。pooled304先封存后184eval，原三LOCO另保留。六ARK主强信号未通过：Uni raw R2三新近W4→8/44、20→31/75、19→22/51且FREE不增，但中远损42/771=5.447%；全184含3RScan中远损71/1447=4.907%，跨源总表达数值条件，不能替换事前ARK主口径。Uni affine R2中远零损但只两条近增益。按封存主决定维持RGB≥0.8m、DAV2 Indoor Large raw/原读出、近带ToF；恢复近信息与局部收益保留，非单目原理否定。旧DepthPro affine、主干/残差/stop与同步融合N/E均保留。

## ToF本阶段成本决定与新留出结论

用户2026-10-10认可：目标最近表面到共向走廊±0.30m边缘横向间隙≤10cm为near-pass，轻通知权重0.25、强1；far-pass与clear任何通知1。沿用gap1及同帧HEAD/BODY最高等级联合计数，成本为权重×通知次数之和；0/0.5只展示敏感性、不选点。contact仍f3–13及时、f14–15晚、全窗静默。此为本阶段效用口径，旧满额成本结果保留。

[新成本＋远pass留出](nearfield/CNH_COST_V2_HOLDOUT_DEV_20261010.md)完成冻结模型EXPLORE：cal/hold各768唯一physical scene×K2，contact各高度128scene、四层pass各64、clear256。hold间隙3.2/8.4/17.3/31.4cm，新family及参数；与21份可访问AABB库存的10,760物理键无交集，不宣称无键/保护来源已比对。

hold原5格HEAD174/BODY143各/256，成本113。both955及时救41/40、损0/0，成本177.5；near轻/强90/62、far10/15、clear19/49。远pass有通知24/256clip（22/128scene），原5格14/256（12/128）；far+clear通知93 vs64。新cal负例完整ties封存匹配阈值7.4555/7.5385/10.1373，主匹配救0/1、损0/0，成本115，far+clear64。956/957匹配均净增1，both净增89/87；主both净增81。

预声明净增≥10/512、far+clear≤110%及seed同向的强信号未出现；保留原5格，不升默认候选。近边界轻提醒降权仍未使当前门控保留收益；下一机制需区分接触窗口与远pass/clear，成本进入监督或读出设计，仅建议、本run未训练。49,152事件/11,039通知与全部ties独立复算PASS；科学保守150/1500s、GPU120/900s、准备评价审计整合550/900s。模型/HGB/旧阈值不变，无保护480/test访问；新hold已消费，后续不当新确认。

## 继承与边界

共向HEAD/BODY不替代外参/时钟核验。[漏检拆解](nearfield/CNH_BASELINE_MISS_DECOMPOSITION_DEV_20261010.md)的已消费解释结果、[冻结E2E](nearfield/CNH_RGB_FROZEN_E2E_RESULTS_20261010.md)、细杆oracle/四类读出失败和停止规则按各run保留。方向/PDR、硬件第二阶段不由本run重启；完整同步RGB/CNH仍NOT_EVALUABLE。当前仅有限AABB模拟Development，不作实机、安全或硬件极限证明。

更新前全文见Git `375ec240`及本run `integration_before/`；历史细节见RUNS及原报告。
ALGORITHM_EXPLORATION / M3_RETAINED / HARDWARE_DEFERRED
