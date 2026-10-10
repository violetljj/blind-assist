# 前视障碍感知：当前状态

更新：2026-10-10。盲杖互补前视感知；原M3、5格局部融合、L2与A+LOCAL保留。

RGB [四官方主干横比](nearfield/RGB_DEPTH_BACKBONE_COMPARE_DEV_20261010.md)完成各496帧，冻结query/56train affine/三折与304cal不调阈值。当前冻结系统近带交ToF、RGB职责≥0.8m；中远带实验首选DAV2 Indoor Large原始米制，三ARK见证440/554、FREE25/199 vs原DepthPro affine354/554、27/199（救138损52），但far292/323 vs319/323、65中远FREE21/103 vs10/103，逐capture无成本替换未获支持。UniDepth affine见证最多468/554但FREE32/199。近带米制误差明显恢复，Metric3D/MoGe raw的全局cut>0.25可结构排除近见证；60唯一增益DAV raw6/11伴FREE1/133，65近POS0使强信号N/E，职责分工不作单目近带原理否定。原DepthPro affine/全部旧失败与stop保留；完整同步RGB/CNH仍NOT_EVALUABLE。

## ToF本阶段成本决定与新留出结论

用户2026-10-10认可：目标最近表面到共向走廊±0.30m边缘横向间隙≤10cm为near-pass，轻通知权重0.25、强1；far-pass与clear任何通知1。沿用gap1及同帧HEAD/BODY最高等级联合计数，成本为权重×通知次数之和；0/0.5只展示敏感性、不选点。contact仍f3–13及时、f14–15晚、全窗静默。此为本阶段效用口径，旧满额成本结果保留。

[新成本＋远pass留出](nearfield/CNH_COST_V2_HOLDOUT_DEV_20261010.md)完成冻结模型EXPLORE：cal/hold各768唯一physical scene×K2，contact各高度128scene、四层pass各64、clear256。hold间隙3.2/8.4/17.3/31.4cm，新family及参数；与21份可访问AABB库存的10,760物理键无交集，不宣称无键/保护来源已比对。

hold原5格HEAD174/BODY143各/256，成本113。both955及时救41/40、损0/0，成本177.5；near轻/强90/62、far10/15、clear19/49。远pass有通知24/256clip（22/128scene），原5格14/256（12/128）；far+clear通知93 vs64。新cal负例完整ties封存匹配阈值7.4555/7.5385/10.1373，主匹配救0/1、损0/0，成本115，far+clear64。956/957匹配均净增1，both净增89/87；主both净增81。

预声明净增≥10/512、far+clear≤110%及seed同向的强信号未出现；保留原5格，不升默认候选。近边界轻提醒降权仍未使当前门控保留收益；下一机制需区分接触窗口与远pass/clear，成本进入监督或读出设计，仅建议、本run未训练。49,152事件/11,039通知与全部ties独立复算PASS；科学保守150/1500s、GPU120/900s、准备评价审计整合550/900s。模型/HGB/旧阈值不变，无保护480/test访问；新hold已消费，后续不当新确认。

## 继承与边界

共向HEAD/BODY不替代外参/时钟核验。[漏检拆解](nearfield/CNH_BASELINE_MISS_DECOMPOSITION_DEV_20261010.md)的已消费解释结果、[冻结E2E](nearfield/CNH_RGB_FROZEN_E2E_RESULTS_20261010.md)、细杆oracle/四类读出失败和停止规则按各run保留。方向/PDR、硬件第二阶段不由本run重启；完整同步RGB/CNH仍NOT_EVALUABLE。当前仅有限AABB模拟Development，不作实机、安全或硬件极限证明。

更新前全文见Git `375ec240`及本run `integration_before/`；历史细节见RUNS及原报告。
ALGORITHM_EXPLORATION / M3_RETAINED / HARDWARE_DEFERRED
