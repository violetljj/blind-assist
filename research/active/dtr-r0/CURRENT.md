ALGORITHM_EXPLORATION / M3_RETAINED / HARDWARE_DEFERRED

# 前视障碍感知：当前状态

更新：2026-10-09。保留5格融合候选继续研究，不升级原M3报警政策。DTR_R2_DYNAMIC_RETAINED仅历史保留，不恢复动态研究。

## 最新决定

用户优先横杆/竖杆/柜体突出物、标牌补充，限定对齐直行、理想位姿、不实测。[形状报告](nearfield/CNH_ALIGNED_SHAPES_DEV_20261008.md)492 AABB×K4、316物理接触、344高度接触、88pass/88clear。原−10°/M3/θ下HEAD517/688、BODY406/688及时；clear46/4576格、25段、24/352clip，pass74/352，非真实提醒频率。

[融合/竖杆](nearfield/CNH_BAR_FUSION_VERTICAL_DEV_20261009.md)预声明5/10格×中段/无下界四点，原K4均同46/4576。推荐5格无下界（raised-M3 OR strong-local，θM=.940418、θL=4.625390）：HEAD543/688救28损2，BODY461/688救56损1；暗4cm BODY1→7/56救6损0，30/90cm为0→3、1→4/28。clear仍25段、clip24→23/352、pass不增；竖杆HEAD/BODY救5/9损0/0。形状评价标签不进门控。

37×64旧draw完整窗复核HEAD119→217、BODY70→157/896，各损3，fixedθ clear均9/832。10格无下界原HEAD551/BODY466更高但BODY损5，MC clear11/832（原9）。不再校准、非独立新几何确认。推荐是Development事后工程选择；融合准备/运行/独立检查.828/180s，竖杆528原观测重投影9.046/600s、分析/审计7.266/180s，无M3推理/新采样/训练。

竖杆448事件T/G/N分解保存，原局部替换损15/22件中支持内峰值中位3.250/3.402、全局3.656/3.646（原localθ4.069）。几何交集不等于目标回波归因，平滑G不等于max(平滑T,N)，不能据此定位因果。下一固定5格候选检验不同背景的预算及原检出保持，NOT_RUN；经验尺度/形状库/跨窗一致性/远箱支路后续开放。

[投影/FP16](nearfield/CNH_BAR_REPRESENTATION_DEV_20261008.md)条件d²保留下界≥96.1847%、远负差几乎未覆盖；[M3读出](nearfield/CNH_BAR_READOUT_DEV_20261008.md)方差失配68/76，d_J仅敏感性。[局部替换](nearfield/CNH_BAR_LOCAL_READOUT_DEV_20261009.md)暗BODY1→9/56，但全批HEAD/BODY损40/55，原失败保留；[−19°](nearfield/CNH_ALIGNED_BOUNDARY_DEV_20261008.md)不推进。

## 其余路线保留

[扰动训练](nearfield/CNH_QUERY_PERTURB_TRAIN_DEV_20261008.md)原配方停止：exact各损7超2件护栏，不加seed/epoch、不换分布。方位微调和稀疏空间/关联L3暂停，方向/Nymeria位移接口暂缓；实质不同机制仍可审查。

[480确认](nearfield/CNH_REAL_HEAD_CONFIRM_RESULTS_20261007.md)原身份保留：E1较exact少39/51及时、EMA收回25/34（/1002）；后续Development及BlindWays已消费回放不重称确认，UNKNOWN/三态/query覆盖待验证。

[Nymeria](nearfield/CNH_NYMERIA_SAMPLE_AUDIT_DEV_20261008.md)仅原IMU姿态前缀通过，准确度未评、位置/PDR未实现，E1整链NOT_EVALUABLE。future pelvis/闭环只作参考、不进估计器。设备、City、保护test、新UE及硬件第二阶段暂停；101/101和53ms属A+LOCAL，M3/CNH实机效果未建立。

[RUNS](RUNS.md) · [总决定](../../../docs/CURRENT_DECISION.md)。更新前全文：Git daed724c同路径。
