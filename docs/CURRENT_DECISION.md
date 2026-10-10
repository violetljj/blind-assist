# 当前研究决定

更新：2026-10-10。主线为盲杖互补前视感知，M3和A+LOCAL保留；原5格局部融合和L2保留。

RGB 当前冻结全局 affine，0.8m以上为候选public query职责；当前近带固定recipe停止调参。两step600残差及全部失败/局部排序结果按[原报告](../research/active/dtr-r0/nearfield/RGB_BODY_QUERY_RESIDUAL_READOUT_DEV_20261010.md)保留，不把配方失败写成单目近带原理否定。后续提前量/语义尚待评价。[本次输入核查](../research/active/dtr-r0/nearfield/CNH_RGB_FROZEN_E2E_INPUT_INVENTORY_20261010.md)确认没有未消费且满足真值合同的同步RGB/CNH输入，完整融合NOT_EVALUABLE，未运行新RGB推理。

## 当前：冻结配置与端到端评价

用户于2026-10-10明确：本阶段暂按“头朝向＝身体朝向＝行进方向”处理，ToF身体/头部query共用这一方向。以ideal/共向条件为主比较；方向估计、PDR及方向不确定性合同暂不作为本阶段前置工作。这是当前任务假设，旧偏差实验的结果和身份保留。

目标回到ToF任务收益与分级提醒：明显风险给强提醒，接触/擦边难分时允许轻提醒；不把精确分界作为继续推进的前提。分级只依据运行时可见的分数与证据质量，真值仅评价；弱证据保留不确定含义。重点是弱/细目标证据在投影、累积和完整bin读出中的保留，以及分级能否增加有用及时提醒。强/轻提醒分别报告接触检出、完整救/损、pass与clear clip/slot成本，强转轻也单列；轻提醒可接受但打扰成本不记为零，不设零损失门槛。共向假设不替代外参、时钟或空间投影核验。

[冻结配置新场景端到端评价](../research/active/dtr-r0/nearfield/CNH_RGB_FROZEN_E2E_RESULTS_20261010.md)完成ToF→通知比较：cal/hold各768新物理scene、K2，新背景family分开，主seed955预声明。主HEAD细横杆厚1.7cm，16scene/32相关事件；原5格与同通知成本匹配臂均2/32，救0损0。HEAD158/256不变，BODY116→119/256救3损0；hold空闲/纯擦边联合通知均28/37。原both为HEAD202、BODY154各/256、细横杆3/32，但通知49/286；956/957匹配HEAD细救回也0，956擦边38比原37多1。不支持主目标同成本收益，不升级App，固定配方结束不在该hold调参。原M3/5格/旧both及head50、L2和所有失败保留；旧剩余检出诊断见Git1311e202同路径。仅解析AABB共向ideal/有限背景、短窗口离线通知，完整RGB＋ToF不可评价。

[HEAD细横杆期望回波诊断](../research/active/dtr-r0/nearfield/CNH_HEAD_THIN_SIGNAL_ORACLE_DEV_20261010.md)完成：厚度/反射率受控扫中，1.7cm在f13前距0.97m最佳bin SNR为1.37/3.59（rho .19/.57）；0.65m原薄杆整盒出横向视场。固定bin移动8帧SNR降至0.78/2.27，真值matched为2.38/6.54，静态√K不可搬用。原M3细杆3/32及时、29静默，但27/32及时窗曾有相容top8，支持中位2帧；query线性投影描述比例约16.4%，非网络因果。sub32敏感性显著，不能宣布硬件物理上限。下一优先可见窗口内随距离移动的稀疏读出，新读出NOT_RUN；不同统计量阈值不可直接比较，不在已消费E2E hold调参，完整融合仍NOT_EVALUABLE。

[局部运动路径读出](../research/active/dtr-r0/nearfield/CNH_GRADED_MOTION_PATH_DEV_20261010.md)完成旧consumed ideal对照，不采用。当前top8 native中心沿公共运动回溯past8，正负值/缺帧/各rank保留；匹配静态同核同窗，两臂各6个固定浅层模型。validation运动相对head50：HEAD救/损0:0、2:3、0:5，BODY2:4、1:4、1:6，各/384；BODY弱横杆0:0、0:1、1:0，各/48。静默弱横杆相容路径诊断Z约3.5 vs静态1.3–1.5，但峰值模型margin仍−1.51/−1.12/−1.49，及时支持4帧；空闲最高路径Z中位3.62，幅度不足分离噪声。cal9/11 clear union slots/pass整clip约束通过，validation成本漂移单列。12cells/6cuts/360cohorts、公共几何与因果前缀通过，51事件/52诊断组补核验；源失败保留。下一候选共享多实例读出联合空间/路径形态，NOT_RUN，不扫本轮参数。[前轮聚合读出](../research/active/dtr-r0/nearfield/CNH_GRADED_WEAK_MASS_DEV_20261010.md)两聚合负结果、原M3/5格/L2/480/工作点/stop保留；未读已消费新E2E hold，仅模拟非实机。更新前当前页见Git b746125b。

已完成的[步行参考](../research/active/dtr-r0/nearfield/CNH_WALKING_CORRIDOR_DEV_20261010.md)、[方向合同](../research/active/dtr-r0/nearfield/CNH_DIRECTION_CONTRACT_DEV_20261009.md)和全部方向聚合/锚定/高度混合结果作为历史诊断保留，不再驱动本阶段主线。步行参考的18–20°是过去/未来轨迹差异，非ToF姿态精度；单位/名义时钟限制仍按原报告。更新前CNH当前正文见Git 53bb1069同路径。

## 停止与保留

[旧扰动训练](../research/active/dtr-r0/nearfield/CNH_QUERY_PERTURB_TRAIN_DEV_20261008.md)、T/T2、旧参照和门控的停止规则仍按各自run适用；不同机制开放。[480确认](../research/active/dtr-r0/nearfield/CNH_REAL_HEAD_CONFIRM_RESULTS_20261007.md)原身份保留，UNKNOWN/三态/query覆盖待验证。原Nymeria位移实现、−19°、设备/City/保护test/新UE及硬件第二阶段按原run暂缓；101/101与53ms属A+LOCAL，M3/CNH实机效果未建立。

模拟AABB、有限背景、人工误差、相关K/帧及探索选择不构成实机、安全或独立确认。更新前CNH当前正文保留于Git 5844eacd同路径；RGB子线继续沿用。

[路线当前页](../research/active/dtr-r0/CURRENT.md) · [RUNS](../research/active/dtr-r0/RUNS.md)。
