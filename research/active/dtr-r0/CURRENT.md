ALGORITHM_EXPLORATION / M3_RETAINED / HARDWARE_DEFERRED

# 前视障碍感知：当前状态

更新：2026-10-10。盲杖互补前视感知主线；原M3、5格局部融合、L2和A+LOCAL保留。

RGB独立子线：[冻结残差读出与条件校准](nearfield/RGB_BODY_QUERY_RESIDUAL_READOUT_DEV_20261010.md)冻结两step600预测，完成absolute/normalized×global/band。主60/65原affine见证/FREE120/12、131/10；CTX归一化global1/13、2/18，绝对band60/20、50/8，归一化band4/7、19/19，不升级。held固定带成本辅助DR130/12、131/10，对强curveaffine122/12、132/10救9损1／救0损1，局部排序空间保留但非校准迁移成功。Ark16band辅助144query N/E，commonPOS158/FREE105；nearPOS44/11/0无稳定恢复。全部4组合/3臂/3capture与5补充cohort、逐query救损、3450ties曲线独立PASS。CPU科学51.607s、独立审计12.806s、整体保守270/1200s；GPU/train/forward/download0，任务资源结束。affine保留，本run停止固定recipe调参，下一优先RGB可观测信息/合规train域覆盖，cal不转train，其它机制开放。仅已消费Development/nativefirstreturn，旧A0全δ/CNH/stop保留；前RGB正文Git24688f0d。

## 当前：共向假设下推进ToF检出与分级提醒

用户于2026-10-10明确：本阶段暂按“头朝向＝身体朝向＝行进方向”处理，ToF身体/头部query共用这一方向。以ideal/共向条件为主比较；方向估计、PDR及方向不确定性合同暂不作为本阶段前置工作。这是当前任务假设，旧偏差实验的结果和身份保留。

目标回到ToF任务收益与分级提醒：明显风险给强提醒，接触/擦边难分时允许轻提醒；不把精确分界作为继续推进的前提。分级只依据运行时可见的分数与证据质量，真值仅评价；弱证据保留不确定含义。重点是弱/细目标证据在投影、累积和完整bin读出中的保留，以及分级能否增加有用及时提醒。强/轻提醒分别报告接触检出、完整救/损、pass与clear clip/slot成本，强转轻也单列；轻提醒可接受但打扰成本不记为零，不设零损失门槛。共向假设不替代外参、时钟或空间投影核验。

[连续提醒与断流重置](nearfield/CNH_GRADED_STREAM_REFRESH_DEV_20261010.md)完成固定episode1的时间戳/可用性适配与单一12秒strong刷新候选，未改Android。停止/不可用/reset、回钟或观测gap>1500ms清事件，重复timestamp不累静默；首正即报/轻到强即升，当前grade2距前强>=12s才刷新、轻档不周期报。主App事件已提醒即抑制与独立ToF Demo布尔12s提醒已核实，候选不称App复现。两方案×baseline/both/head50×三seed全cal/val36cells、110592流、216groups在200ms名义clock/2.4s短窗中均0刷新，通知与前episode1逐帧一致。head50 timely HEAD352/357/351、BODY311/316/320各/384，联合pass221/271/257、clear56/89/69不变。14独立合成工程样例720行：30s强0/12/24s发、30s轻仅首报，断流与升强边界核验；不构成长期风险收益。独立缓存/fixture/hash核验PASS；CPU保守260/330s、GPU/fit/预测/新raw0，无常驻资源。保留episode1离线成本候选及head50/both对照；12s仅工程候选、短窗证据不足，不扩周期扫描。通知层暂收束，下一固定工作点剩余弱/细目标未及时/晚检/全窗无提醒及首次分数/峰支持诊断。惰性reset不能主动取消声振，仍需上游watchdog；未接CNH App，仅已消费模拟Development和合成工程样例。原M3/5格/L2/body truth/fullbin/480/weak_pass及旧stop保留，深度静默证据不足、temporal不默认。前CNH正文Git4653f67a。

已完成的[步行参考](nearfield/CNH_WALKING_CORRIDOR_DEV_20261010.md)、[方向合同](nearfield/CNH_DIRECTION_CONTRACT_DEV_20261009.md)和全部方向聚合/锚定/高度混合结果作为历史诊断保留，不再驱动本阶段主线。步行参考的18–20°是过去/未来轨迹差异，非ToF姿态精度；单位/名义时钟限制仍按原报告。更新前CNH当前正文见Git 53bb1069同路径。

## 停止与保留

[旧扰动训练](nearfield/CNH_QUERY_PERTURB_TRAIN_DEV_20261008.md)、T/T2、旧参照和门控的停止规则仍按各自run适用；不同机制开放。[480确认](nearfield/CNH_REAL_HEAD_CONFIRM_RESULTS_20261007.md)原身份保留，UNKNOWN/三态/query覆盖待验证。原Nymeria位移实现、−19°、设备/City/保护test/新UE及硬件第二阶段按原run暂缓；101/101与53ms属A+LOCAL，M3/CNH实机效果未建立。

模拟AABB、有限背景、人工误差、相关K/帧及探索选择不构成实机、安全或独立确认。更新前CNH当前正文保留于Git 5844eacd同路径；RGB子线继续沿用。

[RUNS](RUNS.md) · [总决定](../../../docs/CURRENT_DECISION.md)。
