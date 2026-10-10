ALGORITHM_EXPLORATION / M3_RETAINED / HARDWARE_DEFERRED

# 前视障碍感知：当前状态

更新：2026-10-10。盲杖互补前视感知主线；原M3、5格局部融合、L2和A+LOCAL保留。

RGB独立子线：[距离分布与查询读出](nearfield/RGB_BODY_QUERY_INTERVAL_DISTRIBUTION_DEV_20261010.md)。新query-independent log-Z分布/解析CDF与后验点深度读出完成，两网络原train1591791点、同初始化600步，仅原cal16/27query选阈值；6臂136帧3672query。新3RScan CDF正见证701/866 vs匹配cal几何555/866，救/损154/8但free射线误支持.111096 vs.097548；点读出675/866、.100255。追加ARKit点读出仍free73/220、79/247 vs几何12/220、10/247，不支持把退化整体归σ/CDF，未形成稳定跨相机候选。原cal只有2个strictfree子盒，百万free query-ray不代替query误报校准；下一补query级负参考并拆点深度迁移/覆盖/工作点。22032新记录与冻结/partition核验通过，GPU40.538s、CPU保守660s，下载0、任务释放；旧head/32臂不续训、575有限相关free/UNKNOWN/真实硬目标/同源锚点/完整身体细障碍步行第三硬件缺口与CNH预算保留。

## 当前：共向假设下推进ToF检出与分级提醒

用户于2026-10-10明确：本阶段暂按“头朝向＝身体朝向＝行进方向”处理，ToF身体/头部query共用这一方向。以ideal/共向条件为主比较；方向估计、PDR及方向不确定性合同暂不作为本阶段前置工作。这是当前任务假设，旧偏差实验的结果和身份保留。

目标回到ToF任务收益与分级提醒：明显风险给强提醒，接触/擦边难分时允许轻提醒；不把精确分界作为继续推进的前提。分级只依据运行时可见的分数与证据质量，真值仅评价；弱证据保留不确定含义。重点是弱/细目标证据在投影、累积和完整bin读出中的保留，以及分级能否增加有用及时提醒。强/轻提醒分别报告接触检出、完整救/损、pass与clear clip/slot成本，强转轻也单列；轻提醒可接受但打扰成本不记为零，不设零损失门槛。共向假设不替代外参、时钟或空间投影核验。

[普通扩充结果](nearfield/CNH_COUNTERFACTUAL_DEV_20261009.md)及weak_pass独立读出保留为Development候选；M3、原5格融合、L2、身体truth/fullbin/480身份与各旧run停止规则保留。下一先用已有ideal分数与账本判断强/轻提醒的可用工作点和细障碍漏检；先声明分级规则，再评价及时收益与分档成本。本次只更新任务假设和提醒方向，尚未实现提醒分级或新阈值，不启动训练/推理/新采集。

已完成的[步行参考](nearfield/CNH_WALKING_CORRIDOR_DEV_20261010.md)、[方向合同](nearfield/CNH_DIRECTION_CONTRACT_DEV_20261009.md)和全部方向聚合/锚定/高度混合结果作为历史诊断保留，不再驱动本阶段主线。步行参考的18–20°是过去/未来轨迹差异，非ToF姿态精度；单位/名义时钟限制仍按原报告。更新前CNH当前正文见Git 53bb1069同路径。

## 停止与保留

[旧扰动训练](nearfield/CNH_QUERY_PERTURB_TRAIN_DEV_20261008.md)、T/T2、旧参照和门控的停止规则仍按各自run适用；不同机制开放。[480确认](nearfield/CNH_REAL_HEAD_CONFIRM_RESULTS_20261007.md)原身份保留，UNKNOWN/三态/query覆盖待验证。原Nymeria位移实现、−19°、设备/City/保护test/新UE及硬件第二阶段按原run暂缓；101/101与53ms属A+LOCAL，M3/CNH实机效果未建立。

模拟AABB、有限背景、人工误差、相关K/帧及探索选择不构成实机、安全或独立确认。更新前CNH当前正文保留于Git 5844eacd同路径；RGB子线继续沿用。

[RUNS](RUNS.md) · [总决定](../../../docs/CURRENT_DECISION.md)。
