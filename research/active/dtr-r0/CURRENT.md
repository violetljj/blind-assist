ALGORITHM_EXPLORATION / M3_RETAINED / HARDWARE_DEFERRED

# 前视障碍感知：当前状态

更新：2026-10-10。盲杖互补前视感知主线；原M3、5格局部融合、L2和A+LOCAL保留。

RGB独立子线：[新session近负覆盖](nearfield/RGB_BODY_QUERY_NEAR_COVERAGE_DEV_20261010.md)。原A0三折与新cal32FREE冻结评分保留；当前affine代表、不提高残差训练优先级，不否定重新训练。新查15额外cal环境全部16未消费rescan各固定均匀8帧，全27query，共128帧3456查询；FREE22/8环境，near仅4仍原1环境。与旧960参考合并1088帧FREE151/13环境、near23→27仍1环境，未改善跨环境近负覆盖；按预定条件不运行新128推理/评分，缓存同scan/新session扩展均停止。旧near8640query中6511已有>=16FREE且零正像素却含UNKNOWN；新near1152中同类903。固定旧15/新16首帧depth missing303059/314458、box前1193/718、FOV外0，中央near格覆盖整张native depth；这是参考缺失诊断，不定位模型所有误差。缓存官方train35环境72scan已无未消费新环境。下一优先未消费ARKit Training visit作独立cal，现有eval不转cal，不放宽严格FREE或填洞。3456XYZ/原生身份及独立27query/分解PASS，CPU保守726/1800s，训练/推理/GPU/下载0，无保留进程。仅Development非身体/整盒/实机证据，CNH与旧停止保留，前文Gitd36e9e4b。

## 当前：共向假设下推进ToF检出与分级提醒

用户于2026-10-10明确：本阶段暂按“头朝向＝身体朝向＝行进方向”处理，ToF身体/头部query共用这一方向。以ideal/共向条件为主比较；方向估计、PDR及方向不确定性合同暂不作为本阶段前置工作。这是当前任务假设，旧偏差实验的结果和身份保留。

目标回到ToF任务收益与分级提醒：明显风险给强提醒，接触/擦边难分时允许轻提醒；不把精确分界作为继续推进的前提。分级只依据运行时可见的分数与证据质量，真值仅评价；弱证据保留不确定含义。重点是弱/细目标证据在投影、累积和完整bin读出中的保留，以及分级能否增加有用及时提醒。强/轻提醒分别报告接触检出、完整救/损、pass与clear clip/slot成本，强转轻也单列；轻提醒可接受但打扰成本不记为零，不设零损失门槛。共向假设不替代外参、时钟或空间投影核验。

[局部峰跟踪与断续支持](nearfield/CNH_GRADED_PEAK_TRACK_DEV_20261010.md)完成3规则×3追加预算×3seed共54split单元；仅在原grade0追加轻提醒，完整single轻档/ordinary OR强档及首次保留。中档单帧峰及时HEAD376/375/380、BODY314/317/322各/384，对完整轻档救25/31/34和7/7/10，clear增12/12/29slots/6656、pass增61/51/58clips/256。轨迹累计HEAD救10/8/9、BODY0，比不匹配5帧累计多救3/1/2但clear多19、pass多21/21/19，不能称等成本胜出；原参照保留，不默认采用追加规则。下一候选联合局部峰存在/支持与身体通道侵入，重点弱BODY，不继续此固定轨迹幅值和调参。峰持续不证明目标/接近/free；cal9/11追加query并集slot上限非validation成本。公共noisy位姿、top8、5帧/缺2帧、当前命中，35字段/156query与54指标/27cutoff/82944台账独立核验PASS，当前几何预检故障修复留证。CPU保守225/900s、GPU/新拟合/前向/采样0，无常驻资源。仅已消费模拟Development、未接入App；原M3/5格/L2/body truth/fullbin/480、weak_pass及旧stop保留；[高度评分候选](nearfield/CNH_GRADED_HEIGHT_CAL_DEV_20261010.md)仍保留，更新前正文见Git c782b6bd。

已完成的[步行参考](nearfield/CNH_WALKING_CORRIDOR_DEV_20261010.md)、[方向合同](nearfield/CNH_DIRECTION_CONTRACT_DEV_20261009.md)和全部方向聚合/锚定/高度混合结果作为历史诊断保留，不再驱动本阶段主线。步行参考的18–20°是过去/未来轨迹差异，非ToF姿态精度；单位/名义时钟限制仍按原报告。更新前CNH当前正文见Git 53bb1069同路径。

## 停止与保留

[旧扰动训练](nearfield/CNH_QUERY_PERTURB_TRAIN_DEV_20261008.md)、T/T2、旧参照和门控的停止规则仍按各自run适用；不同机制开放。[480确认](nearfield/CNH_REAL_HEAD_CONFIRM_RESULTS_20261007.md)原身份保留，UNKNOWN/三态/query覆盖待验证。原Nymeria位移实现、−19°、设备/City/保护test/新UE及硬件第二阶段按原run暂缓；101/101与53ms属A+LOCAL，M3/CNH实机效果未建立。

模拟AABB、有限背景、人工误差、相关K/帧及探索选择不构成实机、安全或独立确认。更新前CNH当前正文保留于Git 5844eacd同路径；RGB子线继续沿用。

[RUNS](RUNS.md) · [总决定](../../../docs/CURRENT_DECISION.md)。
