# 当前研究决定

更新：2026-10-10。主线为盲杖互补前视感知，M3和A+LOCAL保留；原5格局部融合和L2保留。

RGB独立子线：[输入与数值诊断](../research/active/dtr-r0/nearfield/RGB_BODY_QUERY_INPUT_NUMERICS_DEV_20261010.md)完成96帧ARKit输入合约与固定16帧FP16/float32对照。时间戳/尺寸/K/mm→m/focal缩放通过，未发现实现尺度错误；旧48推理缺历史源码SHA如实保留，16/16 FP16缓存精确复现。原色饱和端点14830→14857，未消除；arkit16共同未饱和近正49152点affine ratio中位3.7923→3.7941、轴向inside0，新cal共同未饱和20178点2.6512→2.6505仍inside0；60/65所选端点无近POS，N/E。主absolute及global normalized救损/FREE增减0，辅助normalized band5救3损1/FREE增减0，其中cal near救1保留；cal far无FREE不报零成本。全局log-affine保留，不切换精度或提升训练优先级，未否定真正有界残差；下一现有train/cal残差目标、输入可辨识性与界限，禁止eval倒推bound，不默认范围回退。独立32倒数/336指标/1728配对及1原生frame排序PASS，辅助枚举/引号失败留证。GPU32calls262.191/300s、CPU保守741/1200s，无训练/下载、资源已释放。仅已消费Development/现有参考，非身体/实机/安全；旧A0三折/分带/stop/CNH保留，前文Gite4e6b2cd。

## 当前：共向假设下推进ToF检出与分级提醒

用户于2026-10-10明确：本阶段暂按“头朝向＝身体朝向＝行进方向”处理，ToF身体/头部query共用这一方向。以ideal/共向条件为主比较；方向估计、PDR及方向不确定性合同暂不作为本阶段前置工作。这是当前任务假设，旧偏差实验的结果和身份保留。

目标回到ToF任务收益与分级提醒：明显风险给强提醒，接触/擦边难分时允许轻提醒；不把精确分界作为继续推进的前提。分级只依据运行时可见的分数与证据质量，真值仅评价；弱证据保留不确定含义。重点是弱/细目标证据在投影、累积和完整bin读出中的保留，以及分级能否增加有用及时提醒。强/轻提醒分别报告接触检出、完整救/损、pass与clear clip/slot成本，强转轻也单列；轻提醒可接受但打扰成本不记为零，不设零损失门槛。共向假设不替代外参、时钟或空间投影核验。

[BODY定向追加对照](../research/active/dtr-r0/nearfield/CNH_GRADED_PEAK_BODY_ONLY_DEV_20261010.md)完成固定current中档评分/cutoff、三seed全ideal cal/val18cells；BODY-only保留BODY救回4/6/8及提前53/31/44，将新增clear23/40/27→12/12/18slots/6656、21/31/23→11/8/14clips/512，pass25/57/36→18/22/23slots/3328、1/12/7→0/7/4新增clips/256。代价是HEAD新增救回4/14/6及提前31/59/40全部放弃；救回为横细杆3/10/4＋标牌边1/4/2，真实contact高度联合及时也损4/14/6件，非可忽略成本。保留两高度current候选与原strong/fullsingle，BODY-only仅取舍对照，不默认关闭HEAD或静默；下一分高度calibration在cal控制HEAD实际新增成本，完整计弱细目标与提前保留，尚未执行。此前匹配strict救回clear仅0/1/0、峰更远也见contact，证据不足，不放宽或采用深度静默。独立18metrics/summary、55296ledger、2982897项及24inputhash PASS，CPU保守153/300s、GPU/fit/预测/新raw0、无常驻资源。仅已消费模拟Development、未接入App，原M3/5格/L2/body truth/fullbin/480、weak_pass及旧stop保留；[匹配诊断](../research/active/dtr-r0/nearfield/CNH_GRADED_PEAK_MATCH_DEV_20261010.md)和[联合评分](../research/active/dtr-r0/nearfield/CNH_GRADED_PEAK_JOINT_DEV_20261010.md)保留，更新前CNH正文见Git59696df6。

已完成的[步行参考](../research/active/dtr-r0/nearfield/CNH_WALKING_CORRIDOR_DEV_20261010.md)、[方向合同](../research/active/dtr-r0/nearfield/CNH_DIRECTION_CONTRACT_DEV_20261009.md)和全部方向聚合/锚定/高度混合结果作为历史诊断保留，不再驱动本阶段主线。步行参考的18–20°是过去/未来轨迹差异，非ToF姿态精度；单位/名义时钟限制仍按原报告。更新前CNH当前正文见Git 53bb1069同路径。

## 停止与保留

[旧扰动训练](../research/active/dtr-r0/nearfield/CNH_QUERY_PERTURB_TRAIN_DEV_20261008.md)、T/T2、旧参照和门控的停止规则仍按各自run适用；不同机制开放。[480确认](../research/active/dtr-r0/nearfield/CNH_REAL_HEAD_CONFIRM_RESULTS_20261007.md)原身份保留，UNKNOWN/三态/query覆盖待验证。原Nymeria位移实现、−19°、设备/City/保护test/新UE及硬件第二阶段按原run暂缓；101/101与53ms属A+LOCAL，M3/CNH实机效果未建立。

模拟AABB、有限背景、人工误差、相关K/帧及探索选择不构成实机、安全或独立确认。更新前CNH当前正文保留于Git 5844eacd同路径；RGB子线继续沿用。

[路线当前页](../research/active/dtr-r0/CURRENT.md) · [RUNS](../research/active/dtr-r0/RUNS.md)。
