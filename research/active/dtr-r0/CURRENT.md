ALGORITHM_EXPLORATION / M3_RETAINED / HARDWARE_DEFERRED

# 前视障碍感知：当前状态

更新：2026-10-10。盲杖互补前视感知主线；原M3、5格局部融合、L2和A+LOCAL保留。

RGB独立子线：[冻结分距离校准](nearfield/RGB_BODY_QUERY_DISTANCE_CAL_DEV_20261010.md)完成固定三带each<=5%、五臂absolute主/normalized辅，共30cal-only阈值；cal近395/中180/远13FREE，实际19/9/0、合计28/588，先于eval冻结，原全局与A0三折保留。absolute affine Ark16/60/65总见证13/57/48、FREE35/5/4；原全局95/120/131、5/12/10仍作代表。near Ark16见证0/44、FREE35/96（原0/0），60仍0/11、0/133，65无近POS；原val/3RScan近救21/157但近FREE分母0，不证明近低成本。far13负例floor5%=0，affine far损66/63/83；与near负分布转移失败分开解释。A0三δ完整保留，normalized仅少量near见证且成本高；不采用本固定三带配方、不提高残差训练优先级、不否定其他校准/重训。下一近带点级几何偏差与正/负排序诊断，far稀疏校准方法须独立预定；无事后改预算/合带/扫描。独立30cutoff/276480paired/900LOEO/150matched PASS，score6.347s、CPU保守608/900s，GPU/训练/推理/下载0，无常驻资源；仅已消费Development/采样FREE，非身体或实机安全，旧stop/CNH保留，前文Gitfdb864a3。

## 当前：共向假设下推进ToF检出与分级提醒

用户于2026-10-10明确：本阶段暂按“头朝向＝身体朝向＝行进方向”处理，ToF身体/头部query共用这一方向。以ideal/共向条件为主比较；方向估计、PDR及方向不确定性合同暂不作为本阶段前置工作。这是当前任务假设，旧偏差实验的结果和身份保留。

目标回到ToF任务收益与分级提醒：明显风险给强提醒，接触/擦边难分时允许轻提醒；不把精确分界作为继续推进的前提。分级只依据运行时可见的分数与证据质量，真值仅评价；弱证据保留不确定含义。重点是弱/细目标证据在投影、累积和完整bin读出中的保留，以及分级能否增加有用及时提醒。强/轻提醒分别报告接触检出、完整救/损、pass与clear clip/slot成本，强转轻也单列；轻提醒可接受但打扰成本不记为零，不设零损失门槛。共向假设不替代外参、时钟或空间投影核验。

[局部峰与侵入联合评分](nearfield/CNH_GRADED_PEAK_JOINT_DEV_20261010.md)完成3bank×3追加预算×3seed，18个HEAD/BODY小评分fit、54split单元。原强档/完整single轻档及首次保留，只在grade0追加light；中档current及时HEAD355/358/352、BODY311/316/320各/384，对完整参照BODY救4/6/8、已有及时提前53/31/44，pass增1/12/7clips/256、clear增23/40/27slots/6656。对同fit/cal的score-only BODY救/损3/2、6/0、7/1，早/晚38/15、29/2、21/23；clear差+20/+11/−19、pass差−2/+7/−1。current保留候选，temporal追加BODY救/损4/1、3/5、3/4，收益不稳定，不默认升级；完整参照仍保留。下一建议匹配新增clear与BODY救回的公共距离/支持/背景可见证据，再决定校准或空间证据优先级，不直接静默或重复此固定参数调参。cal9/11追加candidate-query并集slot上限非val等成本，cal8/10仅grade0 fit，runtime无GT；22/57公共描述＋侵入score，峰持续/inner支持不是物体归因或free。特征12661286及评分450826项独立核验PASS，CPU保守216/900s、GPU0，无主干重训/前向/新采样或常驻资源。仅已消费模拟Development、未接入App，原M3/5格/L2/body truth/fullbin/480、weak_pass及旧stop保留；[旧峰轨迹比较](nearfield/CNH_GRADED_PEAK_TRACK_DEV_20261010.md)和高度候选按原身份保留，更新前CNH正文见Git d941e120。

已完成的[步行参考](nearfield/CNH_WALKING_CORRIDOR_DEV_20261010.md)、[方向合同](nearfield/CNH_DIRECTION_CONTRACT_DEV_20261009.md)和全部方向聚合/锚定/高度混合结果作为历史诊断保留，不再驱动本阶段主线。步行参考的18–20°是过去/未来轨迹差异，非ToF姿态精度；单位/名义时钟限制仍按原报告。更新前CNH当前正文见Git 53bb1069同路径。

## 停止与保留

[旧扰动训练](nearfield/CNH_QUERY_PERTURB_TRAIN_DEV_20261008.md)、T/T2、旧参照和门控的停止规则仍按各自run适用；不同机制开放。[480确认](nearfield/CNH_REAL_HEAD_CONFIRM_RESULTS_20261007.md)原身份保留，UNKNOWN/三态/query覆盖待验证。原Nymeria位移实现、−19°、设备/City/保护test/新UE及硬件第二阶段按原run暂缓；101/101与53ms属A+LOCAL，M3/CNH实机效果未建立。

模拟AABB、有限背景、人工误差、相关K/帧及探索选择不构成实机、安全或独立确认。更新前CNH当前正文保留于Git 5844eacd同路径；RGB子线继续沿用。

[RUNS](RUNS.md) · [总决定](../../../docs/CURRENT_DECISION.md)。
