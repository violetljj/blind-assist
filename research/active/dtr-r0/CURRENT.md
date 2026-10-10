ALGORITHM_EXPLORATION / M3_RETAINED / HARDWARE_DEFERRED

# 前视障碍感知：当前状态

更新：2026-10-10。盲杖互补前视感知主线；原M3、5格局部融合、L2和A+LOCAL保留。

RGB独立子线：[查询外推与负参考](nearfield/RGB_BODY_QUERY_QUERY_LEVEL_DEV_20261009.md)。固定27子盒、冻结旧head/cal且不重选阈值；136帧3672格，575格sampledFREE（572来自三ARKit capture），不是整盒/身体空闲。新3RScan正见证head352/866、affine-margin487/866，救/损109/244；旧ARKit采样free支持7/105对0/105；新增两个官方visit组head31/220、22/247，对照12/220、10/247。未证明冻结head对任意query的稳定增量，保留旧RGB距离信息Development收益；下一改距离/不确定性表征和query覆盖，以强几何及全量救回损失检验。第三硬件TUM仍因吞吐停止NOT_RUN，同iPad家族补参考不替代硬件迁移；完整身体/细障碍/步行仍缺。旧query/32特征不续训，真实硬目标/同源锚点限制/CNH预算不变。

## 当前：步行离线参考已接通，因果位置仍缺

[步行走廊参考](nearfield/CNH_WALKING_CORRIDOR_DEV_20261010.md)已接通全部1025个BlindWays缓存片段：10位参与者、615000帧、10250native anchor×T0.5/1/1.5秒=30750窗口。显式60Hz名义相对时钟，无实测时间戳/头部朝向；原XY/Z上合同沿用，米尺度未找到官方明文，公制结果以旧单位假设为条件。

past≥0.30m预定子集的past1s/future-chord差异RMS18.38/18.82/19.62°，不是设备yaw或head-to-travel误差；全部时域/参与者/缺失与低motion都保留，不按结果选T/δ。完整未来折线+0.30m圆盘仅中心轨迹proxy，中心距端点弦>0.30m为1/10250、2/9225、4/9225，非身体面积遗漏/碰撞。片段关联未知、不是iid。

6fixture、30750行账本/66摘要/28700完整路径与100anchor的past公式/prefix核验通过。仅CPU离线整理（本run300s上限、GPU0）；无新模型/投影/训练/采集/数据下载/480。因果位置仍NOT_IMPLEMENTED；下一接合格因果位置与同参考/同时间窗评价，再验证物理T、身体几何、外参和同步。

旧[Charades方向合同](nearfield/CNH_DIRECTION_CONTRACT_DEV_20261009.md)、全部聚合/锚定/高度混合结果和旧2°mixed RMS结论按原run保留，不能与本轮轨迹差异直接换算。M3/5格/L2/bodytruth/bin/480及停止配方保留；更新前CNH正文见Git 4cfcca53同路径。

## 停止与保留

[旧扰动训练](nearfield/CNH_QUERY_PERTURB_TRAIN_DEV_20261008.md)、T/T2、旧参照和门控的停止规则仍按各自run适用；不同机制开放。[480确认](nearfield/CNH_REAL_HEAD_CONFIRM_RESULTS_20261007.md)原身份保留，UNKNOWN/三态/query覆盖待验证。原Nymeria位移实现、−19°、设备/City/保护test/新UE及硬件第二阶段按原run暂缓；101/101与53ms属A+LOCAL，M3/CNH实机效果未建立。

模拟AABB、有限背景、人工误差、相关K/帧及探索选择不构成实机、安全或独立确认。更新前CNH当前正文保留于Git 5844eacd同路径；RGB子线继续沿用。

[RUNS](RUNS.md) · [总决定](../../../docs/CURRENT_DECISION.md)。
