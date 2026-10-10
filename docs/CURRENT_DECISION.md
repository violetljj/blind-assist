# 当前研究决定

更新：2026-10-10。主线为盲杖互补前视感知，M3和A+LOCAL保留；原5格局部融合和L2保留。

RGB独立子线：[距离分布与查询读出](../research/active/dtr-r0/nearfield/RGB_BODY_QUERY_INTERVAL_DISTRIBUTION_DEV_20261010.md)。新query-independent log-Z分布/解析CDF与后验点深度读出完成，两网络原train1591791点、同初始化600步，仅原cal16/27query选阈值；6臂136帧3672query。新3RScan CDF正见证701/866 vs匹配cal几何555/866，救/损154/8但free射线误支持.111096 vs.097548；点读出675/866、.100255。追加ARKit点读出仍free73/220、79/247 vs几何12/220、10/247，不支持把退化整体归σ/CDF，未形成稳定跨相机候选。原cal只有2个strictfree子盒，百万free query-ray不代替query误报校准；下一补query级负参考并拆点深度迁移/覆盖/工作点。22032新记录与冻结/partition核验通过，GPU40.538s、CPU保守660s，下载0、任务释放；旧head/32臂不续训、575有限相关free/UNKNOWN/真实硬目标/同源锚点/完整身体细障碍步行第三硬件缺口与CNH预算保留。

## 当前：步行离线参考已接通，因果位置仍缺

[步行走廊参考](../research/active/dtr-r0/nearfield/CNH_WALKING_CORRIDOR_DEV_20261010.md)已接通全部1025个BlindWays缓存片段：10位参与者、615000帧、10250native anchor×T0.5/1/1.5秒=30750窗口。显式60Hz名义相对时钟，无实测时间戳/头部朝向；原XY/Z上合同沿用，米尺度未找到官方明文，公制结果以旧单位假设为条件。

past≥0.30m预定子集的past1s/future-chord差异RMS18.38/18.82/19.62°，不是设备yaw或head-to-travel误差；全部时域/参与者/缺失与低motion都保留，不按结果选T/δ。完整未来折线+0.30m圆盘仅中心轨迹proxy，中心距端点弦>0.30m为1/10250、2/9225、4/9225，非身体面积遗漏/碰撞。片段关联未知、不是iid。

6fixture、30750行账本/66摘要/28700完整路径与100anchor的past公式/prefix核验通过。仅CPU离线整理（本run300s上限、GPU0）；无新模型/投影/训练/采集/数据下载/480。因果位置仍NOT_IMPLEMENTED；下一接合格因果位置与同参考/同时间窗评价，再验证物理T、身体几何、外参和同步。

旧[Charades方向合同](../research/active/dtr-r0/nearfield/CNH_DIRECTION_CONTRACT_DEV_20261009.md)、全部聚合/锚定/高度混合结果和旧2°mixed RMS结论按原run保留，不能与本轮轨迹差异直接换算。M3/5格/L2/bodytruth/bin/480及停止配方保留；更新前CNH正文见Git 4cfcca53同路径。

## 停止与保留

[旧扰动训练](../research/active/dtr-r0/nearfield/CNH_QUERY_PERTURB_TRAIN_DEV_20261008.md)、T/T2、旧参照和门控的停止规则仍按各自run适用；不同机制开放。[480确认](../research/active/dtr-r0/nearfield/CNH_REAL_HEAD_CONFIRM_RESULTS_20261007.md)原身份保留，UNKNOWN/三态/query覆盖待验证。原Nymeria位移实现、−19°、设备/City/保护test/新UE及硬件第二阶段按原run暂缓；101/101与53ms属A+LOCAL，M3/CNH实机效果未建立。

模拟AABB、有限背景、人工误差、相关K/帧及探索选择不构成实机、安全或独立确认。更新前CNH当前正文保留于Git 5844eacd同路径；RGB子线继续沿用。

[路线当前页](../research/active/dtr-r0/CURRENT.md) · [RUNS](../research/active/dtr-r0/RUNS.md)。
