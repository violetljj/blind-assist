# 当前研究决定

更新：2026-10-09。主线为盲杖互补前视感知，M3和A+LOCAL保留；原5格局部融合和L2保留。

RGB独立子线：[真实细结构首批补标](../research/active/dtr-r0/nearfield/RGB_BODY_QUERY_LABEL_COMPLETION_DEV_20261009.md)。15帧/9session补41core，246格11正/2非空参考负/233UNKNOWN；两负格同一横杆，已知距离全属1session。原7帧正query8→14来自目标域补充，模型没变。新增agent视觉复核非人工GT，现为稀疏监督，完整query训练/验证仍缺可靠距离与独立视觉核验。下一先补跨session距离与核验再训练；旧DepthPro/VDA负结果及事件提前量不可评保留。真实定量硬目标、旧BodyQuery/COPILOT定位与CNH预算/停止规则/M3保留。

## 当前：弱pass改善低擦边独立读出的权衡，旧融合不升级

[工作点、坐标等价与匹配续训](../research/active/dtr-r0/nearfield/CNH_PASS_BOUNDARY_DEV_20261009.md)已完成。先按ideal cal选20/30/40% pass预算（51/76/102个clip），再固定阈值用于validation与+3°；独立读出/OR的cal clear上限65/80格。旧OR pass下限55，20%点不可行。只调阈值会损失接触检出，随后从普通扩充的三个最终checkpoint各做12轮/2808步原mask对照与弱pass续训，共6个模型。

30%点弱pass−匹配对照：独立读出HEAD净+5/+18/+13、BODY+4/+30/+25各/384；实际pass80/73/72→76/77/75各/256，clear0/0/0→3/1/0各/6656，并非严格同成本。1cm HEAD净−2/+2/0、BODY+2/+11/+14各/128。预设40%点1cm HEAD+4/+1/+3、BODY+1/+6/+8各/128；pass减少3/9/5，但clear增加7/4/2。保留弱pass为低擦边独立读出Development候选，不据validation选上线工作点。

OR30% HEAD净−9/−5/−3、BODY−12/+1/+5各/384，clear仍58/6656，1cm HEAD损失8/5/3各/128；+3°独立读出1cm HEAD净−6/−5/−7，成本不稳定。M3、原5格/L2/A+LOCAL保留。32代表历史的世界坐标gauge检查中，实际FP16 token/M3输入与logits逐值相同；只排除所覆盖的坐标依赖，时间、yaw符号及姿态合同仍待查。

弱pass给新增2288个pass query槽label0/mask1；归一化使新增数据的原有效loss权重共同乘.985034，旧39936行不变。因此这是弱监督加共同重归一化的配方比较，未证明原mask导致擦边增加。沿用已消费模拟Development、有限AABB/背景及相关K/帧，无新光子、实机或保护480访问。[此前反事实与摘要负结果](../research/active/dtr-r0/nearfield/CNH_COUNTERFACTUAL_DEV_20261009.md)保留；侧移排序独立候选暂缓，不以弱负例失败为前提，旧扰动配方不重启。

科学阶段266.219/1200s；有记录CPU分析85.249/600s，初始监督映射时长未知。聚焦检查、6模型日程及258048行独立账本核验PASS；6个衍生数组约5.153GB经SHA留证后清理，观察、模型、scores、账本及失败记录保留。

## 停止与保留

[旧扰动训练](../research/active/dtr-r0/nearfield/CNH_QUERY_PERTURB_TRAIN_DEV_20261008.md)、T/T2、旧参照和门控的停止规则仍按各自run适用；不同机制开放。[480确认](../research/active/dtr-r0/nearfield/CNH_REAL_HEAD_CONFIRM_RESULTS_20261007.md)原身份保留，UNKNOWN/三态/query覆盖待验证。方向/Nymeria位移、−19°、设备/City/保护test/新UE及硬件第二阶段暂缓；101/101与53ms属A+LOCAL，M3/CNH实机效果未建立。

模拟AABB、有限背景、人工误差、相关K/帧及探索选择不构成实机、安全或独立确认。更新前CNH当前正文保留于Git d2afbaf1同路径；RGB子线继续沿用。

[路线当前页](../research/active/dtr-r0/CURRENT.md) · [RUNS](../research/active/dtr-r0/RUNS.md)。
