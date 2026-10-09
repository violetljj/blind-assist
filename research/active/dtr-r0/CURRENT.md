ALGORITHM_EXPLORATION / M3_RETAINED / HARDWARE_DEFERRED

# 前视障碍感知：当前状态

更新：2026-10-09。盲杖互补前视感知主线；原M3、5格局部融合、L2和A+LOCAL保留。

RGB独立子线：[尺度、场景信息与度量表征诊断](nearfield/RGB_BODY_QUERY_METRIC_DIAGNOSTIC_DEV_20261009.md)。沿用3RScan96帧/12环境7/2/3隔离，1440格825正/615UNKNOWN，无满足全部可观测采样射线free判据的负query。冻结Depth Pro的RGB预测深度+learned query读出，validation正/free query-ray1071771/1795858，召回/误支持.70048/.10663，匹配geometry .61966/.11535；新增32特征较弱(.65990/.10880)。depth-only打乱退化至.64691/.19574，相同head ray/query的417179近邻反标签pair排序.84421，支持场景信息。三环境和全距离带取舍并列，第三环境召回下降，前两环境误支持增加；cal阈值固定，不以validation改模型。GPU166.096/600s，独立SHA/预测复算通过。保留简洁预测深度读出候选，下一充分参考负query与步行/细结构覆盖；真实射线Development非完整身体/整盒/事件证据。旧两query/SANPO负结果及停止规则、真实硬目标、CNH预算保留。

## 当前：错位query改变完整目标几何，总支持量不能解释漏检

[逐帧机制链](nearfield/CNH_PASS_MECHANISM_CHAIN_DEV_20261009.md)固定seed956、30%点，有目的选择19个救/损/均漏事件，缺组保留。重建209历史×两分支=418输入；两个冻结模型1672个同输入回放分数与保存值逐值一致。无训练、阈值搜索、−3°、新光子或设备采集，不把选例当全seed确认。

全64个1cm场景、f3–13共1408几何行：+3°正x侧HEAD/BODY各16场景在全11帧均无完整目标与contact query交集；负侧各16均仍相交。正侧f10最小x=.36549m，仍有8/11帧与±.4m扩展区相交。表面裁剪与实体交集分开核验；原身体contact真值保持，不能因离开错位query改负标签或判弱监督无价值。

19选例中17个历史正支持增加，包括正侧HEAD全部6个。唯一负侧BODY loss在f8历史带符号支持增加8.130，weak阈值余量+.350→−.047；负侧HEAD rescue也有支持下降却跨入报警的反例。全域membership/gated观察含背景噪声，不能替代bin/zone模式或作为目标专属证据；保留完整bin与[双条件独立读出候选](nearfield/CNH_PASS_POSE_DIAGNOSTIC_DEV_20261009.md)，M3/原5格/L2不升级。

ideal按height×shape×background×rho固定后左右可比cell为0/64，全局rho重叠不能消除条件混杂。此静态fixture两高度共享sensor→query及x范围，未见高度专属横移；不认证实物pose/time合同。下一优先明确身体走廊及方向不确定性合同，冻结读出、校准成本后小试多query/区间汇总；必要bin/zone响应辅助定位。均未执行，不重启旧扰动训练或用零损失门槛否定候选。核验与实际成本见报告。

## 停止与保留

[旧扰动训练](nearfield/CNH_QUERY_PERTURB_TRAIN_DEV_20261008.md)、T/T2、旧参照和门控的停止规则仍按各自run适用；不同机制开放。[480确认](nearfield/CNH_REAL_HEAD_CONFIRM_RESULTS_20261007.md)原身份保留，UNKNOWN/三态/query覆盖待验证。方向/Nymeria位移、−19°、设备/City/保护test/新UE及硬件第二阶段暂缓；101/101与53ms属A+LOCAL，M3/CNH实机效果未建立。

模拟AABB、有限背景、人工误差、相关K/帧及探索选择不构成实机、安全或独立确认。更新前CNH当前正文保留于Git 183358af同路径；RGB子线继续沿用。

[RUNS](RUNS.md) · [总决定](../../../docs/CURRENT_DECISION.md)。
