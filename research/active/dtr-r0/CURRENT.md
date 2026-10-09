ALGORITHM_EXPLORATION / M3_RETAINED / HARDWARE_DEFERRED

# 前视障碍感知：当前状态

更新：2026-10-09。盲杖互补前视感知主线；原M3、5格局部融合、L2和A+LOCAL保留。

RGB独立子线：[冻结迁移与锚点旁路](nearfield/RGB_BODY_QUERY_TRANSFER_DEV_20261009.md)。原模型/归一化/cal实际cutoff冻结，无训练；8个未用3RScan环境64帧，正/free query-ray3400061/4344666，depth-only召回/误支持.59236/.08489 vs geometry .50464/.09513，6/8环境TP净增、2/8净减；相对raw DP .64005/.15911是取舍。ARKitScenes iPadPro单捕获16帧，depth-only .29567/.04188 vs geometry .42204/.05606，两项同时改善与充分近场增量未复现；场景、原图分辨率/K、预测距离及读出适配原因未定。28全采样free格各臂支持0/28，非整盒/身体误报；步行提前量仍缺事件/身体参考。同源实测锚点旁路改善直接几何但不证明ToF增量。GPU154.136/900s，官方数据357569481B/2GiB，独立保存预测复算及旧帧CUDA逐值核验PASS。下一高分辨率不同相机多环境与预测质量诊断、完整query/步行参考；旧两query/32特征不续训，原负结果/真实硬目标/CNH预算保留。

## 当前：中心锚定保住报警，方向增量仍有成本

[校准尾部与中心锚定读出](nearfield/CNH_DIRECTION_ANCHOR_DEV_20261009.md)完成三seed×control/weak_pass×20/30/40%点×ideal/+3°，36center参照与72新政策共108 validation格、54本轮cal记录。仅复用冻结方向raw分数，无训练或模型回放。中心锚定OR：中心θ保持single，偏离两侧逐帧max后原last5平滑；ideal-cal中心未报警联合clear slot及整段未报警pass clip的最大限制分数取nextafter，覆盖f3–15，校准零增量。高度混合为HEAD .25/.50/.25、BODY center，raw后原平滑，在原joint预算下只选一个共享θ。

尾部诊断：max的18个cal阈值全部上移，前驱阻断均为pass clip预算；同θ分数单调且报警无损失，损失来自阈值变化。weighted阈值12升6降，分数和阈值两步分别留证。具体尾部样本/方向仅定位约束，不证明侧墙回波来源。

OR全格对center零损失是超集结构，不能推广为validation零成本。40% weak +3°的1cm H救/损3/0、0/0、5/0，B2/0、4/0、4/0各/128；clear clip148/120/140→155/133/150各/512，pass136/122/122→136/124/128各/256。ideal该点H最多救1、B无新增，pass仍增4/1/2。hybrid +3°浅H3/1、4/1、4/0、B0/0、0/2、0/0各/128；ideal浅H0/6、1/3、1/4、B0/6、0/3、0/0。BODY raw用中心仍会受共享θ影响；不设hybrid零损失门槛，不由validation选赢家或升级。

8项policy/5项tail聚焦检查与独立账本审计PASS；完整82944政策事件、55296分解事件及4548尾部单位保留。CPU实测21.749s+根端保守扣额40s，总61.749/600s，GPU0。原身体truth、完整bin、M3/原5格/L2/weak中心与加权候选保留；人工δ3°与已消费相关模拟Development不构成实机确认。下一围绕有依据的方向误差与独立背景复核附加报警成本；零增量OR严于用满剩余预算，后者未运行，不能判为无空间。完整表、两类clip/slot单位、失败/来源及未验证项见报告。

## 停止与保留

[旧扰动训练](nearfield/CNH_QUERY_PERTURB_TRAIN_DEV_20261008.md)、T/T2、旧参照和门控的停止规则仍按各自run适用；不同机制开放。[480确认](nearfield/CNH_REAL_HEAD_CONFIRM_RESULTS_20261007.md)原身份保留，UNKNOWN/三态/query覆盖待验证。方向/Nymeria位移、−19°、设备/City/保护test/新UE及硬件第二阶段暂缓；101/101与53ms属A+LOCAL，M3/CNH实机效果未建立。

模拟AABB、有限背景、人工误差、相关K/帧及探索选择不构成实机、安全或独立确认。更新前CNH当前正文保留于Git a91cb6c7同路径；RGB子线继续沿用。

[RUNS](RUNS.md) · [总决定](../../../docs/CURRENT_DECISION.md)。
