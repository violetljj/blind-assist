ALGORITHM_EXPLORATION / M3_RETAINED / HARDWARE_DEFERRED

# 前视障碍感知：当前状态

更新：2026-10-09。盲杖互补前视感知主线；原M3、5格局部融合、L2和A+LOCAL保留。

RGB独立子线：[输入配对与可调基线](nearfield/RGB_BODY_QUERY_INPUT_BASELINE_DEV_20261009.md)。冻结模型/归一化/cal实际cutoff，无训练；同capture8个640→256同帧配对，共同正/free query-ray688483/930318，depth-only召回/误支持.34273/.04979 vs .34240/.05062，已知TP≥16见证均45/74，geometry均.45414/.06479；仅很小变化，不定位旧ARKit退化原因，原16帧结果不能直接作因果比较。16/16预测有空间变化，不称常数坍缩。原cal选择δ=0退化为raw本身，尚未回答学习读出的额外价值；下一修正强几何比较合同、检验不同度量深度表征/训练分布，补多相机多环境与身体/步行参考。TUM源4194304B partial，270.830/300s网络限额内未完成，真实帧/GPU NOT_RUN，进程释放。GPU61.358/1200s、接收8697128B/4GiB；配对4臂全量独立复算/冻结身份及新增行为检查PASS。旧新3RScan增量、ARKit负结果及28采样free支持0/28保留，本轮两路各0/15，非整盒/身体误报；提前量仍缺身体/事件参考，同源锚点非ToF增量。旧两query/32特征不续训，真实硬目标/CNH预算保留。

## 当前：中心锚定保住报警，方向增量仍有成本

[校准尾部与中心锚定读出](nearfield/CNH_DIRECTION_ANCHOR_DEV_20261009.md)完成三seed×control/weak_pass×20/30/40%点×ideal/+3°，36center参照与72新政策共108 validation格、54本轮cal记录。仅复用冻结方向raw分数，无训练或模型回放。中心锚定OR：中心θ保持single，偏离两侧逐帧max后原last5平滑；ideal-cal中心未报警联合clear slot及整段未报警pass clip的最大限制分数取nextafter，覆盖f3–15，校准零增量。高度混合为HEAD .25/.50/.25、BODY center，raw后原平滑，在原joint预算下只选一个共享θ。

尾部诊断：max的18个cal阈值全部上移，前驱阻断均为pass clip预算；同θ分数单调且报警无损失，损失来自阈值变化。weighted阈值12升6降，分数和阈值两步分别留证。具体尾部样本/方向仅定位约束，不证明侧墙回波来源。

OR全格对center零损失是超集结构，不能推广为validation零成本。40% weak +3°的1cm H救/损3/0、0/0、5/0，B2/0、4/0、4/0各/128；clear clip148/120/140→155/133/150各/512，pass136/122/122→136/124/128各/256。ideal该点H最多救1、B无新增，pass仍增4/1/2。hybrid +3°浅H3/1、4/1、4/0、B0/0、0/2、0/0各/128；ideal浅H0/6、1/3、1/4、B0/6、0/3、0/0。BODY raw用中心仍会受共享θ影响；不设hybrid零损失门槛，不由validation选赢家或升级。

8项policy/5项tail聚焦检查与独立账本审计PASS；完整82944政策事件、55296分解事件及4548尾部单位保留。CPU实测21.749s+根端保守扣额40s，总61.749/600s，GPU0。原身体truth、完整bin、M3/原5格/L2/weak中心与加权候选保留；人工δ3°与已消费相关模拟Development不构成实机确认。下一围绕有依据的方向误差与独立背景复核附加报警成本；零增量OR严于用满剩余预算，后者未运行，不能判为无空间。完整表、两类clip/slot单位、失败/来源及未验证项见报告。

## 停止与保留

[旧扰动训练](nearfield/CNH_QUERY_PERTURB_TRAIN_DEV_20261008.md)、T/T2、旧参照和门控的停止规则仍按各自run适用；不同机制开放。[480确认](nearfield/CNH_REAL_HEAD_CONFIRM_RESULTS_20261007.md)原身份保留，UNKNOWN/三态/query覆盖待验证。方向/Nymeria位移、−19°、设备/City/保护test/新UE及硬件第二阶段暂缓；101/101与53ms属A+LOCAL，M3/CNH实机效果未建立。

模拟AABB、有限背景、人工误差、相关K/帧及探索选择不构成实机、安全或独立确认。更新前CNH当前正文保留于Git a91cb6c7同路径；RGB子线继续沿用。

[RUNS](RUNS.md) · [总决定](../../../docs/CURRENT_DECISION.md)。
