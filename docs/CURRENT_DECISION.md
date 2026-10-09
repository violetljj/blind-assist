# 当前研究决定

更新：2026-10-09。主线为盲杖互补前视感知，M3和A+LOCAL保留；原5格局部融合和L2保留。

RGB独立子线：[真实连续段评价](../research/active/dtr-r0/nearfield/RGB_BODY_QUERY_EVAL_DEV_20261009.md)。真实定量为硬目标，补充ToF/CNH主要模拟证据。45帧270格为59正/25空域负/186UNKNOWN，空域负剔除，GT定位oracle只评价参考域几何。共同18帧Depth Pro原图/128 IoU .3284/.3261，居中适配更差；VDA全45帧完成但近盒正格0/10，高分辨率/时序收益未建立。金属细杆漏标，下一补真实细障碍标注和非空角域负例，再训练可变query机制；事件提前量不可评。继承旧BodyQuery、对照COPILOT，SANPO重新纳入候选；CNH原预算/停止规则/M3保留。

## 当前：普通扩充有收益，完整交叉反事实未增加稳定收益

[三臂三seed实验](../research/active/dtr-r0/nearfield/CNH_COUNTERFACTUAL_DEV_20261009.md)固定简化8833参数bin-token、BCE、7488步及配对初始化；普通与完整交叉反事实等量扩充。BCE未使用pair身份，本轮检验数据覆盖增量。

不同几何背景验证：普通扩充OR在旧5格融合上H315/313/323、B267/257/282各/384，旧融合298/244；clear均58/6656，pass却由60升至89/103/115各/256。+3°沿用ideal校准阈值，clear1297/1261/1352，旧融合1245；不支持扰动同成本。CF−普通独立读出H−2/+13/0、B0/−4/−7，OR H−7/0/+2、B−10/−2/−10，未获稳定增量。

保留普通扩充为Development候选，不升级M3/原5格/L2。新背景族先分train/cal/validation，同族干预不跨分区；只证明新增族隔离，继承39936训练行及M3的背景相似性未审计。AABB有限几何、相关K/帧及仿真噪声不构成真实材质、实机或独立确认。校准后阈值固定，验证未调参；下一优先公共query/pose输入合同与擦边成本：+3°基线clear M3由64→1255、旧融合58→1245；这不重启旧扰动训练。当前完整交叉配方本轮收敛，未来不同机制开放。

固定3标量摘要（first/peak/signed-total）对完整bin base的独立读出H净+11/−46/−49、B−29/−13/−30，clear102/104/140（base87/46/49）/6656；OR H+2/−6/−10、B−5/−3/−5。保留完整bin读出，但这不是所有压缩不可行或bin机制的普适证明。实验阶段含失败累计1190.955/3600s（失败10.187s）；13项聚焦检查、评价内复算及96训练epoch日程复核通过。12个衍生数组10.305GB及bindings经SHA留证后清理，raw/模型/ledger/失败保留，73728行独立账本核验PASS。

## 停止与保留

[旧扰动训练](../research/active/dtr-r0/nearfield/CNH_QUERY_PERTURB_TRAIN_DEV_20261008.md)、T/T2、旧参照和门控的停止规则仍按各自run适用；不同机制开放。[480确认](../research/active/dtr-r0/nearfield/CNH_REAL_HEAD_CONFIRM_RESULTS_20261007.md)原身份保留，UNKNOWN/三态/query覆盖待验证。方向/Nymeria位移、−19°、设备/City/保护test/新UE及硬件第二阶段暂缓；101/101与53ms属A+LOCAL，M3/CNH实机效果未建立。

模拟AABB、有限背景、人工误差、相关K/帧及探索选择不构成实机、安全或独立确认。更新前CNH当前正文保留于Git 3564255a同路径；RGB子线继续沿用。

[路线当前页](../research/active/dtr-r0/CURRENT.md) · [RUNS](../research/active/dtr-r0/RUNS.md)。
