# GRAIL M1 V2 Off-Center Visual Protocol

日期：2026-08-25（Asia/Hong_Kong）

状态：`FROZEN_BEFORE_V2_COLLECTION_OR_OUTCOME / TARGET_CENTERING_LEAK_REMOVED / NEW_TEST_SALT / B0_B1_B2_GRAIL / STOP_BEFORE_M2_ON_FAIL`

V1 Development 因 query yaw 始终朝向 target 而使 B2 可忽略 goal/reference，formal test 未打开。V2 保持 ProcTHOR source、oracle candidate masks、native pose truth、冻结 DINOv2-S/Depth-Anything-V2-S、reference-goal、heads、K=3、loss、interaction-pose 容差、wrong-target/absence/permutation 指标不变；唯一数据语义修正是 target 不再固定居中。

对每个 target，query position 仍在 1.75--4.0 m reachable set 中按 sample hash 排序；每个位置只从朝向 target 的 `{-30,0,30}` 度偏移中按 sample hash 选择一个 yaw，采用首个 target 在渲染中可见的位置。选择不按 bbox 中心误差或后续模型结果排序。早期 train/dev runtime 曾用五 yaw 逐一试探，因单 house 成本过高在模型/正式 test outcome 前中止；其 partial checkpoint 不进入 V2b 数据，test roster 不变。

V2 使用新 val/test salt。train=`529,407,296,206,768,327,857,482,372,825,628,477,631,708,908,9,696,469,512,368,320,485,486,555`；dev=`663,513,636,403,860,910`；test=`599,898,561,911,339,14,513,169,203,419,651,478`。test 排除 M0 全部已消费 houses 与 V1 已冻结 test roster。冻结 lock 见 [`grail_m1_v2_lock.json`](../../../scripts/research/grail/grail_m1_v2_lock.json)；由 [`freeze_grail_m1_v2.py`](../../../scripts/research/grail/freeze_grail_m1_v2.py) 对固定数据生成的 V2b 完整 manifest SHA-256 为 `1302f0c86eeb0cead5ca5701d7bf2ccf31b1ae48aa96a3422856b0b7be209c76`。

正式 test 必须至少包含 96 positives、24 wrong-target cases、96 absence cases。GRAIL pose success 必须比 `max(B0,B1,B2)` 高至少 10 个百分点；wrong-target rate 不得比最强 candidate-selecting baseline 高超过 2 点；absence false commit 不得高于最强 pose baseline；permutation consistency 必须 100%。dev 只冻结 checkpoint 与 thresholds；任何正式门失败即 `STOP_BEFORE_M2`，不重跑 V2 test。

Claim ceiling：synthetic ProcTHOR RGB + oracle candidate masks + simulator-native interaction-pose truth，reference-goal mode；无自然场景、proposal、text-goal、Android、用户、产品或安全结论。
