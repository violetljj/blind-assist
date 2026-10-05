EXTRINSIC_AUG_NOT_SUPPORTED / M3_RETAINED / HARDWARE_DEFERRED

# 前视障碍感知：当前状态

更新：2026-10-05。已消费Development；`V5_FROZEN_COMPLETE / HARDWARE_DEFERRED`。
Status: `DTR_R2_DYNAMIC_RETAINED`（仅为历史状态）。

## 最新结果：外参增强接续结束

保留冻结M3。原试点6/6还要求严格增加，数学上不可达；校准转弯查询clear 65→28/644、浅及时6→6/6。原FAIL与用户批准事后改为净≥0所得的修订PASS并存，不能隐去修订时点。

接续三seed 0/1/2，校准98000冻结增强阈值1.2440962676079044，再生成新99000全96×40评价。以下按single-M3 / dual-M3 / dual-增强 / single-增强列示：

|指标|共同分母|计数|
|---|---:|---|
|全模式物理clear首停|825|104 / 92 / 104 / 81|
|全模式查询clear首停|4424|192 / 229 / 249 / 161|
|浅0–2cm及时|32|28 / 26 / 30 / 29|
|中2–5cm及时|54|47 / 49 / 52 / 48|
|深>5cm及时|141|138 / 140 / 141 / 137|
|转弯查询clear首停|1299|24 / 81 / 57 / 16|
|转弯浅及时|14|13 / 8 / 12 / 14|

增强dual转弯clear相对single为2.375倍，超过1.5倍（36次）上限，最终NOT_SUPPORTED；不按评价结果调阈值。增强反侧容限0/10/15/20°均48/48、25°47/48、30°14/48，复用旧观测，不是独立验证。历史V训练unit、镜像和yaw同时不同，不能作纯外参消融。[完整六臂、左右转、计划修订与效率](nearfield/CNH_EXTRINSIC_AUG_20261006.md)

## 仍影响下一决定的证据

- 公开G3融合97000：single / OR / G3查询clear首停219 / 271 / 243（n4443），浅及时25 / 25 / 23（n29）。成本门通过、浅及时下降6.90pp超过3pp，GATING_NOT_SUPPORTED；原判读保留。[融合报告](nearfield/CNH_DUAL_GATED_FUSION_20261005.md)
- 覆盖拆分：低可见88条均曾曝光，VD及时26/88；最后可见距离中位1.93m，对照高可见296条为0.97m。FOV_OUT不是全程无信息，下一机制关注近距持续曝光。[拆分](nearfield/CNH_FOV_FAILURE_SPLIT_20261005.md)
- 修正合成提示中dual消除70.1–89.6%几何缺口，软件仍受提示负担和纵向覆盖约束。不是人体/硬件证据；1.7m/0.4s渐进回正仅为模拟参照。[合成诊断](nearfield/CNH_COVERAGE_CUE_CORRECTED_20261005.md) · [回正参照](nearfield/CNH_COVERAGE_POLICY_20261004.md)

## 下一问题、权限与边界

待决定：双路残余转弯代价的机制是否值得继续投入；单路增强收益若要归因，需要匹配训练条件的对照。当前只整理工作流，不开新实验，不续用结束的4小时/150分钟预算。授权问题与预算内自主推进，变更问题、冻结指标和预算按[工作流](../../WORKFLOW.md)处理。

R读出、T2、位姿平均、融合及外参本轮停止，不泛化为永久禁训。[读出收尾](nearfield/CNH_READOUT_CLOSEOUT_20261004.md)和原失败保留。真实回放待设备会话，不继续01/04分析。[审计](nearfield/CNH_REPLAY_EVALUATION_AUDIT_20261003.md)

三级真值：侵入必须报，身体外0–10cm只报告，更远或其他高度计清晰误报。代理分钟不是人体负担；oracle不是物理上界，密输出不增加8×8物理分辨率，UNKNOWN不等于安全。无跨源/实机效果；City、保护test、新UE及硬件第二阶段暂停。

[RUNS](RUNS.md) · [主张台账](THESIS_CLAIMS_20260927.md) · [总决定](../../../docs/CURRENT_DECISION.md)。整理前完整结果链与链接：Git revision `4f174009da62a8fcd9f219bb6758375f3f1ce2aa` 的 `research/active/dtr-r0/CURRENT.md`；旧待决状态不恢复授权。
