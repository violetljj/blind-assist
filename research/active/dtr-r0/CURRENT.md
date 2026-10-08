ALGORITHM_EXPLORATION / M3_RETAINED / HARDWARE_DEFERRED

# 前视障碍感知：当前状态

更新：2026-10-08。方向估计优先，冻结M3不替换。历史`DTR_R2_DYNAMIC_RETAINED`不表示恢复动态研究。
Status: `DTR_R2_DYNAMIC_RETAINED`（历史保留，不表示恢复动态研究）。

## 单种子方向扰动训练筛查未通过，停止配方

用户修订圆桌结论授权[小筛查](nearfield/CNH_QUERY_PERTURB_TRAIN_DEV_20261008.md)：Aug/Control同M3seed0权重、同27456原训练行、同2epoch；保持实际travel的M3±.33m监督，只扰当前query，零扰动原FP16输入逐值复现。冻结五种子M3和seed0另列。完整训练、32个unit评分和独立核验均完成。

测速后、训练及新结果前冻结8cal/24eval，评价47接触/224clear。所有16主格实际56/2240报警格、残差0；E1单路Aug35、Control40、M3为42，双路40/44/45。Aug相对M3 exact救回/损失单0/7、双1/7，超过≤2损失/net≥−2护栏；两配置三门槛全失败，停止本配方，不加seed/epoch或择扰动。GPU墙钟3087.703/3600秒、CPU科研核验<90/300秒，独立6类审计17.188秒PASS，计算释放。一件为2.128pp，绝对门槛非统计优势/非劣；工作点按评价clear选择，段数及clip报警另报。

方位标签微调及稀疏空间支持/关联L3暂停，不作下一优先。首轮不归因方向补偿，不做已撤回的历史位姿exact对照。恢复设备先核对真实处理与逐query真值；手机101/101及53ms是A+LOCAL，未建立M3/CNH实机效果。

## 保留的Development结果

[空间交叉诊断](nearfield/CNH_BEARING_SUPPORT_CROSS_DEV_20261008.md)全96unit/229接触/384clear/3840窗，字段完整NOT_EVALUABLE0。单簇改选四挑战净差单+1/−1/0/0、双全0；集合扩张与双分歧即弃权损正确较多，均未达直接候选投入条件。关联机会过线但竞争存在/整体更近也在全部正确事件成立，不证明可辨识性或收益，最新决定暂停该线。UNKNOWN不否决，粗簇非深度/实例，不改原指标。

[EMA门控L2](nearfield/CNH_EMA_GATED_BEARING_DEV_20261008.md)净支持增益及配对损失保留；[去平滑k1](nearfield/CNH_EMA_BEARING_DESMOOTH_DEV_20261008.md)未稳定改善，保留原L2，不追加本批平滑择参。[步态对EMA](nearfield/CNH_TORSO_EMA_COMPARE_DEV_20261008.md)单路降低优先级、双路混合保留，不自动冷启动或八格求和；输入契约不等，只比较整套方案。支持范围不等于物体归因/用户价值，均已消费Development。

## 证据身份与下一问题

[480新模拟确认](nearfield/CNH_REAL_HEAD_CONFIRM_RESULTS_20261007.md)不与本轮开发混同：E1较exact少39/51件及时（/1002），EMA收回25/34；同校准目标下实际误报不等。源约6分钟、可能同一人，EMA事后选择；非实机或新人群确认。BlindWays虽本轮不进训练/择参，已用于开发回放，不能再称独立确认。[读出账本](nearfield/CNH_EVENT_LEDGER_UNCERTAINTY_DEV_20261007.md)HB净及时区间跨零，启动误报保留。

不开同类确认批，设备继续暂缓。拿到带头朝向的真实行走数据后，先用既有replay评估行进意图估计；方向准确且覆盖充分事件若仍集中漏报，再提升读出优先级。候选query完整覆盖/UNKNOWN/三态待验证，稀疏helper未接pipeline，不授权CLEAR。UNKNOWN不保证安全；City、保护test、新UE及硬件第二阶段暂停。

[RUNS](RUNS.md) · [总决定](../../../docs/CURRENT_DECISION.md)。此前全文：Git `401467ce` 同路径。
