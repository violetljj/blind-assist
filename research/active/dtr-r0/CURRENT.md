ALGORITHM_EXPLORATION / M3_RETAINED / HARDWARE_DEFERRED

# 前视障碍感知：当前状态

更新：2026-10-08。方向估计优先，读出线保留不上调；冻结M3不替换。
Status: `DTR_R2_DYNAMIC_RETAINED`（历史保留，不表示恢复动态研究）。

## 最新决定：粗方位首检完成

[固定头部三查询提示](nearfield/CNH_SECTOR_NOTICE_DEV_20261008.md)复用96物理yaw缓存unit，无新渲染/训练/采集；229事件、384严格clear、3840短窗。按全13输出发出的提示条目总数匹配E1/EMA/top1/all，40工作点预算残差0。这是模拟消息计数，非语音时长、长期频率或部署校准。

top1四挑战首条及时且受接触物方位范围支持，相对同预算EMA单路净−38/−39/−52/−62，双路−28/−39/−41/−52；只看及时检测也全部更少，全sector读出更差。运行前规则按单/双均降低当前读出优先级，本次不进入去重/NAT重渲染或本批角度/阈值择优。几何保留跨界多扇区与3个box1接触目标；支持范围不等于唯一定位/物体归因。11项契约、480raw块/40工作点/4580事件核验通过；prepare2.578s、推理611.531s、分析记账26.374s，计算已结束。下一可考虑原始ToF/重投影空间支持直接读方位，未实施；不否定所有方位方法或断言信息不足。

## 保留决定：EMA对照完成

[步态与EMA检查](nearfield/CNH_TORSO_EMA_COMPARE_DEV_20261008.md)复用96unit、229事件/384clear，各条件实际FA124/4992且残差0。四挑战gait−EMA单路−2/−2/−8/−1，降低步态优先级；双路+5/+1/−5/0混合保留，两配置均不自动冷启动/归附录。配对在报告，不求八格和；零界线非统计优势/非劣。此工作点按FA匹配，与新粗方位总提示预算不混用。

E1/躯干/gait理想60Hz fullclip与EMA带噪5Hz窗口重置输入不等；EMA保留相对旋转、适配理想骨盆原点，非原确认零平移Q。只比较整套方案，不隔离机制。人工yaw、future-conditioned模拟、源重叠/Pxx映射未知及评价选点边界保留；成本与5契约/独立核验在报告，计算结束。[原物理yaw](nearfield/CNH_TORSO_HEAD_YAW_DEV_20261007.md)不改。

## 保留的证据与下一问题

[480新模拟unit确认](nearfield/CNH_REAL_HEAD_CONFIRM_RESULTS_20261007.md)保留原确认身份，不与后续已消费Development混同。同校准2.5%目标下，E1较exact少39/51件及时（/1002），EMA收回25/34，约三分之二；双路及时优势未建立，实际误报并不相等。原θ并集判畅通使静默68→19/21 of1002，UNKNOWN增加15.64/14.23pp，及时不变。头动约6分钟、可能同一人；非实机、新人群确认，EMA事后选择。[章节](thesis/CNH_READOUT_COVERAGE_TRISTATE_CHAPTER_20261006.md)与[答辩问答](thesis/CNH_DEFENSE_QA_20261006.md)已整合。

[读出账本/区间](nearfield/CNH_EVENT_LEDGER_UNCERTAINTY_DEV_20261007.md)保留280事件/728控制：HB策略净及时区间跨零，启动误报与5%损失保留。读出、覆盖、三态有阶段性结果；候选query下完整覆盖/UNKNOWN/三态仍待验证，稀疏helper未接pipeline，不授权CLEAR。

不开同类确认批，设备继续暂缓。行进意图估计为开放问题；拿到带头朝向的真实行走数据后先用既有replay评估估计器；若机制核对显示漏报集中在方向准确且覆盖充分事件，再提高读出优先级。UNKNOWN不保证安全，City、保护test、新UE及硬件第二阶段暂停。

[RUNS](RUNS.md) · [总决定](../../../docs/CURRENT_DECISION.md)。此前全文：Git `6ebb8338` 同路径。
