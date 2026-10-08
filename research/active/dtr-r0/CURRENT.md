ALGORITHM_EXPLORATION / M3_RETAINED / HARDWARE_DEFERRED

# 前视障碍感知：当前状态

更新：2026-10-08。方向估计优先，读出线保留不上调；冻结M3不替换。
Status: `DTR_R2_DYNAMIC_RETAINED`（历史保留，不表示恢复动态研究）。

## 最新决定：去平滑未通过，保留原L2

[唯一k=1对照](nearfield/CNH_EMA_BEARING_DESMOOTH_DEV_20261008.md)全96unit/229接触/384clear/3840窗，只取消额外输出分数平滑；EMA全部提示/首次逐值不变，单输出内部最多8帧观测保留。k1−原L2四挑战clean单0/+5/−3/−2、unique0/+5/−2/−2，双均−2/−1/−2/−2。两配置均RETAIN_ORIGINAL_L2，不追加k2/k3或本批择参。CPU分析7.906/180秒、5测试及480raw/90分层独立重算通过（1.375秒），无新增推理，计算结束。

>10°yaw层没有一致收益，stable标签层仍有共同错误，大margin也不保证支持；仅降低简单去平滑优先级，不能确认因果或错物体归因。下一优先考虑稀疏空间支持/关联，先冻结CNH噪声支持、唯一关联、多簇/跨界弃权；L3本轮未实现且独立开放。主机增量推理成本待实测，取消平滑不能证明便宜；手机验证仍暂缓。

原[EMA门控L2](nearfield/CNH_EMA_GATED_BEARING_DEV_20261008.md)相对L1支持单+22/+39/+32/+33、双+21/+30/+29/+25与局部损失仍保留。支持范围不等于定位/物体归因/用户收益，原[三查询独立触发负结果](nearfield/CNH_SECTOR_NOTICE_DEV_20261008.md)不混同补充标签；本次k1失败也不排除其他时间或空间机制。Development人工yaw及共同理想原点边界保留，不接手机/不晋升确认。

## 保留决定：EMA对照完成

[步态与EMA检查](nearfield/CNH_TORSO_EMA_COMPARE_DEV_20261008.md)复用96unit、229事件/384clear，各条件实际FA124/4992且残差0。四挑战gait−EMA单路−2/−2/−8/−1，降低步态优先级；双路+5/+1/−5/0混合保留，两配置均不自动冷启动/归附录。配对在报告，不求八格和；零界线非统计优势/非劣。此工作点按FA匹配，与新粗方位总提示预算不混用。

E1/躯干/gait理想60Hz fullclip与EMA带噪5Hz窗口重置输入不等；EMA保留相对旋转、适配理想骨盆原点，非原确认零平移Q。只比较整套方案，不隔离机制。人工yaw、future-conditioned模拟、源重叠/Pxx映射未知及评价选点边界保留；成本与5契约/独立核验在报告，计算结束。[原物理yaw](nearfield/CNH_TORSO_HEAD_YAW_DEV_20261007.md)不改。

## 保留的证据与下一问题

[480新模拟unit确认](nearfield/CNH_REAL_HEAD_CONFIRM_RESULTS_20261007.md)保留原确认身份，不与后续已消费Development混同。同校准2.5%目标下，E1较exact少39/51件及时（/1002），EMA收回25/34，约三分之二；双路及时优势未建立，实际误报并不相等。原θ并集判畅通使静默68→19/21 of1002，UNKNOWN增加15.64/14.23pp，及时不变。头动约6分钟、可能同一人；非实机、新人群确认，EMA事后选择。[章节](thesis/CNH_READOUT_COVERAGE_TRISTATE_CHAPTER_20261006.md)与[答辩问答](thesis/CNH_DEFENSE_QA_20261006.md)已整合。

[读出账本/区间](nearfield/CNH_EVENT_LEDGER_UNCERTAINTY_DEV_20261007.md)保留280事件/728控制：HB策略净及时区间跨零，启动误报与5%损失保留。读出、覆盖、三态有阶段性结果；候选query下完整覆盖/UNKNOWN/三态仍待验证，稀疏helper未接pipeline，不授权CLEAR。

不开同类确认批，设备继续暂缓。行进意图估计为开放问题；拿到带头朝向的真实行走数据后先用既有replay评估估计器；若机制核对显示漏报集中在方向准确且覆盖充分事件，再提高读出优先级。UNKNOWN不保证安全，City、保护test、新UE及硬件第二阶段暂停。

[RUNS](RUNS.md) · [总决定](../../../docs/CURRENT_DECISION.md)。此前全文：Git `6ebb8338` 同路径。
