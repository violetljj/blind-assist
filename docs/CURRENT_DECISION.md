# 当前研究决定

更新：2026-10-07。唯一研究主线为盲杖互补的前视障碍感知。用户持续目标为“持续进行算法探索”；保留冻结M3、手机A/A+LOCAL和已有论文证据。

历史路线状态：`L10_R0_PAUSED / DTR_R2_DYNAMIC_RETAINED`。

## 当前算法问题

[最新独立单调标定](../research/active/dtr-r0/nearfield/CNH_QUERYWISE_CALIBRATION_DEV_20261007.md)消除跨高度否决，但原369事件无启动及时query193、event192、原max202，实际误报2.367/2.390/2.629%；HB36事件23/24 vs max25。没有稳健改进，M3保留。

独立精确oracle允许看评价答案选两个固定高度阈值：逐fold误报≤2.5%时最多204/369无启动及时，matched max195，5%278 vs270。受检重标度家族空间有限；不覆盖跨fold预算重分配，不是部署或原观测信息上限。此前[双高度](../research/active/dtr-r0/nearfield/CNH_DOUBLE_HEIGHT_DEV_20261007.md)10unit物理反例仍说明旧浅树不可直接继承。

下一步比较当前两分数、加当前原始径向摘要、再加过去径向变化，分开检验分数压缩与接近时序信息。使用带噪相对旋转、禁止travel/未来路径特征；同unit划分与低容量模型，HB继续作已消费结构检查。新原始观测实验尚未运行。

## 授权与继承

本轮CPU标定38.16/300秒，独立oracle0.422/120秒；5测试、525项重算及219个穷举案例通过，无GPU。持续目标授权内继续探索，不沿用已结束子实验预算或改写其失败；后续运行前明确输入、范围与判断问题。

此前R/T2/V、外参、覆盖、方向扇形/加权和短时预测各轮的停止范围保留。新事件读出是输出端监督和新接触支持范围问题，不恢复旧结构搜索、旧门槛或已停止路径延长。[路线当前页](../research/active/dtr-r0/CURRENT.md)负责详细执行与数字。

## 既有证据与边界

真实头动确认480新模拟单位已完成，作为指定模拟与HEADS-UP来源的论文证据，登记cnh-rhc-20261007为completed；不是硬件或人群总体确认。校准目标2.5%不等于评价实际误报相同。M3保持基线，EMA、双路及新阈值不自动继承。

[章节](../research/active/dtr-r0/thesis/CNH_READOUT_COVERAGE_TRISTATE_CHAPTER_20261006.md)维持读出/三态、合成对照、真实头动确认和方向边界的证据链。BlindWays缺头部朝向，不能复核HEADS-UP的P1/P2；UNKNOWN不保证安全。City、保护test、新UE采集及硬件第二阶段仍暂停，实机回放待会话。

[设备状态](PROJECT_STATE.md) · [主张台账](../research/active/dtr-r0/THESIS_CLAIMS_20260927.md)。此前全文保留于Git bd8852e2 同路径。
