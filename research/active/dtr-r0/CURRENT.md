ALGORITHM_EXPLORATION / M3_RETAINED / HARDWARE_DEFERRED

# 前视障碍感知：当前状态

更新：2026-10-08。本轮对齐直行感知边界完成，冻结M3不替换。
Status: `DTR_R2_DYNAMIC_RETAINED`（历史保留，不表示恢复动态研究）。

## 最新决定

用户授权暂缓方向估计/实机工作，先固定直行与yaw对齐，复用旧证据、补有限场景网格，比较一个安装改动。[边界报告](nearfield/CNH_ALIGNED_BOUNDARY_DEV_20261008.md)完成168cube×K4，HEAD/BODY各144接触事件、48clear×K4×13=2496报警时间格；单传感器−10°对几何预选−19°，exact相对位姿oracle诊断，无训练。

同28/2496实际报警格、残差0，HEAD31→29（救6损8），BODY31→52（救25损4）。每高度损失≤2及净差门槛未过，DO_NOT_ADVANCE_CURRENT_PITCH，保留原−10°。原阈值下成本28→98；匹配时间格后clear段16→20、clip15→20，不宣称提醒次数相等。CUDA渲染/采样/推理墙钟133.594/1800s，两臂CPU期望parity、独立合同及结果重算通过，资源释放。

所有接触cube及时截止前曾可见，4cm及时两角度均7/144；10cm原55、候选74/144。这是新有限模拟网格，不能混入旧直行正视233/234或称普遍准确率。下一用现有光子检验小目标可判别信号，区分观测与读出机会；不再扫角度、不自动训练。原始光子、分数、首命中数、CSV/图及冻结PLAN均在`artifacts.local/work/cnh-aligned-boundary-dev-20261008/`。

## 其余路线保留

[扰动训练](nearfield/CNH_QUERY_PERTURB_TRAIN_DEV_20261008.md)停止：同47事件、56/2240报警格，单Aug/Control/M3=35/40/42，双40/44/45，exact各损7，三条件失败；不加seed/epoch或换分布。本批仅各1件方向失配损失，不足判断补偿能力。方位微调和稀疏空间/关联L3暂停；原L2净增益与配对损失、去平滑负结果、[空间诊断](nearfield/CNH_BEARING_SUPPORT_CROSS_DEV_20261008.md)及步态/EMA整套输入对照保留。

[480新模拟确认](nearfield/CNH_REAL_HEAD_CONFIRM_RESULTS_20261007.md)身份不改：E1较exact少39/51及时（/1002），EMA收回25/34，同目标下实际误报不等；来源窄、EMA事后选，不是新人群/实机证据。后续回放与筛查属于已消费Development；BlindWays不能再称独立确认。读出HB净及时区间跨零，UNKNOWN/三态/query覆盖待验证。

[Nymeria样本/IMU](nearfield/CNH_NYMERIA_SAMPLE_AUDIT_DEV_20261008.md)合法段整体注册为离线参考，原时钟异常保留；原始右首60秒→CPF姿态七处/35项前缀及fixtures通过，姿态准确度未评，位置/PDR未实现、E1整链NOT_EVALUABLE。未来骨盆和闭环轨迹只作参考；因果位移接口/参与者覆盖工作本轮暂缓，不自动重跑配方。

设备恢复先核对M3真实处理与逐query真值；手机101/101及53ms是A+LOCAL，M3/CNH实机效果未建立。设备、City、保护test、新UE及硬件第二阶段暂停。

[RUNS](RUNS.md) · [总决定](../../../docs/CURRENT_DECISION.md)。压缩前全文：Git `fb8641cb` 同路径。
