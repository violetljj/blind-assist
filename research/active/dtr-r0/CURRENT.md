# 前视障碍感知：当前状态

更新：2026-09-24

Status: `DTR_R2_DYNAMIC_RETAINED` (historical algorithm; no new promotion).

当前推进的是 **cane-complementary forward perception**，不是重新启动历史 DTR 动态研究。[中文总览](../../../docs/PROJECT_STATE.md)说明当前手机系统、代表性指标和研究范围。

## 系统与保留对照

- 手机 v10.15.1：原首页手动开始 A+LOCAL 实验模式；保留基础 ToF 切换。名义几何适配未完成物理标定。[硬件说明](../../../docs/HARDWARE_OBSTACLE_DEMO.md)
- 研究：保留 A 基线；LOCAL 的增量检出与误报一起报告。失败的 rescue gate 不进入当前手机组合。
- 历史 A* 是另一套四传感器主机研究/展示路径；其分数不与当前相机 + ToF 结果拼成“最好指标”。[A* 范围](nearfield/corridor_fusion_v1/ASTAR_EFFECT_USAGE_20260917.md)

## 固定对照与现存问题

[新实例受控仿真](nearfield/LOCAL_RESCUE_RESULTS_20260923.md)：576 帧、48 段、32 个正事件；224 正帧、352 负帧。

| 方案 | TP | FP | FN | 检出事件 | 解释 |
| --- | ---: | ---: | ---: | ---: | --- |
| A | 118 | 7 | 106 | 24/32 | 保留低误报对照 |
| A OR LOCAL | 159 | 22 | 65 | 29/32 | 多检出 5 事件，同时多 15 误报帧 |
| 固定 rescue gate | 159 | 18 | 65 | 29/32 | 只去掉 4/15 增量误报，未达门槛；不接入 |

UNKNOWN 均为 538/576，可与障碍提醒并存；无提醒不是无障碍。该数据受控、同渲染器，不是实机自然分布证据。旧稳定性失败仍有效。

当前瓶颈：如何从粗测距与图像中可靠判断近回波属于谁，以及接触位置能否迁移到未见布局。提醒存在、目标归属、接触定位必须分开检验。

## 最近结论，只留影响下一步的部分

- [范围与投影稳定性](nearfield/LEDGER_REPAIR_20260922.md)：账本回执修复不改变先前稳定性结论，不产生新的确认数据。
- [LOCAL 救援](nearfield/LOCAL_RESCUE_RESULTS_20260923.md)：保留增量信号，固定过滤方案失败，不再自动调阈值。
- [接触采样](nearfield/CONTACT_SAMPLING_RESULTS_20260923.md)：几何改善伴随召回下降，仅保留组件价值。
- [原始立体几何](nearfield/FOUNDATION_GEOMETRY_RESULTS_20260923.md)与 [VPP](nearfield/VPP_GEOMETRY_RESULTS_20260923.md)：查询深度提升未变成最终提醒提升；小型平面头部障碍仍失败。
- [立体适配](nearfield/STEREO_ADAPT_RESULTS_20260923.md)：普通适配只保留 challenger；固定平衡配方为负对照，不自动成为手机替代。

## 下一步及停止规则

CNH v3 的用户决定：目标/同类干扰物资产族隔离，普通背景资产允许共享并披露；
同一物理场地与布局不跨分区，参与通道侵入的建筑构件不能按背景豁免。
先完成共同写实几何/实例ID导出，再做Street200V7与City Sample各2布局工程比较；
此项不取消室内、不改变384布局和既有预算，当前未准入七路写实采集或20布局试采。
已完成两来源各1个既有视点的网格导出及稀疏深度诊断，尚无实例ID准入；
这不替代各2个新布局的七路比较。[工程回执](nearfield/CNH_SOURCE_ENGINEERING_20260924.md) ·
[CNH方案第4节](nearfield/CNH_ROUTE_COMPARISON_PLAN_20260924.md)。

此前已完成选定证据的副机备份与恢复校验、短状态页和 Python/冻结文本回归接入。提醒策略保留为后续固定检测输出的比较问题；CNH当前推进来源工程，不启动七组训练或测试选模。

新研究问题/预算/规则由用户确认；已授权问题内自主完成合理实现和验证。每个问题用一个能推翻假设的关键对照，保留成功、失败和不可评估的原始结果。具体规则见[研究工作方式](../../WORKFLOW.md)。

## 历史和恢复

[整理前完整正文](CURRENT_HISTORY_20260924.md)保留旧数字与历史 pending；[详细账本](README.md)保留结果与复现入口。历史文字不授予新执行权限。

ledger303 的回执与指定登记已[修复](nearfield/LEDGER_REPAIR_20260922.md)；未在修复范围内的历史缺失登记不能据此宣称全部解决。
