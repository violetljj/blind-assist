NOT_ESTABLISHED_SINGLE_SEED_DEV

# 前视障碍感知：当前状态

更新：2026-10-02。已消费Development；`V5_FROZEN_COMPLETE / HARDWARE_DEFERRED`。
Status: `DTR_R2_DYNAMIC_RETAINED`（历史保留）。

## 当前结果与决定

逐方向首可见距离/valid读出已训练，但未改善报警。相同初始化、洗牌的BCEO/RAY各seed0十轮；实际native最近8帧和因果姿态输入，只有RAY加几何监督。0–2cm及时M3/BCEO/RAY为26/25/25（n31）；RAY−BCEO 0pp [0,0]，RAY−M3 −3.23pp [−10.34,0]，补回0/丢失1。

整体清晰首停205/214/217次（194.35代理分钟）；外10–20cm清晰69/71/74次（40.69分钟）。RAY对两对照的浅及时及外侧清晰护栏均失败，判NOT_ESTABLISHED_SINGLE_SEED_DEV。不扩五种子、不追加该配方调参；M3继续保留。

RAY中/深及时36/42、161/164；擦身93/207停（44.93% [37.50,52.15]，不计误报）。浅已报警条件提前量1.92秒 [1.59,2.33]。完整分母、区间和独立复算见[结果](nearfield/CNH_RAY_SURFACE_RESULTS_20261002.md)。

## 尚未解决

前轮真值诊断确认角距边缘化误差约径向量化8.18倍；但保留对应关系并学习逐射线距离，没有自动带来任务收益。训练损失下降不等于评估几何准确。下一问题是区分边缘几何预测误差与查询到报警的映射误差；诊断尚未执行，单距离可能平均模糊边界只是候选解释。

[角距诊断](nearfield/CNH_SURFACE_FACTORIZATION_PROBE_20261002.md) · [旧SURF失败](nearfield/CNH_SURFACE_DISTRIBUTION_RESULTS_20261002.md)。HIST/AGG、共享signed场、NEST/MAX/source yaw及原外扩guard、mask失败保留；同批H3单帧小试停止。M3原静态比较与94000方向保持结果保留。

## 必要边界

96评估单位7680查询仅1138覆盖0.9m截止点，6542右删失；浅31来自26单位。新臂单种子，区间条件于固定模型/阈值；无新sensor渲染、跨源或实机效果。128×128预测依赖先验，不增加8×8物理分辨率；1cm软查询与标签舍入精度不代表预测精度。几何valid不等于可检出性、UNKNOWN或畅通。

三级真值：伸入必须报、身体外0–10cm只报告、更远或其他高度才计清晰。13点清晰不保证帧间连续清晰；代理分钟含首停后时间，提前量条件于已报，不代表人安全停止。v5保留A2；手机A/A+LOCAL手动会话与UNKNOWN语义不变；City、保护test、新UE、硬件第二阶段暂停。

[前版全文](CURRENT_HISTORY_20261002_PRE_RAY_SURFACE.md) · [RUNS](RUNS.md) · [总决定](../../../docs/CURRENT_DECISION.md)
