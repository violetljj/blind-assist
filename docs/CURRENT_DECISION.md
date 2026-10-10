# 当前研究决定

更新：2026-10-10。主线为盲杖互补前视感知，M3和A+LOCAL保留；原5格局部融合与L2保留。

RGB [尺度读出与新增近带覆盖](../research/active/dtr-r0/nearfield/RGB_NEAR_READOUT_SCALE_DEV_20261010.md)完成R0/R1/R2/R3×五主干臂；旧496预测不重跑，官方按序新增三Validation visit各16帧，近POS44/75/51。pooled304先封存后184eval，原三LOCO另保留。六ARK主强信号未通过：Uni raw R2三新近W4→8/44、20→31/75、19→22/51且FREE不增，但中远损42/771=5.447%；全184含3RScan中远损71/1447=4.907%，跨源总表达数值条件，不能替换事前ARK主口径。Uni affine R2中远零损但只两条近增益。按封存主决定维持RGB≥0.8m、DAV2 Indoor Large raw/原读出、近带ToF；恢复近信息与局部收益保留，非单目原理否定。旧DepthPro affine、主干/残差/stop与同步融合N/E均保留。

## ToF当前决定

用户10-10成本口径：最近表面到共向走廊±0.30m边缘间隙≤10cm为near-pass，轻通知0.25、强1；far-pass及clear任何通知1。gap1逐query、同帧最高等级联合计数；contact f3–13及时、f14–15晚、全窗静默。旧满额成本及[冻结v2留出](../research/active/dtr-r0/nearfield/CNH_COST_V2_HOLDOUT_DEV_20261010.md)结果保留。

[任务标签成本重训](../research/active/dtr-r0/nearfield/CNH_TASK_COST_RETRAIN_DEV_20261010.md)完成新train3072/cal768/hold768物理scene×K2，12新family，与24可访问键库存12304历史物理键无交集。标签为支持后的contact及时窗正例，near负权重.25/far与clear1；强slot锁原5格。E冻结集成、S固定47维HGB、B原ordinary微调8epoch均完成三seed；cal全负例ties封存后才生成/读hold。

hold原5格H193/B152各/256、成本102、far+clear68；both955救36/38损0、成本184.5、far+clear110。主点E净增0；S三seed净增9/18/0、集成2；B2/7/0、集成8。S956单臂达数值条件但S957无增益，未通过三seed同向；无预声明强信号，保留原5格、不升默认。次点S集成救41/42损0、成本132.5（hold+29.9%）、far+clear74；B集成33/44、140/81，E32/33、130.5/79。cal成本上限不保证hold同倍率。

固定特征/本配方有排序信息但高分尾重叠、seed不稳：S集成cal AUC H.716/B.731，主cut4.348高于正例p95 2.949/3.390。下一建议优先S的通知级成本/尾部排序监督，仅建议、未启动；不作信息不可分原理断言。本run为用户授权的新损失机制，旧扰动/T/T2及细杆各run停止规则仍保留。独立核验和预算见报告。

## 继承与边界

共向HEAD/BODY不替代外参/时钟核验；旧[漏检拆解](../research/active/dtr-r0/nearfield/CNH_BASELINE_MISS_DECOMPOSITION_DEV_20261010.md)、冻结E2E与细杆oracle/读出失败保留。保护480/test未访问，已消费数据不训练/选点，新hold现已消费。仅有限AABB模拟Development，非实机、安全、硬件极限或完整同步RGB/CNH证明；方向/PDR及硬件第二阶段未重启。

更新前全文见Git `40c7e6f3`及本run `integration_before/latest_*`；历史见RUNS/报告。
ALGORITHM_EXPLORATION / M3_RETAINED / HARDWARE_DEFERRED
