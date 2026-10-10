# 当前研究决定

更新：2026-10-10。主线为盲杖互补前视感知，M3和A+LOCAL保留；原5格局部融合与L2保留。

RGB [同FREE排序描述](../research/active/dtr-r0/nearfield/RGB_MATCHED_FREE_ARKIT_CAL_DEV_20261010.md)完成old6/fresh6冻结R0的全部ties事后描述：old6近/中远、fresh6近在全部共同实际FREE点Uni≥DAV；fresh6中远交叉F0：DAV276/Uni227（−49），F104：1047/1046（−1），未达四曲线全域弱支配，按事前门停止阶段2。F7中远上包络仍Uni641>DAV483，近F0为16>13，收益与24capture带局部交叉完整保留；这是事后最大见证，不是部署选点。未重跑预测/训练/affine、未拟合新ARKcal或下载推理，cal域迁移仍未检验。维持DAV2 Indoor Large raw R0、RGB≥0.8m/近带ToF。23683ties与30240逐query配对独立PASS；仅已消费Development、query相关，不作实机/安全确认。旧[主干确认](../research/active/dtr-r0/nearfield/RGB_BACKBONE_CONFIRM_DEV_20261010.md)的见证与FREE代价、尺度读出混合结果、DepthPro affine/残差/各run stop及同步融合N/E保留。

## ToF当前决定

用户10-10成本口径：最近表面到共向走廊±.30m边缘间隙≤10cm为near-pass，轻通知.25、强1；far-pass/clear任何通知1。gap1逐query、同帧最高等级联合计数；contact f3–13及时、f14–15晚、全窗静默。旧满额成本、冻结v2留出及[任务成本重训](../research/active/dtr-r0/nearfield/CNH_TASK_COST_RETRAIN_DEV_20261010.md)的1×主点失败保留。

[S集成1.25×新留出确认](../research/active/dtr-r0/nearfield/CNH_S_ENSEMBLE_CONFIRM_DEV_20261010.md)按WORKFLOW Confirm登记，协议`69e02149`渲染前提交。冻结6HGB/47维，无训练；新cal768/hold1536物理scene×K2、2/4新family，与27可访问库存16936旧物理键、背景及目标参数无交集。cal全负例ties选最低可行点，S集成tau2.04503、成本170.75≤137×1.25；封存后生成/读hold。

hold原5格H398/B315各/512，成本223，far+clear133。S集成救60/91、损0/0，净增151/1024=14.75%；a收益通过，d单seed145/142/140均≥40通过。b far+clear174>146.3（+30.8%）失败；c成本320>292.6875（+43.5%）失败。四项未全通过，保留原5格，不升默认候选。成本增97中near加权+56、far+10、clear+31，不能全部归因近擦边轻提醒。

精确旧tau2.414257迁移救47/86、成本285.5、far+clear149；成本在c容差内，far+clear仍越b上限，仅描述不替换主判。原both955救37/76、382/212；E救41/79、313.75/180。S接触收益与seed一致性复现，但跨family成本约束未复现；保留未晋升研究组件，下一建议针对far/clear高分尾及校准成本漂移，仅建议、未启动，不扫已消费hold阈值。

82944事件/22594通知、104017cal负例ties、模型/特征/聚类与四判据独立PASS；保守GPU200/1200s、准备评价审计整合CPU800/1200command-wall，进程释放。该确认只关闭执行登记，不改变主线/受管复用inheritance。

## 继承与边界

共向HEAD/BODY不替代外参/时钟核验；旧漏检、E2E及细杆oracle/失败，各run停止规则保留。保护480/test未访问，旧已消费数据不选点/训练，新hold已消费。仅有限AABB模拟Development确认，非实机、安全、硬件极限或完整同步RGB/CNH证明；方向/PDR及硬件第二阶段未由本run重启。

更新前全文见Git `69e02149`及本run `integration_before/`；历史见RUNS/报告。
ALGORITHM_EXPLORATION / M3_RETAINED / HARDWARE_DEFERRED
