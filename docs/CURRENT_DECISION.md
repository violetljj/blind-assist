# 当前研究决定

更新：2026-10-09。主线为盲杖互补前视感知，M3和A+LOCAL保留；原5格局部融合和L2保留。

RGB独立子线：[低参数校正与迁移](../research/active/dtr-r0/nearfield/RGB_BODY_QUERY_CALIBRATED_TRANSFER_DEV_20261009.md)。只原train56帧/7环境1591791米制点拟合shift/affine；只原cal16帧选δ+.12/−.02m，实际FPR .143116/.139822非精确目标；head/checkpoint/归一化/实际cutoff冻结。原val affine-margin召回/误支持.704999/.132615 vs head .700483/.106626，TP≥16查询见证149/198 vs163/198；新3RScan .653427/.106888 vs .592361/.084894，384/570 vs411/570，射线取舍且query覆盖减少。ARKit单capture .297995/.014912 vs .295673/.041885，77/140 vs76/140；不能由一次比较裁定贡献。包括预定shift-direct的全部8臂及全部5cohort/环境/距离带/救回损失并列；训练距离监督不同，不作孤立架构归因。独立14400臂query/18000配对及冻结身份PASS，GPU0/600s、CPU保守200/1500s。TUM官方新镜像Python/curl均失败，源0B、第三相机NOT_RUN，官方HTML1110177B，网络保守175.857/600s，资源释放；fr3硬件勘误Asus Xtion。下一补可取得的真实参考/任务可观测覆盖与事件定义，以强基线检验度量表征/时序；旧640→256小变化/28采样free各0/28、完整身体/细结构/步行缺口保留。旧两query/32特征不续训，真实硬目标/同源锚点限制/CNH预算不变。

## 当前：离线走廊参考已接通，因果位移仍缺

[方向合同与离线走廊参考](../research/active/dtr-r0/nearfield/CNH_DIRECTION_CONTRACT_DEV_20261009.md)已交付可运行的离线方向合同接口：原始Nymeria Charades骨盆149170行、8合法段、627native anchor×T0.5/1/1.5秒=1881窗口，未来完整折线+0.30m圆盘proxy只作评价；过去1秒方向仅用给定离线位置的过去，不等于原始传感器在线E1。7时间异常不修补、不跨gap/尾端外推，原始XSens XY/Z上不冒充Aria或CNH坐标。

627anchor中362个过去1秒位移<2cm、8个缺过去窗，单Charades样本不能定典型步行误差δ或物理T。1.5秒未来617可用窗中2个中心距端点弦>30cm，不能把端点方向当完整身体扫掠，也不是身体区域遗漏比例。真实因果位置仍NOT_IMPLEMENTED，姿态前缀通过不等于位移/方向准确度。下一接合格因果位置来源和任务相符的身体路径参考；T/身体几何/外参/时钟误差仍待验证。

旧2°RMS来源已核：bias+逐帧抖动，σ2实际2.2705°，与恒+3°及本轮past-pelvis/future-chord不能直接换算。旧max18pass阈值阻断峰方向17/18朝目标（去重16/17scene/K）；OR +3新增clear主要L形侧墙，仅定位线索，不证明回波因果。全部聚合/锚定/高度混合结果、旧M3/5格/L2/身体truth/bin/480与各run停止规则保留，不按validation选赢家。

5项fixture、1881真实窗口结构/路径核验、1266行past字段截断/未来修改逐值检查通过，原始源/helper SHA绑定。仅CPU离线整理（180s上限、GPU0），无新投影/模型推理/训练/硬件/下载/480。未建立完整身体碰撞、典型行走误差或实机效果；输入合同与全状态/预算见报告。

## 停止与保留

[旧扰动训练](../research/active/dtr-r0/nearfield/CNH_QUERY_PERTURB_TRAIN_DEV_20261008.md)、T/T2、旧参照和门控的停止规则仍按各自run适用；不同机制开放。[480确认](../research/active/dtr-r0/nearfield/CNH_REAL_HEAD_CONFIRM_RESULTS_20261007.md)原身份保留，UNKNOWN/三态/query覆盖待验证。方向/Nymeria位移、−19°、设备/City/保护test/新UE及硬件第二阶段暂缓；101/101与53ms属A+LOCAL，M3/CNH实机效果未建立。

模拟AABB、有限背景、人工误差、相关K/帧及探索选择不构成实机、安全或独立确认。更新前CNH当前正文保留于Git 5844eacd同路径；RGB子线继续沿用。

[路线当前页](../research/active/dtr-r0/CURRENT.md) · [RUNS](../research/active/dtr-r0/RUNS.md)。
