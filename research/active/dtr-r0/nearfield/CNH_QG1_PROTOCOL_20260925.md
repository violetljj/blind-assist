# QG-1 查询几何条件化与响应分解：执行前冻结协议

状态：`FROZEN_BEFORE_NEW_DEV_OUTCOMES`。2026-09-25，用户明确确认新指令替代“仅写方案”的执行限制；[近场验收方案](CNH_NEARFIELD_ACCEPTANCE_PLAN_20260925.md)仍待审，不采集。本协议必须先提交Git才可运行新dev分数。新结果只属于退化、重复布局的Development结构/响应诊断，不恢复独立RGB或泛化证据资格。

## 范围和身份

只用既有六巷道480 train / 480 dev、原标签/查询盒/划分；不采集、不连接设备、不访问或准备任何test/City。保留V2及既有失败，修改以新文件为限。输入绑定：

- collection：`artifacts.local/evidence/cnh-alley-rgb-replay-six-20260925-v2/collection-overlay.json`，SHA256 `f72425577e4e06f9a95e39ba2ef0a9b3d6fc087157e74d31bcfce7927ba4840f`。
- partition：`artifacts.local/work/cnh-route-comparison-20260924/plan/street-alley-merged-partitions-20260924-v1.json`，SHA256 `7ae115183bab26876eedc4d9bd3b11cb0b7613c2b56dddb3055266fe1a586a9b`。
- prepared：`artifacts.local/evidence/cnh-visible-depth-audit-20260925-v1`；native float32 depth SHA256 `444ecc0e1576057d02b88fec7e115600badad5f0ab3e4ae17254d88bffd7cbe9`，perfect H3 SHA256 `801d0b950190225b4a2ff606a3fc38203b0751a581ff3c9cfd0e5e1d9a74e1a3`。
- 逐帧观测/标签/RGB/camera继续验证overlay原哈希、frame_key及布局绑定。train432正/2448负查询，dev408正/2472负，UNKNOWN=0；不改变这些分母。

## A. 模型和输入（固定实现，不搜索）

`cnh_qg1_geometry.py`仅从六公开查询盒与K得到投影边界、相机射线坐标、10维盒描述（中心3/尺寸3/Y边界2/Z边界2）、完整6×64×16重叠权重。移除V2固定列mask；当前数据相机/ToF/Q同坐标，未来下倾坐标方案不混入本轮。

`cnh_qg1_model.py`共享架构：5通道图像分支经3×3 stride2的24通道卷积、GELU、3×3 stride1卷积、GELU；按查询投影框在特征网格上裁剪，非重叠分箱max pool保留8×8空间格（右/下补负无穷至8的倍数，完全空bin置零；支持CUDA确定性反向），展平投影到64维。不先汇聚成ToF区。卷积感受野允许ROI边缘邻近像素参与，不能宣称精确像素级隔离；空ROI给零并由已知投影识别。

ToF signed-log1p直方图先乘查询几何重叠权重，保留区/bin索引展平，与相同支持内的距离/有效位/ambient/age元数据共同经共享MLP得到64维查询特征。图像64维、ToF64维及盒描述10维拼接，经64隐藏单元共享非线性head产生六占用logits。没有解析盒命中分数、标签、实例ID输入模型。

| 臂 | 图像分支输入 | ToF输入 |
| --- | --- | --- |
| full_depth | 原生640×360可见深度，经K回投XYZ（截到±10m后/10）、valid、1/(1+Z)；共5通道 | 禁用，特征零 |
| tof_sim | 禁用，特征零 | 原模拟H3和原同响应公共元数据 |
| tof_rgb | 原修复RGB128×72，RGB/255与两通道公共射线坐标；全图ROI | 与tof_sim逐位相同 |

XYZ只由可观测深度和公开K得到，不读物理真值标签；这是full-depth sanity臂专用输入。RGB编码器本轮随机初始化，不下载预训练权重；预训练后续另议。full-depth与RGB分辨率/通道语义不同，因此不声称两者性能差来自模态信息量单一因素。各臂初始化seed一致，共享同架构参数形状；架构、显式坐标与优化组合一起被测试，不做单因素因果归因。

## Gate 0：固定32帧train拟合

完全复用`cnh_learning_diagnostic.select32`：按train布局字母顺序11/11/10等间隔帧，须与`cnh-learning-diagnostic-20260925-v2/overfit-selection.json`的indices/frame_key完全一致。32帧24正/168负查询，不换样本，不读dev结果。

- seed20260924、batch16、AdamW lr=3e−4（所有活跃参数相同）、weight_decay=1e−4，完整480帧train每查询负/正比作为pos_weight，梯度范数裁剪5。
- 每臂最多1200次更新或1200秒，先到即停；第1步及每50步评估train pooled AP，≥0.99立即停止。
- **唯一备选学习率**：若第600步仍未到0.99，仅将当前优化器lr改为1e−3，沿同一状态继续剩余最多600步；不重置参数、不增加预算。该规则只看train，三个臂一致执行。无其他优化器/seed/结构重试。
- full_depth必须达到0.99才允许Gate1。三个臂均报告AP、更新数、首步梯度及轨迹；ToF/RGB过拟合不作进入Gate1的门槛。
- full_depth失败则`STOP_GATE0`，Gate1为NOT_RUN，保留梯度/汇聚诊断；PartB可继续。该失败定位到当前表征/优化组合，不能直接断言“必然更底层”。

## Gate 1：480/480三seed对照

仅Gate0通过后执行。每臂seed20260924/25/26，各重新初始化；24epochs、batch16，共720次更新，与V2相同epoch预算及train class balance。AdamW所有活跃参数lr3e−4、weight_decay1e−4、cosine24epochs、clip_norm5；不继承Gate0权重或备选lr。不启用Gate1备选学习率，不选择checkpoint，固定最后epoch。记录训练曲线、设备、时间、输入/代码/预测/checkpoint哈希。

最终预测全部960帧，报告每seed/臂train/dev AUPRC（average precision）、AUROC及每布局结果。一个阈值在全部已知train查询上最大化F1，同分选较高阈值；dev给TP/FP/FN/TN和408/2472分母，不用dev找阈值。

冻结判定（严格采用用户门槛）：

1. `STRUCTURE_GATE_MET`：full_depth三个seed的dev AP均≥0.80，否则NOT_MET。
2. `TOF_DIAGNOSTIC_GAIN`：tof_sim三个seed的dev AP均>0.2101且均>0.2314，否则NOT_MET；比较常数使用用户指定四位数，不回改。
3. `RGB_USED_AND_POSITIVE_REUSED_DEV`：每seed tof_rgb AP均高于tof_sim，且同一融合checkpoint做dev布局内跨帧RGB置乱后AP低于原图；任一不成立则`NO_RGB_GAIN`。置乱复用`within_layout_cycle`、固定SHUFFLE_SEED20260925，同一个无固定点排列用于三seed，保持ToF/标签不变；不训练、不多试排列。该门槛只证明当前重复场景上的相关使用及排序变化，不等于任务相关几何使用或独立融合收益。

Gate1完整跑完即停止；某一判定失败不能换阈值/子集/epoch救结论。任何结果都不授权新采集、正式test、App更新或下一模型。

## B. 模拟传感响应损失分解（不训练）

实现于`cnh_qg1_response_decomposition.py`，CPU微小矩阵/元数据处理。使用冻结16×16角采样、原相机深度与原frame_identity随机seed；先验证重新读取的perfect H3与cache逐位相等、完整模拟H3转float32后与原observations逐位相等，再算分数。身份或端点不一致则先报告机械差异，**禁止把差异归因于下表组成**。深度源相同不自动代表两个处理链相同。

原实际组成顺序：归一化角面积 → 4000×ρ×cos/r²原始128bin直方图 → 脉冲核 → 邻区泄漏 → 残余串扰 → 独立计数/ambient估计噪声 → H3每8bin相加。全部参数固定为原运行记录，ρ=.5、cos=1、signal_counts4000、ambient4、noise_scale1、pulse_sigma2raw bins、tail_mass.1、tail_decay4、neighbor_leak.02、xtalk_fraction.02、xtalk_range.04m、range_zero0、gain1；不改模拟器或旧结果。

| 累积阶段 | 新加入的组成 |
| --- | --- |
| AREA4000 | 无噪声角面积H3×4000，重现原规则排名AP0.8452 |
| RETURN_LAW | 同ray能量改为4000×0.5/max(r,.05)² |
| PULSE | 原高斯/单侧尾核；窗口逃逸能量保留损失 |
| NEIGHBOR | 原非周期四邻交换，边缘不绕回 |
| XTALK | 原脉冲形状的残余串扰添加 |
| SHOT_BACKGROUND | 原Poisson计数减独立ambient估计，保留signed输出；应重现AP0.2101端点 |

另外从AREA4000分别只加入RETURN_LAW、PULSE、NEIGHBOR、XTALK、SHOT_BACKGROUND，报告同样指标。isolated XTALK仍用原脉冲形状的串扰核，即使场景脉冲关闭；如实说明此依赖，不伪称物理上完全独立。全局正增益与统一ρ=.5只改变分数尺度，另核验排名不变；加入噪声后增益会影响SNR，不能把该不变性推广到噪声之后。

原实现**没有饱和、ADC量化或截断负值**，这些项目记NOT_IMPLEMENTED，不捏造消融。H3桶化是已有固定表示，不与噪声随机数变化混成“量化”。面积主分数用缓存perfect score×4000保持原float32同分；同时报告raw128→H3再生成面积读出的控制，直方图误差需≤2e−7，有限精度打破同分造成的AP差异单列。

每阶段/单项阈值均仅用train pooled F1选择（同分高阈值），报告train/dev AP/AUROC/混淆及每布局分母。损失用AP**差值**描述，不把0.845→0.210称四倍物理信息损失；累积增量依赖顺序，单项下降也不必可加。列`完整下降−各单项下降之和`作为非加性描述，不能当因果交互效应的完整估计。不据该表调传感参数、重训或选择dev最好版本。

## 验证、交付和生命周期

先通过公开投影/权重、ROI形状/局部性、深度通道、固定32帧选择及模拟端点合成单测；预提交仅运行合成输入检查，不预看新dev结果。register-experiment登记，具体执行代码hash在回执保存。输出分别放`artifacts.local/evidence/cnh-qg1-20260925-v1`与`cnh-qg1-response-decomposition-20260925-v1`，已有目录不覆盖；机械失败保留failure回执，修复不能扩张已消费科学预算。

训练Gate0/Gate1分阶段独立进程，progress.json给当前臂/seed/epoch/时间，终态为gate0.json / gate1.json或failure。单个未完成训练不具备精确断点续训，不自动重启；完整运行保存每臂seed模型/预测。最坏单seed重启损失720步，必须先核实进程退出和故障归属再决定，不能暗中挑较好运行。

交付`CNH_QG1_RESULTS_20260925.md`，列Gate0三臂AP/步数、Gate1九行完整表、RGB置乱、PartB累积与单项表、主要损失与未唯一归因项、检查及限制、提交清单。短更新CURRENT与CURRENT_DECISION，保留无关工作树；普通非force推送origin/master后不等待远端检查。释放任务GPU/进程，保留原始与失败证据。
