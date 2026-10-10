# 局部峰证据与身体通道侵入联合评分

2026-10-10；EXPLORE，起始 `d941e120`。用户“继续”接续[峰跟踪与断续支持](CNH_GRADED_PEAK_TRACK_DEV_20261010.md)，复用相同已消费 ideal 模拟 Development；并发RGB后续提交不属于本轮CNH范围。

**联合当前峰与侵入分数，能用较少额外pass补回部分弱BODY并提前提醒，但clear成本仍有取舍。中档相对完整single参照多救BODY4/6/8件、已有及时事件提前53/31/44件，额外pass1/12/7条序列、clear23/40/27个时间格。相对同训练和校准的score-only对照，BODY救/损3/2、6/0、7/1。当前峰联合评分保留为候选；加入轨迹字段的追加收益不稳定，不默认升级。**

## 本轮机制

相同384scene×K4×f3–15×2query，及时截至f13；HEAD/BODY各384接触事件，pure-pass256 clips/3328 slots，joint-clear512 clips/6656 slots。三个seed955/956/957分别报告。共向不替代位移，几何仍使用原公共noisy位姿；没有新的原始回波读取、匹配、采样或主干推理。

当前峰固定22项来自已审计top8缓存：rank0幅值、top8幅值和、inner_share加权和/最大值/占比、有inner支持的候选数、最强inner原幅值，rank0和最强inner的公共XYZ与支持范围，inner_share、两者深度差与幅值比。最强inner是inner_share>0候选中原始幅值最大者，不是加权幅值最大者；支持范围是粗bin角节点范围，不是物体尺寸、覆盖或free证据。

temporal在当前22项后追加上轮35轨迹字段，包括支持累计、命中/缺帧、残差、原始与补偿距离变化、inner支持与窗口预热。5帧、缺2帧、当前命中及f3启动身份不改；这些变量可表达支持变化，但不预设分数上升就是接近。

固定三评分bank：score-only为ordinary平滑分数减原single阈值、raw5帧斜率、去趋势波动共3维；score-current为score3＋current22＋22缺测指示共47维；score-temporal为score3＋57字段＋57缺测指示共117维。无效字段按NaN送入HGB，并附指示，不当成数值零或free。当前22项在本批全有效；temporal保留源起始帧残差缺测。最强inner不是rank0的query-slots为cal2384/39936、validation2416/39936；其余相同并不证明目标在身体通道内。

每高度分别拟合固定HGB：100轮、学习率.1、7叶/深3、min_samples_leaf50、L2=1、无early stopping、随机种子20261010，与旧小评分器参数相同。只用cal背景实例8/10中完整原grade0的帧；每scene×K×query各自有观测的帧等总权重，再将样本权重归一为均值1。contact事件类别只用于fit标签和评价，不进入特征；训练标签是事件级接触，不是逐帧物体归因。

cal背景9/11只定cutoff，固定追加candidate query并集slot上限(clear,pass)=(5,32)/(15,64)/(30,128)，两类取所需阈值较大者，保留完整ties。HGB logit可为负，非绑定阈值为−∞，JSON以null和nonbinding表示，finite eligibility明确；本批27阈值均绑定，26/27用满clear上限，中档均用满15而pass仅10–29/64。clear是当前校准的主要约束之一，不把未用pass预算解释为能力已饱和。

仅在原grade0追加light：ordinary_OR强档、完整single轻档与全部首时刻保留。新增评分与原评分并行，不过滤、降级或静默已有提醒。18个小评分fit、54 split-cells、27校准、82944 validation事件行。运行某一候选只需一个ordinary模型和两个高度小评分器，无多seed投票；本轮未测手机端推理或提醒全链路时延。

这与上轮直接峰幅值门槛不同，且没有重开上轮停止配方。三个bank拥有相同fit人群、参数、seed和cal程序，可以比较新增字段；旧raw-frame峰没有小评分fit，与其比较是实用取舍，不是隔离机制的严格对照。同cal上限不是同validation实际成本，cal-cutoff半实例也不是全cal。validation已被前轮消费，本轮是探索，未据其结果调阈值/删bank或重新训练。

## 全部validation结果

各斜杠依次955/956/957。救回相对完整single参照HEAD351/344/346、BODY307/310/312，各高度384事件。clear增量为总slots/6656，pass增量为总clips/256。所有原参照及时损失/推迟均0，来自追加构造，不是统计非劣证明。

| bank | cal追加clear/pass上限 | HEAD救回 | BODY救回 | clear增量 | pass增量 | 已有及时HEAD/BODY提前 |
| --- | --- | --- | --- | --- | --- | --- |
| score-only | 5/32 | 0/3/0 | 1/0/0 | 1/11/14 | 0/3/3 | 0/25/14；21/2/28 |
| score-only | 15/64 | 0/3/0 | 3/0/2 | 3/29/46 | 3/5/8 | 0/42/39；29/4/45 |
| score-only | 30/128 | 1/4/2 | 5/2/5 | 35/55/67 | 12/9/17 | 31/58/70；50/17/64 |
| score-current | 5/32 | 2/12/5 | 2/1/4 | 9/12/12 | 1/3/3 | 17/27/23；19/17/32 |
| score-current | 15/64 | 4/14/6 | 4/6/8 | 23/40/27 | 1/12/7 | 31/59/40；53/31/44 |
| score-current | 30/128 | 5/14/8 | 5/6/10 | 40/64/41 | 4/19/12 | 36/81/61；67/38/68 |
| score-temporal | 5/32 | 3/12/4 | 4/4/4 | 13/29/16 | 2/5/5 | 24/44/21；30/15/29 |
| score-temporal | 15/64 | 5/13/8 | 7/4/7 | 36/37/31 | 5/7/8 | 46/55/38；55/21/47 |
| score-temporal | 30/128 | 8/14/9 | 10/9/10 | 60/65/72 | 14/13/14 | 60/87/67；86/29/69 |

中档current及时HEAD355/358/352、BODY311/316/320，clear96/147/107、pass150/180/165；temporal为356/357/354、314/314/319，clear109/144/111、pass154/175/166。

严格配对中档current对score-only，HEAD救/损4/0、12/1、6/0，已有共同及时提前/推迟31/0、41/23、30/30；BODY救/损3/2、6/0、7/1，提前/推迟38/15、29/2、21/23。clear差+20/+11/−19 slots，pass差−2/+7/−1 clips。不能只报三个BODY净增；这些候选之间并非包含关系，即使都保留原参照，也可能失去另一候选独有的救回或提前。

中档temporal对current，HEAD救/损2/1、1/2、3/1，BODY4/1、3/5、3/4；BODY提前/推迟21/19、8/20、17/20，clear+13/−3/+4、pass+4/−5/+1。没有稳定追加改善，因此不默认增加轨迹评分复杂度；这不是否定所有时间积累。

中档current对上轮raw-frame，BODY救/损3/6、3/4、6/8，及时净少3/1/2件，但pass少60/39/51clips，clear+11/+28/−2slots；HEAD救/损1/22、1/18、0/28。共同及时BODY提前52/31/39件、推迟3/4/7件。不能只用pass下降称整体优胜，也不能只用HEAD检出减少否定较早、较少打扰的轻提醒。

中档current相对完整参照，BODY新增竖杆2/2/2、横杆0/3/1、边缘2/1/5，各族每高度96事件，突出物新增0。BODY提前包括突出物21/9/18件，这些是原已检出事件的提醒提前，不计为新救回。完整family/background_family配对见audit/comparisons，不把标签关联当成峰的真实目标身份。

## 强轻成本与持续时间

强档始终及时HEAD315/313/323、BODY267/257/282，clear58slots与28clips/512，pass89/103/115clips、355/362/461slots。强轻在同clip可先后出现，不可相加clips；joint-light按同slot两高度最高等级为1计费。

中档score-only joint-light clear18/78/68slots、pass360/511/335slots、141/161/154clips；current clear38/89/49、pass361/526/328、139/169/149clips；temporal clear51/86/53、pass375/520/320、143/164/150clips。三bank各seed最长light pass连续均5/7/5帧。

单独追加候选中档pass：score-only24/42/44slots，22/37/43段、最长2/3/2帧；current25/58/36slots，24/54/35段、最长2/2/2帧；temporal39/52/29slots，37/48/28段、最长2/2/2帧。候选涉及已有提醒clips，因此不是新增总打扰clips。候选计费、joint最高等级计费和总增量均保留；不以13帧窗口推论真实秒数、长时干扰或用户收益。

## 核验与交付

特征独立逐rank复算全79872 query-slots，共12661286项值、mask、直通与身份核验，最大3.81e−6来自FP32求和次序并在容差内；35track数据和mask精确直通。3个小例验证没有inner时有效count0与缺测NaN、最强原幅值和最大加权幅值的区别、无peak时全invalid。这些小例只验证实现。

评分独立核验18模型hash/固定参数/全量预测、cal-only原grade0人群、sceneKquery权重和加权prior logit、27完整tie阈值、54完整指标/分格、82944台账、原等级与首时刻保留、配对救损/早晚以及全部成本与持续段，共450826项通过。审计不重新拟合模型，训练程序与人群由源码及fit记录核对。

特征1.047s、特征审计命令1.054s、主评分20.281s、评分审计5.891s；保守特征2/120s、主评分25/300s、两审计合计9/120s、统计/文档/交付180/300s，总216/900 CPU command-wall seconds。GPU0；18小评分拟合及小评分预测确已运行，无主干重训/前向、新采样/匹配或设备会话。没有常驻任务进程。

源码 `cnh_graded_peak_joint_features_dev.py`、`cnh_graded_peak_joint_dev.py` 和各自对应audit脚本；载荷 `artifacts.local/work/cnh-graded-peak-joint-dev-20261010/`，TASK/PLAN记录目标预算和停止，features保存22/57项描述及mask/定义，models保存18评分器，scores/grades/calibrations/metrics/summary/event_ledger保存完整结果，audit保存核验和全部对照/持续时间。输入和源码hash有收据，脚本拒绝覆盖已存在PLAN。复现用新输出目录，并同步修改生产和审计的输出入口；本轮无新科学输出失败，上轮几何故障及修复保留。

## 当前决定

保留当前峰联合评分为弱BODY及早提醒候选，完整single轻档和ordinary_OR强档仍作参照。中档是说明取舍的工作点，不是从validation挑出的独立最优政策；全部声明bank/预算已完成，停止本轮固定参数追加调参。

下一建议对新增clear与BODY救回按公共峰距离、支持范围和背景可见条件作匹配诊断，决定优先调整校准还是补充空间证据。最弱假设是粗bin支持能区分侵入与背景；匹配不足允许证据不足，不能据此关闭轻提醒提前的价值，也不直接静默已有提醒。

原M3/5格/L2/body truth/fullbin/480、weak_pass和各旧run停止身份保留。[高度评分](CNH_GRADED_HEIGHT_CAL_DEV_20261010.md)及旧轨迹比较作为历史结果保留。全部仅已消费受控模拟Development，无独立确认、物体归因、free、实机、安全或App效果证明，尚未接入App。
