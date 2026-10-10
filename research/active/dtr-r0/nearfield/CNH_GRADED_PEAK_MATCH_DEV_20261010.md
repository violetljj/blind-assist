# 新增clear成本与BODY救回的匹配诊断

2026-10-10；EXPLORE，起始 `eb7572e7`。用户“继续”接续[局部峰与侵入联合评分](CNH_GRADED_PEAK_JOINT_DEV_20261010.md)。固定score-current中档c15/p64、三seed及全部已消费ideal cal/validation；不修改评分、阈值或提醒规则。

**新增clear成本有明显HEAD贡献，seed956的40格中有28格仅由HEAD追加引起。BODY新增救回与clear严格匹配只有0/1/0对，证据不足；内走廊峰更远同时出现在救回、提前和打扰事件中，不支持将其直接当静默依据。下一优先比较BODY定向追加与当前两高度追加的收益/成本，原有强轻提醒仍完整保留。**

## 事件与匹配定义

原参照为ordinary_OR强档＋完整single轻档。BODY主面板是原f3–13无及时提醒而联合评分新增及时提醒的事件，取新增首次帧；次面板是原已及时而首次提醒提前的事件，另取提前锚点，两者分开报告。每scene×K×BODY只有一锚点，不把单事件连续提醒算成多次救回。

三类对照：joint-clear中首次BODY新增candidate；首次BODY新增且原两高度均静默的joint-newclear；无任何contact且BODY为pass的首次新增pure-pass。对照允许f3–15，主救回/次提前锚点截至f13，帧差完整保留。仅选已有提醒事件作匹配，不能用这个子集评价新的检出率、减少提醒或静默量。

先于匹配写PLAN，固定同split/seed、同作者background_family、background_id和shape_family；再要求公共rank0峰z差≤.3002784m、支持XYZ三轴extent差各≤.15m、inner_weighted_sum_share差≤.2、inner支持候选数差≤1。作者字段只在评价侧分层，不能进入运行时策略；公共峰z和节点范围也不是目标真距离或物理coverage。

每panel内锚点按scene/K/frame排序，选六项差各除其容差后L1和最小的可用对照，tie按control scene/K/frame，一对一不放回。主/次、三对照各为独立panel，可跨panel或seed重复使用同一观察；这不是独立样本池。无匹配标NOT_EVALUABLE，保留每条件候选数与no-reuse用尽原因，不放宽容差，不另选阈值。

完整cal/validation共575锚点/对照记录、2304 BODY接触首时刻、3072 clear clips、1041匹配尝试、133配对记录。133包含两clear面板相同的配对及跨seed重复；去除panel/seed后按观察帧身份为90种配对，也不是90个独立场景。

## 固定validation分布与成本

以下顺序均为seed955/956/957。接触分母每高度384；clear为512 clips/6656 slots。原先中档BODY救回4/6/8、原已及时提前53/31/44精确复现，未新增科学工作点。

| 项目 | 955 | 956 | 957 |
| --- | ---: | ---: | ---: |
| 新增clear总slots | 23 | 40 | 27 |
| 仅HEAD追加贡献 | 11 | 28 | 9 |
| 仅BODY追加贡献 | 12 | 12 | 18 |
| 同slot两高度追加 | 0 | 0 | 0 |
| 新增clear clips | 21 | 31 | 23 |
| 原已有clear clip首次提前 | 1 | 4 | 2 |
| BODY clear对照clips | 12 | 12 | 17 |
| BODY pure-pass对照clips | 15 | 17 | 20 |
| 救回匹配clear | 0/4 | 1/6 | 0/8 |
| 救回匹配pure-pass | 1/4 | 2/6 | 3/8 |
| 提前匹配clear | 6/53 | 7/31 | 9/44 |
| 提前匹配pure-pass | 11/53 | 5/31 | 12/44 |

BODY candidate新增slots本批恰为12/12/18，和joint-newclear BODY贡献相同；两clear面板的事件及匹配亦恰相同。这是本批没有另一高度重叠的事实，不是两种通用计费口径相同。17条seed957 BODYclear clips含18 slots，事件数与slot数不可互换。

主救回与clear只有一对可比观察。955未配4件的首个失败条件为shape3、峰z1；957未配8件为峰z7、shape1。失败条件计数依赖固定cascade顺序，不能说形状或深度是唯一原因。部分pass或次面板未配是已有可行control先被用尽，审计单列；不把这些也当观测证据缺失。

主救回与pure-pass的匹配帧差control−anchor：955为−1，956为0/0，957为−7/0/−1；次面板对照帧差范围−7至+8。匹配公共几何近似，但没有匹配实际接近阶段、真目标距离或提醒时钟，不能作因果分离证明。cal各面板也完整保留，不与validation合成独立确认。

## 内走廊与外环峰深度对照

复用旧已审计88项缓存，在同一BODY事件帧取current内走廊/外环最强正峰的前向z、支持z上下限、membership、native zscore、峰幅值及same-bin share共13项。定义gap=inner峰z−ring峰z；预声明以±.3002784m分为inner更远、同格量级、inner更近；只描述观察，不指定静默或筛除规则。两区域最强峰可能由不同bin或不同角节点支配，较远正峰尚未归因于背景，区间重叠也不是free。

下表给每类事件中“inner更远”的计数。每类分母不同，选中的新增提醒也不是所有contact/pass/clear；不据此宣称检出、静默减少或收益。

| BODY事件 | 955 | 956 | 957 |
| --- | ---: | ---: | ---: |
| 新增及时救回 | 1/4 | 3/6 | 4/8 |
| 提前提醒 | 22/53 | 11/31 | 12/44 |
| 新增clear | 6/12 | 7/12 | 7/17 |
| 新增pure-pass | 8/15 | 6/17 | 11/20 |

主救回唯一clear配对是接触inner更远、clear却inner更近，与“内走廊更远即可静默”的直接解释相反。次面板中inner更远/inner更远同时出现1/2/3对，说明该观察同样可能伴随有用的提前提醒。主救回对pass的gap差中位数−1.619/+0.415/−.659m，样本仅1/2/3对且方向混合；不是稳定判别边界。当前所有13项在575条事件中有效，缺测机制与统计仍保留在源码。

这批观察不能确立“背景可见”证书，更不能当空闲证明。它仍可作为未来组合评分的候选变量；本轮没有用结果调整匹配边界、评分或阈值，没有计算未声明静默规则的收益/损失。匹配不足亦不否定轻提醒提前的任务价值。

## 核验、故障与交付

匹配独立重放575事件、2304 BODYcontact、3072 clearclips、1041 attempts、133 pairs：所有首时刻/及时救回/提前、runtime字段、条件cascade、标准化L1/tie/no-reuse以及HEAD/BODY成本分解通过，共102297标量，内部.547s。

深度对照独立复算575事件13字段的索引/值/有效mask、深度分格及区间overlap，133pair身份与14差值，30事件组和27已有配对面板的分布/线性分位数通过，18146直接CSV标量检查，命令.905s、内部.329s。未重新读raw、重做原几何或拟合评分。

首次深度程序将配对列名误写为anchor_id/control_id，实际为anchor_event_id/control_event_id。错误发生于配对汇总前，初source、PLAN、失败记录和已正确生成的events保留在depth/；修复后的same-fields/same-gaps结果在depth/repair-1，父PLAN hash与输入一致，审计核对修复前后事件和约束。没有覆盖原故障或改科学条件。

匹配内部.437s；深度失败/修复两命令约1.349/1.163s，修复内部.218s。保守匹配2/180s、深度4/90s、两审计合计4/90s、统计/文档/交付180/180s，总190/600 CPU command-wall seconds；GPU0、训练/模型预测/新回波/采样0，无常驻任务进程。单位是主机命令时间，非研究人员思考时间。

源码 `cnh_graded_peak_match_dev.py`、`cnh_graded_peak_match_depth_dev.py` 及两对应audit；载荷 `artifacts.local/work/cnh-graded-peak-match-dev-20261010/`。根PLAN/schema/events/pairs/attempts/stats和audit保存匹配，depth/repair-1保存深度对照与审计，失败记录与收据均保留。所有源/输入hash留存，复现应使用新输出入口，禁止覆盖既有PLAN。

## 下一决定

联合current候选及全部原强轻提醒保留，不采用深度差静默；本轮固定匹配和对照已完成，停止放宽容差或继续同配方找边界。

下一优先检验固定当前分数/阈值的BODY定向追加对照，比较它与两高度追加的clear/pass成本、BODY检出/提前，以及放弃HEAD新增救回4/14/6件和提前的代价；这是后续候选，尚未执行或决定采用。若结果支持，再考虑两高度分别校准。旧高度filter/replace预算实验没有稳定收益的结果仍保留；新追加控制问题不同，不能把它泛化成禁止所有高度预算机制。

原M3/5格/L2/body truth/fullbin/480、weak_pass和所有旧run停止身份保留。全部仅已消费受控模拟Development，不是独立确认、真实目标归因、coverage/free、实机、安全或App效果证明。
