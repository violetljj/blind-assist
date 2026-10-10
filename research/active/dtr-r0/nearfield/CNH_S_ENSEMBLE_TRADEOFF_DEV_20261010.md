# ToF S集成默认候选：池化校准与新family收益–误报取舍

2026-10-10 · EXPLORE · 任务 `CNH_S_ENSEMBLE_TRADEOFF_DEV_20261010`。冻结任务成本HGB与47维特征，无训练，无App接入。

**用户决定：S955/956/957任务成本HGB均值升为ToF模拟层面的默认候选。** 用户接受上一确认run的及时净增151/1024（14.75个百分点）及约2.7个百分点的额外轻提醒；原5格强档逐slot保留，新增提醒均为轻提醒。本run按用户可感知的“有通知clip比例”池化选点，完整报告环境波动，不再作“成本≤k倍原5格”的二元通过判定。

**新hold选定点：HEAD救86/损0、BODY救99/损0，合计净增185/1024（18.07个百分点）；far＋clear有通知clip由112/1536升至155/1536，增量2.799个百分点，逐family最大3.646个百分点。** 池化预期2.995个百分点，实测比预期低0.195个百分点；旧2-family cal的漂移为＋1.042个百分点，绝对漂移本轮缩小81.25%。这是一轮有限新family的描述，不把池化稳定性宣布为已确定因果。

## 决定、旧结果与本轮问题

[上一确认](CNH_S_ENSEMBLE_CONFIRM_DEV_20261010.md)的预声明a/d通过、b/c失败原样保留，Git `1547180d`。本轮默认候选身份来自用户2026-10-10的取舍决定，不是把旧失败改写成通过。用户将旧成本漂移解释为cal仅2个family；已证实的是2-family cal至4-family hold出现成本漂移，本轮比较多family池化后是否缩小。即使观察到缩小，也只能支持池化更稳定的描述，不能仅凭这两次数据对比确立唯一原因。

本轮允许将已消费Development的负例重新用于校准；不将其作为新的确认样本。contact不参与池化选点。新hold的阈值扫描仅描述，已选阈值在新hold渲染和读取前封存。

## 协议、数据与访问顺序

[协议快照](../../../../artifacts.local/work/cnh-s-ensemble-tradeoff-dev-20261010/protocol_snapshot.md)及[场景PLAN](../../../../artifacts.local/work/cnh-s-ensemble-tradeoff-dev-20261010/PLAN.json)在渲染前冻结，PLAN SHA256 `dc1f90aefd7e375bb9a520d09b752e59b9cd88176c679fc711ce1c648959ce8b`；[池化校准封存](../../../../artifacts.local/work/cnh-s-ensemble-tradeoff-dev-20261010/sealed_calibration.json)时间 `2026-10-10T14:40:50.148885+00:00`、SHA256 `474b737fa91a1f679fb47a67c316d10ad3fd6112c7f98827fe04c4d32ed5c3ac`。封存后才生成新hold几何及观测。[源核验](../../../../artifacts.local/work/cnh-s-ensemble-tradeoff-dev-20261010/prepare_receipt.json)逐项比对30份可访问已列元数据库存、19252个旧物理键、68个旧背景几何键；新hold物理键、family、背景几何及target尺寸/rho交集均0。排除范围限于这些可访问、具有同构身份键的来源，保护源不读，缺同构几何键的真实/UE来源不宣称已全面排除。

新hold参数：宽0.44937/0.90937m，薄厚0.02137/0.10137m，rho0.20137/0.62137；四pass层实际间隙3.137/8.137/18.137/32.137cm，clear48.137/60.137cm。标牌edge及突出物完整参数在PLAN。四个新family各384scene/768总clip、192far＋clear scene/384far＋clear clip。失败不换场景源。

池化来源为cost-v2 cal/hold、task-cost-retrain cal/hold、S-ensemble-confirm cal/hold六个cohort，仅pass/clear负例，共14个family、3584个physical scene、7168个负例clip，其中far＋clear5376clip。每个family均256负例scene/512负例clip，其中far＋clear192scene/384clip；K2共享场景不能按两个独立样本计算不确定性。

| 来源 | split | family | 负例clip/family | far＋clear clip/family |
| --- | --- | --- | --- | --- |
| cost-v2 | cal | recessed_side_triplets；split_height_side_beams | 512 | 384 |
| cost-v2 | hold | asymmetric_side_buttresses；offset_side_lattices | 512 | 384 |
| retrain | cal | task_double_side_shelves；task_recessed_cross_blocks | 512 | 384 |
| retrain | hold | task_parallel_offset_panels；task_side_alternating_lobes | 512 | 384 |
| confirm | cal | sconfirm_detached_side_pockets；sconfirm_reverse_step_columns | 512 | 384 |
| confirm | hold | sconfirm_offset_shelf_triplets；sconfirm_side_braced_notches；sconfirm_side_zigzag_slabs；sconfirm_upper_lower_shingled_blocks | 512 | 384 |

逐family来源与分母见[pool_family_counts.csv](../../../../artifacts.local/work/cnh-s-ensemble-tradeoff-dev-20261010/pool_family_counts.csv)。对六个cohort的冻结公开特征仅取负例重算；retrain/confirm的S评分与旧缓存逐值一致，未用contact评分收益校准。

新hold事先固定1536个physical scene×K2、4个新family，与全部可访问且具有同构身份键的已列库存排除交集。配比与确认run相同：HEAD/BODY contact各256scene（各512相关事件）；pass四层各128scene（各256clip）；clear512scene（1024clip）。目标类别覆盖横杆、竖杆、标牌边缘、突出物和暗薄层。类别、target_support与作者几何仅用于生成、标签和评价分组，不进入模型输入。

## 冻结模型、通知与选点

六颗HGB逐字节沿用 `6e441f9a` 的S955/956/957 HEAD/BODY产物。实际47维为ordinary平滑margin与raw趋势slope/residual共3维、current峰22项及有效性/invalidity22项；M3/local仅用于原5格强档锁，joint-parent用于E门控。本轮不加入新均值/方差特征。S集成评分沿用原实现的float32三seed均值。

原5格强档逐slot固定；所有新臂仅在其非强slot追加轻提醒，强转轻为0。gap1逐query通知、同帧HEAD/BODY最高等级联合计数。near-pass最近目标表面到共向走廊±0.30m边缘间隙≤10cm，轻通知权重0.25、强通知1；far-pass与clear任何通知权重1。contact f3–13及时，f14–15晚，全窗无通知为静默。原M3与原both955作为冻结对照保留自己的结局。

主选点为最低阈值，使池化far-pass＋clear中“至少收到一次联合通知”的clip率≤同池原5格clip率＋3.0个百分点。完整枚举负例分数ties和端点，逐候选重算gap1实际通知，不假设阈值成本单调，不用contact收益选点。有通知clip率、通知次数、加权成本分别列出，不能互相替换。

池化原5格445/5376=8.277530%有通知clip，目标上限11.277530%，整数通知clip数允许≤606.28。五个臂均选到606/5376=11.272321%，增量161/5376=2.994792个百分点；完整[池化阈值曲线](../../../../artifacts.local/work/cnh-s-ensemble-tradeoff-dev-20261010/pool_calibration_curve.csv)保留，不假设单调。

| 臂 | 精确阈值 | 负例ties | family增量中位/P90/最大（百分点） |
| --- | --- | --- | --- |
| E | 0.02424664565739236 | 2639 | 2.864583 / 4.166667 / 4.687500 |
| S955 | 2.2415704727172856 | 172728 | 3.125000 / 4.088542 / 4.427083 |
| S956 | 2.068687438964844 | 168926 | 2.994792 / 3.750000 / 4.427083 |
| S957 | 2.1173689365386967 | 169610 | 2.994792 / 3.906250 / 4.687500 |
| Sensemble | **1.9179517030715945** | 183997 | **3.125000 / 3.932292 / 4.427083** |

各S单seed与E为同池同规则的独立选点对照，部署候选包只绑定Sensemble阈值。池化总体≤3个百分点不意味着每个family都≤3个百分点。原both955固定对照未用新规则改阈值；其描述曲线tau0逐slot复现原both，并按冻结margin逐tie删除原额外提醒直至原5格，原额外强档保留自己的原等级。它用于展示原both控制的取舍，区别于S/E新增提醒全部为轻档的结构。

## 新hold选定工作点

所有及时救/损与原5格逐事件配对；HEAD/BODY各512事件，来自各256个physical scene。追加型臂对原5格的零及时损失由强档保留结构保证，不另解释成无风险证据。

| 臂 | H救/损 | B救/损 | H+B净增 | v2加权成本 | far＋clear有通知clip | 增量pp | far＋clear通知次数 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| fixed/m3 | 2/14 | 2/21 | -31 | 207.0 | 118/1536 | 0.391 | 122 |
| fixed/old5 | 0/0 | 0/0 | 0 | 211.0 | 112/1536 | 0.000 | 117 |
| both955 | 70/0 | 75/0 | 145 | 375.25 | 165/1536 | 3.451 | 178 |
| E/selected | 73/0 | 74/0 | 147 | 301.75 | 151/1536 | 2.539 | 162 |
| S955/selected | 82/0 | 93/0 | 175 | 332.25 | 156/1536 | 2.865 | 168 |
| S956/selected | 84/0 | 91/0 | 175 | 326.5 | 158/1536 | 2.995 | 170 |
| S957/selected | 81/0 | 95/0 | 176 | 325.25 | 154/1536 | 2.734 | 167 |
| Sensemble/selected | 86/0 | 99/0 | 185 | 325.75 | 155/1536 | 2.799 | 166 |

原5格及时HEAD372/512、BODY308/512；S集成458/512、407/512。S单seed净增175/175/176，集成185；E147；原both955145。S集成相对原both955的逐事件救/损另保存在metrics与ledger，不能把净差直接当作救回数。

相对原both955的及时救/损配对如下（各高度/512）；S集成另外救46件、损6件，合计净增40，不能把该比较也描述为零损失。

| 臂 | 相对both HEAD救/损 | 相对both BODY救/损 | 净增 |
| --- | --- | --- | --- |
| fixed/m3 | 0/82 | 1/95 | -176 |
| fixed/old5 | 0/70 | 0/75 | -145 |
| both955 | 0/0 | 0/0 | 0 |
| E/selected | 4/1 | 7/8 | 2 |
| S955/selected | 13/1 | 23/5 | 30 |
| S956/selected | 15/1 | 22/6 | 30 |
| S957/selected | 12/1 | 25/5 | 31 |
| Sensemble/selected | 17/1 | 29/5 | 40 |

| 臂 | 暗薄H救/损（/96） | 暗薄B救/损（/96） | 标牌H救/损（/128） | 标牌B救/损（/128） |
| --- | --- | --- | --- | --- |
| fixed/m3 | 2/1 | 2/1 | 0/10 | 0/16 |
| fixed/old5 | 0/0 | 0/0 | 0/0 | 0/0 |
| both955 | 28/0 | 25/0 | 35/0 | 24/0 |
| E/selected | 30/0 | 26/0 | 35/0 | 26/0 |
| S955/selected | 29/0 | 32/0 | 35/0 | 31/0 |
| S956/selected | 32/0 | 32/0 | 35/0 | 32/0 |
| S957/selected | 31/0 | 33/0 | 35/0 | 30/0 |
| Sensemble/selected | 32/0 | 32/0 | 35/0 | 32/0 |

S集成暗薄层各高度净增32/96，HEAD及时43/96、BODY37/96；薄暗目标仍有明显剩余漏检。标牌边缘HEAD及时128/128、BODY103/128。

下表为联合通知次数（同帧HEAD/BODY取最高等级），有通知率分母为clip。0–5/5–10cm为near，10–20/20–35cm为far；各pass层128scene×K2=256clip，clear512scene×K2=1024clip。

| 臂 | 层 | 强次数 | 轻次数 | 有通知clip | 有通知率% |
| --- | --- | --- | --- | --- | --- |
| fixed/m3 | 0-5cm | 67 | 0 | 60/256 | 23.438 |
| fixed/m3 | 5-10cm | 18 | 0 | 18/256 | 7.031 |
| fixed/m3 | 10-20cm | 17 | 0 | 16/256 | 6.250 |
| fixed/m3 | 20-35cm | 19 | 0 | 19/256 | 7.422 |
| fixed/m3 | clear | 86 | 0 | 83/1024 | 8.105 |
| fixed/old5 | 0-5cm | 76 | 0 | 68/256 | 26.562 |
| fixed/old5 | 5-10cm | 18 | 0 | 18/256 | 7.031 |
| fixed/old5 | 10-20cm | 17 | 0 | 16/256 | 6.250 |
| fixed/old5 | 20-35cm | 19 | 0 | 19/256 | 7.422 |
| fixed/old5 | clear | 81 | 0 | 77/1024 | 7.520 |
| both955 | 0-5cm | 121 | 119 | 143/256 | 55.859 |
| both955 | 5-10cm | 28 | 74 | 70/256 | 27.344 |
| both955 | 10-20cm | 17 | 10 | 25/256 | 9.766 |
| both955 | 20-35cm | 19 | 9 | 25/256 | 9.766 |
| both955 | clear | 82 | 41 | 115/1024 | 11.230 |
| E/selected | 0-5cm | 71 | 137 | 149/256 | 58.203 |
| E/selected | 5-10cm | 18 | 66 | 69/256 | 26.953 |
| E/selected | 10-20cm | 17 | 7 | 22/256 | 8.594 |
| E/selected | 20-35cm | 19 | 4 | 21/256 | 8.203 |
| E/selected | clear | 81 | 34 | 108/1024 | 10.547 |
| S955/selected | 0-5cm | 73 | 189 | 177/256 | 69.141 |
| S955/selected | 5-10cm | 18 | 104 | 92/256 | 35.938 |
| S955/selected | 10-20cm | 17 | 7 | 21/256 | 8.203 |
| S955/selected | 20-35cm | 19 | 7 | 25/256 | 9.766 |
| S955/selected | clear | 81 | 37 | 110/1024 | 10.742 |
| S956/selected | 0-5cm | 73 | 169 | 170/256 | 66.406 |
| S956/selected | 5-10cm | 18 | 93 | 88/256 | 34.375 |
| S956/selected | 10-20cm | 17 | 9 | 24/256 | 9.375 |
| S956/selected | 20-35cm | 19 | 7 | 26/256 | 10.156 |
| S956/selected | clear | 81 | 37 | 108/1024 | 10.547 |
| S957/selected | 0-5cm | 72 | 181 | 172/256 | 67.188 |
| S957/selected | 5-10cm | 18 | 92 | 89/256 | 34.766 |
| S957/selected | 10-20cm | 17 | 5 | 20/256 | 7.812 |
| S957/selected | 20-35cm | 19 | 7 | 25/256 | 9.766 |
| S957/selected | clear | 81 | 38 | 109/1024 | 10.645 |
| Sensemble/selected | 0-5cm | 72 | 181 | 175/256 | 68.359 |
| Sensemble/selected | 5-10cm | 18 | 98 | 94/256 | 36.719 |
| Sensemble/selected | 10-20cm | 17 | 5 | 20/256 | 7.812 |
| Sensemble/selected | 20-35cm | 19 | 6 | 24/256 | 9.375 |
| Sensemble/selected | clear | 81 | 38 | 111/1024 | 10.840 |

S集成v2成本325.75，原5格211，增加114.75；near增加65.75、far增加11、clear增加38。near轻提醒新增279次，far＋clear轻49次，不能把全部成本增加归因near。新增slot均为轻档且强slot不变，但gap1轻提醒可以填补静默间隔，减少后续重发强提醒：0–5cm强通知由76降到72。因此“强档slot固定”与“发出强通知总数相等”是两个不同量，slot强转轻为0。

S集成far-pass总体44/512=8.594%有通知clip，原5格35/512=6.836%；clear111/1024=10.840%，原5格77/1024=7.520%。

## family波动与漂移

| 新hold family（各/384far＋clear clip） | 原5有通知 | S有通知 | 原5率% | S率% | 增量pp |
| --- | --- | --- | --- | --- | --- |
| stradeoff_alternating_cross_piers | 32 | 43 | 8.333 | 11.198 | 2.865 |
| stradeoff_forward_side_risers | 25 | 36 | 6.510 | 9.375 | 2.865 |
| stradeoff_nested_recess_frames | 26 | 33 | 6.771 | 8.594 | 1.823 |
| stradeoff_split_side_cantilevers | 29 | 43 | 7.552 | 11.198 | 3.646 |

S新holdfamily增量中位2.864583、P90 3.411458、最大3.645833个百分点；最大来自stradeoff_split_side_cantilevers。完整各臂逐family结果在[hold_family_rates.csv](../../../../artifacts.local/work/cnh-s-ensemble-tradeoff-dev-20261010/hold_family_rates.csv)；池化各family增量在封存记录。

| 校准方案 | cal/池化预期增量pp | hold实测增量pp | 实测−预期pp |
| --- | --- | --- | --- |
| 旧2-family cal（确认run） | 1.171875 | 2.213542 | +1.041667 |
| 14-family负例池 | 2.994792 | 2.799479 | −0.195313 |

绝对漂移从1.041667降到0.195313个百分点（缩小81.25%）。本轮工作点本身与旧确认不同，新hold也不同；不是控制全部条件的family数量消融，不能把全部缩小量归因为family数。

漂移统一使用far＋clear“有通知clip率增量”的百分点单位。旧确认run的133→174是满额通知次数，不能直接作为clip率差；旧cal原5格63/768、S72/768，增量1.171875个百分点；旧hold原5格129/1536、S163/1536，增量2.213542个百分点，因此旧2-family cal的增量漂移为＋1.041667个百分点。原约2.7个百分点来自额外41次满额通知除以1536个far＋clear clip；新口径是其中实际有通知clip的比例差，单位区别明确保留。

## 描述性取舍曲线

![S/E/both收益–误报曲线](../../../../artifacts.local/work/cnh-s-ensemble-tradeoff-dev-20261010/tradeoff_curves.png)

[矢量图](../../../../artifacts.local/work/cnh-s-ensemble-tradeoff-dev-20261010/tradeoff_curves.svg) · [全部hold ties CSV](../../../../artifacts.local/work/cnh-s-ensemble-tradeoff-dev-20261010/hold_tradeoff_curve.csv)；候选阈值行数Sensemble/E/both955分别73771/1763/1788。图中黑点为原5格，S/E选定点标记来自池化封存，不用hold描述改候选阈值。

![0–6个百分点区间放大](../../../../artifacts.local/work/cnh-s-ensemble-tradeoff-dev-20261010/tradeoff_curves_zoom.png)

[放大图矢量版](../../../../artifacts.local/work/cnh-s-ensemble-tradeoff-dev-20261010/tradeoff_curves_zoom.svg) · [描述性预算见证](../../../../artifacts.local/work/cnh-s-ensemble-tradeoff-dev-20261010/descriptive_budget_witnesses.csv)。全曲线保留极低阈值的大打扰端，放大图便于阅读用户关心的局部取舍。

下表为事后描述的有限x区间内最高HEAD+BODY及时净增，括号为实际x百分点，仅帮助阅读完整曲线；不作为新的hold工作点或部署阈值。

| hold x≤多少pp（事后描述） | Sensemble净增/1024 | E净增/1024 | 原both控制曲线净增/1024 |
| --- | --- | --- | --- |
| 0.0 | 22（x=0.000） | 48（x=0.000） | 31（x=0.000） |
| 1.0 | 168（x=0.977） | 130（x=0.977） | 121（x=0.977） |
| 2.0 | 177（x=1.953） | 142（x=1.953） | 133（x=1.953） |
| 3.0 | 187（x=2.995） | 149（x=2.604） | 140（x=2.995） |
| 5.0 | 200（x=4.948） | 149（x=2.604） | 145（x=3.451） |

**零额外far＋clear clip的事后描述中，E净增48，高于S的22（原both31）；允许1/3/5个百分点后，S的168/187/200高于E的130/149/149及原both的121/140/145。** S的优势集中在用户接受少量额外轻提醒的区间，不能宣布全域支配，也不对全部目标类别逐点作同样结论。HEAD/BODY、暗薄和标牌边缘的分面曲线保留各自收益。原both曲线含其自身额外强提醒，成本与S新增仅轻档的结构差异同时保留。

x轴为far＋clear有通知clip率相对原5格的增量，y轴为及时净增，HEAD/BODY及暗薄/标牌边缘分别呈现。每条曲线的hold阈值扫描全部用于描述，不反向选择候选包阈值；原5格对应(0,0)。通知器可能产生非单调计数，原始完整ties保留，不只呈现事后上包络。

## 聚类不确定性

固定seed20261010、2000次配对percentile bootstrap，95%描述区间。physical_scene模式重采样完整scene（K2与各臂配对一起保留）；family模式重采样完整family；hierarchical先family再family内scene。各次比率按实际抽得的对应分母重算，不以K或seed扩大独立样本量。

| 聚类模式 | H净增pp区间 | B净增pp区间 | H+B净增pp区间 | far＋clear增量pp区间 | 加权成本增量区间 |
| --- | --- | --- | --- | --- | --- |
| physical_scene | [12.884, 21.099] | [15.310, 23.529] | [15.275, 20.998] | [2.011, 3.618] | [97.500, 131.250] |
| family | [14.844, 18.359] | [17.773, 20.703] | [16.309, 19.531] | [2.083, 3.451] | [101.500, 129.269] |
| hierarchical_family_scene | [12.749, 21.226] | [14.880, 23.723] | [14.891, 21.287] | [1.814, 3.836] | [94.500, 138.006] |

全部臂与每scene向量在[bootstrap_intervals.json](../../../../artifacts.local/work/cnh-s-ensemble-tradeoff-dev-20261010/bootstrap_intervals.json)；上述区间不作为通过门槛或新的工作点选择依据。

按physical scene保持K2及配对结果一起重采样，并另按family成组重采样。K、帧、评分seed均不当独立样本。仅4个新hold family时，family聚类区间是有限模拟环境的描述，不能解释为对现实环境总体的保证。

## 候选包与独立核验

[候选包manifest](../../../../artifacts.local/work/cnh-s-ensemble-tradeoff-dev-20261010/candidate_manifest.json) SHA256 `34b1c026cf0902f8f667de2fad9ac3e795f362bcfb47981223a6b02a1ec8b5c7`，绑定六颗模型sha、47维实现sha、float32均值、精确tau1.9179517030715945、gap1配置、旧原5格强档来源和池化封存sha。模型保持旧产物位置，manifest使用repo相对路径，不覆写权重。

| HGB | SHA256 |
| --- | --- |
| S_2026100955_BODY.pickle | 824a93352542c6f8648e864385b9bae8c9c31194bd265e59ddabbe6a6d59a196 |
| S_2026100955_HEAD.pickle | b588beef916dbf052b945d602c7ace7720c163b0460e6aab4fbdcbdad7af82b3 |
| S_2026100956_BODY.pickle | dc0804a9693a4488055bc86fa21a25aa7e28bec44fb35104e1002595ce049b96 |
| S_2026100956_HEAD.pickle | 38f95121a15a56c07a1c64103c83feacc705a89428e5bf5efdfcc8ad22952071 |
| S_2026100957_BODY.pickle | 7b7812b893bb85518da9fd66f89c1ebac6518581ad8608b3bf0e814135724e60 |
| S_2026100957_HEAD.pickle | dc968fe2853eca555960b233e13997df635b471304067e73cdf5b0b9354077d2 |

[逐事件ledger](../../../../artifacts.local/work/cnh-s-ensemble-tradeoff-dev-20261010/event_ledger.csv) 49152行；[逐通知ledger](../../../../artifacts.local/work/cnh-s-ensemble-tradeoff-dev-20261010/notification_ledger.csv) 13600行；[逐slot grade/通知归档](../../../../artifacts.local/work/cnh-s-ensemble-tradeoff-dev-20261010/hold_grades_notifications.npz)保留所有8臂，可复算及时、晚、静默和联合分级。

[独立核验](../../../../artifacts.local/work/cnh-s-ensemble-tradeoff-dev-20261010/independent_audit.json) **PASS**，20.938command-wall秒。独立重算数据排除、6模型/3特征实现身份、负例抽样14976个冻结HGB预测、完整ties选点、候选阈值、逐事件/逐通知统计与场景聚类区间；同帧联合计数和强slot锁覆盖。核验也验证元数据路径修复前后科学函数AST一致。

候选包用于以后台架或App接入时明确身份与接口，本轮只生成manifest，不完成App迁移、真实标定或用户测试。收益与打扰曲线支持模拟候选的工作点理解，不扩大证据边界。

## 预算、失败与结论

完整数据阶段实测86.469秒，包含渲染、冻结前向及特征/评分；GPU按整条数据命令保守计100/1200秒。评价、审计、文档与整合command-wall合计保守上界500/1200秒，包含独立子任务、失败命令及剩余交付预留；上界不是每条CPU命令逐项精确计时。阶段实测源准备29.734秒、负例池评分7.078秒、评价9.391秒、独立核验20.938秒；详见[预算记录](../../../../artifacts.local/work/cnh-s-ensemble-tradeoff-dev-20261010/budget_delivery_accounting.json)。没有触及预算上限。

保留三次执行失败：旧缓存字段别名、封存前初始语法、候选manifest的F盘junction路径序列化。均在对应产物完成前修复，未更换场景源、模型、特征或选点规则，未重渲染hold。[元数据修复receipt](../../../../artifacts.local/work/cnh-s-ensemble-tradeoff-dev-20261010/source_serialization_repair_receipt.json)保留原控制器快照与前后hash，独立审计验证科学函数AST相同。任务GPU进程与常驻资源释放由交付receipt核对。

旧确认与本轮clip率漂移另有[独立复算](../../../../artifacts.local/work/cnh-s-ensemble-tradeoff-dev-20261010/independent_drift_recompute.json)；候选阈值不因曲线描述或漂移结论修改。

维持用户决定的S集成模拟默认候选，候选包冻结在池化负例点。新hold在约＋2.80个百分点far＋clear有通知clip代价下，保留HEAD＋16.80、BODY＋19.34个百分点及时收益，优于本轮固定原both的总体收益与打扰组合。池化后总体增量漂移较旧2-family cal缩小，但最差新family仍为＋3.65个百分点，薄暗目标仍有漏检。接下来若接入台架或App，先以manifest复现冻结特征、评分、强档和gap1通知，再检验真实输入下的收益与提醒率；本轮不执行接入。

本轮不训练、不改变冻结权重和特征、不访问保护480/test。库存排除限于可访问且具完整同构键的已列源；背景family仍是有限解析AABB模拟。共向走廊假设、名义外参和短窗口通知，不证明实机误报率、真实用户收益、硬件极限或安全性。已消费负例池用于选点，新hold最终亦成为已消费Development。
