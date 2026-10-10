# ToF 基线漏检拆解（EXPLORE，无训练，2026-10-10）

新 E2E 漏检是混合瓶颈，不能整体归为信号不足或主干失效。原5格未及时检出的238个相关事件中，59个（24.8%）期望最佳单bin SNR≥4、80个（33.6%）<2、99个（41.6%）在2–4。HEAD高SNR漏检38/98，BODY21/140；旧ideal中高SNR占133/226（58.8%），主要是HEAD横杆。两组构造不同，不能合并或把旧4cm杆结果外推到新1.7cm杆。下一应优先解决短促接触证据的及时读出，同时保留暗薄杆等低SNR组为当前模拟条件下的能力边界；本任务不执行重训。

both/955 的pure-pass通知37→286全部位于3.5/7.5cm侧向间隙，确实属于近走廊边界，但这由该数据集的pass几何预设决定，未证明模型只打扰近边界或这些pass在真实感知上不可辨。更远的clear仍有49次通知（原5格28），因此成本也不能全解释为近边界容许轻提醒。

## 口径、分母与复用身份

只读取已消费的`cnh-frozen-e2e-20261010/hold`及`cnh-counterfactual-dev-20261009/validation ideal`。冻结结局来自已保存分数/grades；不读其它分区的观测或保护480/test，不训练、不模型前向、不调阈值、不新采光子噪声。类别/真值/几何只进入评价分组和期望oracle。继承E2E报告ff63612d、细杆oracle8d08aad9；具体文件SHA见`input_manifest.json`。

按原oracle的sub16、FP64物理口径：`T = rho/.65 × (mu_hi − mu_opaque0)`；`mu_present = mu_opaque0 + T`；`V = mu_present + 16 × ambient`；逐帧最佳bin SNR=`max_native_zone,coarse_bin(T/sqrt(V))`。零反射率目标仍遮挡背景，不能与absent混淆。所有bin均为8×8×16 coarse CNH，非128-bin原始直方图。最佳bin覆盖整个目标在传感器内的回波，不保证入侵走廊的局部表面单独有相同SNR。

固定描述档`<2 / [2,4) / [4,8) / ≥8`；高SNR=≥4，低SNR=<2，中间档独立保留，不把SNR当检出阈值。事件最大值与达到2/4/8的帧数只取f3–13。可见首帧及FOV出视场转移取f0–15；另外记录及时窗首可见、最后可见后持续出视场和重入标记。`visible_counts`为无遮挡且落在径向范围的subray；zone占比为原oracle quadrature可见质量，另列总zone质量/64和有支持zone数/64。出视场用无背景遮挡的几何ray命中定义，遮挡与FOV分列。

结局为同高度首个gap1通知：f3–13及时、f14–15晚、全f3–15无通知静默。M3/local均沿用过去至多5输出权重1/2/4/8/16平滑；margin为及时窗最大平滑分数减原M3 θ=.8557642487、raised M3 θ=.9404184588或local θ=4.625390339。both旧validation取已保存`2026100955/score_current/c15_p64`，用相同冻结gap1重放，不重选工作点。

E2E为768唯一物理scene，K2；HEAD/BODY各128物理scene、256相关事件，clear/pass各256scene、512clip。旧validation为384名义行、324唯一世界，K4；HEAD/BODY各96物理scene、384相关事件，pass64scene/256clip，clear128名义行/512clip（其中absence重复同一世界）。统计聚类键为`physical_key`，帧、K、seed不作独立样本；每个contact scene的K相同，事件比例等于scene内比例再平均。表列物理scene“至少一次该结局”，同scene不同K可落入不同结局，scene计数不能跨结局相加。未新增抽样或显著性推断。

## 原5格未及时检出的高、低与中间SNR

百分比的分母为该高度原5格未及时检出数；全部contact分母另列。括号内scene数为至少一次该档漏检的物理scene。

| 数据 / 高度 | 全contact事件 / scene | 未及时 | 高≥4 | 低<2 | 中间2–4 |
| --- | ---: | ---: | ---: | ---: | ---: |
| E2E hold / HEAD | 256 / 128 | 98 | 38 (38.8%; 27scene) | 32 (32.7%; 16scene) | 28 (28.6%) |
| E2E hold / BODY | 256 / 128 | 140 | 21 (15.0%; 14scene) | 48 (34.3%; 24scene) | 71 (50.7%) |
| 旧 ideal validation / HEAD | 384 / 96 | 86 | 73 (84.9%; 25scene) | 0 (0.0%; 0scene) | 13 (15.1%) |
| 旧 ideal validation / BODY | 384 / 96 | 140 | 60 (42.9%; 20scene) | 16 (11.4%; 4scene) | 64 (45.7%) |

## 分组 × SNR档 × 结局交叉表

下表为全组示例；每格为“及时 / 晚 / 全窗静默”。完整`cross_table.csv`包括高度、shape、最小厚度、rho、侧别、侵入/侧向间隙、背景family及shape×rho×厚度×背景组合；`cross_table_sizes.csv`另列宽/高/深及shape×三维尺寸组合。三臂分别列事件数、物理scene数和组分母。尺寸全宽/高/深也保留于事件ledger，形状区别通过shape及三维尺寸解释。

| 数据 / 高度 / SNR档 | 原5格 | 原M3 | both/955 |
| --- | ---: | ---: | ---: |
| E2E hold / HEAD / <2 | 0/2/30 | 0/1/31 | 2/2/28 |
| E2E hold / HEAD / 2-4 | 4/15/13 | 5/9/18 | 10/11/11 |
| E2E hold / HEAD / 4-8 | 74/26/12 | 73/24/15 | 110/2/0 |
| E2E hold / HEAD / >=8 | 80/0/0 | 76/2/2 | 80/0/0 |
| E2E hold / BODY / <2 | 0/0/48 | 0/0/48 | 2/1/45 |
| E2E hold / BODY / 2-4 | 25/10/61 | 22/3/71 | 47/7/42 |
| E2E hold / BODY / 4-8 | 16/0/16 | 16/0/16 | 25/0/7 |
| E2E hold / BODY / >=8 | 75/0/5 | 64/1/15 | 80/0/0 |
| 旧 ideal validation / HEAD / <2 | 0/0/0 | 0/0/0 | 0/0/0 |
| 旧 ideal validation / HEAD / 2-4 | 19/13/0 | 18/10/4 | 26/6/0 |
| 旧 ideal validation / HEAD / 4-8 | 71/25/16 | 69/14/29 | 95/2/15 |
| 旧 ideal validation / HEAD / >=8 | 208/18/14 | 197/25/18 | 234/6/0 |
| 旧 ideal validation / BODY / <2 | 0/0/16 | 0/0/16 | 0/0/16 |
| 旧 ideal validation / BODY / 2-4 | 16/13/51 | 12/6/62 | 27/12/41 |
| 旧 ideal validation / BODY / 4-8 | 117/1/58 | 110/1/65 | 172/0/4 |
| 旧 ideal validation / BODY / >=8 | 111/0/1 | 99/1/12 | 112/0/0 |

## 高SNR漏检最多的组

优先列新E2E的5个不重叠“高度×shape”组，避免将同一事件按多个维度重复占榜。M3/local分布仅覆盖该组高SNR未及时事件，格式中位数 [P25,P75]；为冻结工作点margin，不是可互比的共同概率。

| E2E组 | 高SNR未及时 / 全组 | 漏检scene / 全组scene | 晚 / 静默 | M3 margin | local margin |
| --- | ---: | ---: | ---: | ---: | ---: |
| BODY sign_edge | 19/64 | 12/32 | 0/19 | -2.377 [-3.184,-1.587] | -1.328 [-1.419,-1.097] |
| HEAD sign_edge | 14/64 | 10/32 | 5/9 | -2.302 [-2.976,-1.407] | -1.107 [-1.277,-0.731] |
| HEAD protrusion | 11/64 | 7/32 | 8/3 | -1.472 [-3.078,-1.178] | -1.021 [-1.169,-0.744] |
| HEAD horizontal | 10/64 | 7/32 | 10/0 | -1.324 [-2.049,-0.673] | -1.324 [-1.462,-0.890] |
| HEAD vertical | 3/64 | 3/32 | 3/0 | -0.606 [-1.707,-0.338] | -1.560 [-1.599,-0.875] |

进一步定位：BODY sign_edge高SNR未及时19/64，其中rho .19为14/32，rho .57为5/32；HEAD sign_edge14/64，其中rho .19为12/32。HEAD protrusion为rho .19、最小厚度7.5cm的11/16；HEAD horizontal为rho .19、厚8.5cm的10/16。两个新背景族均出现，不是单一背景族独有：以上rho .19细分高漏检在staggered_side_piers/twin_side_canopies分别8/6、7/5、6/5、6/4。这些rho/尺寸细分只作描述，不据此选择新阈值。sign_edge厚度2.5cm同时涵盖正面宽板与侧缘窄板，不能仅凭最小厚度认定是同一种信号机制。

旧ideal的前5组与margin：

| 旧组 | 高SNR未及时 / 全组 | 漏检scene / 全组scene | 晚 / 静默 | M3 margin | local margin |
| --- | ---: | ---: | ---: | ---: | ---: |
| HEAD horizontal | 53/96 | 16/24 | 28/25 | -2.585 [-3.490,-1.496] | -1.086 [-1.369,-0.729] |
| BODY horizontal | 30/96 | 8/24 | 0/30 | -2.645 [-3.350,-1.783] | -1.326 [-1.594,-1.097] |
| BODY protrusion | 15/96 | 7/24 | 1/14 | -1.184 [-2.182,-0.849] | -1.214 [-1.630,-1.073] |
| BODY sign_edge | 15/96 | 5/24 | 0/15 | -2.981 [-3.949,-1.740] | -1.171 [-1.549,-0.999] |
| HEAD sign_edge | 14/96 | 5/24 | 9/5 | -1.498 [-2.197,-0.992] | -1.256 [-1.377,-0.778] |

旧HEAD horizontal高漏检53/96，其中rho .25厚4cm为31/32、rho .65厚4cm14/32、rho .25厚10cm8/16；旧BODY horizontal30/96分为rho .25厚10cm15/16、rho .65厚4cm15/32。旧BACKGROUND L_sidewall与stacked_side_shelves均有。

时序决定下一解释：新HEAD全部38个高SNR漏检都仅1帧达到4，其中26晚、12全窗静默；新BODY21个中16只有1帧、5有5帧，全部静默。旧HEAD73个中41/14/18分别有1/2/3帧，43晚；旧BODY60个中30/29/1分别有1/2/5帧，59静默。故“高峰存在”不等于持续可分信息。both在这些高SNR漏检中及时救回新HEAD36/38、BODY14/21，旧HEAD50/73、BODY56/60，但这些救回来自已有不同读出和更高pass/clear成本，不能当成本匹配收益或归因主干因果。

## 低SNR漏检与当前能力边界

新HEAD32个低SNR漏检分别是rho .19、1.7cm horizontal16/16与vertical16/16（各8物理scene）；新BODY48个分别是同暗薄horizontal16/16、vertical16/16，以及rho .19、2.5cm sign_edge16/32。原5格低SNR HEAD30静默/2晚，BODY48全静默。旧HEAD低档漏检0；旧BODY16个均是rho .25、2cm sign_edge16/48（4scene），全静默。各组分母为该rho×shape×最小厚度的全部contact，不把全部shape分母混用。

这些低档事件在及时窗都曾有可见subray，不能归为整窗完全出FOV；暗薄目标的期望信号弱仍是实质边界。新BODY所有contact中32个事件在f13前永久出FOV，细项保留ledger；最大SNR与完整visibility历史一并使用，不把最晚窗的缺失等同整窗缺失。both对新低档HEAD/BODY各救2，旧低档未救，低SNR不代表数学上绝对不可检出。中间档在新BODY占71/140（50.7%），必须保留不确定诊断，不能把它们强行归为信号不足或算法失败。

## 擦边通知的几何、信号与等级

侧向间隙为目标AABB最近横向表面到共向身体走廊±.30m边缘的signed clearance；本数据sensor_x全0，f3–15最小值与静态值相同。clip有无通知依最高HEAD/BODY联合通知；同帧合并一次，轻→强升级可再计通知。表中距离为P25/中位/P75（cm），SNR为同目标及时窗最大最佳bin的P25/中位/P75。旧absent clear没有目标，距离N/A、T=0单列。

| 数据 / 类 / 臂 | 有通知clip / 全clip | 物理scene至少有通知 | 通知强 / 轻 | 距离P25/中位/P75 cm | SNR P25/中位/P75 | 无通知clip；SNR中位 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| E2E hold / purepass / old5 | 37/512 | 35 | 37/0 | 3.5/3.5/7.5 | 2.64/4.99/8.30 | 475; 4.28 |
| E2E hold / purepass / both | 193/512 | 122 | 97/189 | 3.5/3.5/7.5 | 5.60/8.30/9.51 | 319; 3.46 |
| E2E hold / clear / old5 | 27/512 | 27 | 28/0 | 13.5/13.5/18.5 | 1.60/2.50/3.28 | 485; 2.84 |
| E2E hold / clear / both | 45/512 | 45 | 29/20 | 13.5/13.5/18.5 | 1.60/2.31/3.47 | 467; 2.84 |
| 旧 ideal validation / purepass / old5 | 60/256 | 25 | 61/0 | 5.0/5.0/5.0 | 6.42/8.37/12.21 | 196; 5.30 |
| 旧 ideal validation / purepass / both | 150/256 | 52 | 92/132 | 5.0/5.0/5.0 | 6.42/8.37/10.37 | 106; 4.55 |
| 旧 ideal validation / clear / old5 | 28/512 | 17 | 29/0 | 15.0/15.0/15.0 | 2.27/3.60/5.53 | 484; 3.90 |
| 旧 ideal validation / clear / both | 61/512 | 27 | 29/33 | 15.0/15.0/15.0 | 2.27/2.90/3.90 | 451; 3.90 |

新pass：both在3.5cm档有通知104/256clip（40.6%，166次），7.5cm档89/256（34.8%，120次）；原5格25/256（9.8%）和12/256（4.7%）。无通知pass分别152/167clip（both），231/244（原5格）。距离档与size绑定，不是保持其它因素不变的距离因果实验。both有通知pass的SNR中位8.30 vs无通知3.46，说明强目标回波仍可能为pass付出成本；clip联合通知286，逐query通知288，其中相对目标高度另一query有20次，不能将这20次直接归因为目标。

所有新pass目标都在10cm内，原37次和both286次没有远处pass来源；新增成本包括有通知clip37→193，以及片段内重复/升级。因此仅在此有限几何支持上，37→286主要表现为近边界pass的接触分离/工作点代价。它未证明噪声下不可辨，也未测试>10cm pass。新clear有通知45/512clip、49次（强29/轻20），全部目标距离13.5/18.5cm；原5格27clip/28次。这些是超过10cm的背景/目标场景打扰，不能用“都是擦边”抹掉。

旧pass全5cm：both150/256clip/224次，原5格60/256clip/61次；同样没有远处pass对照。旧clear both61clip/62次，30clip有15cm目标、31clip absent；原5格28clip/29次，13clip有目标、15clip absent。clear SNR分布表只针对present，不给absence伪造目标距离，名义absence重复物理键聚类保留。

## 下一机制建议（建议，不执行）

优先投入“短促目标证据保留＋contact窗口监督”的读出/主干训练方案，重点为HEAD较厚横杆、突出物及HEAD/BODY sign_edge，并以旧ideal横杆作机制复现；新E2E并非高SNR漏检占绝对多数，所以避免一次全面重训。正例应落在真实contact的及时窗f3–13，以scene为单位平衡短促可见帧与窗口事件目标，区分可观测支持与临近/出视场状态，避免只奖励f14–15晚检；将pure-pass与clear明确纳入负例或显式非接触类别（旧ordinary训练中pass mask/weight=0），使用同scene相关K分组，并将实际强/轻通知、重复/升级以及pass/clear成本纳入损失或预声明cal代价约束。机制重点是目标局部空间/形状与背景对比、入侵关联及短时证据保持，不能仅扩大幅度累积或放低提醒阈值。rho .19的1.7cm暗横/竖杆、BODY暗2.5cm sign_edge及旧BODY暗2cm sign_edge应写为当前sub16/给定光子预算、材质与姿态模拟下的弱信号能力边界；中间2–4档另保留。新机制训练/阈值只可用另行声明的训练与cal，评价需另行授权的新数据，不在这两个已消费解释集选机制/阈值；本建议没有训练授权。

## 独立复算、预算与交付

独立复算PASS：1280contact事件、1792负例clip、18432帧行、178944统计字段、18432标量通知流、7488交叉表单元、12源文件SHA及全部summary/supplement重新计数一致。另用CPU frozen R.expected对两组各HEAD/BODY代表场景f3/f13重新渲染opaque/present端点，核对GPU期望与独立ray visibility。结果与计时见`audit_receipt.json`；补充三维尺寸表的3456单元另见`audit_sizes_receipt.json`。这是一次独立复算加新增尺寸表局部核验，不是新的确认集或硬件检验。

GPU期望阶段两组receipt累计45.078s（含imports/FP64 CuPy渲染、压缩写盘及同进程CPU控制；含command startup保守按50s计），上限600s；CPU预算上限1500 command-wall s，分析主命令2.360s，独立审计9.811s，错误Python启动0.617s留档。准备/发现/辅助统计/文档整合/直推统一保守计300s，分析计3s，全部审计及新尺寸表核验预留并计20s，CPU累计保守323/1500s；GPU50/600s。具体交付收据见`delivery_receipt.json`。无新采样、无模型前向、无阈值选择；仅期望渲染使用本机CuPy，CPU用于缓存、描述统计和独立复算。

持久payload：`artifacts.local/work/cnh-baseline-miss-decomposition-dev-20261010/`（canonical F盘junction）。逐事件`event_ledger.csv`，逐负类clip `clip_ledger.csv`，逐物理scene/帧`frame_ledger.csv`，完整交叉表`cross_table.csv`及`cross_table_sizes.csv`，两组期望与冻结grade/notice NPZ、源hash、PLAN、receipt、supplement及独立audit保留供复核。源分数、geometry、旧失败记录和冻结baseline不覆盖；本任务不改变App。模拟oracle、有限AABB、相关K和已消费描述统计不等于实机检出率、不可检出定理、物理硬件极限或安全证明。
