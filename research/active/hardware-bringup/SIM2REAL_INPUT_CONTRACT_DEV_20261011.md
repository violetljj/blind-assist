# sim→real 第一步：已有 CNH 的训练语义输入适配与差距诊断

2026-10-11，EXPLORE。**适配器和全部真实片段的配对回放已完成，物理输入合同尚未建立，HEAD全触发没有修复。** 保留已有表面回波后，冻结M3、原5格、S集成的HEAD/BODY均12,726/12,726触发。不能把这个结果转写为真实检出率、误报率或有效实机迁移。

这次只读已有真实录制，未新采集、连接设备、刷写、改模型、选阈值或访问保护480/test。原S模拟候选、M3/原5格和所有历史失败保留；本适配器不接入演示/App。

## 交付与执行合同

全部载荷位于[本次目录](../../../artifacts.local/work/sim2real-input-contract-dev-20261011/)。[PLAN](../../../artifacts.local/work/sim2real-input-contract-dev-20261011/PLAN.json)在估计和推理前记录方法、弱参考判据、冻结阈值及预算。原始输入SHA、逐帧解码数组、解析问题、配置/场景清单、仪器估计、两路径分数、逐帧三类标志、补充噪声统计和独立核验均保留。

代码：[录制/适配器](host/cnh_sim2real_input_contract.py)、[仪器估计](host/cnh_sim2real_instrument.py)、[冻结三模型推理](host/cnh_sim2real_frozen.py)、[预算执行器](host/cnh_sim2real_run.py)、[表格/科学图](host/cnh_sim2real_report.py)。复现从仓库根，使用`config/local.toml`的研究Python，依次执行入口的`prepare`、`estimate`、`infer`及报告脚本；已有PLAN不会被prepare覆盖，应使用独立输出目录并重新记录预算。当前OUT常量为本次唯一目录，不允许把重跑覆盖成新的独立证据。

CPU预算≤1800 command-wall s，GPU≤300 s；[预算账](../../../artifacts.local/work/sim2real-input-contract-dev-20261011/budget.json)保守计CPU≤400、GPU≤150 command-wall s，含首轮、修正重放及独立forward。GPU计整段子进程时间，CPU上界显式包含未逐条计时的轻量工具/交付预留，并非CPU利用率或交互耗时。所有任务模型已释放，无新设备或后台服务。

## 录制盘点

13个有帧片段共12,922帧；断序/参照切换后满8曝光计12,726帧，196帧预热不计触发。另两项无帧为NOT_EVALUABLE。[清单](../../../artifacts.local/work/sim2real-input-contract-dev-20261011/inventory.json)逐项保留原始字节路径/SHA、帧数、坏行、缺序、字段、有效距离数量、ambient分布及结构零比例；人读清单在下表。

全部都有`distance_mm/target_status/nb_target`、64×16的`hist_raw/scaler`、64个`ambient_raw/scaler`，没有独立固件signal字段。严格距离有效条件为`D>0 && status==5 && nb_target>0`；早期status5但D=0的格不能算表面。解码`ldexp(raw,-scaler)`是固件输出单位，**不是已识别的光子计数**。H3名义每bin约0.3002784m（8×0.0375348），不把37.5mm称本录制的距离分辨率。initdiag/1MHz有显式配置记录；其他片段按已有H3配置继承，真实bin零点与角响应未标定。

| 片段 | 帧数 | 有效距离zone-frame/总数 | 场景与弱几何 |
|---|---:|---:|---|
| h3-01 | 152 | 9728/9728 | cabinet/background; cabinet approx 63-64cm; measurement origin unknown |
| h3-02 | 306 | 19490/19584 | object introduced before cabinet; cabinet approx 63-64cm; measurement origin unknown |
| h3-03 | 152 | 8973/9728 | object movement; no per-frame positions; cabinet approx 63-64cm; measurement origin unknown |
| h3-04 | 153 | 9792/9792 | restored cabinet/background; cabinet approx 63-64cm; measurement origin unknown |
| h3-1mhz | 186 | 5091/11904 | I2C 1MHz static engineering recording; unknown |
| h3-static | 228 | 6927/14592 | I2C 400kHz static engineering recording; unknown |
| h3-initdiag | 13 | 350/832 | initialization diagnostic short recording; unknown |
| live-demo02 | 5894 | 352975/377216 | stationary bench; actions not timestamp-labelled; unknown |
| live-opt01 | 3952 | 241507/252928 | stationary bench; actions not timestamp-labelled; unknown |
| small-01 | 102 | 6528/6528 | cabinet background; cabinet approx 64cm; weak recollection |
| small-02 | 152 | 9728/9728 | prompted pen but prior RGB inspection showed background; not positive truth; cabinet approx 64cm; weak recollection |
| small-04 | 102 | 6528/6528 | restored background; no introduced object per prior RGB inspection; cabinet approx 64cm; weak recollection |
| preview | 1530 | 97785/97920 | PREVIEW_ONLY_NOT_GUIDED_COLLECTION: pen/hand/cylinder mixed; no frame truth; unknown |
| h3-framerate | 0 | NE/NE | NOT_EVALUABLE: error records only;  |
| small-03 | 0 | NE/NE | NOT_EVALUABLE: concurrent serial connection rejected;  |

所有真实姿态未测，推理统一采用名义静止−10°，设备运动片段也只能按这个工程假设解释。柜面63–64cm是用户粗测，测量起点未知；固件全域中位约639mm。small-02曾提示摆笔但历史RGB抽查仍是背景，不作为阳性。preview保留`PREVIEW_ONLY_NOT_GUIDED_COLLECTION`，手/笔/圆柱混杂，不从目录名称或提示语制造真值。

## 仪器特性与名义模拟器对照

逐zone×bin的中位、时间方差、前后半差、四连续块中位范围见[仪器数组](../../../artifacts.local/work/sim2real-input-contract-dev-20261011/instrument_arrays.npz)；方法、覆盖、留一参数、弱平面拟合见[仪器报告](../../../artifacts.local/work/sim2real-input-contract-dev-20261011/instrument_report.json)。这些块范围是描述性不确定性，连续帧不能视为独立置信区间。

**固定模式/持久残差。** row4 col2 bin1/2，01中位为480.839/2142.850，04为524.167/2229.000，差43.328/86.150；04四块范围11.349/48.335。估计远场bias仅5.033/4.256，说明旧01→04的局部差远大于这两格的远场模板。该zone距离约0.617–0.619m，恰处于bin1/2表面回波区域；持久场景回波和仪器固定模式不能由这些录制完全分离。它是固定格敏感性诊断，不能把柜面回波直接称串扰。

**串扰/bias模板。** 只取严格有效且距离`>max(1.5m,(bin+2.5)*0.3002784m)`的zone-frame，每cell至少8帧取中位；invalid range不当开阔证据。覆盖340/1024格，逐bin覆盖zone数51/51/51/48/38/25/19/19/19/13/6/0/0/0/0/0。未覆盖格标NOT_EVALUABLE、减0；这不是测得bias=0。最强固件返回不能排除未报告的近表面/多径，所以远场模板仍可能受场景污染。

覆盖zone的bin0/1/2中位模板约67.667/4.571/3.516（仅覆盖单元统计）；名义冻结模板为9.460/0.188/0.0247 pseudo-counts。形状与尺度均有差距，但真实光子gain未识别，不能直接用比值宣称物理串扰增强。完整zone/bin轮廓及MAD宽度保留。

**噪声关系。** 对连续序号、稳定距离≤3cm、ambient变化≤20%的远场帧对，排除固件峰±2bin；用差分二阶矩`E[(H[t]-H[t-1])²]/2`拟合`V=c0+cA*A+cB*positive_bias`。24,574块的NNLS为`c0=.0624152,cA=0,cB=.444294`，R²=.002401、相对RMSE=.998354，**该三参数模型几乎不解释差分变化，不接受为物理噪声映射**。留一台架后cB约.1486至.9959且截距变动，显示跨会话不稳；ambient系数落0边界不证明真实无ambient噪声。

[独立均值—方差及时间相关分析](../../../artifacts.local/work/sim2real-input-contract-dev-20261011/audit/noise_law_diagnostic.json)重算1079个可用通道：corr(μ,V)=.303893，`V≈.242238+.504591*max(μ,0)`的R²=.092351；`V/|μ|` P5/中位/P95=.106849/.604013/3.723720。01/04近bin1–4 lag1相关中位约−.086，四帧和方差/(4单帧方差)约.877/.879；远bin8–13则约1.003/.981。差分矩是`Var−Cov(lag1)`，不是独立单帧方差。物理Poisson/Skellam对应**NOT_IDENTIFIED / NOT_EVALUABLE**：缺少真实双计数过程、光子单位、固件平均/处理合同；不能仅凭离散/均方差判定光子分布。

**脉冲及幅值映射。** 固件稳定单返回zone作为弱单平面代理，用距离估计轴向平面、ρ=.3、实际冻结tier6 electronics渲染期望。名义`signal_counts=606.1126027316143,ambient=4`，不是一般proxy默认4000。[独立electronics等价核验](../../../artifacts.local/work/sim2real-input-contract-dev-20261011/electronics-parity.json)在.64/1.5m平面上最大差≤3.411e−13，与训练四角象限×8原始bin聚合一致。

仅拟合一个非负effective gain；未知反射率、填充率、姿态、零点与gain混淆，因此这些数值是条件映射，不是物理计数换算。01/04 gain≈.804，相对L2残差.713/.726；small静止≈.695–.704，残差.350–.367；其它弱代理差异更大。没有稳定单返回覆盖的片段标NOT_EVALUABLE。全片段参数/残差如下。

| 片段 | 弱平面轴向m | 有效gain (rho=.3) | 相对L2残差 | ambient/名义4 | sigma真实/名义(聚合bin) |
|---|---:|---:|---:|---:|---:|
| h3-01 | 0.609 | 0.804 | 0.713 | 0.695 | 0.446/0.466 |
| h3-02 | 0.623 | 0.514 | 0.424 | 0.695 | 0.656/0.429 |
| h3-03 | NOT_EVALUABLE | — | — | — | — |
| h3-04 | 0.609 | 0.804 | 0.726 | 0.698 | 0.446/0.467 |
| h3-1mhz | 1.230 | 1.925 | 0.834 | 0.687 | 0.549/0.300 |
| h3-static | NOT_EVALUABLE | — | — | — | — |
| h3-initdiag | 1.243 | 1.616 | 0.847 | 0.711 | 0.574/0.301 |
| live-demo02 | 0.650 | 0.183 | 0.910 | 0.683 | 0.624/0.347 |
| live-opt01 | NOT_EVALUABLE | — | — | — | — |
| small-01 | 0.610 | 0.695 | 0.350 | 0.570 | 0.419/0.464 |
| small-02 | 0.610 | 0.695 | 0.350 | 0.577 | 0.419/0.464 |
| small-04 | 0.609 | 0.704 | 0.367 | 0.587 | 0.422/0.467 |
| preview | 0.607 | 0.698 | 0.375 | 0.580 | 0.430/0.448 |

01/04聚合脉冲sigma中位约.446bin，名义约.467，FWHM中位1bin，峰外局部质量约15%；某些弱单返回片段真实sigma .55–.62 vs名义.30–.35，可能含多目标/手/场景混合。每片段保存偏移−2..2归一化kernel；没有把它当已测native128光学脉冲。用16个粗bin无法唯一识别光学核、固件滤波和多径。

![已有真实CNH与冻结名义electronics对照](../../../artifacts.local/work/sim2real-input-contract-dev-20261011/instrument_comparison.png)

图中noise散点只画正均值/正差分矩，其余仍在数组中；模板只统计覆盖格，不把未覆盖0当测量。

## 适配器与两路径触发

旧路径：simple01–04用01场景参考，small用small01，live按已保存100帧参考和推理记录的reference_version切换；早期工程/preview无独立参照，用本段均值/样本SD自回放，不能作为独立对照。live收集参照时的帧也保留，参照切换同时重置两路径。

新路径：`z=(H−far_selected_bias)/sqrt(c0+cA*max(A,0)+cB*max(bias,0))`，保留符号与已有表面；不减整场景、删格、裁剪或选阈值。少量映射参数与逐zone模板都只从原始仪器字段拟合，不使用模型触发。**这是训练的“仪器bias减法+噪声底”语义适配原型；噪声拟合质量差、684格bias未可评，尚不满足已校准训练物理单位合同。** effective signal gain/ambient比值/pulse kernel作为差距参数交付；它们与ρ混淆，不硬塞成全片段光子单位换算。

三模型统一使用静止名义投影，历史8曝光、最后5logit指数平滑；每连续段第4曝光开始推理，满8计触发。原模拟几何FP32、特征显式FP32工程模式用于两条路径，区别于模拟FP16与旧台架FP64几何；无精度静默回退。冻结阈值M3=.8557642487、原5格M3 raised=.9404184588或local=4.625390339；S三seed FP32均值tau=1.917951703，只加轻档，强档完全来自原5格。

| 模型/输入 | HEAD触发 | BODY触发 |
|---|---:|---:|
| m3/old | 7875/12726 (61.88%) | 1951/12726 (15.33%) |
| old5/old | 10795/12726 (84.83%) | 6661/12726 (52.34%) |
| Sensemble/old | 10799/12726 (84.86%) | 6726/12726 (52.85%) |
| m3/new | 12726/12726 (100.00%) | 12726/12726 (100.00%) |
| old5/new | 12726/12726 (100.00%) | 12726/12726 (100.00%) |
| Sensemble/new | 12726/12726 (100.00%) | 12726/12726 (100.00%) |

| 片段 | decoded/满8帧 | 旧M3 H/B | 旧5格 H/B | 旧S H/B | 新三模型 H/B |
|---|---:|---:|---:|---:|---:|
| h3-01 | 152/138 | 1/0 | 1/0 | 1/0 | 138/138 |
| h3-02 | 306/299 | 291/1 | 299/294 | 299/294 | 299/299 |
| h3-03 | 152/145 | 97/0 | 145/133 | 145/133 | 145/145 |
| h3-04 | 153/146 | 3/1 | 144/1 | 144/8 | 146/146 |
| h3-1mhz | 186/179 | 4/82 | 6/92 | 6/93 | 179/179 |
| h3-static | 228/221 | 101/118 | 205/221 | 205/221 | 221/221 |
| h3-initdiag | 13/6 | 0/0 | 0/0 | 0/0 | 6/6 |
| live-demo02 | 5894/5852 | 3699/0 | 4786/3426 | 4788/3441 | 5852/5852 |
| live-opt01 | 3952/3896 | 3039/1674 | 3886/2310 | 3886/2311 | 3896/3896 |
| small-01 | 102/95 | 1/2 | 1/2 | 1/3 | 95/95 |
| small-02 | 152/138 | 1/4 | 1/4 | 1/6 | 138/138 |
| small-04 | 102/95 | 38/1 | 95/64 | 95/71 | 95/95 |
| preview | 1530/1516 | 600/68 | 1226/114 | 1228/145 | 1516/1516 |

这是新任务的统一三模型回放，旧H3审计是另一个v5 NN/A2六查询链；原128/128及10-10的248/248满窗台架结果保留，不与本次全片段分母或统一工程入口混同。HEAD命名/盒顺序照继承代码，不从未测身体坐标推断人体位置。

## 三类触发解释

预定义弱支撑：严格有效距离，4×4名义zone射线的径向点经−10°变换，在相应中央HEAD/BODY盒及z=.3..3m内的射线比例≥25%，任一zone满足即“有表面支撑”。HEAD盒y=−.2.. .42、BODY=.42.. .9、x=±.3m。invalid距离仍UNKNOWN，“无观测支撑”不能称已知空走廊。

第三类用事前已定位的row4 col2 bin1/2消融：只作为诊断，若清零这两格标准化输入取消触发，则优先记“两格残差敏感”；其余再按弱表面支撑划分。**不把这种必要输入贡献等同于已证实的仪器固定模式来源**，且不作为修复候选。主适配器不删任何zone/bin。

| 模型/输入/查询 | 有表面支撑 | 无观测表面支撑 | 两格残差敏感 | 合计 |
|---|---:|---:|---:|---:|
| m3/old/HEAD | 7744 | 4 | 127 | 7875 |
| m3/old/BODY | 1874 | 73 | 4 | 1951 |
| old5/old/HEAD | 10631 | 4 | 160 | 10795 |
| old5/old/BODY | 6039 | 618 | 4 | 6661 |
| Sensemble/old/HEAD | 10643 | 4 | 152 | 10799 |
| Sensemble/old/BODY | 6056 | 664 | 6 | 6726 |
| m3/new/HEAD | 12722 | 4 | 0 | 12726 |
| m3/new/BODY | 10140 | 2586 | 0 | 12726 |
| old5/new/HEAD | 12722 | 4 | 0 | 12726 |
| old5/new/BODY | 10140 | 2586 | 0 | 12726 |
| Sensemble/new/HEAD | 12722 | 4 | 0 | 12726 |
| Sensemble/new/BODY | 10140 | 2586 | 0 | 12726 |

旧HEAD三模型的127/160/152个残差敏感帧全部同时有固件表面支撑；不能当成真实假警。04旧5格HEAD144触发中133对两格敏感，剩11有表面支撑，S为127敏感+17有支撑。新路径两格消融取消数0：**饱和时其它输入仍足够触发，不能据此认定固定模式已消失**。新BODY的2586无观测支撑帧是优先审查对象，仍不能称误报。两格消融也会新增少量触发，完整新增数量已保留，避免只报下降。

## 差距排序与下一步草案

以下按本次诊断对下一决定的影响排序，并非已隔离的因果效应大小。

1. **噪声/单位/固件处理合同。** 当前线性floor R²仅.0024，新路径全部饱和；单位gain、平均次数、差分相关和shot-noise项未建立。优先模拟输出scaler/量化、跨zone异方差、时间相关与ambient波动，保留实测矩与拟合失败。先用原始噪声窗口检验改进模型是否重现均值、单帧/时间和方差，避免用触发率选择参数。
2. **空间串扰模板与未覆盖单元。** 训练近bin模板同形而很快衰减，真实候选有更长尾和zone差异；340格覆盖不足，最强固件距离可能漏近表面。把模板中心、zone变化、跨会话漂移及估计缺口作为不确定量补进模拟器；不能把场景表面均值当模板。
3. **输入语义与姿态/角响应。** 整场景减法抹掉柜面，新路径则保留；名义射线/外参只提供弱支持。未来以量尺、安装角和query标注区分“已有表面”“新增物体”“走廊外”。模拟器训练背景表面应保持而仪器bias单独建模。
4. **幅值与ambient尺度。** ρ=.3条件gain在弱片段约.18–1.93，静止柜面约.70–.80，且残差高；一个全局gain不足。使用固定材质/距离锚点分离rho与gain，在不改变旧阈值的独立描述集上验证映射；本轮不强推物理换算。
5. **固定格漂移与时间相关。** row4 col2重复变化能驱动旧5格大部分04 HEAD响应，但也处在柜面距离bin。模拟device/zone/bin持久偏置、慢漂移和相关噪声，避免仅独立Skellam；区分背景内容与电子残差。
6. **聚合脉冲/多径。** 静柜面的粗宽度已接近名义，部分片段更宽但几何混杂；优先级低于floor/模板。模拟可用有效kernel不确定范围与secondary echo，但native光学核仍NOT_EVALUABLE。

**建议重训/微调方案（未执行）。** 先完成electronics随机化与真实统计矩复现，保存原名义模拟器及冻结权重对照；逐次加入空间模板、相关噪声/固件平均、gain×ambient与聚合kernel，采用相同几何/seed的配对机制检查。训练从M3与ordinary骨干小规模微调开始，保留原5格strong及S原候选；若native分数分布变化，再按新特征合同训练S读出，不把旧HGB无条件套新骨干。按录制会话/设备状态隔离真实诊断数据；当前已消费录制只能做描述，不重命名为独立确认。验收先看真实输入数值分布、表面保留和未支撑触发，再在未来query真值上评估配对救/损、时效和提醒成本；阈值校准、模型升级另行授权，不用本次触发压线选参。

**以后最小有标注录制清单（本轮未采集）。** 固定支架记录真实俯角/高度、传感器到目标距离和走廊坐标；逐query标记已有表面、走廊内/外、距离窗内/外，保留UNKNOWN。六条件：开阔/远墙>4.8m；同一大平面.64m；同平面1.5m；HEAD细杆（不用手持）；BODY细杆；匹配距离的走廊外侧杆。各两档ambient×两次独立开始×30s，约12分钟、约3700帧。至少一次重启/重新安装作为独立会话条件；保留固件距离/status/target_count、hist/ambient raw+scaler和明确动作标记，如固件已有signal/平均次数则一起记录。拍尺/标定照片用于人工query标注，不从固件距离生成真值。校准与评价分开会话；这是最小诊断集合，不是自然使用总体或安全验证样本量。

## 独立核验、修正与局限

[原始与数值核验](../../../artifacts.local/work/sim2real-input-contract-dev-20261011/audit/audit_final.json)256项PASS：独立原始字节解码/帧数、far模板340格、噪声24,574块与SciPy NNLS/留一、弱支撑、冻结manifest/hash、触发和三类闭合。[真实forward核验](../../../artifacts.local/work/sim2real-input-contract-dev-20261011/audit/gpu_forward_audit.json)在三片段各16帧、两路径六次fresh forward，末帧M3/local/S分数差0、触发一致；[冻结功能smoke](../../../artifacts.local/work/sim2real-input-contract-dev-20261011/frozen-smoke.json)M3/ordinary与原入口差≤4.768e−7、local差0。

首轮在推理启动后发现ordinary趋势residual的窗口均值需保留FP32，原实现误用FP64；已完整保留`pre-trend-repair/`并重放。M3/原5格不受影响；S分数最大变化约.3031，本数据触发与三类计数不变。没有省略失败或用旧缓存作为修正后的结果。

所有参数均用同批已消费录制探索性拟合，不是交叉场景确认；长台架会话主导总体分母，不能把连续帧当独立事件。far筛选无法排除弱近返回，bias覆盖不完整；mean/variance/pulse/gain含物体与固件处理混杂；真实姿态、光子单位和逐query真值未建立。**当前可交付的是表面保留适配原型与差距证据，不能交付已校准实机报警或迁移成功结论。**
