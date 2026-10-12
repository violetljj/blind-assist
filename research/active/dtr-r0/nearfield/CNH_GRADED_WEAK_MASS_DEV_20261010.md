# 弱支持读出：正负分离与局部距离 profile 对照

2026-10-10，EXPLORE，基线1c6fc341。承接[原生支持诊断](CNH_GRADED_TARGET_SUPPORT_DEV_20261010.md)，用户再次“推进”，实际实现特征、训练新浅层读出、校准成本并完成配对评价。

**两种读出均不采用。** 正负质量先分离再累积没有改善主要BODY弱横杆群；改成保留前向距离位置的profile后，仍无BODY弱横杆及时救回，且其它群有损失。控制模型精确复现旧分数，强提醒和ordinary轻提醒逐slot保留，故这次负结果不是基线被改动或分数缓存漂移。停止这两套聚合配方，不继续扫阈值、窗口或树参数；下一方向转向保留zone/range局部身份的运动对齐短窗读出。

## 实现、训练与匹配关系

复用旧counterfactual ideal cal/validation，不读新端到端hold，不生成场景/光子，也不访问保护480。每split384 scenes×4 replicas×13输出帧×2query；contact每height/seed384流，族96、族内rho48。及时f3..13，晚f14..15，整窗无提醒为静默。955/956/957简称完整2026100955/956/957。两高度union clear为512 clips/6656 slots，purepass为256 clips/3328 slots。

原base由冻结M3/local融合与ordinary OR强提醒、ordinary_single轻提醒组成，全部保留。只替换原base grade0处的joint追加轻提醒，允许相对head50发生救回和损失，不要求零损。新候选没有通知去重、episode规则、App或硬件修改。

三种读出如下；原强/轻分级基础、训练人群与树配置一致：

- current_control：3个原ordinary分数/趋势描述＋current22峰描述及其缺失指标，共47列。重新拟合，与旧score_current六个split/seed缓存最大差全部0。
- weak_mass：在控制上加28个allbin公共query统计及缺失指标，共103列。inner/ring各14项，包含当前正负密度、质量参与度、top8-bin质量份额、历史逐帧正密度mean/max/std、负密度mean、有效支持帧数、当前/历史比、bounded正幅度和正负抵消量等。所有1024 bins保留；这里的top8质量只是8个最大weighted bin之和，不是local峰筛选或原top8截断。
- weak_profile：首方案负结果后，在同任务剩余预算内执行不同空间表征，原方案不变。inner/ring分别保留12个固定前向深度格[.3,3]m，格宽.225m，每格当前正/负、历史分离正/负密度，共96项＋缺失指标；与控制合成239输入列。native cell按原public geometry支持node的平均forward-depth分桶，是近似profile，不是原生zone/range完整保留、精确物体对齐或全node splat。

历史为真实past≤8，包含可用f0..2；每个历史观测先用当前query的公共membership归一化，再分开正/负signedlog后求统计量。区别于旧“先有符号历史平均、再取正”造成的抵消。历史总量和当前量分开，没有2-of-3或固定幅度门槛。无geometry/分母为零时保留NaN及valid false，零支持不代表free。特征路径只读hist、ambient/bias、public sensor/query/membership及缓存身份，不读目标box、rho、shape、类别或前次目标支持mask。

每个ordinary seed/height分别训练，cal背景8/10的完整原base grade0人群；每scene/replica/query的eligible帧总权重相等，再缩放mean1。固定HGB100迭代、learning_rate.1、leaf7、depth3、min_leaf50、L2=1、random_state20261010、不early stop；三seed来自既有ordinary网络，不是三个新的独立数据集。global12个fit，profile新增6个fit并复用同训练配方的控制checkpoint。类别仅训练标签和评价/成本，不进模型特征；未重新训练或前向M3/ordinary网络，新浅层模型的fit/predict实际已运行。

## 成本校准合同

只在cal9/11选择每seed/候选共享HEAD/BODY cut：总clear union slots和purepass整clip数都不超过同子集冻结head50实际成本。先扣掉保留base已覆盖的成本单位，再对候选分数取最低whole-tie cut。clear同帧任一query提醒算1 slot；purepass整窗任一query提醒算1 clip，不能把旧pass-slot cap当clip cap。该共享cut也用于control，隔离新特征与成本规则变化的效果。

| seed | head50 cap：clear slots/pass clips | current actual | weak_mass actual | weak_profile actual |
|---|---|---|---|---|
| 955 | 126/88 | 125/88 | 126/87 | 126/88 |
| 956 | 108/87 | 108/87 | 108/86 | 108/83 |
| 957 | 111/87 | 111/87 | 111/87 | 111/85 |

这仅是cal9/11的“不超”，不是全cal或validation精确等成本；ties及两项约束会留下预算。全cal含训练背景，成本可高于全cal head50，完整数值保留；没有用validation找cut、配成本或选seed。

## Validation效果与成本

所有三元组按955/956/957，及时/晚/静默每height/seed各/384。表中救/损均是相对原head50及时配对。

| 方案 | HEAD及时 | HEAD救/损 | BODY及时 | BODY救/损 |
|---|---|---|---|---|
| head50 | 352/357/351 | — | 311/316/320 | — |
| current_control | 354/357/351 | 2:0 / 0:0 / 0:0 | 310/313/317 | 0:1 / 0:3 / 0:3 |
| weak_mass | 354/353/350 | 2:0 / 0:4 / 1:2 | 311/311/315 | 1:1 / 1:6 / 0:5 |
| weak_profile | 353/348/346 | 2:1 / 0:9 / 0:5 | 308/310/312 | 0:3 / 0:6 / 0:8 |

相对同成本规则control，weak_mass HEAD救/损0:0、0:4、1:2，BODY2:1、1:3、1:3；weak_profile HEAD1:2、0:9、0:5，BODY0:2、0:3、0:5。955 weak_mass HEAD的+2与control相同，不可称新弱质量特征净贡献。各方案共同及时的提前/延后、完整窗救损和各族/rho结果均在metrics/cohorts与独立comparison保存，未把只提前的子集当总体。

| 方案 | clear slots /6656 | clear clips /512 | purepass slots /3328 | purepass clips /256 |
|---|---|---|---|---|
| head50 | 89/131/102 | 55/84/65 | 711/871/783 | 150/176/164 |
| current_control | 91/122/93 | 56/78/59 | 714/857/775 | 150/171/162 |
| weak_mass | 101/124/97 | 64/77/62 | 722/858/777 | 154/172/161 |
| weak_profile | 85/111/84 | 49/69/50 | 719/845/758 | 154/170/158 |

955 weak_mass比head50增加12 clear slots/9 clear clips及4 purepass clips；955 profile减少4 clear slots却增加4 purepass clips，不能称validation同成本增益。956/957 profile成本部分更低，但contact损失也更多。

主要弱群rho=.25，各/48：BODY横杆head50 timely16/18/18，weak_mass16/16/18、profile16/16/18；两候选及时救回均0/0/0，损失均0/2/0。HEAD弱横杆mass救/损1:0、0:1、0:1；profile2:0、0:3、0:3，单seed救回伴随其它seed损失。BODY弱边缘mass救/损0:0、1:0、0:1，profile0:0、0:0、0:2，没有稳定净改善。

完整窗也没有被“只是变晚了”解释掉：weak_mass HEAD完整救/损2:0、0:2、0:0，BODY2:0、1:7、0:6；profile HEAD1:0、0:6、0:0，BODY0:2、0:7、0:9。mass HEAD晚/静默15:15、16:15、18:16，BODY14:59、9:64、11:58；profile HEAD15:16、17:19、22:16，BODY13:63、9:65、11:61。所有cal、种子、族/rho和强/轻成本均保留，不只报告成功的955。

## 决定、修复与核验

这次实现排除了两个具体配方：增加全局分离质量统计，或进一步加粗距离profile，并未在此固定浅层训练/校准下建立主要弱群收益。这不证明allbin信息无用，也不证明ToF物理上限。profile丢失zone和细range身份，两个候选均未沿一条运行时运动假设对齐局部证据；只是公共query变换，不能称目标移峰或matched读出。

下一优先机制是**当前公共native候选沿已观测传感器运动回溯局部zone/range支持，保留正负与缺帧后再读出**，避免先压成query总量或粗profile。与[另线薄杆期望诊断](CNH_HEAD_THIN_SIGNAL_ORACLE_DEV_20261010.md)提出的可见短窗运动对齐方向一致；其真值matched结果只提示空间，不能进入候选输入或借已消费hold调参。最弱假设是噪声下公共对齐能维持正确支持，而不会把背景跟成目标；比例合适的下一检查是先验证局部运动路径/前缀不读未来，再在原Development全clear/pass成本下比较真实救损。新运动局部读出NOT_RUN，本run不再调这两套聚合特征或校准规则；原M3/5格/L2及旧停止结果保留。

profile首次训练因全NaN列触发HGB binning错误，失败内部2.844s、命令5.632s及原源码/PLAN保存。repair1只在每height eligible cal8/10训练数据上判定finite.any列：HEAD保留227/239、BODY231/239，缺失指标全部保留，推理固定使用同列。没有把未知填成free，也没有用validation删列；特征缓存和统计量不变，修复后6fit完成。全局方案和其原收据未改写。

两特征CPU/CUDA focused等价通过，最大绝对差2.861e−6与1.192e−7；f8以后改变不影响f3..7输出，历史索引无未来访问。独立核验不import新producer/builder：24个28项向量用FP64重算、18 score/grade cells、9最低whole-tie cuts、baseline/strong逐slot、18fit人群/权重/schema/hash、540族/rho cohorts和完整窗时序通过；profile全部训练列保留/删除逐元素重建一致。该核验未重fit或重复网络forward。

预算为CPU1050 command-wall s（feature240、fit/eval360、verify180、integration180、contract90），GPU feature阶段120s；后验profile没有扩大预算。两特征命令合16.219s，GPU阶段9.315s；global fit/eval内部20.593s、profile成功31.859s，含启动与失败保守扣64/360s；独立审计命令4.972s及定向检查保守扣30/180s；合同及交付其它命令分别按90/180上限扣，总CPU保守380.219/1050s，非精确性能计时。所有任务进程/GPU缓存释放，无付费、新数据下载或常驻服务；必要模型/特征/失败留复算。

载荷`artifacts.local/work/cnh-graded-weak-mass-dev-20261010/`：原TASK/PLAN、features、12模型、分数/grades、calibrations、完整metrics/cohorts/verification；`localized/features/`保存96项表征、`localized/`保留首失败、`localized/repair1/`保存成功6模型及独立核验。源码为两features builder、两readout driver及两audit同名脚本。仅已消费ideal共向AABB模拟Development、小量背景实例、相关replicas/帧及后验表征选择，不是新确认、真实峰身份、硬件/用户或安全收益证据。
