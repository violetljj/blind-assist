# 共向条件下的分级提醒与原始回波诊断

2026-10-10；EXPLORE，复用已消费的受控模拟Development。源为`CNH_COUNTERFACTUAL_DEV_20261009`，本轮起始提交`167181b5`；不升级M3、原5格融合、L2或App策略。

**轻提醒具有额外及时检出和提前量，但pass成本明显。当前query内双轴代理的降级收益很小；更值得保留的探索参照是“沿用ordinary OR强提醒，新增ordinary single证据给轻提醒”。短窗max3有小量额外收益，暂作次选。背景可见尚不能作为静默依据。**

## 范围与规则

用户“推进”授权实现、诊断、评价及记录。预算在载荷`PLAN.json`声明：分数分析CPU命令累计600秒、原始抽查CPU命令累计600秒；GPU/训练/模型推理/新光子采样均0。只访问ideal cal/validation和历史公开输入，不访问保护480、设备或新采集。原停止规则保留。CPU用于小型缓存分数、I/O和统计（TASK_NOT_GPU_SUITABLE）。

固定ordinary三个seed `[2026100955,2026100956,2026100957]`分别报告；运行时每个候选只用一个ordinary读出，未用多seed投票。HEAD/BODY共向；每split384scene×K4×13帧，HEAD/BODY各384接触事件，joint-clear6656slots/512clips，纯pass3328slots/256clips。及时为f3–13，晚提醒为f14–15；提前量以帧计，不冒充实机反应时间。

管线核实：新旧都有过去≤8帧输入和≤5帧`[1,2,4,8,16]`因果加权分数平滑；新ordinary读出保留current/历史signed-log均值及membership门控历史，新M3基线才经过tot/cnt/lst体素投影。ideal使用exact公共位姿；旧NEAR的noisy位姿不能移植为本批描述。因此本次max/k-of-n改变打分后的时间读出，并非首次引入累积。

初始双轴`dual`：存在代理为旧query**内部**局部patch支持≥4.625390338985158；侵入代理为ordinary smoothed logit≥已有standalone-cal阈值。二者均不是独立存在/侵入真值估计器。

- 强：`M3>=.9404184587540165 OR (local过线 AND ordinary>=singleθ) OR ordinary>=oldOR追加θ`。
- 轻：未达强档、且`oldfusion OR ordinary>=singleθ`。
- `dual_max3`/`dual_k2of3`只向轻档追加过去≤3帧raw logit最大值/第二大值过线；k2至少要2帧。各自仅在ideal cal匹配原single实际65 joint-clear slots，整tie匹配残差保留；并集不保证成本仍65。

三个seed的singleθ `.565796455/−.530725539/.269024428`，追加θ `3.17705393/2.27108908/2.40092087`，均继承旧校准。max3θ `2.343435287/.702963650/1.973130345`，cal实际slots65/64/66；k2θ `.542996824/−.607477605/.091469213`，实际65/64/65。cal最终dual并集为141/144/143slots，不称等成本。

初dual结果显示降级成本收益有限，随后先声明`strong_preserve/PLAN.json`再运行同覆盖比较：强档=`oldfusion OR ordinary>=oldOR追加θ`；轻档=`ordinary>=singleθ AND未达强档`。它与dual总提醒逐slot相同，仅恢复原ordinary OR强档。此选择发生在观察初结果之后，属于探索；没有修改阈值或宣称独立确认。

## 检出、及时性与打扰

以下均为validation ideal。旧5格HEAD/BODY及时298/244各/384；ordinary OR为315/267、313/257、323/282。`strong_preserve`的强档恰好保留这些OR提醒。

| seed尾号 | 总及时HEAD/BODY，各/384 | 仅轻及时HEAD/BODY | vs ordinary OR救/损HEAD；BODY | vs旧5格救/损HEAD；BODY | clear强/轻slots，各/6656 | clear总clips /512 | pass强/轻/总clips，各/256 |
|---|---|---|---|---|---|---|---|
| 955 | 351/307 | 36/40 | 36/0；40/0 | 53/0；63/0 | 58/15 | 40 | 89/138/149 |
| 956 | 344/310 | 31/53 | 31/0；53/0 | 46/0；66/0 | 58/49 | 65 | 103/156/168 |
| 957 | 346/312 | 23/30 | 23/0；30/0 | 48/0；68/0 | 58/22 | 47 | 115/140/158 |

强/轻query档互斥，joint slot按两个高度中的最高档计费；同一clip可先轻后强，因此强/轻clip数不能相加。旧5格clear58slots、28clips、pass60clips；新增轻档后总clear73/107/80slots、pass149/168/158clips。相对ordinary OR，新增轻档clear为15/49/22slots，pass总clips增加60/65/43。轻提醒不是零成本。

同为及时检出的事件，相对ordinary OR，HEAD提前127/121/130件，BODY提前99/118/97件；完整advance直方图见metrics。全部共同及时事件的提前中位数仍0帧，不能只报发生提前的子集。相对原M3总提醒HEAD救/损67/0、61/1、62/0；BODY86/0、89/0、91/0。seed956保留1件原M3及时损失；其它共同及时的原M3事件亦有推迟，详见逐事件账本。

初dual降级相对ordinary OR：HEAD有1/0/1、BODY有3/2/3件及时强提醒变为仅轻档；另有HEAD各1件、BODY5/6/5件强档推迟。validation仅省强clear2slots、pass1clip/seed。保留强档恢复这些事件/时序，增加成本仅上述量；因此本轮不采用该query内代理降级，不否定更好的存在/侵入证据或分级机制。

| 时间追加，相对初dual | 新及时HEAD/BODY，955/956/957 | 共同及时提前HEAD/BODY，955/956/957 | 新增clear slots，955/956/957 | 新增pass clips，955/956/957 |
|---|---|---|---|---|
| max3 | 0/1；5/1；4/1 | 9/10；14/20；8/11 | 9/41/14 | 2/9/4 |
| k2of3 | 1/0；2/1；0/1 | 8/8；12/8；13/13 | 11/24/19 | 4/3/5 |

追加是并集，零总损失来自构造而非非劣证明。max3比k2有更多新增及时事件，但各seed成本不同；未据validation选seed/新阈值。最长连续轻提醒clear为max3各3/4/3帧、k2各3/4/3帧；pass为max3各6/9/6、k2各6/9/6帧。它们只覆盖13帧模拟窗口，不代表长期会话打扰。

相对旧5格，横杆HEAD救27/23/23、BODY26/24/27；标牌边缘HEAD20/17/19、BODY17/19/18；竖杆BODY仅5/8/8，HEAD本来96/96饱和。新增证据的主要机会仍在横杆/边缘及BODY弱目标，不能把整体涨分当成所有细障碍都解决。

## 分格与原始回波

`deadline_grid.csv`按f13 scene×replica×query分格：local过线、ordinary single过线、raw过去≤5帧趋势>0、去趋势RMS>cal全部f13的中位数。残差cutoff分别`.700782752/.739146281/.737786118`，单位logit；趋势logit/帧。门槛仅作描述，不进入候选策略。背景可见全量字段为UNKNOWN，覆盖缺口不会被当作负证据。

截止快照中“local高、ordinary低”contact仅5/5/4，clear33/46/33；“local低、ordinary高”contact173/174/187，同时pass114/118/136。这里clear指逐query clear（包括接触clip的另一高度），不等于joint-clear clip分母。当前局部存在代理不能把轻档误报清楚排除；两轴低也不证明上游没有信号。五维表的背景列尚未建立全量测量。

原始抽查、相近分数格的趋势/残差诊断与核验见下述载荷；原始峰仅为观测峰，不自动归因目标。允许“证据不足”，不会据少量未见峰关闭峰值保留或全部分级方向。

带对照原始抽查最终按背景族×HEAD/BODY×全3seed及时错/对分层，各非空格取首例，共8锚点、32角色样本、24独特原始观测、416帧、1293局部峰。包括两个validation背景族，每族仅一个实例；角色间可重复观测，不能扩大样本量。同一次hist读取完成标准化、原始峰/径向距离及center/extent公共query门控摘要。全部所选有效FP16历史与旧histories缓存逐值相同；原compact feature缓存已清理，该缓存门控对照为NOT_EVALUABLE。原生angular节点支持量不等于实际物理可见覆盖，也没有重建M3体素或运行模型。

抽查miss及成功/pass/clear都有观测峰，但目标回波归因均为证据不足；“存在峰”不证明目标未丢，也不证明CNN该报。初lexicographic抽样仅覆盖单一背景，已保留并补分层版本；首版将空compact feature核验误报为max差0，错误结果和更正均留证，最终明确缓存不可评价。不同场景光子噪声不同；形态/反射率不匹配字段保留，未称严格匹配。

另先声明`raw_echo/background_visible_v1/PLAN.json`再试轻档静默候选：当前公共走廊相同x/y、z>3m的far观测峰normalized_z≥2.5，旁侧near(.3..3m)峰≥2.5，far depth比旁侧near远≥.75m，query内near峰<2.5；只对grade1静默，grade2全部保留。匹配要求同background_id/query/replica/frame/shape/rho/group/side，以及far和near距离各差≤实际一bin（.3002784m）。去重24个query事件的312slots中候选6slots，严格contact/pass匹配0对，因此**匹配比较NOT_EVALUABLE**。

未匹配小样本评价仅作候选失败线索：每seed静默1个contact light slot并丢1个及时接触事件，pass减少0/0/1slot、clear0。当前这条背景可见候选不进入策略；少量有损样本不能否定所有背景证据。新增阈值为观察初分级结果之后的探索声明，未按此子表反复调参，不是独立确认。

追加`audit/PLAN.json`预先固定ordinary分数margin格`(-∞,−1,0,1,+∞)`，然后比较相近分数格的趋势和去趋势残差。144行表中，validation两个正margin格的contact趋势中位数各6/6高于pass；全部24组seed×height×score格的趋势及残差p10–p90区间均重叠，负margin的contact残差并未稳定更高。它支持保留“支持增强”描述，不支持当前用波动单独解释侵入难判或据此降档。

## 复现与证据

- 脚本：`cnh_graded_evidence_dev.py`；初始命令无参数，同覆盖后续命令`--preserve-strong`。初始PLAN/source哈希与后续不同，原记录保留。
- 载荷：`artifacts.local/work/cnh-graded-evidence-dev-20261010/`，包括PLAN、cal阈值、18单元完整metrics、55296逐query事件记录、576分格记录、等级数组；`strong_preserve/`为6单元同覆盖对照。
- 初分数命令1.203秒、同覆盖后续0.360秒；追加抽查/审计时间以各receipt为准，失败与修复一并计量。无常驻进程或GPU分配。
- `audit_cnh_graded_evidence_dev.py`独立重建18原单元及6同覆盖单元，复算全部grades/及时救损/晚提醒/first、分格/分组、joint强轻clear/pass成本和CSV；57118项通过，复核1.516秒。13个future-mutation和前缀核验验证策略只用过去。首次FP64严格残差容差失败保留；生产FP32均值与独立FP64 OLS最大差2.5513e−7，追加3e−7容差复核通过，分格计数仍要求完全相等，无实际指标缺陷。
- 原始脚本`cnh_graded_raw_echo_dev.py`；最终载荷`raw_echo/stratified_v3/`，失败/更正保留在raw_echo首版及`repaired_v2/`。背景规则最终子比较见`raw_echo/background_visible_v2/`，同规则v1保留。原始子任务命令墙钟约21秒（含修复与两次CPU参考工具启动），保守扣30/600秒；独立524288标准化值逐值相同，center/extent公共query算术参考最大差6.16e−6/1.59e−6，见`raw_echo/independent_audit_v3.json`，full-feature缓存缺口如实保留。分数分析含两次审计保守扣10/600秒；没有扩展旧run预算。

这些是有限背景、AABB目标、exact位姿和相关K/帧的已消费Development结果。证明了候选轻档在该批数据的收益/成本；未证明实机风险分级、背景可见即可free、真实安全或独立泛化。下一方向是先补更有区分力的公共扩张走廊/局部峰证据和匹配背景观测，评价能否降低轻档pass成本；这些机制仍开放，当前阈值和原基线保留。
