# 去 bin0 与仪器域随机化重训 PLAN

2026-10-11；EXPLORE。设计来自已消费宿舍 pilot 的诊断，真实回放仅为开发观察。没有独立实机确认，不访问保护 480/test，不接入 App。

目标是去掉 query 范围外且失配最大的输入，验证模拟及时收益是否保留，再描述真实行为。旧 M3、原5格、S 权重与旧候选切点保留。新模型切点只按模拟 cal 规则计算，不读真实结果选点。

## 输入与训练

主臂在全部 64 zone 将 bin0 置零，保持原 16-bin 张量布局；这是删除输入信息，便于继承架构，覆盖 CVR 投影、ordinary 原生特征、LOCAL 和 S 当前峰特征。敏感性臂将 bin0、bin1 置零；bin1 在任务范围内，因此只作描述，不据此选择默认候选。其余 signed-log、公共姿态、历史长度与预处理合同继承原配方。

每个模拟训练物理场景×噪声副本独立抽取一次仪器参数，在该样本整个历史内保持一致；参数在训练准备时抽取并留存，训练 epoch 复用该样本。重新渲染未卷积的可见表面能量，再按 128 原始 bin 积分脉冲和聚合，避免重复展宽已经卷积的直方图。

| 参数 | 事先固定的独立均匀采样范围 | 来源/解释 |
|---|---|---|
| 有效增益 | 0.42–0.73 | 94d28cef 墙2有效增益剖面 |
| Gaussian σ | 0.070–0.115 m | 用户指定的扩展工程范围，非置信区间 |
| 正向拖尾比例 | 0–0.05 | 墙2拟合附近工程范围 |
| 拖尾尺度 | 0.50 m | 墙2拟合上界固定，不单独再调 |
| range 零偏 | −0.10…+0.10 m | 墙距与零偏不可辨识的容许范围 |
| 噪声 a | 0…0.25357091365 | 稳定窗奇偶拟合包络 |
| 噪声 b | 0.08671592722…0.09326653791 | 同上 |
| 噪声 c | 0.58070959930…0.68960300094 | 同上 |
| ambient | 名义 4 × U[1,2.6] | 覆盖4/4b约2.6倍照明差 |

噪声采用 Var=a·positive_signal+b·ambient+c 的有效 Gaussian 代理。上述系数是 decoded 单位的经验值，R²=0.182496，并未识别光子单位；将其用于模拟器能量单位是本次明确的单位桥接假设。训练归一化仍沿用原训练合同；不把弱墙2 bin0 背景当作完整空间真值插入模拟器，不减真实场景。

M3 使用原 CVR 配方：五 seed 0–4，20 epoch，batch256，AdamW lr0.002/wd0.0001，cosine20。数据为原 near train 27,456 行加现有 ordinary augmentation 19,968 行（共47,424行），所有表面按0.33m半宽 author 规则标注，补入 paired-side-slabs/staggered-side-blocks 等现有训练族。ordinary 使用原59,904行（旧39,936+augmentation19,968）、三 seed 2026100955–57、32epoch/batch256/lr0.001/wd0.0001、原 masked weighted BCE 与调度。两网络共享渲染/特征，避免按seed重复生成。

敏感性臂预算优先采用 M3 seed0、ordinary seed955；同一权重复制到接口槽仅为兼容，统计上仍是一模型。S 保留三个随机 seed 的 HGB：100iter、lr0.1、7leaves、depth3、minleaf50、l2=1、无earlystop。主、敏感性各自使用自己的 strong-slot ignore 掩码，在现有 task-cost train 上重拟合47维S，保持原v2成本与特征。

长训练前 profile 代表性渲染、特征与优化步骤，写出各阶段时长推算。总 GPU command-wall 上限5400s（保守计入数据生成、训练与CUDA评分），CPU科学计算/评价上限1800 command-wall s；达到上限停止。若 profile 预计超额，按预计耗时减少 seed 数并先留 receipt，不改变 epoch、训练数据、随机化范围或评价标准。训练先主臂，敏感性使用剩余额度；未完成须明示。

## cal、留出与切点

强档按原模拟负例的 whole-score ties、原 M3/raised/local 匹配成本规则重算新模型尺度；ordinary single 同原 cal 规则。固定解析 LOCAL 若无学习尺度变化可沿用其原切点，具体继承与重新计算需在切点 receipt 中写明。切点均为模拟 cal，不能读取新 hold/真实。

S 池化 cal 继承9e150dd8六组已消费 pass/clear、14family、7168负例clip（far+clear5376），重新计算新模型及输入臂的47维特征。contact不参与选点。按全部 whole-score ties 选择满足相对于旧原5格池化 baseline 的 far+clear 有通知clip增量≤3pp的最低tau，与9e150dd8相同绝对通知预算；gap1真实通知计算，不能用slot计数代替。如果新strong本身已超预算，记录不可行（S取∞不能修复strong），不放宽预算。封存权重hash、所有切点和代码后才生成并打开新hold。

新hold沿用tradeoff目标/布置采样数量，1536物理scene×K2、HEAD/BODY各512接触event，四pass层、clear；所有目标尺寸与rho在旧源标量上加0.00191。背景采用新结构的 alternating-double-bridges、stepped-twolevel-panels、skew-separated-tripod-posts、offset-asymmetric-nested-backs 四family（每族两镜像实例），具体盒体在配套source预先固定，改变侧物连接与数量，不只改名或平移。必须对允许的全部库存元数据（含已消费9e150dd8）检查物理key、背景几何hash、family零交集。只读允许库存metadata，不打开保护数据。新hold冻结名义光子模型，与旧候选在同一观测上配对，不能给新候选独享较容易的观测。

报告及时 f3–13、晚 f14–15；near-pass≤10cm轻0.25/强1，far/clear任何通知1，逐query gap1、同帧联合最高档。按物理scene/背景family重抽样作描述区间；暗薄、sign-edge拆组，不选点。

主接受标准在同一新hold上计算：以旧原5格为共同参照，旧默认S及时净增G_old，新候选净增G_new；G_old>0时要求G_new≥0.8G_old，否则要求新候选及时总数≥旧默认S。far+clear有通知clip率新≤旧S+1pp。以上为预定点估计工程门槛，描述区间不冒充统计非劣性确认。

## 真实开发回放与判读

十段1507帧已消费。旧候选与新两臂都跑：(1)原v2 numerator/denominator；(2)最简只减墙2拟合bin0 bias，其余表面不减，denominator保持v2不变，以隔离背景减法。先适配再去bin；静态公共名义−10°、恒等历史输运，full8作主分母。不按真实结果改权重/切点。

条件预期继承4cd95b2d：墙2/3、杆5/6、头横杆8应有提醒（8主要HEAD）；7外侧杆不应强；1阳台UNKNOWN。4/4b/9空通道单列边界场景：尽头约3m，不能当干净clear或报告实机误报率。描述强/纯轻报警query-frame率及相对于旧候选的变化。

墙2拟合bin0背景的无表面、无噪声空输入按两适配路径回放，报告强/轻。另做任意bin0污染前后所有新score/grade一致性，敏感性同时检验bin1。空输入不是一段真实clear录制。

“墙/杆段保留有提醒”按响应存在性解释：2/3/5/6每段至少一个满窗S响应（HEAD或BODY），8至少一个HEAD响应；不增加最低报警率门槛，低比例不能声称稳定提醒保留。该解释在真实结果打开前明确。

仅当主模拟两项通过、bin0空输入不再触发且墙/杆段保留有提醒，并且主S池化校准状态为SEALED，写“建议新骨干+S替换为默认候选（待独立实机确认）”；否则列明失败项，旧默认候选保留。校准状态要求是原池化预算规则的有效性检查，不是新增性能门槛。真实空通道下降只算开发观察，不作独立确认。后续录制clear选择尽头>4.5m楼道；域随机化与未来训练建议不扩大本次范围。

交付主报告、模型/hash/切点、模拟与逐条件表、输入诊断和独立核验；更新dtr-r0 CURRENT/RUNS与hardware CURRENT，正常直推master。
