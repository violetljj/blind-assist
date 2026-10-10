# HEAD/BODY分别评分、校准与轻提醒预算

2026-10-10；EXPLORE，起始`fb61588d`。用户“继续”接续[公共走廊峰值比较](CNH_GRADED_CORRIDOR_DEV_20261010.md)，使用相同已消费ideal模拟Development。

**按高度分别拟合恢复了部分BODY检出，seed956的空间评分收益尤其明显；按高度硬分预算没有稳定追加收益。分别评分保留为候选，当前完整single轻提醒仍作参照。下一转向距离峰跟踪后的断续支持，不继续在本批固定参数上调阈值。**

## 比较机制

HEAD/BODY共向，强档ordinary_OR、旧M3/5格/L2、ordinary原阈值及全部输入保持原身份。每split384scene×K4×f3–15×2query；及时f3–13，HEAD/BODY各384接触事件；纯pass256clips/3328slots，joint-clear512clips/6656slots。三个seed2026100955/956/957分别报告，不以三seed投票运行。无新回波、投影重建、主干重训/前向、保护480或设备访问。

固定四组合，分别用于score-only和score+spatial两个对照：

| mode | 评分 | cutoff/预算 |
|---|---|---|
| shared_shared | 上一轮共享HEAD/BODY模型与分数精确复用 | 原联合成本预算、共享阈值 |
| shared_height | 仍用共享评分 | 将原联合预算分配到两高度，分别定阈值 |
| separate_shared | 两高度分别拟合评分 | 原联合成本预算、共享阈值 |
| separate_height | 两高度分别拟合评分 | 分高度预算和阈值 |

分数/空间字段沿用上一轮固定3/34维及缺测指示，运行时不使用真值。新拟合每高度使用相同浅层HGB参数：100轮、.1学习率、7叶/深3、min_samples_leaf50、L2=1、无early stopping、seed20261010；非强帧按scene×K×query等总权重、均值权重1。分别评分改变模型数与每模型样本支持，不能只归因于分数尺度。每候选运行时使用一个ordinary模型及一个或两个小评分器，无多seed模型需求。

cal实例8/10拟合，9/11只选cutoff，validation12–15。filter只删原single新增轻档；replace可在原single以下补轻提醒，也可丢原轻提醒。各自固定25%/50%/75%原轻档cal成本目标，完整score ties决定阈值；共288split-cells，12新评分拟合，另6共享模型复用。分区、参数、分配规则在评价前记录于PLAN，未按validation选参数/seed/阈值。

分高度预算先取与shared相同的总目标floor，再按原light各高度的pass clip/clear slot成本比例分摊，用largest remainder补整数，余数同值HEAD优先。两高度target之和等于总target；校准各高度actual≤各自target，query并集≤actual之和≤联合target。并集重复与未用预算逐条保存，**这是保守分配，不保证相同实际成本，也不是validation等成本**。

校准预算沿用原脚本的实际口径：candidate light-query并集；若同slot另一高度为强档，该轻query仍计入预算。评价中的joint-light cost按两个高度最高等级为1计费，排除有强query的slot。两者不能互换；且预算只用于cal-cutoff半实例，全cal384scene的metrics不能直接与半实例目标比较。`audit/cutoff_budget_accounting.json`并列保留半实例预算、query并集和最高等级实际light成本。

## 任务结果与成本

下表是空间评分filter50工作点，便于比较四组合；完整25/50/75及replace保存在summary/metrics，不以这一表作独立确认。及时HEAD/BODY各/384，clear slots/6656，pass clips/256。

| seed尾号 | mode | 及时HEAD/BODY | clear slots | pass clips |
|---|---|---|---|---|
| 955 | shared_shared | 345/290 | 69 | 107 |
| 955 | shared_height | 345/289 | 69 | 110 |
| 955 | separate_shared | 345/295 | 70 | 106 |
| 955 | separate_height | 345/295 | 69 | 103 |
| 956 | shared_shared | 334/278 | 78 | 111 |
| 956 | shared_height | 334/280 | 79 | 112 |
| 956 | separate_shared | 335/299 | 74 | 125 |
| 956 | separate_height | 335/299 | 74 | 126 |
| 957 | shared_shared | 341/308 | 75 | 133 |
| 957 | shared_height | 342/300 | 72 | 129 |
| 957 | separate_shared | 342/300 | 72 | 130 |
| 957 | separate_height | 342/299 | 75 | 126 |

空间separate_shared/filter50相对shared_shared，HEAD救/损0/0、1/0、1/0；BODY救/损8/3、22/1、0/8，不能只报净+5/+21/−8。BODY共同及时事件提前8/11/2、推迟9/6/18件；提醒延迟同样是代价。pass变化−1/+14/−3clips，clear+1/−4/−3slots。因此seed956恢复BODY并非没有打扰成本，seed957也真实丢掉接触。

相对完整single轻档351/344/346和307/310/312，空间separate_shared/filter50及时HEAD损6/9/4、BODY损12/11/12，无救来自filter子集构造；共同及时推迟HEAD23/45/28、BODY26/42/25。仍相对固定强档额外及时HEAD30/22/19、BODY28/42/18，总损0来自保留强档构造；强档等级/时序及强转轻0均核验。不把构造上的强档保留称非劣证明。

其joint-light clear12/16/14slots、pass84/84/77slots、pass58/61/60clips；强/轻slot互斥，但同clip可先轻后强，不能相加。最长pass轻连续3/3/2帧；全部共同及时的提前中位数仍0帧，只覆盖13帧模拟窗口。

score-only也表现出高度取舍：filter50的共享评分及时HEAD337/332/338、BODY294/290/303，分别评分＋共享cutoff为332/320/337、303/297/298；HEAD净−5/−12/−1，BODY+9/+7/−5，pass122/130/136→116/129/133，clear66/69/75→65/68/71。分别评分并非仅空间特征才起作用，也不能默认它总提高总检出。

空间separate_shared的BODY净变化在filter25/50/75分别为+1/+15/−2、+5/+21/−8、−1/+9/+1；replace25/50/75为+1/+11/−4、−1/+22/−2、+2/+14/−9。seed956的恢复跨工作点存在，但其它seed仍有损失；全部救/损、时序和成本见`audit/comparisons.json`。没有要求全seed必胜或零损失，不默认采用的理由是弱BODY取舍与时序还不稳定，而不是可分性不足就关闭分级。

弱族对照以空间filter50为例，各族各高度/96：seed956分别评分使BODY横杆49→54、边缘62→72、竖杆78→80、突出物89→93；seed957横杆59→54、边缘73→71、突出物95→94、竖杆81→81。新增收益确含弱目标，失去的也含弱目标，不能用整体BODY净值掩盖它们。

## 交付与下一步

源脚本`cnh_graded_height_cal_dev.py`；载荷`artifacts.local/work/cnh-graded-height-cal-dev-20261010/`含PLAN、models/参数及哈希、各split分数/grades、144calibration、288完整metrics、442368事件记录。`audit_cnh_graded_height_cal_dev.py`独立核验缓存分数读取/特征适配、score预测、完整ties、整数预算分配、强档、所有cost/paired与账本；18任务自产模型先核hash，24预测数组精确、72原shared_shared split-cells与上轮精确相同，2,396,768标量检查通过。

主命令19.313s、审计15.234s；本轮CPU命令墙钟预算600s，含解释导出/记录保守90s，GPU0。小型缓存评分用CPU，无新主干训练/推理、远端/付费资源或常驻进程；Python加`-B`，不新增源码旁缓存。上轮CPU/CUDA特征计数不一致与原失败身份继续保留。

当前完整single轻档保留参照，按高度评分保留候选；固定参数/全部工作点完成，不追加本配方调参。下一优先在公共空间跟踪距离峰和支持范围，允许断续回波累积，再与完整轻档/空间交互比较首次提醒及pass持续时间；峰身份需检验，约.30m径向bin、背景峰和峰跳转仍是难点。支持增强不自动等于接近，背景可见仍未建立free/静默依据。

此前结果选择了本机制，本轮仍是有限背景、AABB目标、exact位姿、相关replica/帧的已消费Development；未接入App，不证明实机风险分级、用户打扰或独立泛化。旧M3/5格/L2/body truth/fullbin/480及各自停止规则保留。
