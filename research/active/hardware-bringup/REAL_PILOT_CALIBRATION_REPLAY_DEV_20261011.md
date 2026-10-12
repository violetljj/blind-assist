# 宿舍实机 pilot：仪器参数与冻结模型首次十条件回放

2026-10-11，EXPLORE，描述性。**判读：仍不可迁移。** 宽松稳定窗使噪声/幅值参数可估，但v2未让空通道安静：原5格和S在4/4b/9均为HEAD/BODY全满窗强报警，与v1相同；墙、杆、头部横杆也报警，因此不能把正场景报警解释成迁移成功。M3 v2在空通道的HEAD同样全报警。本次只改变适配表示，冻结模型、阈值及保护480/test边界保持原合同。

数据为`772878a5`的[十条件头戴录制](REAL_LABELED_CAPTURE_PILOT_20261011.md)：1507帧，1362满8历史帧、145 warmup，阳台/重启段23缺序号使历史重置。所有帧均回放；稳定筛选只用于仪器估计。预算CPU≤1500累计command-wall s、GPU≤300；结果/代码/独立核验均已完成，资源已释放。完整payload入口：`artifacts.local/work/real-pilot-calibration-replay-dev-20261011/`，本轮主读数在`relaxed-pilot/`。

## 稳定窗与数据角色

[PLAN](REAL_PILOT_CALIBRATION_REPLAY_PLAN_20261011.md)先写严格口径；严格筛选只有2/3各152帧、其余0，原参数NOT_EVALUABLE与fallback回放、代码snapshot完整保留于payload根目录。用户随后明确要求“不用对条件那么苛刻”，在第二次拟合前写入宽松修订：全传感器总正回波相邻变化中位≤20%/P90≤40%；有距离参考时允许80mm中位/250mm P90的窗口变化；排除近零距离伪读数，距离参考不足允许ECHO_ONLY并标明不可评。不再用有效状态变化作硬门槛。这一修订在strict回放之后，属于用户授权的探索，非独立预注册确认；没有根据宽松回放结果继续挑规则。

|条件|宽松稳定帧/总帧|占比|距离+回波窗 / 仅回波窗|严格稳定帧|
|---|---:|---:|---:|---:|
|1|128/143|89.5%|0 / 16|0|
|2|152/153|99.3%|19 / 0|152|
|3|152/153|99.3%|19 / 0|152|
|4|152/153|99.3%|19 / 0|0|
|4b|152/153|99.3%|19 / 0|0|
|5|152/153|99.3%|19 / 0|0|
|6|152/153|99.3%|19 / 0|0|
|7|152/153|99.3%|19 / 0|0|
|8|152/153|99.3%|19 / 0|0|
|9|112/140|80.0%|1 / 13|0|

全局低变化不证明每个zone静止；阳台全部ECHO_ONLY、9主要ECHO_ONLY。严格口径失败包含有效range不足、状态跳变和弱回波相对噪声，不能称“测得头动很大”。本轮宽松口径适用于pilot描述，不替代固定台架噪声测量。

1–4b用于参数：1模板、4/4b噪声、2增益/核；3只检验2的幅值和核外推，不回流校准。5–9只作报警评价（固定残差跨段比较也只诊断，不拟合适配器）。4/4b报警是拟合内诊断，9是独立条件评价，但均来自同次会话。

## 仪器参数

**噪声有可描述的关系，物理单位仍未识别。** 4/4b各19个稳定窗，共31934个窗×zone×bin观测；排除结构全零、有效固件峰±2bin与ambient过快变化。NNLS：

`Var(H) = 0.145813659 × max(mean(H),0) + 0.089664644 × ambient + 0.081290590`

样本方差ddof1，R²=**0.533699**，相对L2 RMSE=.637048，rank3、列归一条件数1.720。明档单独R²=.676139，暗档=.384189（暗档c落0边界）；奇/偶窗交叉检验R²=.437106/.692942。相邻差²/2口径a=.160950、b=.089139、c=.111394，R²=.513562；样本方差/差分方差中位.9768，P10–P90 .6886–1.6100，未证明帧独立。

与旧v1 R²=.0024相比拟合关系清晰，但样本/筛选/解释变量不同，不能作配对拟合提升。冻结模拟器底是`16×ambient+bias`，其计数域参数约为a=1、b=16、c=0；实测b/16=.00560只是**不同单位下的数值比**，不能直接把模拟器16替换为.08966而保留其余单位。需同时对齐scaler/平均次数、bias、gain、时间相关及固件处理；本次ambient不是lux。

**阳台近bin模板覆盖128/1024单元。** 仅raw rows4–7、bins0–3，可能含栏杆的raw rows0–3不参与模板；其余896格标NOT_EVALUABLE，适配中减0。不补空通道/墙场景均值、不删输入zone。阳台上半场景来自RGB朝外信息，多数固件range无效，模板是`SCENE_INFORMED_WEAK`，不是固件证明无近表面；不能排除未检测近回波。参数/coverage/mask在[instrument arrays](../../../artifacts.local/work/real-pilot-calibration-replay-dev-20261011/relaxed-pilot/instrument_arrays.npz)。

**旧zone34/bin1,2异常未在所有场景重现。** 下表为raw row4 col2，两bin的稳定中位与同bin跨zone空间中位之差；正残差分别只在8/10、6/10条件出现，10条件都没有超过3×空间MAD。近墙大幅值位于墙回波区域，不能认作仪器固定模式；不能称旧固定模式被消除，也没有在适配器清零这两格。全部1024cell跨段残差保存在数组中。

|条件|zone34 bin1/2 中位|相对同bin跨zone中位的残差|超过3×空间MAD|
|---|---:|---:|---|
|1|-0.112 / -1.817|-0.425 / -0.395|[False, False]|
|2|51.600 / 262.417|6.454 / 7.929|[False, False]|
|3|1.217 / 0.233|0.193 / -0.061|[False, False]|
|4|1.453 / 0.717|0.112 / 0.321|[False, False]|
|4b|1.167 / 0.533|-0.180 / 0.102|[False, False]|
|5|1.167 / 0.239|0.259 / 0.051|[False, False]|
|6|0.909 / 0.084|0.061 / -0.057|[False, False]|
|7|1.306 / 0.158|0.413 / 0.016|[False, False]|
|8|1.100 / 1.358|0.185 / 0.167|[False, False]|
|9|1.303 / 0.262|0.254 / -0.051|[False, False]|

**近墙有效增益能外推总能量，不能外推逐bin形状。** 2的64个有效zone、ρ假设.3，fit2 gain(real/nominal)=**.749184686**，能量相对L2残差12.30%。冻结5bin有效核offset−2…2为`[0, .113236, .843131, .039872, .003761]`，σ=.404742聚合bin（约.1215m），只指粗bin有效形状，不是native128bin光学脉冲宽度。3保留29个有效稳定zone：固定gain后总能量相对L2残差**11.16%**、实测/预测能量中位.95070；固定近墙核逐bin外推残差**109.98%**，fit2本身65.51%。固件bin边界/零偏、径向/轴向混淆、墙角度与未覆盖bias可使核移位，不能把全部残差归因于脉冲宽度。轴向弱代理2=.59357m、3=1.33532m，区别于录制中央径向.623/1.452m。

![观测方差拟合与近拟合/远检验](../../../artifacts.local/work/real-pilot-calibration-replay-dev-20261011/relaxed-pilot/figures/instrument_fit.png)

## v1 / v2 冻结回放

v1严格复用`06d49473`保存的bias/噪声参数及SHA，指该报告的“新训练合同适配器”，不是全场景背景减法。v2：

`Z = (H − balcony_bias) / sqrt(max(.145813659×max(bias,0) + .089664644×max(ambient,0) + .081290590, .0006384201))`

噪声底只带仪器bias均值，保留表面回波和负残差，不按当前表面均值缩小Z。有效gain若同单位作用于信号与噪声会相消，故只交付模拟器映射诊断；核用于外推检验，未反卷积、裁剪或删峰。所有实际预测底都高于数值下界，未靠下界选报警。

两路径原始zone顺序、名义Rx(−10°)、恒等历史输运、FP32 engineering/FP32冻结投影一致；每段/断序重置，最近8曝光与最近5logit平滑。实际自然头动、俯角未测，显示方向虽粗核对但没有metric外参，因此此投影是工程假设。M3参考τ=.8557642486787612；原5格strong由raised M3 .9404184587540165或local 4.625390338985158；S均值τ=1.917951703只新增light，strong逐帧完全继承原5格。M3此处单档按strong记，原5格无light；S本次两路径的light均0。

|条件|满窗/总帧|v1 M3强 H/B|v2 M3强 H/B|v1 原5格/S强 H/B|v2 原5格/S强 H/B|S轻 H/B（v1；v2）|
|---|---:|---:|---:|---:|---:|---:|
|1|105/143|0.0%/0.0%|38.1%/0.0%|100.0%/100.0%|36.2%/0.0%|0.0%/0.0%；0.0%/0.0%|
|2|146/153|100.0%/100.0%|100.0%/100.0%|100.0%/100.0%|100.0%/100.0%|0.0%/0.0%；0.0%/0.0%|
|3|146/153|100.0%/100.0%|100.0%/100.0%|100.0%/100.0%|100.0%/100.0%|0.0%/0.0%；0.0%/0.0%|
|4|146/153|56.2%/28.8%|100.0%/100.0%|100.0%/100.0%|100.0%/100.0%|0.0%/0.0%；0.0%/0.0%|
|4b|146/153|98.6%/100.0%|100.0%/100.0%|100.0%/100.0%|100.0%/100.0%|0.0%/0.0%；0.0%/0.0%|
|5|146/153|3.4%/8.9%|100.0%/14.4%|100.0%/100.0%|100.0%/100.0%|0.0%/0.0%；0.0%/0.0%|
|6|146/153|2.1%/2.7%|100.0%/1.4%|100.0%/100.0%|100.0%/100.0%|0.0%/0.0%；0.0%/0.0%|
|7|146/153|83.6%/100.0%|100.0%/100.0%|100.0%/100.0%|100.0%/100.0%|0.0%/0.0%；0.0%/0.0%|
|8|146/153|100.0%/76.0%|100.0%/100.0%|100.0%/100.0%|100.0%/100.0%|0.0%/0.0%；0.0%/0.0%|
|9|89/140|6.7%/7.9%|100.0%/14.6%|100.0%/100.0%|100.0%/100.0%|0.0%/0.0%；0.0%/0.0%|

表中H/B为HEAD/BODY，百分比是满历史帧报警占比，不是实机检出率/误报率。原5格与S相同，所以合列；逐query、逐model、逐路径计数/strong/light比例及符合计数均在[segment_alerts.csv](../../../artifacts.local/work/real-pilot-calibration-replay-dev-20261011/relaxed-pilot/segment_alerts.csv)（120行），逐帧在[frame_alerts.csv](../../../artifacts.local/work/real-pilot-calibration-replay-dev-20261011/relaxed-pilot/frame_alerts.csv)。

|条件|预期 H/B|v1 S符合/不符合/不可判（H；B）|v2 S符合/不符合/不可判（H；B）|
|---|---|---:|---:|
|1|UNKNOWN / UNKNOWN|0/0/105；0/0/105|0/0/105；0/0/105|
|2|ALARM / ALARM|146/0/0；146/0/0|146/0/0；146/0/0|
|3|ALARM / ALARM|146/0/0；146/0/0|146/0/0；146/0/0|
|4|SILENT / SILENT|0/146/0；0/146/0|0/146/0；0/146/0|
|4b|SILENT / SILENT|0/146/0；0/146/0|0/146/0；0/146/0|
|5|ALARM / ALARM|146/0/0；146/0/0|146/0/0；146/0/0|
|6|ALARM / ALARM|146/0/0；146/0/0|146/0/0；146/0/0|
|7|NO_STRONG / NO_STRONG|0/146/0；0/146/0|0/146/0；0/146/0|
|8|ALARM / UNKNOWN|146/0/0；0/0/146|146/0/0；0/0/146|
|9|SILENT / SILENT|0/89/0；0/89/0|0/89/0；0/89/0|

S每路径全部2724 query-frames：符合1314、不符合1054、不可判356；原5格相同。M3 v1符合1114/不符合1254/不可判356，v2为1121/1247/356；总体计数不能把墙/杆与空通道取舍混成性能提升。不可判含1两query及8 BODY；warmup另列，不当静默。2/3/5/6/8 HEAD符合事前报警预期；4/4b/9静默预期全部不符合，7不强报警预期全部不符合。5/6混有筒、目标高度/位置无逐query实测，符合仅指场景预期，不确认细杆检出。8 BODY虽全报警，按UNKNOWN只描述。

主空通道4/4b/9：原5格/S任意通知均**762/762 query-frames**，v1→v2零下降；其中拟合内4/4b为584/584，评价9为178/178。M3的部分未报警在v2反而消失，尤其HEAD全报警。阳台S H/B从105/105各报警变为38/105、0/105，但该条件UNKNOWN，不能作为空通道迁移收益。按事前描述规则，判读**仍不可迁移**，不是否定模型在未来校准/重训后的可能性。

## 不符合典型帧

每条件固定取v2 S首个不符合帧；没有不符合就取首个满窗帧，10张图与逐frame/seq/固件距离记录在[typical_frames.json](../../../artifacts.local/work/real-pilot-calibration-replay-dev-20261011/relaxed-pilot/typical_frames.json)。4/4b/7为frame7；9因早期断序重置，首个满窗不符合在frame13/seq442（后续断序同样重新warmup，未填补）。图中invalid是UNKNOWN，不是无表面；1–3mm伪近读数用mm显示。最强Z所在zone不是已证实报警因果贡献。

![4明档空通道：两路径强报警](../../../artifacts.local/work/real-pilot-calibration-replay-dev-20261011/relaxed-pilot/figures/condition_4.png)

![4b暗档空通道](../../../artifacts.local/work/real-pilot-calibration-replay-dev-20261011/relaxed-pilot/figures/condition_4b.png)

![7外侧杆](../../../artifacts.local/work/real-pilot-calibration-replay-dev-20261011/relaxed-pilot/figures/condition_7.png)

![9重启空通道](../../../artifacts.local/work/real-pilot-calibration-replay-dev-20261011/relaxed-pilot/figures/condition_9.png)

其他条件：[1](../../../artifacts.local/work/real-pilot-calibration-replay-dev-20261011/relaxed-pilot/figures/condition_1.png)、[2](../../../artifacts.local/work/real-pilot-calibration-replay-dev-20261011/relaxed-pilot/figures/condition_2.png)、[3](../../../artifacts.local/work/real-pilot-calibration-replay-dev-20261011/relaxed-pilot/figures/condition_3.png)、[5](../../../artifacts.local/work/real-pilot-calibration-replay-dev-20261011/relaxed-pilot/figures/condition_5.png)、[6](../../../artifacts.local/work/real-pilot-calibration-replay-dev-20261011/relaxed-pilot/figures/condition_6.png)、[8](../../../artifacts.local/work/real-pilot-calibration-replay-dev-20261011/relaxed-pilot/figures/condition_8.png)。

## 模拟器补项与方案草案（未执行）

排序依据是当前输入残差和饱和行为的关联，尚不是干预因果排名：

1. **全空间近bin串扰/bias模板。** 128格覆盖不足以处理另一半视场的强bin0残差；典型空通道图仍有数十至百级Z。先补全zone×bin、设备/boot/温度变化及其减法合同，保留真实表面，不能整场景背景归零。潜在收益最大，最弱假设是剩余强bin0主要为仪器来源而非真实近物体；需固定设备朝开阔视场做全zone仪器测量。
2. **噪声系数/固件处理/单位。** 当前R²中等，异方差/时间相关、scaler和光子域未对齐。用本次明暗观测矩作为模拟输出约束，先同时匹配均值、样本方差、差分方差和相关；单换b会放大未减bias。本轮v2底较小而报警未降，不能把噪声拟合好等同迁移好。
3. **脉冲宽度＋range零偏/聚合bin相位。** 总能量外推近似可行，但固定核逐bin残差大。模拟native核再按实际16bin配置聚合，加入range在bin内相位和可能电子零偏，核/零偏分开估；不要用远墙结果回调本次核。
4. **固定空间模式/相关结构。** 旧zone34 bin1/2没有普遍异常证据，优先模拟当前重复的bin0空间纹理及zone相关，而非特判旧两格。与串扰模板区分静态项/boot变化项，不能把同一现象重复算收益。
5. **有效gain与材料/占用/入射角。** .749只是ρ=.3假设下的有效比例；总能量已有11.2%远检验残差，且一致单位映射会相消，故单改gain预期不能解决当前强报警。以后用同材质多距离固定平面拆分gain、ρ和角度。

**推荐先更新模拟器并重训输入骨干，再重新拟合S。** 保留原模拟器/M3/原5格/S作为冻结对照；逐项加入electronics/空间模板、异方差与固件聚合，再核/零偏和gain随机化。先核对模拟与实测输入矩、弱表面能量/形状，不用报警来拟合仪器参数；小规模重训M3及ordinary/native读出，局部统计的归一底和模拟阈值也按新合同重新校准（另行任务）。同几何/seed做配对，评价空/墙/杆/横杆的报警、救/损和通知成本；随后以新骨干新47维特征重训或微调S HGB，并在新的cal上重新固定轻档参考。

**只微调当前S不足以关闭强报警**：本次没有light，strong由原5格强制继承；任何S参数都不能撤销strong。若未来要允许S仲裁strong，那是新的策略/授权，不能称本轮S微调。真实监督不足，不直接用这次条件弱标签训练安全判决。未来用新会话做pilot复验；本次数据已消费，门槛/参数选择与确认应分离。本任务没有执行重训/微调/新阈值/保护集/App接入。

## 核验、资源与限度

独立[严格核验](../../../artifacts.local/work/real-pilot-calibration-replay-dev-20261011/audit/independent_audit.json)2,000,024项、[宽松核验](../../../artifacts.local/work/real-pilot-calibration-replay-dev-20261011/relaxed-pilot/audit/independent_audit.json)2,276,089项PASS：从raw逐cell解scaler、筛窗、scene-informed模板/mask、SciPy NNLS复算全部31934样本与明暗/奇偶、fit2/check3核和残差、模型SHA、逐frame阈值/grade/符合闭合。计数是算术检查，不是独立科学样本。

独立[CUDA fresh forward](../../../artifacts.local/work/real-pilot-calibration-replay-dev-20261011/relaxed-pilot/audit/forward_gpu_audit.json)4/5/9每段前16帧×两路径PASS：M3/ordinary最大差≤2.3842e−6、local/S差0，ready/grade/reset/history逐值相同，原5格/S strong完全一致。CPU跨后端spot曾发现S seed分数差.127989，失败留证且未冒充逐值通过；同CUDA复算通过。HGB边界可能对特征浮点差敏感，本报告只依赖明示的FP32工程CUDA后端，不宣称CPU/App数值一致。

资源账见[budget receipt](../../../artifacts.local/work/real-pilot-calibration-replay-dev-20261011/budget_receipt.json)。累计CPU保守计入300/1500command-wall s，GPU保守计入65/300command-wall s（strict30、relaxed20、独立fresh15，含初始化）；实测脚本内部strict23.25s、relaxed11.95s，独立GPU命令10.004s。CPU计入全部读写/绘图/核验与CPU spot失败，并预留交付命令，未扩额。原始Python缺NumPy的启动失败、strict不可评、核验精度修正和CPU跨后端失败均保留。

代码：[校准与适配器](host/real_pilot_calibration.py)、[表与绘图](host/real_pilot_report.py)；仪器参数[JSON](../../../artifacts.local/work/real-pilot-calibration-replay-dev-20261011/relaxed-pilot/instrument_parameters.json)。本任务没有常驻模型/worker/硬件连接；原用户采集页面及相机服务为用户资源，未关闭。

单次宿舍头戴pilot、自然头动/俯角未测，近/远杆和筒混杂、7横移未实测、阳台栏杆位置不精确、4/4b帘子改变回波，固件距离不是独立几何真值。恒等历史不能补偿头动；模板大面积未覆盖、弱场景模板和拟合方差可能含真实表面。连续帧不是独立事件，未报告真实性能检出率/误报率、及时性或安全收益。以上限制影响迁移解释，不影响本次冻结回放和参数算术复现。
