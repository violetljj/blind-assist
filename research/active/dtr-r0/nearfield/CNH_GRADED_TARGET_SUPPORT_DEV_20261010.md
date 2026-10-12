# 弱目标原生支持与并行读出诊断

2026-10-10，EXPLORE，基线1311e202；承接[剩余检出诊断](CNH_GRADED_REMAINING_DETECTION_DEV_20261010.md)，固定 baseline/both/head50、三seed、cal/validation及ideal共向条件。

**静默目标有对应的原生支持；主要问题更接近支持弱、短及读出未能过线，而不是 top8 把目标完全丢光。** 全13824个冻结contact事件在及时窗均至少一次有直接目标bin为正且进入top8，包括所有晚检/静默事件。HEAD静默横杆的对应top8支持首次出现中位f10、只持续3个输出帧；BODY静默横杆支持4帧，对照及时8帧。BODY静默标牌边缘则中位f3已经有对应top8，却仍整窗不提醒，不能统一解释成支持来得晚。这里的对应bin可混合背景/噪声，不等于网络识别了目标。

## 身份、目标支持定义与读出结构

物理数据每split384 scenes×4 replicas×16输入帧；本次每split192 contact scenes×4 replicas×13输出帧，合计19968逐帧记录。冻结结果按3政策×3seeds×2splits复用这些物理输入，共13824事件；每seed/policy/split每高度384 contact，每族96、族内每rho48。f3..13及时、f14..15晚检、整窗无提醒为静默。955/956/957简称2026100955/956/957。没有新光子噪声抽样、fit、网络forward、阈值或通知改动。

目标定位只在评价侧读取 authored AABB。采用原渲染器每zone16×16射线、opaque first-return及原遮挡/并列规则；把首个可见目标返回所在raw bin映射到同一zone/coarse bin。这是direct mask，不用高斯/尾波的“任意非零期望”当目标归属。目标期望发射T由同一opaque几何rho=.65与rho=0两端点差分，再乘实际rho/.65；被遮挡背景、ambient和独立crosstalk相消。没有目标存在/不存在世界差分，也没有实测光子差分。GPU期望量与冻结CPU渲染和first-return抽样一致。

源码核查修正了之前拟议的串行链表述，实际是并行支路：

- 全1024 bins进入M3/local的有限体积投影，过去8帧变换到当前query，构造历史/当前等通道。
- 全bins进入ordinary token；中心到query盒的soft membership是一个通道，没有把其余bins硬删除。
- 独立的9 angular nodes expanded gate→正signedlog→径向局部峰→stable top8进入joint追加分数；top8不喂M3。

因此top8覆盖只能诊断joint峰支路，不能用来解释M3输入是否丢失。门控/峰保留使用实际有噪声输入，逐帧重建与冻结top8 index/valid一致；T仅度量这些选择保留多少模拟目标发射。直接bin中的正观察也可能主要来自混合背景。

## 支持弱与支持时序

以下为validation head50，同一事件先取及时11帧的最大值，再在各outcome事件间取中位。各列最大值可能位于不同帧，不能拼成一条逐帧衰减链；物理指标可跨seed相同，因为复用相同模拟输入。投影T为归一化T经过query内原有限体积线性敏感度、过去8帧求和的描述量，单位为该归一化积分量，**不是M3网络分数、体素激活或物体贡献的因果估计**。

| 高度/族/结果 | 事件数955/956/957 | T期望counts最大中位 | top8保留T比例最大中位 | query投影T历史质量最大中位 | 对应top8及时帧数中位 |
|---|---|---|---|---|---|
| HEAD横杆及时 | 71/75/69 | 198.12/198.12/198.12 | .897/.897/.897 | 21.97/21.97/21.97 | 6/5/6 |
| HEAD横杆静默 | 17/13/16 | 58.16/58.16/58.16 | .784/.784/.784 | 4.73/4.73/4.73 | 3/3/3 |
| BODY横杆及时 | 61/62/63 | 152.81/152.81/152.81 | .459/.459/.459 | 26.31/26.31/26.31 | 8/8/8 |
| BODY横杆静默 | 35/34/33 | 58.77/58.77/58.77 | .426/.416/.446 | 7.18/7.18/7.18 | 4/4/4 |
| BODY边缘及时 | 74/75/78 | 215.29/215.29/215.29 | .595/.597/.597 | 31.01/31.01/31.01 | 10/10/10 |
| BODY边缘静默 | 21/20/17 | 27.82/27.82/27.82 | .613/.613/.613 | 7.84/7.84/7.84 | 6/6/6 |

BODY静默横杆的expanded gate T比例最大中位.588，top8约.42；及时横杆gate约.695、top8约.459。但HEAD静默横杆gate约.963、top8约.784，仍不提醒。BODY边缘静默top8比例甚至不低于及时组，而绝对query投影T质量明显较低。单独扩大top8或改成“有峰就提醒”没有直接证据支持。

BODY静默横杆的物理inner目标hit比例最大中位.0846，中心soft历史保留比例最大中位.2669；及时组为.2678/.2967。两者度量不同，且目标其余表面可能在query外，不能把“总目标发射未进入query”全部算成错误丢失。BODY静默边缘物理inner比例最大中位1、中心soft历史比例.9764；该群不支持“中心门控丢光”解释。几何/反射差异和输出条件相关，以上组间比较为描述，不是配对机制效应。

rho对照完整保留：BODY弱横杆静默32/30/30各/48，仍都有对应top8；支持帧中位4，对照弱横杆及时16/18/18的7帧。高rho横杆仍有3/4/3静默，对应支持4..6帧；不能把全部失败归于低rho。全部政策、seed、split、族/rho、placement与timely/late/silent共1782个摘要分组保留，未筛掉不符合解释的事件。

## 首分数与首次目标支持对齐

后验描述的时序对齐只复用frames与父events，不产生运行时特征。f3 joint margin和及时最佳joint margin均按原head50分数减原cut，负值不等于完整多分支政策一定不提醒。

| 静默群 | f3 joint margin中位955/956/957 | 及时最佳margin中位 | 首次对应top8帧中位 | 对应top8及时帧数中位 |
|---|---|---|---|---|
| HEAD横杆 | −2.607/−1.372/−2.164 | −2.176/−.811/−1.880 | 10/10/10 | 3/3/3 |
| BODY横杆 | −1.932/−1.411/−1.587 | −1.369/−.969/−1.140 | 6/6.5/6 | 4/4/4 |
| BODY边缘 | −1.825/−1.235/−1.620 | −1.643/−.947/−1.339 | 3/3/3 | 6/6/6 |

BODY及时横杆首次对应top8中位f4、支持8帧；HEAD及时横杆f7、支持6/5/6帧。HEAD晚检边缘首次支持f7/f8.5/f6.5，首次提醒距该帧中位7/5.5/7.5帧；该差值可以包含模型及策略，不是设备端延迟。原raw机会进一步经[固定短促对照](CNH_GRADED_RAW_BURST_DEV_20261010.md)落实：HEAD及时救8/5/9、BODY9/15/7各/384，clear新clip51/87/45各/512；不默认采用raw_single，固定2-of-3几乎无救回，不续扫窗口阈值。

## 决定、核验与边界

下一优先改检测读出对**少量、断续、低幅度的公共query支持**的利用：保留当前证据与过去累计证据的区别，明确支持出现帧数/幅度及query覆盖，不再先压成一个持续性判据。用运行时可见的bins/public geometry，目标mask不得进入候选输入。最弱假设是这些支持能与clear/pass背景噪声分开；下一小试应同初始化对照旧读出，只用既有Development训练和cal校准，在完整clear/pass成本匹配下报弱横杆/边缘及时救损。该读出改造本run未实现、未训练，不能预报收益；冻结M3/5格/L2/480/weak_pass及所有旧结果/stop不变。

已执行主检查：全部19968输出的top8 index/valid/amplitude一致、保留比例顺序、跨split public geometry一致；期望/first-ray抽样CPU parity。独立核验未import producer：16个shape/split/frame冻结CPU target期望及ray、全部center soft几何、10个原CPU投影aggregate（最大绝对差3.875e−6）、全部13824事件及1782摘要所有成员计数/min/median/max、16个有噪声观测帧与24项metric通过。时序30个非空cohort/1152事件全部重join，首次/最佳margin直接与冻结分数核对；共424448 scalar checks通过，核验命令保守不足25s（含一次NPZ schema修复失败留证）。收据见载荷verification；其网络模型未运行，目标相容top8首次帧不保证峰身份，也不是目标首次可见时刻。

三次实现失败原样保存：首次归档脚本相对路径不可搬迁；repair1 CuPy CUDA12 cublas DLL未注册；repair2零目标发射帧fraction为null却参与断言。修复源码绑定、DLL生命周期、零分母null处理后repair3完成，原PLAN/失败/各执行源码保留，未改科学标准或清洗失败数据。完整主执行40.750s，GPU适用阶段37.610s（含该阶段CPU统计/I/O，不是kernel时间）；失败主命令保守50s、失败GPU阶段保守47s扣原预算。CPU总预算1200 command-wall s：主480、核验180、整合180、raw对照300、合同检查60；GPU阶段180s。成功+失败GPU阶段保守84.610/180s；主累计保守91/480s；独立核验及交付定向命令见receipt，余项按上限记账，总CPU保守不超过91+180+180+96.877+60=607.877/1200s。无新下载/模型forward、Android或硬件修改；渲染器、DLL handle、GPU缓存、临时环境变量全部释放，无任务常驻资源。

源`cnh_graded_target_support_dev.py`；载荷`artifacts.local/work/cnh-graded-target-support-dev-20261010/repair3/`：PLAN、public_weights、两split target_geometry、frames、summary、receipt、first_score_alignment及timing_receipt；父目录保留所有失败与源码。每个无T帧fraction保留null，摘要字段n明确有效分母，direct count仍为0并计入。仅已消费解析AABB模拟、ideal共向、有限背景与相关seed/replica/frames，非新确认、实机、身体安全或用户收益证据。
