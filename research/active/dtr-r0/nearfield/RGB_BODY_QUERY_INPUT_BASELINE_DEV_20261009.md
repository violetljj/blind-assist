# RGB输入配对、可调几何与多相机参考

2026-10-09，EXPLORE / COMPLETE_WITH_REFERENCE_DOWNLOAD_INCOMPLETE / PAIR_SMALL_EFFECT_DELTA_ZERO。8个同帧640→256配对的query净变化很小，未解决本组近场问题；δ校准按预定规则选0、退化为重复raw DP，学习读出的额外价值仍未回答；第三相机TUM下载到网络限额，真实帧/GPU未运行。用户“推进”授权并行执行[冻结迁移之后](RGB_BODY_QUERY_TRANSFER_DEV_20261009.md)的输入质量诊断、可调几何基线与多相机参考；实施参数在本执行阶段确定，不冒充圆桌已冻结合同。旧模型、train归一化和实际cutoff不变，不训练/在新数据调参。

本轮预算：CPU辅助累计2400s、GPU阶段执行壁时1200s、新数据下载≤4GiB、新模型0B。payload `artifacts.local/work/rgb-body-query-input-baseline-dev-20261009/`。可调基线CPU600s、输入结构CPU240s、同帧数据CPU240s/下载≤2GiB/网络300s、多相机CPU240s/下载≤1.6GiB/网络300s，其余供核验/收尾；配对DP最多GPU800s、多相机DP240s、冻结head及复现最多GPU160s。到限保存partial，不扩旧run/CNH预算，已授权范围内不增审批门槛。

可调几何：原query射线区间[entry,exit]保持，δ是米制有符号对称余量，预测Z∈[entry−δ,exit+δ]；负δ收窄，反转区间不支持，原不可达射线不扩。运行前固定候选−1..1m、步长.01；仅原cal16帧选|实际FPR−.14190356447903094|最小，tie选|δ|小、再δ小，不用recall选择；先写选定δ/cal实际FPR再评旧validation、新3RScan和ARKit。δ=0理论上已精确匹配，因为目标来自raw DP原cal；若如此，报告校准退化与完整cal曲线，不称调参基线失效或学习头价值已证明。比较全量/环境/距离带及救回损失，工作点取舍明确。

输入结构：现有ARKit16帧全量预测分布/IQR/log-IQR/空间相邻变化、实测深度仅诊断、1000m尾部全保留；固定indices0/5/10/15展示RGB/预测/参考，不按输出挑。分带实测/预测比不证明预测常数。输入配对：官方VGA640×480与原16评分帧有8个exact timestamp，取其固定交集；从同VGA原帧生成256×192面积降采样与半像素K，publicK映射复用同reference。核实时间/视场/K对应才执行；generated低分辨率不是原官方lowres stream。640→256同帧改善仅支持该配对条件中的分辨率作用，未改善不排除交互，不定位原ARKit退化主因。1920wide若不可访问，明示未运行。

多相机参考：核查公开TUM Kinect fr3 long_office_household，许可/K/单位/注册依据有效则预算内下载，预先均匀选16对RGB/depth（最近timestamp≤.02s），640×480，原15query、第一返回稀疏参考。不是完美同步，不由pose补身体/事件真值；不提交账号/联系人或签协议。不以额外机型单捕获结果宣称广泛泛化。

决策检查：旧候选新3RScan .59236/.08489 vs geometry .50464/.09513，在ARKit .29567/.04188 vs .42204/.05606为取舍，近场未充分迁移。各新增评分域正/free分母先未知，无分母记NOT_EVALUABLE；相关query-ray非独立N，UNKNOWN不当负。28全采样free支持各臂0/28是已完成证据，非整盒/身体误报，步行提前量仍缺身体/事件参考。若可调基线追平，仅当前数据和比较条件未证明learned读出额外价值，RGB距离相对几何先验收益保留。本轮不以合成锚点限制替代独立ToF证据，旧8×8同源旁路不重命名为新阶梯。

## 输入分布与空间结构

既有ARKit全部16帧、原validation24帧、新3RScan64帧均按原全量执行。ARKit786432 finite-positive native像素全部处于公开RGB覆盖及共同传感器域；16/16预测水平/垂直相邻log差非零，逐帧IQR .0723–5.0371m，log-IQR .0431–1.0978。预测逐帧median帧间中位2.4375m，传感器1.1580m；这些不证明常数预测，也不定位距离偏差成因。

≥1000m尾4737/786432（.6023%），来自固定frames4/5/11/13的190/36/169/4342像素，最大10000m；主结果全部保留。补充排尾后IQR中位仍.4473m，没有据此裁输入。固定indices0/5/10/15的图已由主agent查看，确有场景结构，存在结构不等于度量准确或查询可用。控制集不同分辨率/相机/场景，非因果配对。

常数/非零梯度fixture及104帧source SHA/K/shape/map检查通过。首次strict浮点边界误排1279边缘像素，按已观察到<1e−5px映射数值误差加入边界容差后覆盖786432/786432；此前输出留`structure/prior_strict_boundary_attempt/`，原native分布未受影响。正式9.681s、首轮11.709s，CPU保守35s。全帧CSV、固定色标.2–10m及magenta尾标图在`structure/`，不称全组“坍缩成常数”。

## 同帧分辨率干预

[ARKitScenes官方RAW说明](https://github.com/apple/ARKitScenes/blob/main/raw/README.md)把lowres_wide和vga_wide标为同wide camera。原16评分帧与VGA流有8个exact timestamp/source-id交集，旧frame编号0/3/4/6/7/10/13/14，预先固定全部交集；公共K×.4与原lowres K最大误差.0006px，BOX生成图与官方lowres图相关.9965–.9995、MAD .818–1.756。这支持对应核查，不把两流当像素等价。

原640×480帧经BOX生成256×192，K按half-pixel resize，public depthK→colorK投影；高、低两臂ref深度/labels/query完全相同，每帧共同域48705射线，447边界射线两臂均不计参考评分。半像素映射fixture误差1.53e−5px、ZIP CRC/source SHA/derived逐像素BOX/K/共同参考检查通过。评分参考仍为原lowres LiDAR，非FARO/高分辨率细障碍真值；不能将此8帧公共域结果与旧16帧全域数值直接作因果比较。

两路DP及原冻结head均完成，actual cutoff depth-only=.5618626475334167、geometry=.6106688380241394，原train归一化和200步模型不变：

| 冻结臂 | native640召回/free误支持 | derived256召回/free误支持 | native正救回/损失 | free误支持移除/新增 |
| --- | --- | --- | --- | --- |
| depth-only head | .342730/.049789 | .342398/.050621 | 8275/8046 | 913/139 |
| geometry head | .454139/.064790 | .454139/.064790 | 0/0 | 0/0 |
| raw DP直接几何 | .270827/.010248 | .270066/.011662 | 13225/12701 | 1441/126 |
| 原cal尺度DP直接几何 | .261821/.009442 | .260034/.010570 | 12991/11761 | 1149/100 |

相关query-ray正/free分母688483/930318，UNKNOWN550270，非独立N；120格74POS/31UNKNOWN/15全采样free。head native TP235964/FN452519/FP46320/TN883998，derived235735/452748/47094/883224；正净+229，召回+.0333个百分点，free净−774、误支持−.0832个百分点。geometry312667/375816/60275/870043两路逐概率相同。raw DP native186460/502023/9534/920784，derived185936/502547/10849/919469；所有对照和逐帧/query/距离带救回损失全表保存，不只展示改善切片。

head full-reachable query支持均47/74，已知TP≥16见证均45/74，不能用前者冒充实测命中。两路各臂15个全采样free格均支持0/15，单capture相关单位、非整盒清空。`full_vs_common_query_support.csv`显示native边界head支持1300、geometry1614、raw340、cal332；derived为0/1614/0/0，均为跨120query-record累计单位，去边界后≥16判定变化全部0。参考评分排边界一致，输出支持仍按原full-reachable定义，没有悄改阈值/评分。

共同metric像素389640；native/derived全量meanAbsLog .910333/.938753、medianAbsRel .552953/.641551，meanAbsRel180.022696/190.196263被≥1000m尾5122/5536显著影响。主指标保留全部尾，不用好看的补充统计替代全量；深度质量略改善未转为明显query收益，known-witness无变化。仅支持640→256在这组配对条件下影响输出，不定位旧ARKit退化主因，不排除分辨率与其他因素交互。1920×1440 wide/upsampling指定URL本次403，未运行，不能写成整个capture确定没有高分辨率图。

独立两路4臂×120记录/summary/增量全部精确复算，模型/归一化/cutoff SHA冻结检查通过；geometry概率逐值相同。固定pairs0/2/4/7科学图主agent已查看。正式含边界表6.172s，首轮6.065s留`prior_without_boundary_table/`；保守CPU25s，不重跑已通过图形。完整结果在`paired-comparison/`。

## 可调几何校准的退化结果

选择和读出规则按开头plan运行，仅原cal16帧/2环境，free1309953、positive703611：δ=0m，TP479917/FN223694/FP185887/TN1124066，recall .682077、FPR精确等于目标.14190356447903094。201点得到200个不同FPR，范围0–.73121478，最小非零相邻粒度7/1309953≈.0000053437；完整cal曲线和选择时间在`margin/calibration.json`及CSV，先写选定值再迁移评分。

原validation24帧、新3RScan64、ARKit16的δ0全部计数复现raw：原.581300/.187899、新.640047/.159107、ARKit.205829/.010578。全量/环境/距离带及与冻结head救回损失记录齐全。240cal直接区间、50040点closed boundary/反转区间fixture、三组7800臂query-record及配对恒等式复算通过，执行9.791s+独立.489s，CPU保守20s。

这一合同以raw自身FPR作为单调余量的目标，δ0必然满足，未增加有效比较信息。不能据此说可调几何无价值、学习头已经胜过强校准基线，或一次比较决定独立贡献；RGB距离相对geometry先验的既有增量保留。下一合同需避免这个退化，例如在原train拟合小参数距离校正、只原cal定工作点，或预先定义多个cal工作点；这仍是待设计路线，没有在本轮另选迁移阈值。

## 第三相机参考与未完成项

实时核查[TUM官方格式](https://cvg.cit.tum.de/data/datasets/rgbd-dataset/file_formats)和[许可](https://cvg.cit.tum.de/data/datasets/rgbd-dataset)：Kinect640×480 registered RGB/depth，Freiburg3去畸变、公用K535.4/539.2/320.1/247.6，opticalZ=uint16/5000，官方深度已预缩放。新adapter预定16个uniform RGB索引、最近depth≤20ms，选择先于深度读取，缺失/遮挡UNKNOWN，不叫完美同步/身体事件参考。4项新增选择/单位行为检查通过。

官方long_office_household源在约270.830s仅接收4194304B，不能在300s网络限额完成约1.48GB包；task-owned PID67556已释放，部分文件SHA和失败原因保存。`multicamera-sensor/`只有NOT_RUN收据，没有真实manifest/观察，真实帧准备、第三相机GPU、算法成绩均NOT_RUN。准备脚本可复用，但未实测完整源，不能称已补到第三相机或完整步行评价。这个网络失败不产生模型判断，也未扩大预算/留下后台下载。

## 下一决定、成本与交付

不沿“继续提高这组图像分辨率就会修复近场”推进；原始预测存在结构，当前同帧干预收益小，输入分辨率主因假设仍未被建立。下一优先是修正强几何比较合同和检验不同度量深度表征/训练分布，并补真正多环境、多相机参考；先将输入距离与读出适配各自可解释地比较。旧候选保留，但没有独立贡献升级；旧两query/32特征不续训，原真实评价硬目标、28采样free证据及完整身体/步行缺口保留。8×8同源锚点不冒充新阶梯或真实ToF。

GPU总61.358s/1200s：native DP27.118、derived DP27.229、冻结head共7.011；包含启动/存储，不作部署效率。CPU辅助保守200s/2400s（margin20、structure35、hires20、TUM10、pair25、主评分/收尾余量），不是性能测量。ARKit实际range/索引/K/README3618223B，未下载670242432B完整VGA包；TUM partial4194304B，另官方网页884601B，总已记接收8697128B/4GiB。TUM网络270.830s/300s，ARKit网络保守10s/300s；模型下载0B。所有计算/下载进程释放，原始段/索引/模型/参考/预测/审计/失败与partial为durable复算/续传证据留在canonical artifacts树；续传属于后续新范围/预算，不把旧网络限额重置。

代码：[结构诊断](rgb_body_query_depth_structure.py)、[同帧准备](rgb_body_query_hires_prepare.py)、[配对独立复算](rgb_body_query_resolution_compare.py)、[可调几何](rgb_body_query_margin_baseline.py)、[TUM准备](rgb_body_query_tum_prepare.py)。复用[DP执行器](rgb_body_query_3rscan.py)和[冻结读出](rgb_body_query_frozen_transfer.py)，不更改旧输出。源码/输入/输出SHA、分阶段成本、实际partial与资源释放在payload总`completion_receipt.json`，热链接与scoped diff检查用于交付。
