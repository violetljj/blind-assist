# RGB固定子体积查询与参考覆盖

完成136帧3672个预定子盒的8臂评价，补到575个严格采样空闲查询参考。冻结学习头在新query和追加ARKit capture上没有稳定胜过校正几何；保留旧RGB距离信息收益，后续优先改距离/不确定性表征和query覆盖。完整身体、细障碍、步行和第三硬件评价仍缺，不能由子盒结果宣称这些能力成立。

2026-10-09，EXPLORE / COMPLETE_QUERY_EXTENSION / THIRD_HARDWARE_NOT_RUN。用户“继续”授权承接[低参数校正与迁移](RGB_BODY_QUERY_CALIBRATED_TRANSFER_DEV_20261009.md)：把射线成绩推进到查询支持、救回损失和参考可观测覆盖，并以预定空间栅格补采样空闲查询的评价机会。旧模型/归一化/实际cutoff与两距离校正fit/cal冻结，不训练、不在新query挑阈值或位置。

新预算：CPU辅助累计1500s（旧query诊断300、固定grid准备300、相关复算300、第三相机源300、主评分/交付300），GPU阶段壁时900s（固定grid冻结head600、条件参考源DP240/head60；剩余条件槽用于追加ARKit同家族参考，不称第三硬件），网络/跨机传输900s，新源/跨机接收≤768MiB，新模型0B。生成的预测/参考为本地计算输出，不冒充新下载。payload `artifacts.local/work/rgb-body-query-query-level-dev-20261009/`；不重置旧TUM失败run预算。第三相机改用已注册secondary worker独立网络或有出处的其他源路径，固定候选与公开参考条件，没有源就不触发相机算法评分。

实施合同先于新grid标签/成绩：camera-local x edges[−.9,−.3,.3,.9]，y[−.55,−.18,.18,.55]，opticalZ[.3,.8,1.5,3]，全3×3×3=27子盒；名称/顺序固定，全部保留。旧validation24、新3RScan64、ARKit16原全部帧，RGB/K/native参考/observed不变，缓存Depth Pro重用并标来源，不重新执行DP；原15query结果独立保留。子盒查询超出原head训练query集合，是可观测占用/空闲的Development任务扩展，不称完整身体体积或独立确认。

参考：首表面在盒内为POS ray，首表面超过盒远端为FREE ray，盒前遮挡/孔洞为UNKNOWN，不可达255。query POS须≥16个positive采样ray；sampledFREE须0positive、0UNKNOWN、≥16free。有限ray全部free不自动证明整盒空闲；参考覆盖率和未知分母并列，不能用≥95%支持比例替代严格free状态。模型输入仅RGB/publicK/query，参考仅准备/评分。评价全部8臂和全部网格，不按输出挑空间位置。

决策检查：旧15query3RScan无充分free格，ARKit28格单capture相关；新27grid分母未知，若仍缺free记NOT_EVALUABLE而不放宽判据。查询正支持分别报告full prediction≥16及known TP≥16；前者可能含UNKNOWN，后者是采样正见证。同时保留射线召回/free误支持及每环境/距离带、query配对救回/损失；不把相关query-ray当独立N。新query不匹配原cal FPR、不判定旧原始query完整身体误报/步行提前量；这轮检查数据可评价机会及冻结模型对查询变化的适用性，不凭一次结果裁定独立贡献。

## 旧15query：见证优势与参考缺口

全部旧5cohort/8臂14400记录与40summary按保存结果复算一致。新3RScan head相对geometry的query见证救90、损0，全部在.3–1.5m；但相对affine-margin救71、损44，.8–1.5m救1/损44，1.5–3m救39/损0，3–6m救22/损0，center-close救9/损0。总411/570对384/570不能概括为所有强基线下身体近场占优；同query条件的场景信息仍要与空闲参考和饱和状态共同判断，不能凭分组认定因果。

head对UNKNOWN query仍有总支持≥16：原val89/162、新3RScan272/390、ARKit63/72，affine-margin分别14/162、44/390、20/72；这些既不计误报也不计真实检出。旧原val/新3RScan可达射线域UNKNOWN占46.86%/46.18%，不是空闲；新6644217 UNKNOWN query-ray中3290690为第一表面在query前，3353527为不可用参考；原val分别822495/1706436，ARKit1203389全部为前方第一表面遮挡。

旧新3RScan7744727 known query-ray来自1879896不同(frame,pixel)，平均复用4.120次、最大7；原val2867629来自627352，4.571次；ARKit3169200来自786432，4.030次。跨帧/世界坐标也未证明独立。仅增加同视点first-return重复不能补被遮挡的盒内真值。全5cohort/8臂、环境/带/位置/复用直方图及参考拆分在`legacy-query/`；旧数字不改，补UNKNOWN支持前输出也保留。

## 固定27子盒：准备与初始三组评价

104个原RGB/native参考数组和K/SHA保持，全部2808格独立XYZ标签一致，public observation不含参考字段；Depth Pro104帧缓存明确REUSED，本轮DP推理0。新query超出原head训练集合，全部保留且实际cutoff不变：depth-only .5618626475334167、geometry .6106688380241394；shift/affine的原train参数与原calδ继承未改。

| 原全部帧 | query数 | POS | UNKNOWN | 严格sampledFREE | 正/free/UNKNOWN query-ray |
| --- | ---: | ---: | ---: | ---: | --- |
| 原validation24 | 648 | 295 | 351 | 2 | 532479/1724743/1938626 |
| 新3RScan64 | 1728 | 866 | 861 | 1 | 1689002/4258496/5240750 |
| ARKit16 | 432 | 228 | 99 | 105 | 703186/1755908/1004836 |

108个严格sampledFREE中105来自同ARKit capture，负参考并未达到多环境均衡。旧15query的28格和新grid的105格是不同空间查询，0/28不能推成新空间定义也零误支持；新105也不证明整个物理子盒/完整身体空闲。三个cohort全部8臂数据如下，括号为sampledFREE支持数／分母：

| 臂 | 原val TP见证/295（free/2） | 新3RScan TP见证/866（free/1） | ARKit TP见证/228（free/105） |
| --- | --- | --- | --- |
| 冻结depth-only | 150（0/2） | 352（1/1） | 77（7/105） |
| 冻结geometry先验 | 91（0/2） | 226（1/1） | 60（3/105） |
| raw DP直接几何 | 161（1/2） | 512（0/1） | 73（0/105） |
| 原cal全局尺度DP | 162（1/2） | 510（0/1） | 71（0/105） |
| shift direct | 168（0/2） | 467（0/1） | 46（0/105） |
| shift margin | 205（0/2） | 584（0/1） | 54（0/105） |
| affine direct | 196（0/2） | 502（1/1） | 78（0/105） |
| affine margin | 188（0/2） | 487（1/1） | 74（0/105） |

| 臂 | 原val射线召回/误支持 | 新3RScan | ARKit |
| --- | --- | --- | --- |
| depth-only | .401426/.088908 | .285037/.080424 | .165483/.035582 |
| geometry | .244057/.056517 | .171298/.049289 | .115204/.023588 |
| raw DP | .386457/.143932 | .463562/.124338 | .123845/.005550 |
| 原cal尺度DP | .390860/.139427 | .465591/.119571 | .117851/.005086 |
| shift direct | .435136/.081422 | .438153/.065327 | .059247/.001760 |
| shift margin | .573348/.151482 | .577244/.131659 | .071109/.002992 |
| affine direct | .546818/.098494 | .485218/.077146 | .151694/.009927 |
| affine margin | .515570/.087560 | .461573/.070965 | .140259/.007581 |

head对affine-margin已知query见证救回/损失为原val52/90、新3RScan109/244、ARKit28/25。新3RScan校正几何支持更多正query且射线误支持较低；ARKithead多3个见证但新增7个采样free支持。head的UNKNOWN总支持原val120/351、新3RScan367/861、ARKit70/99，affine-margin为61/351、160/861、73/99；完整8臂UNKNOWN和每query计数保存，不将未知支持当FPR或检出。

结论限定本组冻结query外推：旧固定query收益不能推广任意边界，小head对新增子盒的适用性不足；不由此确定是表征、训练query范围、读出形式或深度迁移哪一种原因。原cal FPR也没有在新grid重新匹配。子盒评价获得了一些严格采样负参考，但完整身体、细障碍和步行提醒评价仍缺相应证据，原真实评价硬目标保留。

## 相关核验与补参考分支

初始三组全部8臂×2808=22464记录、157248逐ray pair恒等式、480summary分组及600父脚本配对聚合独立复算一致，冻结fit/cal/checkpoint/归一化/实际cutoff SHA通过。首轮准备缺depthpro目录失败保留，修复后全量完成；没有改原native数据或阈值。GPU冻结head累计25.471s、DP0；原val/new/ARKit head分别12.879/8.662/3.930s，包含启动/保存，不作部署效率。

第三相机工作机路径已完成不同机制检查：注册worker在线，Python因缺CA issuer失败8.408s/0B；Windows/.NET原生信任库不关闭验证即成功HTTP200，长度497551823B，但90.440s只收1487832B，触发预写吞吐停止。source及11309B薄证据接收合计1499141B，网络/传输保守160/900s，原TUM真实参考/推理仍NOT_RUN。两个scheduled task已注销，所有记录PID释放；部分源留worker canonical G树，owner本run，具体路径/SHA见`third-camera-source/lane_completion.json`，没有活跃资源。

在剩余同一source/transfer预算内另设补参考分支，旧TUM停止与NOT_RUN不改：复用Apple已验证HTTP Range ZIP目录/成员读出，从官方Training metadata按video_id数值排序排除旧47333462，下载前固定两capture及全序列exact timestamp交集uniform16帧；若metadata提供scene/visit分组则在图片/深度读取前核实分组关系，不能从capture ID声称独立物理环境。constructed member ZIP明确是摘取子集而非整官方包，CRC/SHA/时间/K对应先核实；仍用27固定query，不按空闲或预测挑位置。这只补同iPad/LiDAR家族capture，不能代替第三硬件或TUM真值；源与预算/实际选择写`third-camera-source/arkit-additional/`，新增成绩另外报告，不覆盖初始三组。

## 两个追加 ARKit capture

源选择在读成员前修订并留旧plan：最前两个ID的visit_id缺失，因此按数值ID排序排除旧47333462和缺visit，固定前两个不同非缺失官方visit组40777060→369709、40777065→369708；旧capture→467138。不同visit组不证明独立物理环境。全ZIP目录exact timestamp交集uniform16帧，每capture原RGB/depth/pincam同时间，256×192，公共K，注册optical-Z毫米/1000；0值UNKNOWN。HTTP206/Content-Range、成员CRC/SHA、32帧单位/K/public observation及864格独立XYZ通过。下载的是明确标注的constructed subset ZIP，非整个官方包；只本地非商业研究，不再分发。

| capture | query | POS | sampledFREE | UNKNOWN | 正/free/UNKNOWN query-ray |
| --- | ---: | ---: | ---: | ---: | --- |
| 40777060 | 432 | 202 | 220 | 10 | 393956/2840328/229388 |
| 40777065 | 432 | 179 | 247 | 6 | 334862/3018677/114471 |

追加源没有DP缓存，实际原backend运行32次Depth Pro推理；读取原RGB与公共fx，没有GT、重映射或尺度拟合。原冻结head与全部校正参数/实际cutoff不改，没有新数据匹配FPR。源准备收据的NOT_RUN是推理之前的时点，后续完整预测manifest单独留存。所有8臂如下：

### Capture 40777060

| 臂 | 射线召回 / 误支持 | 已知正见证 | sampledFREE支持 | UNKNOWN总支持 |
| --- | --- | --- | --- | --- |
| depth_only | 0.486524 / 0.114509 | 120/202 | 31/220 | 10/10 |
| geometry | 0.355002 / 0.073034 | 81/202 | 20/220 | 7/10 |
| depthpro_raw | 0.009156 / 0.000000 | 6/202 | 0/220 | 0/10 |
| depthpro_cal_scale | 0.008336 / 0.000000 | 6/202 | 0/220 | 0/10 |
| log_shift_direct | 0.000000 / 0.000000 | 0/202 | 0/220 | 0/10 |
| log_shift_margin | 0.000000 / 0.000000 | 0/202 | 0/220 | 0/10 |
| log_affine_direct | 0.409822 / 0.022853 | 118/202 | 12/220 | 9/10 |
| log_affine_margin | 0.401555 / 0.021877 | 118/202 | 12/220 | 9/10 |

### Capture 40777065

| 臂 | 射线召回 / 误支持 | 已知正见证 | sampledFREE支持 | UNKNOWN总支持 |
| --- | --- | --- | --- | --- |
| depth_only | 0.590315 / 0.124811 | 122/179 | 22/247 | 4/6 |
| geometry | 0.497733 / 0.077716 | 86/179 | 15/247 | 3/6 |
| depthpro_raw | 0.083506 / 0.001256 | 23/179 | 0/247 | 0/6 |
| depthpro_cal_scale | 0.082147 / 0.001033 | 23/179 | 0/247 | 0/6 |
| log_shift_direct | 0.036738 / 0.000000 | 9/179 | 0/247 | 0/6 |
| log_shift_margin | 0.042191 / 0.000007 | 9/179 | 0/247 | 0/6 |
| log_affine_direct | 0.476337 / 0.032312 | 131/179 | 10/247 | 3/6 |
| log_affine_margin | 0.467249 / 0.030776 | 131/179 | 10/247 | 3/6 |

head相对affine-margin正query见证救/损为40777060的25/23、40777065的15/24。两组head射线召回较高，但射线误支持及sampledFREE总支持也较高；未在新增数据上调工作点，不能把这解释为同FPR优势。raw/全局尺度/shift在两组几乎不支持，affine改善明显，但一次迁移比较不能定位相机、深度或读出因果，也不能裁定整体RGB价值。全环境、距离带、每query/配对结果在`additional-arkit-evaluation/`，不挑切片。

现有3个ARKit capture共572个sampledFREE，head支持60/572、affine-margin22/572，原3RScan另3格；合计575格均为相关的有限采样参考，不是575独立空闲物理体积。与旧15query的0/28空间定义不同，旧结果不改。

## 当前决策与收尾

本轮将可评价负query从旧单capture28格扩展到预定27子盒、三ARKit visit组的572格，并补3RScan三格；完整身体、细障碍和步行事件仍没有对应参考。首轮新query结果加追加capture都未证明冻结学习头比校正几何稳定更好，不能续用旧固定查询优势概括任意身体空间查询。保留RGB预测距离路线和旧Development增量；下一项实现宜改变可查询距离/不确定性表征与训练query覆盖，以固定全grid、强几何、unknown覆盖、救回损失和同工作点评价检验。是否采用时序须建立运动/事件参考，不能由本轮直接断言有效。旧两query/32特征不续训，真实硬目标/同源锚点限制/CNH预算保留。

源lane新收5996966B，加worker TUM/薄证据合计7496107B/768MiB，网络保守240/900s，CPU90/300s；追加源执行37.563s，源核验另5.428s。新增DP阶段壁时37.071+32.984s，追加head3.912+3.729s，加初始head25.471s，总GPU103.166/900s，模型新下载0B/训练0；包含启动、保存、清理，不作部署时间。TUM第三硬件仍NOT_RUN且原吞吐停止保留，无活跃源进程。

追加32帧864query、8臂6912记录、48384逐ray pair恒等式、128summary及160父pair聚合独立复算一致；UNKNOWN支持16臂聚合、真实DP32calls与全部冻结身份通过。加初始三组共29376臂query记录通过；旧15query14400条单独保留，不混入新query分母。追加主审21.544s，累计审计保守145/300s。

全run CPU辅助保守500/1500s（旧query30、初始grid准备35、审计145、源90、主评分/摘要/核验/交付200）；主评分实际32.256+12.332s，GPU103.166/900s、网络240/900s、新源及跨机7496107B/768MiB。完成相关复算后不加训练或扩预算。所有本轮计算/下载任务结束；canonical本地源码/参考/预测/失败/全量CSV/身份收据持久保留，worker partial也按原owner保留，没有活跃计算。partial路径与释放凭证见`third-camera-source/lane_completion.json`；总`completion_receipt.json`绑定计划、源码与所有必要收据。

新增代码：[旧query覆盖诊断](rgb_body_query_query_coverage.py)、[固定query准备及几何读出](rgb_body_query_fixed_grid.py)、[追加ARKit源子集](rgb_body_query_arkit_additional.py)。已有baseline/head/校正代码复用未修改。交付检查为本轮真实执行、相关独立复算、scoped diff及docs index（15热页126本地链接）；未跑无关Android检查。三热文档仅替换RGB段，CNH并行记录保持。旧route文本由基线5844eacd和本轮交付前d4226481保留，原数字与失败不改。
