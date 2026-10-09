# RGB跨场景距离参考：Development续推

状态：REAL_SENSOR_TRAINING_PIPELINE_COMPLETE / RGB_GAIN_NOT_ESTABLISHED，2026-10-09。继承[首批补标](RGB_BODY_QUERY_LABEL_COMPLETION_DEV_20261009.md)的真实定量目标；已补跨环境参考并完成两种query模型/控制，不扩旧停止run，也不训练CNH。

目标：解释SANPO稀疏core的跨场景距离缺口，利用已有真实RGB/实测深度形成可按采集组隔离的空间查询参考，并检查现成单目深度基线。原六个相机局部query、UNKNOWN及旧失败保留；身体外参和行走事件指标不凭相机深度补齐。

资源上限：本轮CPU 1800秒（含子任务）、GPU执行壁时600秒、新增下载2GiB。优先已有缓存；允许调整数据源/样本规模，不调整旧评分阈值来凑已知标签。阶段到限即停该阶段，保存部分结果。不采集手机/实机，不访问受保护final，不将无许可数据再分发。payload位于`artifacts.local/work/rgb-body-query-cross-session-dev-20261009/`。

判定：先检查参考的投影、距离单位和数据身份，至少跨两个独立采集组才构造分组训练/验证接口。若只能得到可见表面正支持和局部负支持，明确作为局部监督，遮挡后/缺测处保持UNKNOWN。先做CPU参考构造和一项几何/配对核验，再按可用数据运行有意义的基线；不由基线预测选参考区域或样本。每个已知单位及分母可追溯，现成深度比较不自动证明query模型贡献。

参考构造预设：3RScan官方train缓存按环境reference的SHA排序取12组，每组一scan，8个配对帧均匀抽样，共96帧，前7组train、2组cal、3组validation（均为探索Development）。每个环境及其rescans只能在同一侧。保留旧六query；另加同横向/高度的0.3–0.8、0.8–1.5、1.5–3米九query，作为新任务的距离分带，不重算或修改旧SANPO阈值。

使用原生depth点，按各自K投影至RGB；仅支持缓存中已标定identity传感器外参。不简单resize两相机、不渲染填补深度孔洞。逐射线：实测第一表面在query内为正，在query后为FREE_RAY，在query前遮挡或缺测为UNKNOWN；不把query前表面当盒外负例。至少16正支持为POSITIVE；全部可观测采样射线free且非空可记FREE_ON_SAMPLED_RAYS，仍不是整个实体盒无障碍真值。相机视锥外与射线间空间未被证明。基线全图RGB+公开K，参考仅在评分端使用；此次可评价独立RGB在实测可见射线上的空间判断，暂不评价细障碍/身体外参/事件。

在传感器参考形成后，启动独立query稀疏监督pilot：冻结已缓存ImageNet MobileNetV3Small原图特征，点解码器输入RGB特征/公开ray方向/query六边界，已知occupied/free射线做BCE，UNKNOWN排除。固定seed7、200步上限、train7环境；同采样/初始化/日程的无RGB几何先验控制；cal2环境只选与Depth Pro像素FPR匹配的阈值，validation3环境及RGB跨环境打乱做Development诊断，不选validation checkpoint。该子任务GPU240秒，DepthPro子阶段330秒，终端余量30秒；各实际执行壁时合并不超过600秒，CPU辅助120秒计总1800。不证明高分辨率/时序机制或整盒无障碍。

## 跨场景参考已形成

96帧/12环境，分组train56帧/7环境、cal16帧/2环境、validation24帧/3环境。15query每帧共1440格：825格有实测表面正支持，615UNKNOWN，完整采样射线free query为0。三侧正格501/126/198，UNKNOWN339/114/162。此处不使用panoptic/core定位oracle，输入全图RGB/公开K/query；Tango测距只在监督与评分端。可开始射线级稀疏监督，完整二分类负查询与误报事件仍缺，不能把615UNKNOWN改负。

本轮固定样本含墙体、天花板、家具、室内细腿与扫描旋转视角；不是步行佩戴者数据，也不是已经核验的细杆子集。参考分辨率224×172不能用来证明高分辨率细障碍收益。它补独立RGB米制查询的真实传感器参考与环境隔离接口；SANPO真实步行/细结构线及先前负结果保留。

缓存核查72scan/22481帧/35环境均为official train，reference及rescans按环境归组。官方[FAQ](https://github.com/WaldJohannaU/3RScan/blob/master/FAQ.md)与[回投源码](https://raw.githubusercontent.com/WaldJohannaU/3RScan/master/c++/rio_lib/src/rio_lib/sequence.cc)确认毫米光学Z、RGB-D配准。72扫描外参identity、depthShift1000；按半像素及横纵各自比例缩放K的最大差0.0476px，官方resize说明与缓存吻合。本实现仍按两K投影，前期把不同横纵比例当配准疑点的推断已撤回。ZJU检查15个H5均无K，暂不猜测内参。

数据仅用既有本地研究缓存；[数据条款](https://docs.google.com/forms/d/e/1FAIpQLSfl9Xm1qWiGmN2HXzbRIecVns_V-n-4bwrzPEE4ZezEpOKT9Q/viewform)限非商业研究/教学，工具MIT不代表数据开放再分发。未核当年账户批准记录，未提交表单/再分发。源码/条款核查回执在payload，新增数据/模型下载0B。

## Depth Pro全图基线

96/96原图调用，无参考选crop/尺度拟合/未来帧。仍是给公开fx的raw-native臂，未做主点居中；不能把其失败归因为独立主点或分辨率机制。GPU阶段壁时114.492s/330s子限，模型调用合计85.045s、启动19.364s、峰值3.977GB；桌面GPU成本不是手机效率。

| 评价域 | 实测正query检出 | 正射线召回 | free射线误支持率 |
| --- | ---: | ---: | ---: |
| 全96帧/15query | 620/825 | .58518 | .20660 |
| validation24帧/15query | 146/198 | .58130 | .18790 |
| validation/旧六query | 70/80 | .73113 | .23891 |
| validation/新增九距离带 | 76/118 | .42955 | .17037 |

validation全域TP623020/FN448751/FP337440/TN1458418，正射线1071771、free射线1795858；这些是相关query-ray单位，不是独立事件。旧六与新增九分别保存，不能用新增query扩大分母后冒充旧六成绩。cal实测FPR .14190356，供新query模型匹配阈值用；validation不选工作点。没有完整负query分母，所以query误报率/事件提前量不评价。

## 首个query训练与对照

首pilot已完成：冻结576通道粗特征，seed7固定Gaussian压到32通道；head40→64→64→1，已知点未加类别权重BCE，RGB/geometry各200步、每步8192点，同初始化/采样。RGB cal FPR .14190356448与DP一致，geometry .1419020377因ties少2FP。validation正/free射线分母1071771/1795858：

| 臂 | TP/FN/FP/TN | 正射线召回 / free射线FPR | query支持 / 正格中已知TP≥16 |
| --- | --- | --- | --- |
| RGB query | 549769/522002/246908/1548950 | .51295 / .13749 | 193/198 · 176/198 |
| 无RGB几何控制 | 650352/421419/206511/1589347 | .60680 / .11499 | 134/198 · 133/198 |
| RGB跨环境打乱 | 533175/538596/273610/1522248 | .49747 / .15236 | 192/198 · 175/198 |
| Depth Pro | 623020/448751/337440/1458418 | .58130 / .18790 | 146/198 · 137/198 |

首RGB在像素召回更低且误支持更多的情况下弱于geometry控制；query高支持率与打乱后几乎不变不能当独立图像贡献。已知TP≥16是补充定位见证诊断，旧支持指标保留；没有完整负query分母，不作事件判定。各臂cal匹配不能保证validation相同FPR，上表列实际成本，不称同误报收益。GPU阶段28.445s/240s、两head训练7.368s、原图特征/投影模型时间8.687s（特征阶段11.395s）、峰值506.55MB。104个预测SHA/1680格独立复算一致，两final head实际更新。

续推不同候选：取消固定Gaussian预压缩，保留MobileNet原图较细低层及final576高层，在query head内学习投影后接公开ray/query的56维输入。实际执行idx3/24通道，原记录把它写成stride4/16通道有误；实测idx3输出68×120、约stride8，idx1输出135×240/16通道才是stride4。本轮保留执行的idx3，不另跑idx1；源快照/原receipt及更正说明保留。实际投影576→32、24→16。RGB/zero仍各200步、seed7、同初始化/采样/cal原则，受首pilot Development结果启发，不是独立确认；低层保留与学习投影同时改变，不能单独归因高分辨率。

| 多尺度臂/同validation | TP/FN/FP/TN | 正射线召回 / free射线FPR | query支持 / 已知TP≥16 |
| --- | --- | --- | --- |
| RGB query | 540025/531746/261955/1533903 | .50386 / .14587 | 196/198 · 184/198 |
| 无RGB几何控制 | 651013/420758/210606/1585252 | .60742 / .11727 | 134/198 · 134/198 |
| RGB跨环境打乱 | 507064/564707/319954/1475904 | .47311 / .17816 | 196/198 · 177/198 |

RGB/geometry的cal实际FPR均.14190356448。RGB训练最后样本BCE .26424、geometry .49298，但RGB cal召回.45380、validation.50386仍弱于geometry validation.60742；这提示场景泛化不足，未单独确定成因。打乱RGB使召回下降3.075个百分点、误支持增加3.230个百分点，有图像响应，仍没有相对无RGB控制的有效增益。query支持196/198和打乱后196/198不证明细结构识别；已知见证数与像素计数更能暴露代价。第二版也保留为负结果，不增加步数/以validation换阈值。

多尺度GPU阶段41.630s/240s，特征阶段23.124s（冻结模型合计4.368s、含缓存保存等），两head训练6.158s，评价5.556s，峰值4.657GB。600通道缓存4.438GB保留以复算；它是中间表征，不是模型大小。两版pipeline成本未包含手机或实际报警全链路。

## 下一决定

真实跨环境监督/校准/验证接口已经可运行，首个RGB独立query训练已执行，贡献尚未成立。保留有公开K的真实传感器参考和geometry强先验控制，下一优先使用具度量几何预训练的RGB表征检验跨环境泛化，同时补真实步行场景/薄结构距离与负query覆盖；不继续堆同一弱表征的支持数。高分辨率与因果时序仍是待验证机制，完整身体外参/负query/事件提前量尚未补齐，真实定量硬目标保持。

## 缺口诊断与核验

旧跨session14core×6格独立复算84UNKNOWN：57角域空、27非空不完整。1929个唯一core像素1153双目一致/776缺ZED；按非空query重复计4032像素中2292一致且全部在query远端外、1740缺ZED，没有不一致/跨边界/正支持。1像素远杆缩放后为空。此数据既远，又缺一路参考；继续描同14core或放宽agreement无法补跨scene近场正例。

新适配器6项fixture通过：毫米/光学Z、公开K映射、遮挡UNKNOWN/后表面free、闭区间、环境隔离、只读预测插值。独立直接XYZ复算96帧×15query正掩码逐值一致，RGB/reference SHA核对；最大实测depth7.368m，没有65535哨兵值，耗时1.266s。该检查不重复模型调用。准备6.504s、DP评价2.810s；GPU三个阶段合计184.566s/600s，任务进程均释放。CPU准备/独立核验及终端辅助总保守扣额200s/1800s（不是性能测量），其中库存20s、fixture12s为保守计，GPU阶段CPU辅助已包括在其执行壁时，不当纯推理效率。新增数据/权重0B，网页源码文本留存量另记回执、wire bytes未计量。模型/参考/衍生feature cache为任务durable复算证据留存，无常驻worker。

payload：`artifacts.local/work/rgb-body-query-cross-session-dev-20261009/`。`distance-gap/`留旧缺口诊断；`real-depth-inventory/`留环境/配准/来源回执；`sensor/selection.json`固定采样，`dataset_manifest.json`路由稀疏参考、`observations.json`只路由公开RGB/K、`evaluation.json`逐格基线、`independent_geometry_check.json`独立核验，`rgb_depth_contact.png`已实际检视。`pilot/`与`multiscale/`保存两种query模型、相同日程geometry控制、跨环境shuffle预测、校准阈值/指标、源快照及核验。数据和模型不入Git。

## 复用入口

[传感器适配/DP/评分](rgb_body_query_3rscan.py)、[首query训练](rgb_body_query_sensor_pilot.py)、[多尺度query训练](rgb_body_query_sensor_multiscale.py)、[聚焦检查](test_rgb_body_query_3rscan.py)。新执行使用新output目录，既有payload拒绝隐式覆盖。PowerShell从checkout运行前设置`$env:PYTHONPYCACHEPREFIX = (Resolve-Path artifacts.local/cache/python-pycache).Path`；已有缓存目录存在，字节码路由canonical F树。本轮发现并清理4种任务模块的checkout字节码，保留共享旧cache。
