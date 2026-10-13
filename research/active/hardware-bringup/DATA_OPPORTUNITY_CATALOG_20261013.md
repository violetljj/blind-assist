# 扩展数据机会目录：现有资产与公开来源

2026-10-13，EXPLORE。承接[首轮数据潜力检查](DATA_POTENTIAL_DEV_20261013.md)，用户追加要求多子代理、更多现有和网上数据。五条支线分别负责本地资产、原始ToF、材质、地面障碍与连续视频/位姿；主线程整合并实际读取公开原始波形。来源卡片与小样保存在`artifacts.local/work/data-potential-dev-20261013/expansion/`。

## 优先结论

最值得优先做的仍是**已有完整RGB流的持续跟踪、细结构距离机制、原始ToF的区内混合与噪声机制**。新增素材让这三件事可以分开验证，也补出了透明物体、台阶/落差和头身运动的后续研究入口。

数据身份分三层：本机确认存在的已消费/train素材；本轮真正下载、解析过的小样；仅由官方文档/文件树核验的候选。下载路径或标签名称不等于米制真值，额外帧也不等于独立新场景。未执行模型推理或训练，所有结果属于数据利用空间和合同检查。

第二轮扩展形成**9项本地资产卡、27项公开来源卡**（ToF5、材质7、连续序列7、地面8）。公开来源之间仍可能共享原始场景，特别是同作者坑洼集、Matterport派生标注；27不是27个独立验证域。另有待获取/低优先线索保留在各JSON，不混入这27项。

**第三轮已进一步读通OctoNet的232帧真实ST 64×18直方图与LLRGBD真实照明配对，确认现有Hypersim原生几何的采样损失，并生成54个额外RGB复核入口。** 最新具体结果、来源缺口和优先级见本页后半“第三轮”；所有变化都是数据开发证据，没有模型增益宣称。

## 新确认的九项本地入口

入口均相对仓库；完整字段、已消费身份、来源报告及许可见`expansion/local_assets.json`。新增像素读取仅限已消费或明确train，角色未知的源只查元数据。

| 优先用途 | 实际存在的数据 | 可立即提出的问题 | 边界 |
| --- | --- | --- | --- |
| 姿态变化、低安装、玻璃 | `artifacts.local/work/phone-imu-capture-20261012/raw`：27段，8710 RGB、2356 ToF、46430 IMU（manifest计数） | 逐帧姿态与RGB运动能否解释地面侵入；走近/退后尺度是否一致；阳台玻璃何时让深度失效 | 已消费弱场景描述，无独立杆距/玻璃真值；本轮未重复全payload完整性审计 |
| 明暗/重启/仪器响应 | `artifacts.local/work/real-labeled-capture-20261011`：十条件1507 CNH帧，每段30秒 | 稳定短窗下ambient与峰形；重启与背景持续性；近/远墙响应 | 头动、窗帘背景、姿态随条件变化，不能直接当固定视点噪声重复试验 |
| 真实区内前景被背景淹没 | `artifacts.local/datasets/zjul5/ZJUL5`：483项train清单、1021 H5库存，schema读取限train | 用RGB-D与zone像素矩形检验细前景何时输给背景主距离 | `hist_data(64,2)`是均值/方差，**不是CNH**；无已核验时序；数据独立许可未声明 |
| 原生分辨率与已知几何 | `artifacts.local/datasets/hypersim-ba-nfo`：8000 train RGB/深度对、338场景，路径全存在，depth1024×768 | 原生→256×192/模型输入的细结构损失；几何边界/遮挡；已知相机位姿下的坐标检查 | 合成已消费；稀疏抽样帧，不能称连续行走视频；本轮未确认专门坑洼/玻璃子集 |
| 真实多区波形 | `artifacts.local/datasets/lcspcdata`：128帧、9×128 bins、参考脉冲 | 相对参考脉冲的峰形；距离主返回和完整波形的信息差 | TMF8820，单物体128姿态；time全0、pose末行全0需按已记录合同修复；无空场配对，不能估计ST重复噪声 |
| 头部转动压力 | `artifacts.local/downloads/heads-up-20261007`：raw/refined pose包，本地两个独立录制 | 头朝向与行进方向偏离、转头/丢帧时的时序稳定性 | 本地**没有RGB/depth**；easy是hard坐标旋转副本，不能重复计独立来源；VIO为估计 |
| 真实身体摆动 | `artifacts.local/downloads/blindways-20261007/Motion`：1025文件，24关节/60Hz | 慢走、停顿、转弯、头与骨盆的平移摆动压力 | 无头部朝向、RGB、障碍物距离；步行描述不是逐像素危险标签 |
| 多时钟和头身运动 | `artifacts.local/work/cnh-nymeria-sample-audit-dev-20261008/sample/`：一原始样本 | 实际IMU重置/间隙；头身变换；因果匹配与离线参考差异 | 合法长段约333秒，7处原时钟异常需断开；无RGB，闭环轨迹含未来信息，仅作评价参考 |
| 协议/早期波形 | `artifacts.local/hardware-bringup/captures`：三段266帧、16区、约5.1Hz | 旧/新输入合同、序号/时钟异常、近前景响应形式 | 不是当前8×8×16；另一些目录只有零成功帧的失败记录，不计新有效数据 |

**排除目录幻觉：** Bonn archives为空；SEVN RGB在9月24日已删除，只剩图/位姿/标签/恢复记录；SceneNN当前核到播放工具与合成三帧ONI，没有确认真实序列；TUM/ETH3D/TartanAir旧报告存在也不代表原始payload仍在。CID-SIMS/CARLA的当前payload和访问身份未闭合，不计可用库存。

## 本轮额外读通的小样

### SP-TransientBench：真实原始光子波形

[官方数据](https://huggingface.co/datasets/shuinb/SP-TransientBench)固定revision `04bbf64cb63ff3eda498c16d96be4a0825628790`，下载编号1直方图和Livox参考，另保存IRF、offset、外参、配置、官方parser和README，共**101,849,954字节**。CC BY-NC-SA 4.0。原始两个大文件与官方LFS SHA一致。

实际读取hist为**49152×672**非负整数，48000/49152像素有计数；共33030144个bin中5415480个非零，最大计数2164。offset为49152项。Livox CSV共480576行，但**246270行为全零坐标，只有234306行有限非零坐标**；非零点范数p10/50/90为2.652/4.400/5.232m。不能把所有“有限值”当有效点，也不能把这个以中远背景为主的样本说成已覆盖近杆。

初版每100行的抽样恰好命中交错空点，摘要保留为`summary_before_zero_xyz_check.json`；改为全量零/非零分解后完成上述合同。官方parser含17bin偏移及逐像素offset，默认畸变参数与config不同，因此**本轮没有计算校准后距离准确率或Livox投影成绩**。这是ADS6311外域机制数据，无配对RGB，也不等价于ST空间响应。

复现：`host/sp_transient_source_pilot_dev_20261013.py`；`sp_transient_sample/`含预选清单、SHA收据、两版摘要与直方图强度/峰bin预览。首次16.552秒，修复零点读出4.684秒，不重复下载已有文件。

### ClearGrasp：透明材质小样

[官方仓库](https://github.com/Shreeyak/cleargrasp)公开`real-val/d435/000000080`，固定revision `0688647e49380139e440bacbe8562132bb2afbd7`。取RGB、raw EXR、opaque-twin参考EXR、mask与K，**668737字节**。真实透明场景以相同几何的喷涂不透明物体替换得到参考，非独立激光测距。

实际解码1280×720；mask有123112像素，其中参考深度有效101269，原始深度在该有效参考区域缺失**45981/101269=45.40%**。剩余55288像素的raw−reference中位+0.04834m；参考本身全图有81078非有限像素。目标有效参考中位0.500m。此为单公开验证样本像素描述，不是总体漏检率或本模型准确率。

单位通过官方采集代码的`depth_scale`和保存流程核到米；K/尺寸一致，叠图已目检。当前Python缺现成EXR模块，使用样本专用ZIP/HALF解码并检查chunk长度、行覆盖与三通道一致；局部脚本及receipt/contract保留于`material_samples/`，不泛化为通用EXR工具。

### GMRPD：近地小物与路面参考

[官方仓库](https://github.com/hlwang1124/GMRPD)固定revision `a7a442857ea4f765d518b79a1186588702130b8a`，两公开demo的RGB、u16/u8 depth、label及README/LICENSE，**3431504字节**。图像为1280×720，包含室内小瓶/反光地面，以及室外纸盒、小物和路沿排水沟。

两张u16深度零值占31.54%/19.81%，第二张65535占8.29%；mask值为0/1/2。**深度米制单位、K、标签值映射、65535的正式含义未核实**，因此当前只通过格式/内容可用性检查，不能让这些像素进入严格metric/FREE评价。源码与图像在`ground_samples/`，未消费官方test。

这三个小样加首轮VisTa3D已经把“公开页面有数据”推进到真实文件、字段与失败约定的检查；各自仅适合特定问题。

## 扩展公开来源目录

以下表格均指**新增候选**；首轮VisTa、Middlebury、Booster、DDOS、DENALI等不重复列入。详细传感器字段、下载URL、许可证据与小切片方案见各`*_sources.json`；“可下载”不表示本轮已下载。

### ToF、波形、噪声与反射率

| 来源 | 实际保留的信息与最小已核入口 | 适用性与限制 |
| --- | --- | --- |
| [HistNet](https://researchportal.hw.ac.uk/en/datasets/dataset-for-robust-super-resolution-depth-imaging-via-a-multi-fea) | 真实32×64×15 bins＋128×256 intensity；大学档案393MB，CC BY | 紧凑直方图形态/次峰；`_RGB.mat`实际为强度图，真实序列未核到公制GT，zero label placeholder不能当标签 |
| [ZHAW MPI mesh](https://zenodo.org/records/7892452) | 10mm网格＋墙＋0.3m可控杂散光；320×240×14频幅相/逐次重复，最小ZIP347760327B，CC BY 4.0 | 很贴近弱前景与强背景；幅相不是photon histogram，有限频率不唯一决定CNH |
| [WISC SinglePhoton](https://github.com/felipegb94/WISC-SinglePhoton3DData) | 真实timestamps/hist、同步/异步、光通量/pile-up、双峰IRF；单hist包约100MB–1GB | 可拆仪器峰与场景多返回；不能先执行双峰消除再研究弱峰。代码MIT，独立数据许可未核 |
| [FLAT](https://github.com/NVlabs/FLAT) | 合成多跳sparse transient `prop_idx/prop_s`；CC BY-NC-SA 4.0；单transient大小待核，全集576GB | 可以重新施加仪器响应/zone聚合，机制信息多于单张depth；未提供实测ST噪声，禁止全量先下载 |
| [945nm材料反射率](https://zenodo.org/records/8322076) | 实测反射率随入射角变化；表＋说明204250B，CC BY-NC-ND 4.0 | 可约束暗材料/斜入射的模拟取值；无波形，保留原数据许可与出处，不当ST标定 |

DeepToF、iToF2dToF、PMD Selene、TMF8801塑料源单列为待获取线索：分别有当前NAS文件未列出、表单、登录、向作者索取等缺口。本轮未申请或发送信息，未把论文中数据规模计为现成可用量。

### 透明、反光和深度参考失效

| 来源 | 保留的信息/获取情况 | 为什么有用与弱点 |
| --- | --- | --- |
| [ClearGrasp](https://github.com/Shreeyak/cleargrasp) | 上述真实小样已读通；公开repo还有real-val 123/130/153；库代码Apache-2.0，外部分发整数据许可另核 | 首选材料小试；要区分raw缺测、错误远距和参考自身无效；桌面瓶子不能直接泛化走廊玻璃 |
| [GlassRecon](https://github.com/JMamie/GlassRecon) | 917真实室内图，RGB/raw/mask/K/完成depth/过滤评价depth，官方Drive包入口已核；许可/小包大小未核 | 玻璃门窗最贴近导航；参考由共面点＋平面拟合构造，依赖原RGB-D，不是独立激光GT；须用过滤后的evaluation depth |
| [TRansPose](https://sites.google.com/view/transpose-dataset) | 87序列，99透明＋12不透明物；RGB/raw/完成depth、robot相机位姿、CAD/6D pose、mask、TIR；最小公开序列2.81GB，CC BY-NC-SA 4.0 | 多视角下材质失效；robot/hand-eye比纯SLAM更独立，depth仍为CAD插入背景。BlindAssist没有热像输入，不能借TIR当可用推理通道 |
| [ClearPose](https://github.com/opipari/ClearPose) | >350k真实RGB-D、63透明物，depth_true/label/metadata，factor_depth1000；Dropbox入口，逐scene大小未核 | 遮挡、液体、弱光、多视角；参考依赖人工物体/SLAM配准，背景仍raw。repo MIT不自动覆盖外部全部数据 |
| [DREDS＋STD](https://github.com/PKU-EPIC/DREDS) | 合成主动双目材质随机化＋真实材质数据；官方包URL，CC BY-NC 4.0；小切片未核 | 同几何不同材质机制对照；主动双目噪声不能当CNH；真实参考为photogrammetry mesh＋位姿渲染 |
| [TODD](https://borealisdata.ca/dataset.xhtml?persistentId=doi%3A10.5683%2FSP3%2FZJJAJ3) | 14659真实RGB-D，depth/mask/6D pose；CC0；train约2.11GB、val约889MB | 透明器皿/液体的可控比较；AprilTag/CAD构造参考，几何种类少，最小已核包仍偏大 |
| [Mirror3D](https://3dlg-hcvc.github.io/mirror3d/) | 7011镜面实例mask/平面注释、delta-depth；repo有小样，关联Matterport/ScanNet/NYU原数据 | 先修正评价器把镜中虚空间当真值的问题；mask存在不等于平面/精细参考可用，必须检查flags及原数据授权 |

Trans10K只提供透明物体位置mask，无公制深度/K/轨迹；可用于材质先验，不能为身体空间标FREE。旧透视/单目数据凡只给相对深度，均不计当前公制缺口已解决。

### 连续RGB-D、动态物体与位姿

| 来源 | 可用字段与获取成本 | 可回答的问题/参考边界 |
| --- | --- | --- |
| [TUM RGB-D](https://cvg.cit.tum.de/data/datasets/rgbd-dataset) | 30Hz RGB-D、100Hz外部动捕、K；PNG/5000米。fr1/desk23.4s约0.36GB，rpy27.67s约0.42GB，CC BY 4.0 | 近家具与纯转动对照；位姿外部独立，深度仍Kinect测量；加速度不是六轴IMU。本地旧目录未核现货 |
| [Bonn动态RGB-D](https://www.ipb.uni-bonn.de/data/rgbd-dynamic-dataset/) | 外部OptiTrack＋独立Leica静态扫描＋动态RGB-D；synchronous182.8MB、balloon243.8MB；明确数据许可未找到 | 历史表面何时应该失效、遮挡重现；静态扫描不为动态人/箱子提供逐物体真值，本地archives为空 |
| [OpenLORIS](https://huggingface.co/datasets/shixuesong/openloris-scene) | RGB-D/两IMU/轮速/外参；office外部动捕，其余多离线LiDAR SLAM。当前office包10618337280B，CC BY-ND 4.0 | 比较相机朝向与移动方向；若确认本地现货可先复用，目前本轮未核其本地office。旧Drive已失效，不承诺小片下载 |
| [EVIMO2v2 RGB train](https://better-flow.github.io/evimo/download_evimo_2.html) | 2080×1552真RGB、扫描对象＋Vicon相机/物体位姿、metric depth/实例mask；RGB组22/33GB，数据许可未核 | 最值得保留的独立动态几何源，可拆相机/目标运动；参考由扫描投影，覆盖/动捕间隙显式处理；没确认单段小包 |
| [VOID](https://github.com/alexklwong/void-dataset) | 30Hz RGB-D＋400Hz IMU、K/validity/sparse depth；48train序列；当前整ZIP、独立小片未核，研究用途条款 | 深度覆盖与时序支持；XIVO的pose和稀疏点相关，不能互相当独立GT；depth不是8×8硬件ToF |
| [M3ED Spot train](https://m3ed.io/sequences/) | RGB/IMU/LiDAR/标定；obstacles85s H5约30.8GB，stairs100s约31.5GB，CC BY-SA 4.0 | 台阶和姿态变化；depth_gt在左事件相机系，来自4/40扫描聚合，不能直接当当前RGB因果观测；不整包先下载 |
| [WHU i2Nav-Robot](https://github.com/i2Nav-WHU/i2Nav-Robot) | 10Hz彩色双目/LiDAR、200Hz IMU、轮速/转角、独立导航级INS姿态/速度；先小轨迹文本，bag大小未核 | 身体轴与速度方向不一致、转弯/后退；室内GT仍用LiDAR地图匹配，曝光时长未计入时间戳，README同时写GPLv3及学术用途 |

连续深度源的最小有价值问题是：**时序累积增加稳定性的同时，是否留下已经离开的障碍？** TUM/Bonn最适合较低下载成本的短序列对照；EVIMO2是更强的后续对象运动参考。人头/躯干独立运动仍应优先用现有HEADS-UP/BlindWays/Nymeria，固定机器人相机不能替代这一点。

### 台阶、路沿、坑洼与负障碍

| 来源 | 字段与最小入口 | 关键适用性/限制 |
| --- | --- | --- |
| [SurDis](https://github.com/kuanyewleong/surdis) | 17302图/200序列/10地点；胸前单色双目、下阶/上阶/排水沟/落差/混合坡度框；CC BY-NC-SA 4.0。repo一对图＋NPY＋XML约2.17MB | 负障碍语义很贴任务；**生成脚本直接存左右x坐标差，depth_maps实际为视差**，未闭合米制换算；不是RGB或mask。整包临时OneDrive未验 |
| [RGB-D stair](https://data.mendeley.com/datasets/6kffmjt7g2/1) | 2276train＋556val；凸/凹边及踏面/踢面mask，CC BY 4.0 | train/val补边缩为512²，需要对应K/变换/单位；凸凹边不自动代表上楼/下楼。154正式test有额外原分辨率几何，本轮未读取或消费 |
| [GMRPD](https://github.com/hlwang1124/GMRPD) | 3896 RGB-D、30场景、18类异常；两公开样本已读通；repo MIT | 低小物与反光地面，详见上文；尚未闭合米制/标签合同 |
| [Pothole激光几何](https://github.com/ruirangerfan/rethinking_road_reconstruction_pothole_detection) | 真实坑洼双目＋石膏坑模独立激光PLY＋stereoParams；model3一对图/GT/标定约8.53MB，repo MIT | 负障碍独立几何最有价值线索；先核单位、图与PLY对应、坐标和石膏模正负翻转，未下载不称已对齐 |
| [TrueSight](https://documents.uow.edu.au/~phung/TrueSight.html) | 5000 ZED2左右图/自动depth/手工16类，Steps与Dangerous Surfaces分开 | 需签Release Agreement并邮件申请；仅科学研究、商业须书面授权、不得再分发。本轮只读协议未联系作者；depth黑色是undefined |
| [Stereo pothole](https://github.com/ruirangerfan/stereo_pothole_datasets) | 官方示例合计67帧：RGB、像素坑mask、原/变换视差、点云，可逐文件取，repo MIT | 视差来自双目估计，变换视差不当深度；与上一坑洼源可能同源，先去重 |
| [GOOSE](https://goose-dataset.de/docs/dataset-structure/) | RGB/NIR、实测LiDAR语义/实例、标定；curb/pole/wire/tree_root，CC BY-SA 4.0 | 可做地面杂物/杆线多模态对照；无单独坑/落差类，LiDAR未返回不是连续地面证据，最小包未核 |
| [RELLIS-3D](https://github.com/unmannedlab/RELLIS-3D) | RGB/双目、两LiDAR、pose/K；杆/木头/碎石/水坑/深水，CC BY-NC-SA 3.0；3MB图/24MB点云示例 | 能补地面复杂度；示例之间是否对应待验，越野视角与头戴近场不同 |

其他线索保留而不抢优先级：ORFD的dense depth是稀疏LiDAR插值，且README Apache与LICENSE MIT冲突；RUGD只有已核语义没有米制几何；ORG反光地面序列有mask/odometry但无稠密GT；GND广域校园多传感器虽含stairs，当前不是逐像素负障碍的最短接入路径。

## 推荐的实际推进次序

| 次序 | 使用数据 | 可判别问题与最小检查 | 何时有理由继续 |
| --- | --- | --- | --- |
| 1 | 原18段完整RGB＋新核27段逐帧IMU | 在已固定候选点之间利用额外RGB跟踪，比较断链与背景持续成本；再查姿态突变时地面误入，不给传播点虚增检出 | 真杆链更稳定且背景残留代价可解释；单纯帧数增加不是收益 |
| 2 | 已落地VisTa3D＋已有SANPO mask/UE配对几何 | 分开定位、可测覆盖、绝对距离误差与拒绝代价；原生/降采样同帧对照；不用逐图GT尺度拟合遮蔽失败 | 定位与测距失败能被分解为不同机制，且可复核地减少其中一种 |
| 3 | 本地ZJUL5/LCSPC＋已落地SP-TransientBench | 看区内前景占比、主返回与完整峰形差异；再决定取HistNet或受控mesh/wall | 发现深度摘要丢失而波形仍保留的具体可用信息；跨传感器转换身份单列 |
| 4 | ClearGrasp＋本地玻璃弱片段 | 明确“无深度”“有深度但穿透”“有效参考缺失”；再查GlassRecon门窗 | 有效参考覆盖足以暴露错误远距；不把透明mask直接当米制障碍 |
| 5 | SurDis＋stair＋GMRPD，后续激光坑模 | 建立正障碍、负障碍、观测连续地面、UNKNOWN四种读出；先核视差/深度单位与参考地面 | 几何能够支持符号判别，再扩大样本；无返回不推成无障碍 |
| 6 | TUM/Bonn小段；后续EVIMO2 | 测量历史证据何时失效，拆自运动与目标运动；保留当前/聚合参考差别 | 时序收益超过陈旧表面残留成本；大包仅在该问题确实需要时取 |

上述次序是本次推荐；只完成来源筛选/小样接入，没有把这些后续检查写成已运行，也没有改变模型、报警阈值、App或冻结v1/v2。当前最强方向是第1–3项，地面危险和材质为新补出的可执行支线。

## 前两轮产物、成本与核验

`expansion/catalog_index.json`汇总36张卡片并保留每张原始记录与来源文件；`local_assets.json`、`tof_sources.json`、`material_sources.json`、`sequence_sources.json`、`ground_sources.json`分别保存完整合同/许可/链接/选择计划。原始数据及派生产物均留在F盘canonical artifact树，无新旁路数据根。

扩展公开样本payload为**105950195字节**（SP101849954＋Clear668737＋GMRPD3431504），加首轮VisTa7357909字节，共**113308104字节**；少量网页/源码/元数据另计，扩展所有落盘素材仍小于150MB。无全量集下载、账户注册、表单提交或外发信息。

五支线本地command-wall保守计≤320/780秒（序列支线按其全部150秒预算保守计，其他分别30/40/40/60秒）；主线程下载解析/目录整合保守≤180/300秒。GPU/模型推理/训练/付费均0。公开源小样的格式、SHA与分母检查通过；关键叠图实际查看；本次脚本语法和目录36项唯一ID/原JSON哈希/参考点分母守恒检查通过，文档索引15热文件/203本地链接通过，并核验本次diff；无全仓测试。进程/会话均已释放，固定输入及必要诊断文件保留供复用。

## 第三轮：把潜力推进到实际字段与信息损失检查

用户继续要求“还能继续挖掘吗”。本轮仍由五子代理并行，主线程整理RGB外观覆盖。范围为固定train切片的区内混合、原生几何、公开坑模、材料/真实ToF源及新增细结构/照明源。五支线各≤180 command-wall秒、主线程≤300秒；GPU/训练0，新增下载≤32MB（地面20MB、ToF5MB、细结构5MB、其他元数据2MB）。只读已消费/train/明确公开示例，不打开保护480或新冻结eval。产物在同根`deeper/`，旧失败和冻结规则保留。

### 完整RGB：5407个逐帧描述与54个复核入口

实际解码6746张原图；4871张未缓存图有同source/boot的缓存参照，536张没有，后者全部为070657抬头段。使用80×60灰度缩图、分别减中位数后的未配准MAE；4871张差异p10/p50/p90为0.516/1.514/7.737（0–255灰度单位），与最近缓存的接收时间间距p50为72.63ms、p90为3556.49ms。这是外观与采样间距，不能解释成语义新颖度、曝光延迟或算法收益。

`deeper/rgb_appearance/per_uncached_rgb.jsonl`逐帧保存来源、灰度统计、最近缓存及间距；每段挑差异p50/p90和最大时间缺口，完全无参考的段挑首/中/尾，共54个入口、6张拼图。抬头段没有缓存不等于可以绕过原姿态门控。目检房间、抬头、杆/横杆拼图，额外图中既有外观相近的邻帧，也有构图/头动/目标运动变化。优先073821连续杆流、073554/073635/073740低频杆/横杆，以及072939/073020台阶/地面变化；没有标出新的目标真值。

源码`host/mine_rgb_appearance_dev_20261013.py`。首版遗漏无缓存段的展示入口（逐帧统计已包括），补齐后重跑，原摘要保留；54入口覆盖18段、路径/唯一ID及全体统计不变检查通过。两次实际读取共65.859秒。

### Hypersim：旧单点采样确实丢掉原生近表面支持

先固定已有train字典序前四个不同scene首帧，均属ai_001一个family。原生1024×768光轴深度转换及旧256×192 RGB/depth均逐元素精确复现；旧深度每4×4块只取左上角一个值，之后才生成8×8 proxy。四帧原生、K/姿态均在，但未核到这些帧的instance/normal/position逐像素文件。原姿态位置为asset units，须乘约0.0254m/unit。

| 固定诊断 | 分母 | 旧表示丢失支持 |
| --- | ---: | ---: |
| 原生>10cm邻接边界所在粗格 | 8063格 | 734格无旧边界支持（9.10%） |
| 全有效4×4块，跨度>10cm；距块最小Z≤2cm仅1像素 | 438块 | 372个旧采样比最小Z远>10cm（84.93%） |
| 同上，近表面占2–4像素 | 2050块 | 1376块（67.12%） |
| 同上，近表面占5–16像素 | 3439块 | 1401块（40.74%） |

不是细杆漏检率；遮挡边缘、斜面和细部都可能进入分母，采样相位也固定。真正改变下一步的是：可在**不下载、不推理**的情况下，从现有原生几何保留每格有效数、min/max、近表面占比，先定位resize和proxy各自的信息损失。固定选择、公式、精确复现、逐帧统计与已查看拼图保存在`deeper/hypersim_native/`。

### ZJUL5：找到真实区内混合切片，保留几何与语义复核

固定原train顺序前32帧，全部library2。2048区中1201区原mask有效且代表返回可用；117区裁切后为空，保留UNKNOWN。1040区参考覆盖充分，78区/17帧同时有0.3–2m与>2m参考，10区/5帧近参考占比≤20%；再要求两组中位数间隔≥0.5m，剩3区/3帧。

三例的near/far/返回分别为1.962/2.473/2.497m、1.224/2.068/2.012m、1.422/2.222/2.148m；代表返回数字均更接近far，但不能直接说回波来自远面。fr行列次序已对照上游loader；区足迹、轴向/径向口径未独立重标定。`hist_data`第二列旧来源称sigma、上游代码称var，本轮原样保存，**不把它作为已确认的标准差或概率分布**。选择/全部分母/独立重算在`deeper/local_zju/`。

三例已生成RGB/depth/zone/near/far/UNKNOWN叠图并实际查看：第一例是台面/墙面交界；第二例为器件擦zone边角、临近缺失带，适合作投影/配准敏感反例；第三例为柜体前缘与后方墙面，是最值得优先复核的分离表面入口。保存原坐标掩码与逐例弱判读，不称细杆真值。绘图没有修改固定统计，原摘要内容SHA复核一致。

### 激光坑模：可用独立表面，暂时不能与RGB直接配对

实际取[官方model3示例](https://github.com/ruirangerfan/rethinking_road_reconstruction_pothole_detection)：PLY有162677个有限点；MATLAB标定已解出左右K、畸变、R/T，WorldUnits明确毫米、基线119.493103mm。独立激光PLY自身的单位/朝向没有随相机标定自动确认。

关键缺口是发放图2081×1040而标定1920×1080，Rectification未初始化、Q为单位阵；5对图与4份PLY无明确对应表，扫描到相机和坑模翻转无已核外参。因此保留为**未注册几何**，没有制造投影精度或叠图。SurDis源码的V/H翻转及Undistort已核，但仍无数值K/baseline/Q。数据、源码、SHA收据及标定在`deeper/ground_geometry/`。

### 新找到并读通：OctoNet的真实ST 8×8直方图

[OctoNet官方项目](https://aiot-lab.github.io/OctoNet/)使用ST VL53L8CH。[公开HF](https://huggingface.co/datasets/hku-aiot/OctoNet) revision `47a68995d135344c43356840e9e14cc630f62c35`可列出976个ToF文件，均在node_4；数据CC BY-NC 4.0，代码许可单列。本轮读取预选jump录制`4_ToF_1_10_jump_1_20240530180318.pickle`，2326728字节，**232帧，每帧tof_bins[64,18]与tof_depth[8,8]**，另有timestamp。

用pickletools静态解释字面量和数值字节，不执行网络文件中的构造函数。267264个bin均有限/非负，16131为0；14848个深度元素中251为0，原数值范围0–4688，样本不编码单位，因此没有将它当毫米或距离真值。时间跨度31.595375秒，231个正间隔均值136.7765ms；这是文件记录节奏，时钟来源/时区/曝光同步未知。

这是比单纯距离摘要更贴本机的新波形入口，但**未证实是未加工固件CNH**：没有ambient/status/invalid mask，也没有start_bin/subsample/ref_residual/积分时间/标定；18bins不能直接截成现有16bins。当前样本是室内人体动作，不证明细暗杆/强日光覆盖；项目提到RGB-D，但配对RGB文件与注册未核实。最优先补采集配置和bin物理定义，再决定是否可进入域差异/峰形检查。原始样本、SHA及静态parser在`deeper/tof_material/`。

同支线还实际读了[Consumer NLOS](https://github.com/sidsoma/consumer-nlos)三个连续person_tracking样本及一个cam样本，单文件约29.5KB。虽然来自VL53L8，作者loader明确已去掉一次反射并对齐时间原点，数组实际为**4×4×128**、所读样本非负且大量为零；代码有补零操作，但存档是否已补零未独立追溯。particles是算法粒子而非GT，不能当原始8×8 CNH或静态重复噪声。可作弱非直视返回形态旁证。

[945nm材料表](https://zenodo.org/records/8322076)实际下载204250字节、解析314曲线×9角度，共2826值。它来自Infineon IRS1125C强度经Lambertian标定的方向相关量，6值为负、168值>100%，全范围约−0.022%至201.884%；不能简单除100作为ST 940nm表面albedo。适合约束入射角/材料变化趋势，保留仪器、方向和参考归一化身份。239份描述非空，其余不猜材料；CC BY-NC-ND 4.0原表保留。

### 新读通的真实照明配对：LLRGBD

[LLRGBD/LISU官方仓库](https://github.com/noahzn/LISU)提供同场景低照/正常RGB及人工语义标签。本轮先封存按三个train目录共同文件名词典序选首项规则，再通过HTTP Range读取375427289字节ZIP的目录及`10.png`三件，**实际接收951005字节**，未下载整包或test。415个train文件名在三个目录完整对应。

小样均640×480，RGB平均强度52.60→90.51/255，CRC32/大小/SHA和标签分母通过。实际查看是餐桌/椅子同构图照明变化；人工语义mask合并了椅背细条间部分孔洞，因此新增价值是**真实照明对照**，不能当精细杆线mask或独立米制参考。当前未下载depth，不将传感器/补全depth称作独立几何。选择、精确Range收据与样本在`deeper/thin_sources/`。

LLRGBD明确的数据许可尚未在已读入口找到，论文许可不代替数据许可；本轮仅作来源检查。论文的100对test在代码命名为val，本轮只取显式train，未消费该val。其他条件候选如下，未计作已经接入的小样：

| 新来源 | 相对已有数据的增量 | 当前缺口 |
| --- | --- | --- |
| [ThinObject-5K](https://github.com/liewjunhao/thin-object-selection) | 辐条、绳网、细长形状的人工筛修alpha | 原论文5743透明前景、train4743，合成背景；非5748真实场景。未取train GT小样，无米制/轨迹 |
| [DIS5K](https://xuebinqin.github.io/dis/index.html) | 复杂孔洞与边界的人工二值mask，train3000 | demo结果是预测，未拿来充当GT；非商业研究/教育条款禁止再分发，不能套代码Apache许可 |
| [RGB-NIR Stereo](https://github.com/divisonofficer/Pixel_aligned_RGB_NIR_Stereo) | 真实暗处多模态与独立稀疏LiDAR | 小暗室包仍约339.67MB；train身份/单位/许可未闭合；BPNet稠密视差是模型派生，无实际样本读取 |

现有SANPO还可沿[官方附录](https://openaccess.thecvf.com/content/WACV2025/supplemental/Waghmare_SANPO_A_Scene_WACV_2025_supplemental.pdf)区分人工标注与AOT传播帧；附录直接举出细树/杆传播失败。先恢复已存mask的来源身份，比把所有mask都当同等级细杆GT更有价值。本轮仅核该文档，尚未完成本地mask逐项归属。

### 第三轮的下一优先级与交付核验

最值得继续的三件事：**OctoNet采集配置/bin物理合同 → Hypersim原生近表面支持与ZJU分离面 → LLRGBD真实照明配对**。前者补更贴ST的外部波形，第二项不需新下载就能定位既有表示的信息损失，第三项能把照明变化与换场景分开；原54个RGB入口供时序与标注复核。没有把任何公共来源升级为本机细杆/安全真值，也没有因此改模型或报警阈值。

`deeper/INDEX.json`汇总六支工作目录、已读取样本、条件候选和可复用入口；各目录保留选择、来源、脚本、摘要及必要原始小样。第三轮已知样本/官方源码传输至少13717470字节（地面10008559＋ToF及源码2757906＋LLRGBD Range951005）；搜索/API元数据未全程逐字节计量，不把落盘大小当网络流量。所有支线的下载限额合计32MB，未下载整库。

支线command-wall保守上界333/900秒（30＋20＋70＋33＋按全部180秒计的照明支线），主线程≤120/300秒；仅RGB两次解码实测65.859秒。GPU/推理/训练0，任务进程全部结束。各实读分母/字段/来源哈希与相关图像检查已完成，新增RGB脚本/热文档/diff作范围内核验；交付SHA及检查记录见`deeper/delivery_receipt.json`。
