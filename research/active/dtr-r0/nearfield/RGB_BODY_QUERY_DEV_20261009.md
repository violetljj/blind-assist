# RGB真实身体空间查询：独立子线（2026-10-09）

状态：EXPLORE / 数据核查、真实输入诊断及基线烟测已执行；用户授权“推进”。

## 本阶段目标与资源

目标：把RGB从融合辅助提升为独立研究对象；真实数据定量评价为硬目标，补足ToF/CNH当前主要模拟证据。先核查数据/文献、原图信息损失及强深度基线执行条件，再确定可训练任务。已有RGB-only BodyQuery必须继承；身体查询、视频碰撞和热图本身不直接构成新贡献。

本阶段预算独立于CNH：CPU准备/图像诊断/核验累计600秒，GPU基线启动及推理累计180秒；最多复用50帧/10session已消费真实Development，基线烟测最多10帧。外部只读搜索及元数据不计计算预算；新增下载上限256MiB，仅公开小样本/元数据，不购买资源、不新采集、不访问保护test。预算是本阶段实施范围，不是整个RGB研究上限。超过当前预算的训练不在本阶段启动。

调整范围：数据候选、输入分辨率和基线执行方式；保留旧负结果和停止规则，不能通过改名扩展旧SANPO候选质量验收。交付：来源可评价性表、最接近文献/贡献定位、真实原图诊断、基线运行收据或明确阻塞、下一可执行任务。

候选机制：可变米制区域查询＋保留薄结构的高分辨率因果时序证据。运动估计与ToF增量后续分开检验。对照须包含强单帧metric depth＋几何及因果视频metric depth＋几何，匹配图像、帧、内参、尺度/额外信息、校准与阈值选择；不得使用未来帧为在线基线增益。

## 路线调整与既有证据

`PROJECT_STATE.md`原将SANPO列为“不作独立推进线”；本次明确调整为RGB真实评价候选，不升级旧模拟结果。已有[SANPO-Real筛查](SANPO_TARGET_COVERAGE_20260927.md)50帧/10session在原1–2m光轴通道细类有效深度支持为0，其候选验收停止保留。新图像诊断仅重新消费该Development观察及人工掩码，不能变成独立确认或细障碍事件评价。

V2低分辨率融合和RGB扰动负结果保留；缩图影响是待实测的输入机制，不是融合失败既定原因。深度缺失、遮挡、不完整身体坐标或事件时间契约不能标成无障碍。真实图像中的双目估计深度也不是独立物理真值。模型效果/新颖性/实机成本均未验证。

## 结果

### 数据：SANPO-Real优先，真实图像不等于真实碰撞标签

| 条件 | SANPO-Real | HEADS-UP |
| --- | --- | --- |
| 真实视频/获取 | 701session，15fps，头/胸相机；公开GCS单对象获取，数据CC BY 4.0 | 真实头戴RGB，论文30fps；HF Apache-2.0，仍gated；10-07已有用户同意获取记录 |
| 空间参考 | 原生rectified K、双目CREStereo及ZED估计depth、VIO pose可得；不是独立小目标距离真值 | 米制SDK depth/VIO可得，但本次未闭合公开标定/同步/缺值契约 |
| 细障碍 | pole/tree/sign等人工与传播掩码；树枝无独立类，细杆传播存在已知失败 | 主要标签为行人轨迹，静态细杆/横杆标签缺口明显 |
| 时间/身体 | pose无timestamp列；frame/fps只能称nominal time。身体安装外参、体积与实际碰撞事件均缺 | 轨迹经YOLOv8/ByteTrack/2.5fps均值/Kalman，不能作为人工碰撞真值 |
| 当前用途 | 原生真实输入与相机局部可见表面query可起步；审核连续几何接近事件后再计算首次提醒 | 动态行人/头动补充；RGB包hard约37.06GB、unconstrained12.36GB，本阶段不下载 |

来源：[SANPO官方仓库](https://github.com/google-research-datasets/sanpo_dataset)、[WACV论文](https://openaccess.thecvf.com/content/WACV2025/papers/Waghmare_SANPO_A_Scene_Understanding_Accessibility_and_Human_Navigation_Dataset_WACV_2025_paper.pdf)、[补充材料](https://openaccess.thecvf.com/content/WACV2025/supplemental/Waghmare_SANPO_A_Scene_WACV_2025_supplemental.pdf)、[HEADS-UP论文](https://arxiv.org/html/2409.20324v1)、[HF模型卡](https://huggingface.co/datasets/Yassaman/HEADS-UP/blob/main/README.md)、[公开文件目录](https://huggingface.co/api/datasets/Yassaman/HEADS-UP/tree/main)。

实读官方train session d3CKba…的description/pose/annotation：449pose行、449标签帧、75人工帧，READY435/449，fps15，pose无时间戳列。官方2025-06-05给出fixed_camera_poses.csv修正，但后来[issue12](https://github.com/google-research-datasets/sanpo_dataset/issues/12)仍报告同session姿态翻转/深度对齐问题；[作者回复](https://github.com/google-research-datasets/sanpo_dataset/issues/12#issuecomment-3362537386)当时在修复，本次未找到完成确认。故跨帧世界累积前须针对样本核验，不因文件名叫fixed而默认正确。世界为OpenGL右手Y-up、−Z前，不能与相机光学+Z混用。

官方loader只解码semantic=R、instance=G×256+B，未证明instance0无效；本轮保留pole类别/ID0掩码组，并标身份UNKNOWN。不能用semantic unlabeled=0推导instance0应删。

### 文献：贡献候选收窄，强基线必须含时序

| 近邻/基线 | 已有内容与本线定位 |
| --- | --- |
| [COPILOT，ICCV2023](https://arxiv.org/html/2210.01781) / [官方代码](https://github.com/leobxpan/COPILOT) | 第一视角视频→全身/关节碰撞和图像热图，已有Root-Only单相机版本；真实四段视频为定性迁移。身体查询、单相机或热图单独均不是本线新颖性。 |
| [LACO，CoRL2023](https://proceedings.mlr.press/v229/xie23b.html) | 图像＋机器人状态＋语言直接查碰撞；直接query替代完整重建已有先例。 |
| [薄结构多帧检测，CVPRW2017](https://openaccess.thecvf.com/content_cvpr_2017_workshops/w4/papers/Zhou_Fast_Accurate_Thin-Structure_CVPR_2017_paper.pdf) | 边缘、里程计和多帧3D；单目版含IMU尺度。时序检细杆本身不是新机制，不能给RGB-only偷偷增加IMU。 |
| [Forecasting TTC，IROS2019](https://arxiv.org/html/1903.09102v3) | 单目历史视频预测动态行人接近，已有真实走廊数据；不是静态头部横杆事件。 |
| [MonoMPC](https://arxiv.org/html/2508.07387) | 单目深度和状态/控制序列预测轨迹最小间距分布；轨迹条件风险也已有工作。 |
| [Depth Pro](https://github.com/apple/ml-depth-pro) / [UniDepthV2](https://github.com/lpiccinelli-eth/UniDepth) | 强metric深度空间基线。前者已在本阶段真实输入烟测；后者支持相机/置信度，作为标定复杂时备选，CC BY-NC 4.0。 |
| [Depth Anything V2 metric](https://github.com/DepthAnything/Depth-Anything-V2/blob/main/metric_depth/README.md) | 可复现实践基线，有indoor/outdoor权重；Small Apache2.0，大模型非商业许可按checkpoint核对。 |
| [Video Depth Anything](https://github.com/DepthAnything/Video-Depth-Anything) | 已发布metric模型和实验streaming。评价在线提前量须用因果streaming，离线未来帧不得混入；本阶段NOT_RUN。 |

最强候选是显式可变米制query驱动的薄结构高分辨率因果证据积累，并在真实视频改善同误报下检出/提前量或实测成本。尚未证明文献新颖性与收益；如果强深度/视频深度已覆盖效果，不能凭更小模型或更好attention图宣称贡献。

### 已执行：50真实帧输入诊断，10帧Depth Pro配对烟测

固定复用原50人工anchor帧/10session，均2208×1242；本次没有把旧1–2m通道筛查重开。全图PIL BILINEAR降/升采样后，在同一原生像素格比较灰度绝对一阶差分均值：128×72中位比0.126945，256×144为0.204239，640×360为0.392841。它是图像细节指标，不是目标检出率；真实原图→128的压缩比也不同于旧V2的640→128。

人工pole class/id掩码共82组帧，其中ID0为20组、身份UNKNOWN；128输入的平均行掩码支持<1px为25/82、<2px为65/82。该量=area/bbox_height×128/native_width，不是物理杆宽，也不把82组当独立物体。非零ID子集62组帧/23个session-camera标识。首版错误跳过ID0（62、25、55）已保存并修复；旧原始结果未修改。

固定每session第一人工帧共10帧，Depth Pro以原图和128输入各推理一次、公开fx按宽度缩放，不做GT尺度拟合。20/20 CUDA调用完成，配对输出全有效；全图输出相对变化的逐帧中位数再取中位为0.201760，它仅说明输入改变输出，不说明哪一臂更准。权重SHA核对原参考，torch2.11.0+cu130/FP16；启动至输出46.665s、模型调用22.908s（首native7.593s含冷启动）、中位每调用0.807s。设备是桌面/笔记本GPU，不是手机时延；fx烟测没有处理偏心cx/cy，不能称完整几何标定。均未读stereo/depth/mask/pose来作模型输入。

独立float64复算50帧细节比例/非零ID支持及fx比例通过；首版统计问题和修复保留。源码：[输入诊断](rgb_body_query_input_diagnostic.py)、[基线烟测](rgb_body_query_depthpro_smoke.py)。payload为`artifacts.local/work/rgb-body-query-dev-20261009/`，含输入SHA、v1/修复、逐帧表、固定裁剪对比和深度对比；固定裁剪为每session第一帧中心三分之一区域，没有GT选crop。

进一步完成[米制区域读出](rgb_body_query_geometry.py)：相机右/下/前坐标，左右中三横向区×[0.3,3]/[3,6]m，共6个固定query/帧；只投影预测的可见表面，不给完整自由空间或身体碰撞标签。10帧共60query/臂，native估计可见支持33/60、UNKNOWN27/60，128输入为38/60、UNKNOWN22/60。这些计数含地面且两臂像素格不同，不能把支持更多称为更准确。几何读出正确处理公开cx/cy及half-pixel缩放，未消除模型本身忽略主点的限制。6项[聚焦数学测试](test_rgb_body_query_geometry.py)通过，覆盖轴向/径向差别、K缩放、invalid→UNKNOWN和闭合边界；初次浮点边界、junction相对路径两次失败已保存并修复。读出应用3.699s，无新GPU推理。

## 下一可执行阶段

后继执行已完成：[真实连续段参考与基线评价](RGB_BODY_QUERY_EVAL_DEV_20261009.md)。以下保留第一阶段交付时的计划与NOT_RUN状态，当前决定以后继结果为准；前阶段预算/负结果不重写。

已从旧train样本的可见栏杆场景物化连续片段：-PqSDmi…/camera_chest，首人工anchor414前后帧392–436，45帧/15fps（样本首末跨度44/15=2.933s，名义片段3s）。RGB/CRES/ZED/mask各45，共180/180文件，新增228,777,854B（218.18MiB），引用复用16文件22,446,439B；大小/MD5/SHA与PNG/gzip格式完整性核验，GCS CRC32C仅保存元数据、未独立重算。7/45人工、38/45传播，READY42/45、NOT_READY3/45，READY不等于姿态精确。首次4次TLS EOF造成176/180，按原计划只补4缺失文件；累计metadata/download76.449s/200s、实际新增字节不超256MiB，失败与执行源码留存，不重选、不重下成功文件。HTTP/线程已释放。[获取脚本](rgb_body_query_real_sequence.py)，payload的`sequence/`保留完整计划/收据，另有原生RGB低分辨率GIF预览。

下一步完成该真实段相机局部query与双目参考的可评价mask。缺身体外参时任务名称严格是“相机局部可见米制区域查询”；只有公开安装/地面变换成立后才映射HEAD/BODY。事件定义为审核后的参考几何阈值接近，时间先按frame/fps，不能叫真实碰撞提前量。

先跑Depth Pro＋同等K几何及VDA metric因果streaming。评价端可用stereo/人工核验，模型端仅用匹配RGB历史、K和双方共享的公开尺度信息。深度缺失、遮挡、参考间不一致区域UNKNOWN；用人工真实薄结构子集查标注质量，保留覆盖分母。此阶段尚无事件标签，任务训练/检出/误报/提前量均NOT_RUN，不降低真实评价目标。

标签可评价后执行低/高分辨率×单帧/因果多帧匹配消融；先用全图特征和query驱动的高分辨率局部读取，任何训练/推理crop必须由RGB/query本身选择，GT只在评价端。固定prompt预算、按session配对，同误报比较事件和首次提醒；手机成本另测。文献/标签/尺度三项检查决定是否继续这个切口，而不是旧融合结果决定RGB整体价值。详见本次[RUNS](../RUNS.md)。
