# 现有数据再利用与公开细结构参考

2026-10-13，EXPLORE。用户授权“挖掘现有数据的潜力，也可以去网上找找用得上的数据”。当前问题是细结构候选已经可见，但公制距离、持续跟踪及身体区域仍不稳定。

本轮交付：现有已消费18段RGB/CNH数据的覆盖与复用清单、已有公共资产适配判断、经原始来源核实的公开数据优先级，以及一个公开细结构数据小切片的实际读取检查。只读既有数据、生成独立派生产物，不修改冻结v1/v2、模型、阈值或手机行为。旧几何/可信度失败保留，新来源检查不是重开旧停止运行。

预算：本地统计与小切片检查合计CPU≤600 command-wall秒，GPU0、训练0、付费0；公开数据payload下载≤100MB，本轮新增payload≤200MB。数据走`artifacts.local/work/data-potential-dev-20261013/`（canonical junction，物理F盘，开始约22GB可用）；源码和此报告进Git。网络资料只核实有关字段，停止于可决定来源适用性和下一项实验；不批量下载完整数据集。

已有头戴记录仍为已消费Development，额外帧不增加独立场景数。公开参考的传感器、制作过程与适用限制分别记录；官方GT也不等同于本机测距真值。不得读取保护480/test或未打开的新头戴冻结eval。

## 结果与推荐

**现有数据还有明显的时序和形状利用空间；公开数据已经找到并实际读通了近场细结构公制参照。** 最优先组合是：用现有完整RGB流改善候选持续性，用VisTa3D检验细结构距离/尺度机制，用SANPO已有实例掩膜检验杆与背景的区分。UE四状态配对适合解释几何失败。它们回答不同问题，不拼成同一份“实机准确率”。

本轮没有训练或新模型推理，尚无检测/测距提升结果。完成的是数据可用性诊断、可复用索引和公开样本接入；下一实验可以直接消费这些产物。

## 1. 已有头戴数据：5407张图像尚未进入当前深度缓存

读取已消费18段的原始RGB/ToF索引和现有cache manifest，按source＋boot＋frame ID去重，同时核对既有图像SHA。**6746张原RGB全部独立，当前1677个ready ToF行只有1344个深度缓存，对应1339张独立RGB；另有5407/6746=80.15%未缓存。** 其中5154张未被ready ToF最近帧选择，253张被选中但没有缓存。

这不意味着5407张都可直接当有效融合帧：现有配对/姿态门控仍适用，新增RGB没有插值姿态。它们最直接的用途是RGB时钟上的图像跟踪，按需要再运行深度，而不是跟着ToF抽样丢掉图像。

| 片段（弱场景名称） | 原独立RGB | 缓存独立RGB | 额外RGB | 原RGB接收间隔中位数 |
| --- | ---: | ---: | ---: | ---: |
| 073554 正前杆 | 102 | 40 | 62 | 173.55 ms |
| 073635 侧杆 | 70 | 49 | 21 | 185.09 ms |
| 073740 横杆 | 104 | 67 | 37 | 147.26 ms |
| 073821 低头正前杆 | 378 | 77 | **301** | **37.62 ms** |

全18段原RGB接收间隔中位38.53ms，现有深度cache对应ToF间隔197.32ms；不能把这个整体比值套给每个片段。**首选073821零基索引74–377：13.159秒、304张图，其中238张未缓存。** 其余三杆段原RGB已较慢，补帧收益空间小得多。

已生成5407行`local_mining/uncached_unique_rgb.jsonl`与54个`continuous_tracking_slices.json`切片：同source/boot、两个时钟均正向、相邻间隔各≤250ms、至少5图且含额外图。切片不依赖检测成绩，不增加目标标签，也不增加18个录制之外的独立场景。

额外诊断：1677个ready行选未来接收RGB870次/过去807次；1344缓存中为702/642。这是旧离线最近接收匹配的描述，不能当曝光同步或因果在线协议；本轮没有修改或重跑冻结协议。

## 2. 已有公共/模拟资产的再利用

下列是只读库存、源元数据与已消费报告核验；没有跑新模型。SANPO整库枚举曾包含test文件名和PNG尺寸头，所有像素/深度统计先按manifest筛到train；未读取其test像素结果或保护480、新头戴eval成绩。

| 已有资产与入口（均相对仓库） | 核实规模与新用途 | 关键限制 |
| --- | --- | --- |
| `artifacts.local/nearfield/contextual-worker1000-20260908/spec-1000-v1.json` | 1000张640×360；250组同视点CLEAR/BODY_ONLY/HEAD_ONLY/BOTH；横杆252、斜杆248等。精确相机/佩戴者/目标位置厚度可隔离“候选形状丢失、尺度外推、身体盒投影” | 单站点已消费合成；dense native实际仅8张，不能称1000张都有稠密深度。旧999/1000 agreement及849 VISIBILITY_GAP保留；实际复用按UE admission规则 |
| `artifacts.local/datasets/sanpo-real-coverage-20260927/` | 10段、50套2208×1242 RGB/实例掩膜/双深度，2头戴/8胸戴；旧全图统计有杆82、树73、标牌31实例帧。可以评价候选沿杆还是背景边缘 | 官方train、已消费；旧1–2m通道细类0/50及停止结论不改。CRES/ZED同源双目不作独立测距真值 |
| `artifacts.local/datasets/sanpo-synthetic-ba-nfo/` | 按`artifacts.local/work/ba-nfo-20260919/prepared-manifest.json`只取train2000/40段、val500/10段；原生RGB/语义/深度2208×1242。40个train段各首帧，37/40有pole，共706390 pole像素 | 同一固定抽样0/40有源深度<2m的pole。适合原生→模型尺寸的细结构损失诊断，不补近场真值；光轴Z/range旧语义未完全闭合 |
| `artifacts.local/datasets/3rscan/` | 72 scan ZIP中央目录合计22481 RGB-D-pose帧；RGB960×540、深度224×172、K/外参/instance mesh。适合RGB可见而depth缺失/跨视角重现诊断 | 35环境已消费，rescans不是新环境；低分辨率深度和mesh可能抹掉细结构，不能把缺失当FREE |
| `artifacts.local/work/sync-rgb-tof-feasibility-dev-20261010/sources/` | 41069021和41069048各231/152张1920×1440 FARO depth，合计383，旁有K/轨迹。可先筛细结构附近参考支持 | 配套现用融合RGB仅256×192；高分辨率参考时间覆盖稀疏，尚未证明有可用细杆真值，不以更多普通帧替代检查 |

## 3. 公开来源：按补缺口的价值排序

以Exa发现来源，再核对作者项目/论文/官方文件树；下表“已验证下载”仅指本轮VisTa3D切片，其他项目不冒称已下载。

| 优先级与来源 | 能补什么 | 获取/规模与边界 |
| --- | --- | --- |
| **1 [VisTa3D](https://huggingface.co/datasets/shaniaguo/VisTa3D)** · [作者论文](https://arxiv.org/html/2608.20740v1) | 真实细结构RGB、raw depth、对象扫描重投影GT、mask、K、pose，直接检查细物的绝对尺度与深度缺失 | 真实225序列/52物体/8环境，另162合成；HF标MIT、可逐文件取。**本轮已下载12图对应模态，7.36MB**。物体GT来自扫描配准重投影；背景不是同精度真值，pose是ORB-SLAM3估计，非外部测量。排除透明/高反光物 |
| **2 [SP-TransientBench](https://huggingface.co/datasets/shuinb/SP-TransientBench)** · [实时说明](https://huggingface.co/datasets/shuinb/SP-TransientBench/raw/main/README.md) | 真实全直方图的弱峰/强背景/多返回、IRF与域差异诊断 | ADS6311 Hawk，256×192×672，750ps bin，CC BY-NC-SA 4.0；1对深度hist+Livox参考约101MB，全量约168.7GB。本轮未下载，约101MB已超过本轮100MB上限。没有核验配对RGB，降成8×8不等于ST CNH |
| **3 [Middlebury 2014](https://vision.middlebury.edu/stereo/data/scenes2014/)** | Cable/Sticks/Pipes等真实高分辨率细物，检查缩放、边界混合和深度分辨率损失 | 静态双目、结构光disparity、K及毫米基线，可恢复尺度；Cable-perfect完整包89.7MiB。未下载，独立数据许可本轮未核实；Staircase为隐藏GT测试场景，不当现成真值 |
| **4 [Booster](https://amsacta.unibo.it/id/eprint/6876)** | 玻璃/镜面等材料导致的伪远距、错误空域；补VisTa排除的材料 | 公开train38场景/228图，4112×3008，GT disparity、材质/遮挡mask，CC BY-NC 4.0；静态源，未核验包大小/下载，不当连续头戴数据 |
| **5 [Diffuse LiDAR calibration](https://github.com/NikhilBehari/diffuselidar_calibrate)** | 研究zone的非均匀宽空间响应，改变现有模拟中的混合机制，而非只增加噪声 | 随仓库4个NPZ约490KB，`responses(3,3,K), xs, ys`；TMF8828校准，不是ST标定或完整CNH。顶层数据许可未核实，仅作机制/候选资料 |
| [DENALI](https://github.com/NikhilBehari/denali) | 粗zone真实128bin直方图、RGB-D/位姿与digital twin | TMF8828，72000 hist；作者表单申请非商业研究，未提交个人信息/未获得下载。反光目标、NLOS墙面三跳光，域差较大 |
| [DDOS](https://huggingface.co/datasets/benediktkol/DDOS) | 显式ultra_thin/thin_structures标签，宽度/天气/背景机制矩阵 | 合成无人机，30k train，CC BY-NC 4.0；必须先验证0.2–3m支持，不能把薄物标签自动当头戴近场 |

次选：[ETH3D](https://eth3d.ethz.ch/documentation)提供独立激光几何但原生深度不稠密，薄物覆盖需核；[TartanAir V2](https://tartanair.org/modalities.html)适合合成位姿/时序机制，未核实具体近杆覆盖。[DELTAR/ZJU-L5](https://github.com/zju3dv/deltar)有真实VL53L5CX 8×8距离统计+RGB-D，**不是原始CNH**；[ZHAW木格栅序列](https://zenodo.org/records/8205692)有ToF depth+RGB，CC BY 4.0，但模型构造GT与非同步配对限制其测距解释。[HRWSI](https://github.com/KexianHust/Structure-Guided-Ranking-Loss)相对深度不能补当前公制缺口，不优先。

没有找到已经核验开放、可直接下载的“ST 8×8 CNH＋RGB＋日常近细杆公制真值”完整组合。SP-TransientBench搜索缓存的MIT/空README信息已被实时官方README推翻，采用当前CC BY-NC-SA 4.0；其重建表写10场景但文件树列9包，保留差异。

## 4. VisTa3D实际小切片检查

固定revision `6a24d271559ed378f8f0649fc6652a37e323c405`；预先选择`wires_hallway`和`bluecable_hallway`，各取共同文件名排序六个等间隔位置，共12帧。先保存选择清单，再解码，不按模型成绩挑图。实际目录`ground_truth_depth`与README写法`gt_depth`不同；raw除1000、GT除256得到米，Windows本地将时间戳冒号改为下划线，收据保留远程原路径。

- 下载总计**7,357,909字节**，含12×RGB/raw/GT/mask、两套K/pose/frames表及README；50个文件与官方LFS SHA一致，其余小文本/表保存本地SHA和固定来源。不是整数据集111GB的镜像。
- 12/12图像/深度/掩膜尺寸对应，K有限且焦距正；所选12个时间戳均能与frames表及pose匹配，四元数有限且单位范数通过。匹配只验证文件合同，不确认轨迹精度。
- 12帧对象mask内有效GT共**63631像素**，其中raw sensor depth缺失**14597/63631=22.94%**；这是对象像素分母，不是对象漏检率。
- 各帧对象GT中位深度**0.410–0.605m**。叠图已逐格检查，确实为桌面线缆/细线，mask对应目标；不能把这两个近距桌面场景称1–2m直立杆或头戴迁移验证。
- 保存`sample_contact_sheet.jpg`、`selection_before_decode.json`、`download_receipt.json`及逐帧`summary.json`；无模型输出、无GT校准模型尺度。

它能马上用于“模型是否保住细结构、背景锚定是否外推失真、只判距离失效是否比删除候选合理”的Development检查。主要弱点是桌面视角、很近的对象、扫描配准误差和背景深度来源；扩展时按物体/场景隔离，先固定比例/切片，不据成绩挑距离或逐图拟合GT来遮蔽绝对尺度错误。

## 5. 下一项最值得做的检查

1. **先用已有073821完整RGB流与旧ToF抽样流做同片段持续跟踪对照**。共享现有候选检测点，中间帧只做图像跟踪；不把传播位置计为新检出，比较真实独立观测链、断链和背景候选成本。机制假设是跳帧造成关联损失；若额外流不能减少真杆断链就停止这个方向，不能把301额外图像直接说成增益。
2. **用VisTa3D对象区域检验距离机制**。先在固定Development切片上比较原尺度/距离状态与对象参考；分别报告定位、可测覆盖、距离误差及拒绝代价。禁止用逐图GT scale/shift使绝对尺度成绩虚好。需要时用UE同视点四状态精确几何分解失败。
3. **用现有SANPO-real全图实例掩膜扩展形状评价**，区分杆、台阶、门框等背景边；保持旧近场通道失败结论。只有这些切口显示明确缺口，才按需取SP-TransientBench真实波形或Booster材料样本。

以上是执行优先级和可直接消费的输入，不是已经取得的算法收益。本轮的数据发现工作在索引、来源、样本合同与证据边界可复核后结束；真实独立头戴检验和固定几何采集仍有各自价值，但不再把新采集当作继续数据研究的前提。

## 复现与验证

源码：`host/mine_existing_rgb_coverage_dev_20261013.py`与`host/vista3d_source_pilot_dev_20261013.py`，均可用项目已有Python运行；后者需要NumPy/Pillow，前者仅标准库。输出统一位于本页计划路径。

本地索引检查覆盖6746条RGB路径/长度，1677条ready的RGB/ToF/最近帧对应，1344个cache存在且无额外cache；集合守恒与5407行manifest回读通过，54张实际图像SHA抽验通过，没有重算全量图片SHA。公开源完成上述12帧合同与图像检查。没有Android/全仓测试需求。

脚本运行：本地挖掘两次合计8.759秒；公开pilot25.637秒，补pose合同检查复跑6.900秒并保存首版摘要，重复使用已下载文件。均为command内wall而非GPU计时；其他库存查询/文档检查计入600秒上限。GPU/训练/付费为0，公开payload仅7.36MB，未启动硬件或持有后台计算资源。最终总payload与交付核验见`delivery_receipt.json`。
