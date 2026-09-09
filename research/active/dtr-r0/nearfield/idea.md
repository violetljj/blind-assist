# RGB + 多区 ToF：文献核对后的算法想法

2026-09-10。类型：文献与代码分析，候选机制尚未验证。
本次没有新训练、采集或实验结果；[当前路线](../CURRENT.md) 中 MZ0 的组件角色、
MZ1 的 challenger 角色以及 MZ3/MZ4 的已完成状态保持原有含义。
以下建议来自原始论文、作者项目/代码说明、ST 文档和本地实现的交叉核对。

## 判断的变化

最值得优先检验的是：**在局部空间信息被平均之前，用 ToF 的区域测量条件化 RGB
特征，再按身体和头部的实际查询几何读出。** 将“增加一个证据门控”降到次要位置。
当前结果暴露了融合读出的错误，但没有证明其唯一成因，更没有排除空间融合方法。

三类不确定性需要分开表达：测距数值的噪声、返回值在区域内的表面归属、跨帧对应与
位姿误差。它们不能合并成一个 confidence，也不能都由传感器 sigma 表示。

例如，一个区域里有近处细横杆和远处墙壁。返回墙壁距离可能是一次正确测量，仍然
没有回答横杆是否挡头。RGB 可以帮助定位横杆的方向；其绝对距离仍可能不确定。
合适的中间表示应保留这些可能解释，让后续观测消除歧义。

## 现有实现实际检验了什么

- [mz1_features.py](mz1_features.py) 缓存的是 12×64 个查询特征，加 4 个 JOINT
  概率；ToF 是固定顺序的 64×2 距离及有效位。
- [body_query_context_evidence.py](body_query_context_evidence.py) 将 144×256 RGB
  编码到 18×32 特征场；每个查询采样至多 27 个投影位置，然后对有效位置平均。
  查询顺序仍然存在，查询内部各位置的区别已被汇总。ToF 展平也保留 zone 索引，
  所以不能说 MZ1 完全没有空间信息；它缺少显式承载局部对应的特征结构。
- [mz1_tiny_fusion.py](mz1_tiny_fusion.py) 用 1028→128→4 MLP 读取这些缓存，
  RGB 编码器冻结。因此“联合训练压制了弱模态编码器”的论文解释不能直接套用。
- [MZ3](MZ3_ERROR_ATTRIBUTION_RESULTS_20260910.md) 中，两个单模态读出都正确的
  1192 行，融合只保留 1180 行；仅 RGB 正确时保留 106/118，仅 ToF 正确时保留
  60/135。这支持检验共享读出偏置，但没有确定内部因果。
- [MZ2](MZ2_RESULTS_20260910.md) 的 4×4 与 8×8 只差一行，是当前数据、表示和
  训练预算下的结果，不是 8×8 信息价值的上限，也不决定硬件最优分辨率。
- [MZ4](MZ4_MOTION_RESULTS_20260910.md) 中 stationary / moving-framewise /
  exact-pose joint 分别为 224/481/495 个完整解出案例。主要收益来自新视角，
  联合约束额外增加 14 个。固定 0.5° yaw 偏差导致 338 个空集，混合了位姿误差、
  整数位置场景词典与硬兼容判据的影响，不是实测硬件容差。

上述数值均来自已消费的受控 Development，不是本次新测量。

## 对设计有直接影响的原始文献

| 文献与实际接口 | 值得借的机制 | 对我们的适用边界 |
| --- | --- | --- |
| [DELTAR，ECCV 2022](https://arxiv.org/abs/2209.13362)，§4、表2；RGB + 8×8 区域分布 | 保留 RGB 特征图，用已标定 patch 与区域分布的对应限制交互 | 取消局部对应后 RMSE 从 0.436 到 0.512；直接 feature concat 为 0.454。支持检验对应表示，不能据此断言 attention 必需。论文的分布均值/方差接口不能直接等同于 CX 的 distance/sigma。 |
| [CFPNet，3DV 2025](https://arxiv.org/abs/2411.04480)，引言与作者实现 | 显式向 ToF FoV 外传播特征，区分有测距覆盖和仅由图像补全的区域 | 主要解决覆盖范围不同，不能证明 FoV 外预测拥有直接测距证据。[作者代码](https://github.com/denyingmxd/CFPNet) 已核对 README，未运行。 |
| [SelfToF，IROS 2025](https://arxiv.org/abs/2506.13444)，§III、表I/IV/V | RGB 视频重投影 + 区域深度一致性，训练时联合估计位姿；显式处理无效区域的特征传播 | 推断只需当前 RGB + ToF。报告实验的 ToF 从 RGB-D 数据模拟；区域一致性对齐的是均值/标准差，不是 CX strongest-return 的真实输出律。强一致性损失会把细节压向粗输入。[作者代码](https://github.com/denyingmxd/selftof) 可作实现参考，未运行。 |
| [Prompt Depth Anything，CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/html/Lin_Prompting_Depth_Anything_for_4K_Resolution_Accurate_Metric_Depth_Estimation_CVPR_2025_paper.html)，§3.2–3.3 | 在已有深度 decoder 多尺度注入零初始化的轻量残差；边缘辅助监督 | 可借“保留已有视觉能力、局部加入测距条件”的方式。其 ARKit 输入是 192×256 深度，且已由手机 LiDAR 与 RGB 处理生成，不能当作原始 64-zone 输入的结果。作者实验中通用 cross-attention 还不及简单注入，说明结构应取决于接口与对应关系。 |
| [DEPTHOR++，2025 预印本](https://arxiv.org/abs/2509.26498)，§III、表XI–XII | 模拟缺失、错位和错误回波；融合冻结的单目先验；移除强制回填深度点 | 值得借训练数据的失配建模。其异常检测按单目与 ToF 的深度排序/局部相似度删点，会受单目先验错误影响，不宜直接赋予它近障否决权。论文也报告无噪声时异常检测会损失精度。 |
| [LiteSense，CVPR 2026](https://openaccess.thecvf.com/content/CVPR2026/html/Li_LiteSense_Lifting_Lightweight_ToF_with_RGB_for_High-Resolution_Metric_Depth_CVPR_2026_paper.html)，§3–4 | 轻量 RGB-D encoder、局部 CNH 条件注入；对比全局与局部融合 | 真实装置用 VL53L8CH，输入含 18-bin CNH。它支持局部融合的思路，但完整方法需要比 CX 标准测距结果更多的信息。其报告的边缘 NPU 延时也不能直接当作本项目设备表现。 |
| [DVSR/HVSR，CVPR 2023](https://openaccess.thecvf.com/content/CVPR2023/papers/Sun_Consistent_Direct_Time-of-Flight_Video_Depth_Super-Resolution_CVPR_2023_paper.pdf)，§3–5、表1–2 | RGB 光流加可变形特征对应，因果的 forward-only 版本可利用过去帧 | DVSR 使用低分辨率峰值深度，不要求直方图；HVSR 才要求直方图。GT 光流用于评估而非推断输入。论文的 dToF 评测是合成观测，不能视为真实 CX 验证。 |
| [Light-weight ToF SLAM，ICCV 2023](https://arxiv.org/abs/2308.14383)，§3、表3/4 | 在原始传感器域与像素域共同约束场景和位姿 | 借“预测应能解释观测”的思想。其 zone rendering 实际采用粗网格与中心射线近似，且依赖 DELTAR 稠密深度辅助；不是精确 photon forward model，也不是可直接部署的避障方案。 |
| [Depth on Demand，ECCV 2024](https://arxiv.org/abs/2409.08277)，§3、§4.5 | 利用不同采样率的 RGB 和深度，联合当前与历史视角 | 几何编码需要相对位姿，输入稀疏深度点具有像素归属；多区混合回波需要重新建模。作者明确报告动态物体会损害准确率。 |

## 时序与多模态优化的补充参考

[Ruget 等，Optics Express 2025](https://doi.org/10.1364/OE.550516)
确实展示了 ST VL53L8 8×8 + RGB 的运动人物重建。
[作者正文](https://arxiv.org/html/2412.09427v1) 使用 144-bin 光子直方图、四次 4×4
子采样、六帧联合处理，五次迭代各约 23 秒。它是运动辅助的直接参考，但硬件输出与
处理时序都不等于 CX 标准结果流。不能把这项离线演示称为我们的实时能力。

[Real-Time Non-Rigid Multi-Frame Depth Video Super-Resolution，CVPRW 2015](https://orbilu.uni.lu/bitstream/10993/21434/1/CVPR2015_MSF_Camera%20Ready%20%282%29.pdf)
将二维光流和径向深度变化分开，通过深度/速度及协方差递推融合。
可借因果状态更新这一简单起点；原设备是 120×160 ToF，不能直接套到 8×8 区域回波。

[Gradient-Blending，CVPR 2020](https://openaccess.thecvf.com/content_CVPR_2020/papers/Wang_What_Makes_Training_Multi-Modal_Classification_Networks_Hard_CVPR_2020_paper.pdf)、
[Uni-Modal Teachers，2021 预印本](https://arxiv.org/abs/2106.11059)
说明多模态优化可能丢失单模态能力，但它们训练模态编码器，和 MZ1 冻结编码器的情况不同。
[Beyond Modality Fusion，2026 预印本](https://arxiv.org/html/2607.05019v2)
使独立单模态 logit ensemble 成为值得补上的简单比较；其分类结论不证明几何交互没有价值。
[Calibrating Multimodal Learning，ICML 2023](https://proceedings.mlr.press/v202/ma23i/ma23i.pdf)
可启发模态缺失/腐蚀的置信诊断，但“移除模态就应降低置信”是建模约束，冲突证据下并非物理定律。

## 硬件接口对算法的实质约束

[ST UM3109 Rev12](https://www.st.com/resource/en/user_manual/um3109-a-guide-for-using-the-vl53l8cx-lowpower-highperformance-timeofflight-multizone-ranging-sensor-stmicroelectronics.pdf)
§4.8–4.10、§5.1 给出这些容易混淆的事实：

- range sigma 是**报告距离的噪声估计**；区内多个表面的深度分布宽度是另一个量。
  DELTAR/SelfToF 的高斯分布监督应重新核实与实物字段的对应。
- CX 默认每区最多输出一个目标，默认 strongest；可配置最多四个，手册注明双目标
  检测的最小距离分离为 600 mm。因此两个几何表面存在，不等于两个目标一定被报告。
- sharpener 会影响近目标在邻区的扩展；背景反射、环境光和弱回波也影响报告内容。
  当前独立、理想的 zone 读出不能直接代表实物响应。
- 记录目标数、status、signal、ambient、sigma、配置和时间戳，保留它们各自含义。
  缺测不是空旷，返回远处目标也不能清空该区域内所有近处射线。

[ST 工程师关于 distance matrix 的说明](https://community.st.com/imaging-sensors-49/vl53l8cx-distance-matrix-138858)
明确指出 CX 结果经过 radial-to-perpendicular 转换。工程接入需按实际固件核验并
声明轴向/径向语义；本项目 generic radial packet 不能直接改名为 CX 真机输出。

若器件还可选择，优先确认能否保留更丰富的返回信息。
[VL53L8CH 官方接口](https://www.st.com/en/imaging-and-photonics-solutions/vl53l8ch.html)
提供 CNH；这让直方图路线成为可选项，但目前没有本项目条件下的成本、传输预算或
薄障碍收益比较，尚不构成采购结论。CX 路线仍可采用 range/status 等标准输出。

## 建议的候选结构

1. **视觉局部表示。** 先保留现有 backbone 的二维特征或查询内逐点特征；复用视觉
   表达能力。若后续证据显示 144×256 输入本身已抹去细杆，再单独比较输入分辨率。
2. **区域测量表示。** 每个 zone/target 保留距离、质量字段及角域 footprint。
   用外参、距离候选和查询几何建立可能对应；实际双传感器基线下投影范围随深度变化。
   一个有效测距表示某个表面可能产生该回波，不代表整个区域均处于该深度。
3. **局部条件融合。** 在这些可能对应内做小规模交互。简单残差注入和受几何约束的
   局部交互都可成为候选；暂不把 Transformer 或硬置信门控指定为答案。
4. **任务读出。** 从融合后的局部几何表示回答 BODY/HEAD 查询。局部深度或边界可作
   辅助输出/训练监督，其价值由近 HEAD 漏报、错误高度/距离归属与覆盖决定。
5. **短时更新。** 以 RGB 对应估计和 ToF 距离变化共同更新少量历史候选。显式表达
   对应误差、共同标定偏差与动态遮挡；只使用当前及过去信息，不假定存在额外 IMU。

这套表示可理解为：图像提供候选表面布局，ToF 提供区域测量似然，身体查询决定需要
消除哪种歧义。若引入观测一致性训练，应匹配实际返回律与噪声，不能直接对每区所有
像素强加同一个测距，也不能把模型预测出的历史内容再次算成独立传感器证据。
稠密重建不是必须输出，但相关论文中的空间 decoder、自监督与辅助几何值得复用。

## 最小、能改变判断的推进次序

| 比较 | 保持什么 | 回答什么；结果如何影响推进 |
| --- | --- | --- |
| 固定 0.5/0.5 的 RGB_ONLY、TOF_ONLY logit 平均 | 直接使用已有单模态输出，不选权重或阈值 | 检验联合 MLP 是否值得其复杂度。凸平均必然保留双方均正确的 1192 行，但总正确数及其它错误尚未计算。部署时两个读出的实际成本需计入，不能声称天然匹配参数。 |
| 提前平均 vs 保留局部位置的融合 | 第一轮保持 RGB 输入、backbone、ToF packet、标签和训练预算；匹配或披露新增参数/计算 | 检验查询内空间信息是否真的有用。围绕相同几何改变外观、同一物体改变 BODY/HEAD 占据设置控制，关注近 HEAD 及负例。若只有参数变多而无机制收益，不继续堆叠。 |
| 当前帧 vs moving-framewise vs 柔性对应的因果融合 | 相同历史长度、采样和实际可用姿态信息 | 区分新视角收益和融合收益。保留固定对应基线；解释不了的量化/位姿误差不能通过调兼容阈值隐藏。 |

以上是后续候选比较，本次没有执行。旧 EVAL 可用于明确披露的 Development 诊断，
不可重新称作独立确认；后续模型选择仍使用声明的 TRAIN/DEV。

实物到位后的首批数据应回答观测问题：细横杆/悬挂物与背景同时在 zone 内时，改变
反射属性、距离分离和姿态，究竟报告了什么。用这类记录校正模拟返回律，再考虑
借 SelfToF 的视频自监督扩大数据量。BODY/HEAD 真值与关键实测评价仍须独立于
融合模型和其单目 teacher，尤其保留无可见支撑与无有效回波的案例。

本轮新增的是可检验的结构假设与参考依据；它尚未证明新方法优于当前组件。
