# Nymeria 原版样本入场核验

2026-10-08，EXPLORE。目标是判断原版Nymeria能否提供可靠转换的头部/身体运动数据，供后续方向估计开发；不重新运行已停止的扰动续训配方，不评价M3实机效果。

## 范围与获取

用户要求未下载就自行下载。本轮只取一个原版Nymeria v0.0样本，网络载荷上限1.5GiB、下载墙钟1800秒，CPU样本分析180秒，无GPU训练/推理。决定检查：当前样本分母跑前未知，缺少有效时钟/姿态字段记NOT_EVALUABLE；头部锚定后的恒定相对位姿不作为独立对齐通过条件。结果用于决定数据获取，不设检测收益门槛。

官方[原版入口](https://explorer.projectaria.com/nymeria)的公开下载清单与metadata均含1100序列，release v0.0。三组metadata_json、body_motion、recording_head以及timesync/body_motion/head_slam元数据声明完整的候选为1100；声明不是逐序列核验结果。按三组总字节最小选择`20230619_s1_eric_martin_act3_g091qh`，总包998941258字节，场景S13-Charades。选样服务于几何/时钟入场检查，不是行走覆盖抽样。

入口浏览器停在Loading；公开GET接口可以访问，没有使用账号或API key。全清单首次45秒请求只得到部分内容，保留该失败；随后150秒请求完成。SDK使用PyPI官方Windows版projectaria-tools1.7.1，放在本轮ignored runtime目录，不修改共享环境。

用[获取脚本](cnh_nymeria_sample_fetch_dev.py)对官方ZIP按字节区间提取`body/xdata.npz`、头部`motion.vrs`、closed-loop轨迹、online_calibration和summary，省略RGB/眼动/点云/腕部/观察者。**所选包没有open_loop_trajectory.csv**；首次下载在该字段停止，保留失败receipt，修复后复用已校验文件完成其他字段，缺失没有静默忽略。metadata全文件SHA1与官方相符；ZIP选择性获取只验证各entry的CRC32，保存本地SHA256，**没有验证整包SHA1**。完整清单、源代码快照、样本和receipt在`artifacts.local/work/cnh-nymeria-sample-audit-dev-20261008/`，不提交或重新分发原始人体数据。

## 官方契约及因果边界

使用[nymeria_dataset_legacy](https://github.com/facebookresearch/nymeria_dataset/tree/nymeria_dataset_legacy)，不混用NymeriaPlus。骨架23段，Head零基6、Pelvis零基0；segment_tXYZ是米，segment_qWXYZ为WXYZ，均描述段到漂移XSens world。[字段与索引](https://github.com/facebookresearch/nymeria_dataset/blob/nymeria_dataset_legacy/nymeria/xsens_constants.py)

XSens timestamps_us在官方provider中当global TIME_CODE；MPS tracking_timestamp_us是device time。转换来自motion.vrs的同步记录，不能直接将两个CSV时间轴拼接，也不能假设存在time_sync.csv。实际转换用SDK双向映射检查，并保留不支持、范围外和最近邻残差。[录制provider](https://github.com/facebookresearch/nymeria_dataset/blob/nymeria_dataset_legacy/nymeria/recording_data_provider.py)

官方注册为`T_Wd_Wx = T_Wd_Hd @ T_Hd_Hx @ inverse(T_Wx_Hx)`。T_Hd_Hx由全序列手眼拟合求得；默认常量拟合及居中的平滑版本都可用未来信息，closed-loop轨迹也来自离线优化。最近邻匹配还可能选未来帧。注册结果只能作为离线参考/评价标签，不能直接宣称在线因果输入。[注册](https://github.com/facebookresearch/nymeria_dataset/blob/nymeria_dataset_legacy/nymeria/data_provider.py)、[手眼](https://github.com/facebookresearch/nymeria_dataset/blob/nymeria_dataset_legacy/nymeria/handeye.py)

逐帧注册Head等于devicePose×外参，因此头—眼镜相对位姿恒定是构造恒等式，不能独立证明无漂移或同步正确。本轮额外看运动增量残差、分段外参差异、原骨架和注册骨盆连续性；这些仍是内部一致性诊断，缺少外部标定真值。

Aria CPF为X左/Y上/Z前，device是左SLAM相机坐标，不能直接用作CNH的X右约定；人体Head的解剖前向轴尚未建立。要形成头朝向，应结合device到CPF校准及重力水平化，行进“去向”由轨迹评价标签定义。[坐标约定](https://facebookresearch.github.io/projectaria_tools/docs/data_formats/coordinate_convention/3d_coordinate_frame_convention)

审核器不调用官方BodyDataProvider的时间间隔/坏四元数修补，先保留原始缺失；另报告套用其修补会影响多少行。future pelvis仅可进入评价。纯过去open-loop位移proxy只验证prefix截断及未来位置扰动是否改变已有输出；即使实现检查通过，也不证明上游MPS整条链路在线因果或这个proxy有方向估计收益。

## 样本结果

**可继续把合法连续片段作为离线运动参考；整段原始时钟和在线输入链路均不能直接判为通过。** 获取脚本累计551.219/1800秒，实际取回468160269字节（446.47MiB，不含小型清单/SDK），最终状态COMPLETE_AVAILABLE_FIELDS。原始缺失、失败receipt及首次全段不可评价结果保留，没有新推理、模型训练或扩大抽样。

| 核验项 | 实际结果与分母 | 解释 |
| --- | --- | --- |
| 骨架原始行 | 149170帧、23段、240Hz；五个被检字段非有限值0，坏四元数0 | 其他加速度等字段未核验 |
| frame_index | 0..149169连续，非单位步长0 | 未发现索引丢帧；不能将时钟异常直接称为丢帧 |
| 原始时间 | 149169间隔中7个偏离240Hz名义间隔超过1ms，含2个非正间隔 | 不修改原时间，按7处切为8段 |
| 异常定位 | interval8889、8898、8906、68971、69091、69210、69330 | 有−61400us、+24003269700us、−24003195800us，以及两个3100us、两个5200us间隔 |
| 官方修补的反事实 | 若套用，终点差仅62us、可过其10ms终点条件；最大局部改变量约24003200ms | 实际未套用；终点条件不能证明局部时钟正确 |
| SDK时钟映射 | 全149170帧转换，unsupported0；往返误差−2..0ns；8帧在VRS支持范围外 | SDK边界外仍可给数值，往返小不证明物理同步 |
| MPS闭环 | 807460行，非正间隔0、坏姿态0、graph变化0；149162/149170骨架帧在轨迹内 | 其余8帧保留在分母，非可用匹配 |
| 最近邻 | 范围内残差−0.531..+0.529ms；74591个匹配位于查询未来 | 是离线关联，不能用于在线因果声明 |
| CPF校准 | 实际VRS的device←CPF变换可读，旋转det≈1 | CPF前向在device约为[−.0415,−.6088,.7923]，不能直接把device某一轴当头前向；不是物理标定精度证明 |

按原始帧数选最大合法段`[69331,149170)`，79839帧、332.6532秒；选段不看拟合残差。与VRS/MPS范围相交并各端裁2秒后，注册78878帧`[69812,148690)`。其余70292帧保留，但本轮不做注册；其他7段逐段NOT_RUN，未桥接异常点。首次审核因全段非单调使注册NOT_EVALUABLE；后续将审核范围明确为最大连续段，两个结果分别保留，未覆盖旧文件。

手眼stride2为78876个局部运动对，平移正规矩阵rank3、条件数3.966。运动残差P95平移0.821mm、旋转0.093°，最大4.384mm、0.757°。注册骨盆相邻78877对的步长P95 2.330mm、最大5.947mm，非正dt0；head–pelvis距离约0.535–0.562m。三分块外参相对整体分别偏移1.269/2.046/2.036cm、旋转0.453/1.732/1.239°。这些是拟合内部一致性和局部连续性描述，**没有外部真值，也不证明全段无漂移**；小增量残差不等于绝对路径准确。

纯过去proxy在真实样本上为NOT_EVALUABLE：包内缺少open-loop轨迹。prefix截断、未来位置扰动和坐标重置的实现fixtures通过，仅证明该函数的索引依赖；没有把它包装成真实在线链路的因果性验证。未来骨盆路径未进入任何估计器。

审核器[源码](cnh_nymeria_sample_audit_dev.py)的语法、已知手眼/SE3、精确us→ns、SDK失败值、非单调原时间分段、prefix及UID重置fixtures通过；Range ZIP/CRC、服务器忽略Range拒绝、字节预算fixtures通过。真实首次审核4.969秒、分段审核8.063秒、CPF补核0.187秒、索引补核0.015秒；含修复fixtures和一次失败fixtures的本轮CPU分析/核验墙钟保守<25/180秒，不重复旧训练测试或全仓测试。

主要载荷：`sample/download_receipt.json`（首失败）、`sample/download_receipt_attempt2.json`（完成可用字段）、`sample_audit.json`（原全段限制）、`sample_audit_segmented.json`（最终分段）、`device_cpf_check.json`、`frame_index_check.json`与`provenance.json`。源码/输入/输出SHA256与取得方式在provenance中，SDK、原始样本和失败记录保留用于复现，任务计算及下载已结束。

## 下一决定

方向扰动配方继续停止；更接近真实的分布或更多损失余量不构成自动重跑条件。Nymeria有可转换的离线片段，值得保留为后续运动标签来源；获取策略应先确认可因果处理的输入来自何处，并按参与者规划慢走/停走/转弯覆盖抽样，再决定更大下载。单个Charades样本及最小包选择不代表整份数据覆盖。当前不直接开始新训练；实质不同训练机制需另行审查，冻结M3/L2及历史负结果保留。
