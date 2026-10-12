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

## 原始IMU与因果接口补核

用户最新圆桌复核要求先查原始IMU、明确到E1的链路，并把初始化/校准纳入前缀检查。本轮继续EXPLORE，复用同一个样本，科研CPU墙钟上限180秒，不下载新人体载荷、不训练或运行M3。计划和载荷在`artifacts.local/work/cnh-nymeria-imu-causal-dev-20261008/`。决定检查：本轮验证实现依赖，不设方向精度/检测收益门槛；前缀不同即实现因果检查失败，缺位置则E1整链NOT_EVALUABLE，不能拿姿态通过替代它。

在方向误差评价前冻结：**眼镜朝向为CPF的+Z变换到重力/world坐标后的水平投影**，归一化水平模长小于0.1则UNKNOWN；不等同于人体解剖前向或标定的CNH传感器前向。本轮未计算方向误差。工程探针用初段2秒经gyro去旋转的平均specific force估计up，假设平均平动加速度近似零，未验证这一假设；gyro只用固定厂参，不把运动均值当静止偏置拟合，初始化完成前不输出。输出只证明实现可运行，尚无姿态精度或漂移验证。

实际`motion.vrs`包含两路IMU、baro和mag，没有位置或摄像机流：

| 流 | 全文件记录数 | 标称/时间跨度实测Hz | DEVICE_TIME非正间隔 |
| --- | ---: | ---: | ---: |
| imu-right / 1202-1 | 785144 | 1000 / 1002.022 | 0 |
| imu-left / 1202-2 | 630221 | 800 / 804.305 | 0 |
| baro0 / 247-1 | 39170 | 未查配置 / 49.988 | 0 |
| mag0 / 1203-1 | 7711 | 未查配置 / 9.841 | 0 |

两IMU均声明accel/gyro有效能力、无mag能力；factory JSON各6543字节，配置online_calibration为空。这里的全文件检查仅含流配置和时间戳，未冒称两路全记录数值有效。厂参从配置JSON独立解析，固定偏置/rectification和`T_device_imu`、`T_device_CPF`，不使用MPS online_calibration.jsonl。实际读数上的`M⁻¹(raw−bias)`与SDK accel/gyro校正各一致，工厂JSON SHA256为`4134dc292e9fc083877a761973043724682530a237190e072ecc8a3d19e08f10`。

按跑前固定的右IMU首60秒做[姿态输入探针](cnh_nymeria_imu_causal_dev.py)，原生DEVICE_TIME，不做TIME_CODE插值。60129/60129条记录accel/gyro flag与数值均有效；前2005条初始化UNKNOWN，随后58124条有CPF水平输出。1、1.999、2.002、3、10、30、60秒七处截断，均从原始记录重新执行**读取→厂参解析/校正→初始化→gyro传播→CPF投影/有效掩码**。与完整60秒运行同前缀逐值一致，未来gyro/accel/valid的大幅变更也不改变过去输出。

另以HOST_TIME自身起点截断保存记录前缀，再从raw重跑，并修改该前缀之后的数据，七处同样一致；共七处×五类比较=35项一致。**HOST_TIME/arrival是写入设备记录的时刻，不能当应用接收时刻或手机全链路延迟**；不直接拿HOST_TIME和DEVICE_TIME差值证明延迟。[官方时间戳定义](https://facebookresearch.github.io/projectaria_tools/docs/data_formats/aria_vrs/timestamps_in_aria_vrs) 本轮验证记录级前缀，未做物理VRS文件截断、decoder启动边界或实时传输验证。

实际到E1的链路和缺口如下：

| 环节 | 当前实现/证据 | 状态 |
| --- | --- | --- |
| motion.vrs原始accel/gyro→固定厂参校正 | 实际读取，单位m/s²/rad/s，变换到device坐标 | 首60秒实现已核验 |
| 校正IMU→初始化/姿态→CPF水平前向 | 仅过去2秒初始化及gyro传播，无MPS/骨架输入 | 记录级前缀通过；精度未评价 |
| 原始流→可信位置/因果PDR | 当前未实现；motion.vrs没有位置流 | NOT_IMPLEMENTED |
| 因果位置+CPF朝向→过去1秒位移E1 | 缺少上述位置输入，不能执行完整链路 | NOT_EVALUABLE_NO_CAUSAL_POSITION |

没有把纯IMU双积分或零初始速度假设包装成可信位置，也没有从MPS/未来骨盆/手眼拟合补初始位姿、速度或漂移校正。原始IMU解决了姿态输入来源，**尚未解决现有E1的位置依赖**；姿态前缀通过不等于E1整链通过，更不等于方向补偿或避障收益。

针对新实现运行七类聚焦fixtures：已知yaw与真实非对称厂参/外参、初始化前UNKNOWN、无效记录重启、时间缺口重启、近垂直CPF UNKNOWN、重暖机初始化元数据清空、零specific-force保持UNKNOWN。复核发现后两处边界问题，已修复并重验；最终真实60秒结果使用修复后源码。最终分析30.156秒；全流inventory14.219秒、初探19.891秒、HOST边界补核24.312秒、两轮fixtures0.234/0.265秒，累计已记科研分析89.077/180秒，不含少量shell启动，含启动和最终语法/载荷检查保守<100秒。没有GPU计算，任务进程已结束。

保留初探`causal_probe.json`、中间`arrival_aware/`及最终`final/causal_probe.json`，主结论只用最终版本；`imu_inventory.json`、`fixture_check_final.json`、最终raw/CPF输出和源代码快照/hash在同目录。全部原始样本仍本地保留，未重新分发。

## 三份头动数据的统一口径

| 数据 | 已使用的估计器输入 | 支持的因果程度与证据范围 |
| --- | --- | --- |
| HEADS-UP | 发布的ZED SDK VIO位置/朝向，E1为过去1秒位移 | 给定位姿下的位置索引只读过去，上游完整VIO因果链未核验；真实位姿描述及模拟Development误差回放。[报告](CNH_HEADS_UP_HEADING_DEV_20261007.md)、[估计器](cnh_heads_up_heading.py) |
| BlindWays | Xsens人体模型头部关节位置，60Hz，无头部朝向 | 给定位姿下E1/EMA只读过去，动捕上游未核验；跨参与者位置描述及模拟Development回放。[报告](CNH_BLINDWAYS_HEADING_DEV_20261007.md)、[估计器](cnh_blindways_heading.py) |
| Nymeria | 骨架与closed-loop整体注册用于离线参考；新探针独立读取原始IMU厂参 | 注册使用所选合法连续片段内整体拟合，不能作在线输入；首60秒姿态记录前缀通过，因果位置/E1仍不可评价 |

现有模拟E1的[实现](cnh_query_direction_dev.py)取5Hz位姿最近至多5帧位置差及当前yaw；[noisy_poses](cnh_track_a_readout.py)对模拟真实sensor增量按假设噪声积分，源码已明示不是IMU-only证明。480个新模拟unit保留原确认身份，后续误差注入回放/扰动筛查属于已消费Development；这里的因果限定不修改原实验身份，也不把给定位姿下的过去依赖提升成原始传感器全链在线验证。

## 下一决定

方向扰动配方继续停止；更接近真实的分布或更多损失余量不构成自动重跑条件。下一优先补因果位移接口与实现成本，不能仅凭有IMU扩大行走样本：官方清单另有该序列`recording_head_data_data_vrs`组，声明11076461225字节（约10.316GiB），RGB视频组701352268字节；本轮未下载、未核验实际摄像机字段，清单存在不等于已经具备可复现在线VIO。原始姿态可继续开发，PDR或VIO需完整位置处理及初始化的前缀资格检查，再接过去1秒E1。

接口确认后用metadata挑行走为主的序列，按参与者统计直行/停走/转弯；3–5名是Claude建议，尚非冻结规模或预算。单个Charades最小包不代表数据覆盖。未来骨盆/closed-loop只作带参考误差限制的离线评价参考；恢复M3/CNH设备验证仍先核对真实输入处理和逐query真值，定位原因后再评估优先级，不自动切换主线。实质不同训练机制需另行审查，冻结M3/L2及历史负结果保留。
