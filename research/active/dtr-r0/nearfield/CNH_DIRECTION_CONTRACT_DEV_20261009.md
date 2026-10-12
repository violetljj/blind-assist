# 方向合同与离线身体中心走廊参考（Development）

2026-10-09，用户“继续”。**已把方向合同接成可运行的离线参考接口，真实因果位移输入仍未实现。** 全部保留原始Nymeria样本149170行、8合法时间段、627个原生帧anchor×0.5/1/1.5秒，共1881窗口、447252个未来路径点；不修补时钟、不重注册、不跨gap或外推尾端。未来路径只供评价，估计侧仅从给定离线位置的过去1秒取方向，不宣称上游动捕是在线输入。

该单样本是按获取字节成本选出的Charades片段，不是步行覆盖样本。627个anchor中，过去1秒不完整8个、骨盆位移低于2cm的362个、可定义过去方向257个。新接口避免强行为低运动赋予方向；数据可以验证字段/几何/依赖，不能据此确定典型步行方向误差、实物δ或物理预测T。保留M3、原5格/L2、旧中心/OR/weighted/hybrid读出及原身体接触标签；没有新模型推理、投影、训练、硬件、下载或保护480访问。

下一最有价值的是把合格的因果位置来源接到该接口，并取得任务相符的离线身体路径参考。已有姿态前缀通过不能填补位置缺口；继续调人工方向聚合不是本轮执行项。不同机制仍开放，不建立新的零损失门槛。

## 本轮范围与固定选择

CPU命令阶段上限180秒，含启动/I/O/失败；GPU0。缓存切片、离线路径几何、聚焦检查及文档交付各有记录，源编辑与交互等待另计。T={0.5,1,1.5}秒是描述性grid，不选择赢家，也不把f3–13及时窗当预测时域。圆盘半径0.30m沿用几何proxy量级，不代表实测身体宽度；方向位移下限0.02m仅数值有效性规则，不是步行判据。全部窗口和不可用/低运动状态保存。

源为缓存原始XSens `segment_tXYZ.reshape(N,23,3)[:,0,:]` 的Pelvis，`timestamps_us`为TIME_CODE，乘1000成整数ns。官方缓存 [XSens字段约定](../../../../artifacts.local/work/cnh-direction-contract-dev-20261009/../cnh-nymeria-sample-audit-dev-20261008/sources/xsens_constants.py) 与 [provider](../../../../artifacts.local/work/cnh-direction-contract-dev-20261009/../cnh-nymeria-sample-audit-dev-20261008/sources/data_provider.py) 注明XSens world/Z上/米；水平取XY，不推断人体前向，也不直接套用CNH的XZ/Y上符号。原始xdata SHA256 `8c2298f0041eb820b407e46c52e3d7be95d0f16966f62b5198a8b92b95175e95`；source/PLAN/官方缓存身份见 [绑定](../../../../artifacts.local/work/cnh-direction-contract-dev-20261009/reference_inputs.json)。本轮没有已注册骨盆NPZ，旧审计仅保存注册统计；不会把raw位置写成Aria/device world。

7个异常时间边（包括2个非正间隔）按原行序切8段，规则为与240Hz周期偏差>1ms或非单调。每段从段首开始，每1秒目标选首个不早于目标的真实原生帧，以该帧时间/位置为anchor。过去端点插值的两个样本都不晚于anchor；未来端点仅在同段观测内插值，缺失不外推。完整3D/XY未来折线、时间和ragged offsets均保存；不跨段拼接时间或轨迹。

## 可调用接口

[源码](cnh_direction_corridor_reference_dev.py)提供`corridor_reference(timestamps_ns, pelvis_xyz_m, horizontal_axes=(0,1))`，返回逐窗`columns`、合法段/非法边、`future_path_offsets`及完整未来XY/XYZ/时间数组。数值缺失用NaN并配独立状态，CSV空值不当零。当前分段采样合同固定240Hz；接30Hz HEADS-UP、5Hz模拟或其他位置流时须显式换采样合同，不能原样调用后把大多数间隔判成异常。

原始包到参考包的本轮驱动保存为 [build_reference.py](../../../../artifacts.local/work/cnh-direction-contract-dev-20261009/build_reference.py)，源码与输入绑定已保留。驱动拒绝覆盖既有CSV，复算应使用新载荷目录并沿用同一PLAN定义；这不是新增实物方向/预测功能。

## 从端点弦到完整路径

未来走廊proxy用完整未来XY中心折线与半径0.30m圆盘的Minkowski和（各线段capsule并集）表示；输出是折线+半径，不虚构实际身体边界、头部高度或碰撞标签。端点弦只作为简化方向参照。下面偏离值为所有未来中心点到端点**线段**的最大距离，不是身体区域遗漏比例。

| T秒 | 全部窗口 | 未来几何可用 | 未来尾截/缺段 | 过去/未来方向均可用 | 中心偏离p95米 | 最大米 | 中心超过0.30m |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0.5 | 627 | 620 | 7 | 195 | 0.0257 | 0.0851 | 0/620 |
| 1.0 | 627 | 619 | 8 | 213 | 0.0834 | 0.1576 | 0/619 |
| 1.5 | 627 | 617 | 10 | 217 | 0.1215 | 0.4267 | 2/617 |

1.5秒下2/617个可用窗口的中心离开直线±30cm代理带，说明端点方向不能完整替代曲线路径；其他窗口中心未超带，不等于完整身体扫掠区域与直线带相同。窗口重叠且来自一个相关片段，不是独立事件世界。直行/L转弯fixture同时验证路径长度、弦长和偏离可不同，不依赖真实样本有大量转弯。

## 位移方向代理的描述性结果

这里比较“过去1秒骨盆位移方向”与“未来T秒骨盆端点弦方向”；不是原E1的头部位置/姿态输入，也不是传感器yaw或head-to-travel误差。角度按XSens XY的`atan2(dy,dx)`，字段`wrapped_error`为future−past绕回±π；如对外合同使用estimate−reference须显式取反，不能直接当作旧Ry query符号。较大角差可以来自行为变化、停走和参考定义，不能归因IMU漂移或M3。

| T秒 | 有效方向对n | RMS° | 绝对中位° | 绝对p95° | 过去位移≥0.30m子集n | 子集RMS° | 子集绝对中位° |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0.5 | 195 | 72.35 | 32.68 | 159.60 | 91 | 36.11 | 19.12 |
| 1.0 | 213 | 79.29 | 40.47 | 162.49 | 92 | 48.18 | 24.10 |
| 1.5 | 217 | 86.08 | 48.14 | 171.98 | 90 | 59.82 | 28.53 |

≥0.30m是过去1秒骨盆**端点位移量**的描述子集，不是步行标注或剔除规则，全部母表保留。低运动、过去窗口缺失、未来窗口缺失和未来弦低运动分别列状态；总status采用优先级显示，不能用它替代各边际缺失计数。[全状态/逐段统计](../../../../artifacts.local/work/cnh-direction-contract-dev-20261009/reference_summary.json) · [1881窗口](../../../../artifacts.local/work/cnh-direction-contract-dev-20261009/reference_windows.csv) · [原始位置与完整未来路径包](../../../../artifacts.local/work/cnh-direction-contract-dev-20261009/reference_pack.npz)。原始人体载荷本地留存，不随Git推送再分发。

## 可实现的四项输入合同

本段为可实现接口建议，未宣称设备合同已通过。原M3、原身体接触标签、480身份与旧停止结果保留；不因目标落在错位query外改负标签，不要求新增硬件采集。

### 1. 输入与输出身份

- 每条观测建议字段：`sequence_id, sample_id, sensor_id, measurement_time, available_time, measurement_clock, available_clock, source_kind, provenance, validity, unavailable_reason`。未知字段值为`null`，状态为`NOT_AVAILABLE`，不填零、当前墙钟或拟合值冒充。
- 原始位置、姿态分别记录`position_xyz_m, orientation_quaternion, quaternion_order, pose_frame`；位置不可用不妨碍姿态接口独立输出，但完整位移方向须标`NOT_AVAILABLE_NO_POSITION`。
- `past_only`估计器输出建议为`estimate_time, latest_input_measurement_time, latest_input_available_time, history_start/end, heading_vector, heading_frame, heading_uncertainty, estimator_id/version, status`。不确定性未建立时记`NOT_AVAILABLE`，人工±3°另标`declared_simulation_assumption`。
- 已知：模拟E1取过去≤5帧x-z位置差及当前sensor yaw；位置来自对真实模拟运动增量加噪积分，非IMU-only。真实Nymeria探针有姿态记录前缀证据，因果位置仍`NOT_IMPLEMENTED`，不能把head姿态等同身体行进方向。

### 2. 时钟与可用时刻

- `measurement_time`描述测量发生时刻，`available_time`描述该输入对指定消费者可用的时刻；两者须有单位与clock domain。只有已知时钟映射时才能比较，不直接拼接XSens TIME_CODE与MPS DEVICE_TIME。
- 建议映射字段：`clock_mapping_id, source_clock, destination_clock, mapping_valid_interval, mapping_method, mapping_residual, mapping_availability`；映射缺失则跨域同步状态`NOT_AVAILABLE`。
- 以决策时刻`decision_time`筛输入时，同时满足测量不晚于决策、且输入已对该消费者可用；不同域先按明确映射转换。保存的前缀测试只证明被测试的读记录／估计实现，不替代上游在线生成资格。
- VRS `capture_timestamp_ns`／DEVICE_TIME与记录的`arrival_timestamp_ns`／HOST_TIME应原样保留。HOST_TIME arrival不是应用收包时间；未有应用接收字段时`app_received_time=NOT_AVAILABLE`，不据两时钟数字差宣称app或端到端延迟。

### 3. 坐标、外参与query服务目标

- 每条pose建议字段：`pose_frame, parent_frame, axis_convention, length_unit, quaternion_order, gravity_reference`；每个外参使用明确`T_destination_source, calibration_id, calibration_source, validity_interval, uncertainty, availability`，并写明齐次矩阵乘法方向。
- 本轮离线参考来源为原始XSens pelvis位置，`source_kind=raw_xsens_offline_reference`，`pose_frame=raw_xsens_world`，米、Z上、水平XY；保留其漂移world身份。它不是注册后的Aria/device world，也不是在线位置估计。不能直接套用CNH水平XZ／Y上的公式，轴变换需单列。
- Aria CPF X左/Y上/Z前与CNH X右约定不同；人体Head解剖前向尚未建立。CPF水平姿态输出也不自动给出行进方向。物理sensor→body外参与误差未认证时记`NOT_AVAILABLE`，不能由注册头部相对位姿恒定证明正确。
- query建议记录`query_time, sensor_to_query_transform, transform_convention, query_frame, service_goal, estimated_heading_source, uncertainty_contract_id`。`service_goal=body_corridor`与literal query方向分别保存；历史传输`Q_t·inverse(S_t)·S_i`及人工yaw左乘是现有模拟约定，不能冒充物理外参／时钟验证。

### 4. 过去估计器与未来评价参考分离

- `estimator_inputs`只提供在决策时刻可用的原始观测、合格过去pose及已可用固定校准。未来骨盆、中心/未来轨迹chord、全段手眼拟合、closed-loop注册轨迹及评价标签不得进入这个输入包。
- `evaluation_reference`独立保存`reference_time, future_horizon_s, bodypath_timestamps, bodypath_xyz_m, reference_frame, reference_source, clock_mapping_id, reference_validity, reference_error_limits`；明确`EVAL_ONLY`。原始XSens pelvis未来路径是离线路径proxy，未认证完整身体占据走廊。
- 本轮`future_horizon_s∈{0.5,1.0,1.5}`是描述性grid，非已选择的物理预测T。pelvis中心路径与身体宽度／高度／包络、走廊边界、转弯/停走有效性须分字段；未定义的几何或误差标`NOT_AVAILABLE`，不由轨迹向量补出身体真值。
- 现有缺口：真实因果位置与其可用时刻、head/CPF到身体服务方向的映射、物理外参与同步误差、未来身体包络参考均未完整建立。仍可开发离线proxy与可逆接口；不把这些缺口变成采集要求或对候选的零损失门槛。

依据：`cnh_query_direction_dev.py:25–38`、`cnh_track_a_readout.py:149–162`、`cnh_counterfactual_data_dev.py:334–337`；`CNH_NYMERIA_SAMPLE_AUDIT_DEV_20261008.md:17–27,72–85`、`cnh_nymeria_imu_causal_dev.py:190–196`。本轮原始XSens pelvis参考和T grid遵循主端冻结计划；注册仅作历史离线证据，不接在线估计器。


## 已有角误差实验的口径核对

- [旧方向不确定性](CNH_HEADING_UNCERTAINTY_DEV_20261007.md)：每序列bias＋逐帧独立抖动，`e=σb+(σ/2)w`；σ2/4/8°实际RMS2.2705/4.5411/9.0822°，全部5760序列f3–15统计。旧“约2°RMS”是该模拟量级，不是设备门槛；旧并集各方向先平滑再max，与后续raw先max再平滑不同。
- [旧扰动续训](CNH_QUERY_PERTURB_TRAIN_DEV_20261008.md)：每unit/config固定delta，全部帧/两epoch复用，25%zero否则截断Normal(0,10°)。不是逐帧抖动。E1是带噪head yaw减过去≤1秒位移方向，无RMS输入；配方停止仍按原run保留。
- [方向聚合](CNH_DIRECTION_AGGREGATION_DEV_20261009.md)与[锚定OR](CNH_DIRECTION_ANCHOR_DEV_20261009.md)：只测ideal和恒定+3°中心偏差；±3°是声明网格。只有参考方向/符号/误差分量/统计窗一致时，恒偏3°才可标RMS3°；不能据此直接换算旧配方损失，更不能与本轮past-pelvis/future-chord RMS混用。

## pass阈值尾部的方向核对

18个max阈值记录全部由pass clip前驱阻断；每记录1个最大query/frame，全部为pass query，HEAD11/BODY7，18峰值与90贡献帧、raw argmax ties0。17/18峰值查询轴偏向目标侧，时序带符号logit贡献主导也17/18；去重17个scene/K（15个scene）中16/17同向。13/18窗口全程朝目标，5/18为混合方向，峰帧argmax不能替代整个last5。

实际`Ry(δ)`左乘world→query，原坐标中的query轴`x=−z·tanδ`，所以+3°偏negative_x、−3°偏positive_x。唯一反例weak956/40的BODY f14：前4帧朝目标、末帧背离，时间权重15/31与16/31，朝/背logit贡献2.129877/2.731600。成本窗f3–15包含晚帧，不能只读及时f3–13。

这些仅支持几何冲突线索，不证明具体回波来源、目标被“看成接触”或损失不可避免。[18条完整记录](../../../../artifacts.local/work/cnh-direction-contract-dev-20261009/pass_blocking_directions.csv) · [来源/汇总](../../../../artifacts.local/work/cnh-direction-contract-dev-20261009/cached_slice_summary.json)。

| arm | seed后3位 | 点% | scene/K | 目标侧 | 高度 | 峰帧 | 实际5帧max方向 | 朝目标权重 | 峰方向/贡献主导朝目标 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| control | 955 | 20 | 279/1 | positive_x | BODY | 12 | -3/-3/-3/-3/-3 | 100.00% | True/True |
| control | 955 | 30 | 14/1 | negative_x | HEAD | 13 | 3/3/3/3/3 | 100.00% | True/True |
| control | 955 | 40 | 327/0 | positive_x | HEAD | 14 | -3/-3/-3/-3/-3 | 100.00% | True/True |
| control | 956 | 20 | 13/1 | negative_x | HEAD | 13 | 3/3/3/3/3 | 100.00% | True/True |
| control | 956 | 30 | 276/0 | positive_x | BODY | 12 | -3/-3/-3/-3/-3 | 100.00% | True/True |
| control | 956 | 40 | 255/0 | negative_x | BODY | 15 | 0/3/3/3/3 | 96.77% | True/True |
| control | 957 | 20 | 15/2 | negative_x | HEAD | 13 | 3/3/3/3/3 | 100.00% | True/True |
| control | 957 | 30 | 13/0 | negative_x | HEAD | 13 | 3/3/3/3/3 | 100.00% | True/True |
| control | 957 | 40 | 108/2 | negative_x | HEAD | 13 | 3/3/3/3/3 | 100.00% | True/True |
| weak_pass | 955 | 20 | 277/1 | positive_x | BODY | 10 | -3/-3/-3/-3/-3 | 100.00% | True/True |
| weak_pass | 955 | 30 | 325/2 | positive_x | HEAD | 14 | 3/0/0/-3/-3 | 77.42% | True/True |
| weak_pass | 955 | 40 | 230/2 | positive_x | HEAD | 13 | -3/-3/-3/-3/-3 | 100.00% | True/True |
| weak_pass | 956 | 20 | 205/2 | negative_x | HEAD | 13 | 3/3/3/3/3 | 100.00% | True/True |
| weak_pass | 956 | 30 | 13/0 | negative_x | HEAD | 13 | 3/3/3/3/3 | 100.00% | True/True |
| weak_pass | 956 | 40 | 183/1 | positive_x | BODY | 14 | -3/-3/-3/-3/3 | 48.39% | False/False |
| weak_pass | 957 | 20 | 279/3 | positive_x | BODY | 10 | -3/-3/-3/-3/-3 | 100.00% | True/True |
| weak_pass | 957 | 30 | 324/3 | positive_x | HEAD | 14 | 3/0/-3/-3/-3 | 90.32% | True/True |
| weak_pass | 957 | 40 | 87/3 | positive_x | BODY | 10 | -3/3/-3/-3/-3 | 93.55% | True/True |

## OR新增clear背景分层

已保存的36个center/OR格×2背景共72行，仅使用冻结结果。cal为segmented_backwall/side_gate_posts，validation为L_sidewall/stacked_side_shelves；名称不同是事实，切片不能区分一次背景迁移和一般阈值脆弱。ideal全格新增clear clip0；+3°主要集中L_sidewall。40% weak seed955/956/957分别新增L形侧墙3/13/9、货架4/0/1，各背景/256clip。下面保留全部点和两模型。

OR结构保证中心报警保留，因此按背景的OR clip减center clip就是新增clip数；72行加总逐一对齐36个总体clear增量。新增clip0也可能在已报警clip中增加slot；clear clip/slot和pass clip分别列，不折算。跨seed/点重复出现不能算独立新世界。

| arm | seed后3位 | 点% | 分支 | 背景 | center→OR clear/256 | 新增clear slot/3328 | 新增pass/128 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| control | 955 | 20 | ideal | L_sidewall | 0→0 | 0 | 0 |
| control | 955 | 20 | ideal | stacked_side_shelves | 0→0 | 0 | 0 |
| control | 955 | 30 | ideal | L_sidewall | 0→0 | 0 | 0 |
| control | 955 | 30 | ideal | stacked_side_shelves | 0→0 | 0 | 2 |
| control | 955 | 40 | ideal | L_sidewall | 1→1 | 0 | 0 |
| control | 955 | 40 | ideal | stacked_side_shelves | 0→0 | 0 | 0 |
| control | 956 | 20 | ideal | L_sidewall | 0→0 | 0 | 0 |
| control | 956 | 20 | ideal | stacked_side_shelves | 0→0 | 0 | 1 |
| control | 956 | 30 | ideal | L_sidewall | 0→0 | 0 | 0 |
| control | 956 | 30 | ideal | stacked_side_shelves | 0→0 | 0 | 0 |
| control | 956 | 40 | ideal | L_sidewall | 0→0 | 0 | 0 |
| control | 956 | 40 | ideal | stacked_side_shelves | 1→1 | 0 | 1 |
| control | 957 | 20 | ideal | L_sidewall | 0→0 | 0 | 1 |
| control | 957 | 20 | ideal | stacked_side_shelves | 0→0 | 0 | 0 |
| control | 957 | 30 | ideal | L_sidewall | 0→0 | 0 | 3 |
| control | 957 | 30 | ideal | stacked_side_shelves | 0→0 | 0 | 0 |
| control | 957 | 40 | ideal | L_sidewall | 0→0 | 0 | 1 |
| control | 957 | 40 | ideal | stacked_side_shelves | 0→0 | 0 | 0 |
| weak_pass | 955 | 20 | ideal | L_sidewall | 0→0 | 0 | 0 |
| weak_pass | 955 | 20 | ideal | stacked_side_shelves | 0→0 | 0 | 0 |
| weak_pass | 955 | 30 | ideal | L_sidewall | 2→2 | 0 | 1 |
| weak_pass | 955 | 30 | ideal | stacked_side_shelves | 1→1 | 0 | 4 |
| weak_pass | 955 | 40 | ideal | L_sidewall | 4→4 | 0 | 0 |
| weak_pass | 955 | 40 | ideal | stacked_side_shelves | 4→4 | 0 | 4 |
| weak_pass | 956 | 20 | ideal | L_sidewall | 0→0 | 0 | 0 |
| weak_pass | 956 | 20 | ideal | stacked_side_shelves | 0→0 | 0 | 1 |
| weak_pass | 956 | 30 | ideal | L_sidewall | 0→0 | 0 | 1 |
| weak_pass | 956 | 30 | ideal | stacked_side_shelves | 1→1 | 0 | 0 |
| weak_pass | 956 | 40 | ideal | L_sidewall | 3→3 | 0 | 1 |
| weak_pass | 956 | 40 | ideal | stacked_side_shelves | 2→2 | 0 | 0 |
| weak_pass | 957 | 20 | ideal | L_sidewall | 0→0 | 0 | 0 |
| weak_pass | 957 | 20 | ideal | stacked_side_shelves | 0→0 | 0 | 0 |
| weak_pass | 957 | 30 | ideal | L_sidewall | 0→0 | 0 | 0 |
| weak_pass | 957 | 30 | ideal | stacked_side_shelves | 0→0 | 0 | 0 |
| weak_pass | 957 | 40 | ideal | L_sidewall | 1→1 | 0 | 0 |
| weak_pass | 957 | 40 | ideal | stacked_side_shelves | 1→1 | 0 | 2 |
| control | 955 | 20 | yaw_plus3 | L_sidewall | 87→128 | 373 | 15 |
| control | 955 | 20 | yaw_plus3 | stacked_side_shelves | 0→0 | 0 | 0 |
| control | 955 | 30 | yaw_plus3 | L_sidewall | 132→132 | 252 | 0 |
| control | 955 | 30 | yaw_plus3 | stacked_side_shelves | 5→7 | 2 | 0 |
| control | 955 | 40 | yaw_plus3 | L_sidewall | 134→134 | 201 | 0 |
| control | 955 | 40 | yaw_plus3 | stacked_side_shelves | 8→9 | 2 | 0 |
| control | 956 | 20 | yaw_plus3 | L_sidewall | 7→10 | 3 | 0 |
| control | 956 | 20 | yaw_plus3 | stacked_side_shelves | 0→0 | 0 | 0 |
| control | 956 | 30 | yaw_plus3 | L_sidewall | 58→107 | 150 | 24 |
| control | 956 | 30 | yaw_plus3 | stacked_side_shelves | 1→2 | 1 | 0 |
| control | 956 | 40 | yaw_plus3 | L_sidewall | 122→135 | 338 | 2 |
| control | 956 | 40 | yaw_plus3 | stacked_side_shelves | 5→11 | 11 | 0 |
| control | 957 | 20 | yaw_plus3 | L_sidewall | 82→129 | 601 | 27 |
| control | 957 | 20 | yaw_plus3 | stacked_side_shelves | 0→0 | 0 | 0 |
| control | 957 | 30 | yaw_plus3 | L_sidewall | 122→130 | 481 | 6 |
| control | 957 | 30 | yaw_plus3 | stacked_side_shelves | 1→1 | 0 | 0 |
| control | 957 | 40 | yaw_plus3 | L_sidewall | 128→130 | 290 | 1 |
| control | 957 | 40 | yaw_plus3 | stacked_side_shelves | 5→5 | 0 | 0 |
| weak_pass | 955 | 20 | yaw_plus3 | L_sidewall | 90→129 | 259 | 20 |
| weak_pass | 955 | 20 | yaw_plus3 | stacked_side_shelves | 1→1 | 0 | 0 |
| weak_pass | 955 | 30 | yaw_plus3 | L_sidewall | 127→134 | 342 | 7 |
| weak_pass | 955 | 30 | yaw_plus3 | stacked_side_shelves | 7→9 | 2 | 0 |
| weak_pass | 955 | 40 | yaw_plus3 | L_sidewall | 134→137 | 263 | 0 |
| weak_pass | 955 | 40 | yaw_plus3 | stacked_side_shelves | 14→18 | 7 | 0 |
| weak_pass | 956 | 20 | yaw_plus3 | L_sidewall | 65→65 | 0 | 0 |
| weak_pass | 956 | 20 | yaw_plus3 | stacked_side_shelves | 1→1 | 0 | 0 |
| weak_pass | 956 | 30 | yaw_plus3 | L_sidewall | 96→100 | 14 | 3 |
| weak_pass | 956 | 30 | yaw_plus3 | stacked_side_shelves | 4→4 | 0 | 0 |
| weak_pass | 956 | 40 | yaw_plus3 | L_sidewall | 115→128 | 60 | 2 |
| weak_pass | 956 | 40 | yaw_plus3 | stacked_side_shelves | 5→5 | 0 | 0 |
| weak_pass | 957 | 20 | yaw_plus3 | L_sidewall | 34→126 | 264 | 32 |
| weak_pass | 957 | 20 | yaw_plus3 | stacked_side_shelves | 0→0 | 0 | 0 |
| weak_pass | 957 | 30 | yaw_plus3 | L_sidewall | 105→132 | 548 | 15 |
| weak_pass | 957 | 30 | yaw_plus3 | stacked_side_shelves | 4→6 | 3 | 0 |
| weak_pass | 957 | 40 | yaw_plus3 | L_sidewall | 124→133 | 395 | 6 |
| weak_pass | 957 | 40 | yaw_plus3 | stacked_side_shelves | 16→17 | 4 | 0 |

[完整背景CSV](../../../../artifacts.local/work/cnh-direction-contract-dev-20261009/OR_background_costs.csv)。原cal预算、validation阈值与M3结果没有重选或修改。

## 实际核验、预算与未完成项

5项helper fixture通过：直行/native anchor、L转弯完整路径、gap/非单调分段、恰好端点与尾截、低运动状态。真实输出全1881行检查原生anchor身份、段内端点、单调ragged时间、路径长度复算、原8段边界；给定离线位置的past依赖检查对1266行×5字段逐值比较，截断与修改未来位置均不改变过去输出。此检查不认证上游XSens/MPS的在线因果性，也不验证姿态准确度。

源helper SHA `350156efed06907a9193a3c95cf2da65fba251275692c7335b5cfc5cf5ed320c`，真实数据执行后保持；PLAN在输入绑定中冻结。[fixture](../../../../artifacts.local/work/cnh-direction-contract-dev-20261009/focused_reference_check.json) · [past依赖](../../../../artifacts.local/work/cnh-direction-contract-dev-20261009/past_dependency_check.json) · [预算](../../../../artifacts.local/work/cnh-direction-contract-dev-20261009/budget_final.json) · [PLAN](../../../../artifacts.local/work/cnh-direction-contract-dev-20261009/PLAN.json)。CPU命令实测8.334s（盘点4.625、fixture0.704、切片0.665、参考1.575、past依赖0.764）＋根端盘点/文档/Git保守扣额40s，总48.334/180s；40s是扣额而非性能测量，GPU0，内部函数时间不重复累加。科学阶段无失败；根端一次旧RECEIPT路径缺失已改读provenance，计入根端扣额。

未实现：原始传感器→合格因果位置、物理head/body外参及同步精度、完整身体占据真值、任务相符的步行方向分布。未运行：新yaw投影/模型回放、训练、采集、下载、保护480、新δ/权重/阈值选择。原身体truth、M3/5格/L2、480身份及已停止配方均保留。所有命令结束，无常驻任务模型/GPU；小型离线参考、CSV/JSON、源码快照、旧原始样本留作本地复算。

交付检查/commit及远端一致性记于payload delivery_receipt.json；该文档交付不等于物理方向合同已经成立。
