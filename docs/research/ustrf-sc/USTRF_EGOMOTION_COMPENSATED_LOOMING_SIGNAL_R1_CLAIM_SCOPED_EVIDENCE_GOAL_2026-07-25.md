# USTRF 相机运动补偿 Looming R1 声明级证据目标（2026-07-25）

状态：`EXECUTION_OCCURRED / NONAUTHORITATIVE_EVALUATION_QUARANTINED / INPUT_AUTHORITY_BLOCKED`

当前可执行边界：`R1A_ORACLE_PROTOCOL_AND_CLAIM_SCOPED_SOURCE_SUBSET_FREEZE`

最大权限：`CONTINUOUS_GEOMETRIC_SIGNAL_EXISTENCE_ONLY`

执行说明：Bonn discovery 的 base/oracle traces 已实际生成；一次评分也已发生，但
其 truth input 已被审计为 diagnostic-only，因此评分被
[隔离结果](USTRF_EGOMOTION_COMPENSATED_LOOMING_SIGNAL_R1_NONAUTHORITATIVE_EXECUTION_QUARANTINE_RESULT_2026-07-25.md)
永久排除出接受/停止权限。当前没有权威算法结果。

## 一、研究问题

R1 只回答两个连续几何问题：

1. `C1_ROTATION_LEAKAGE_SUPPRESSION`：旋转补偿能否降低纯旋转产生的局部径向扩张泄漏？
2. `C2_TRUE_CLOSING_RETENTION`：旋转补偿能否保留相机接近静止表面或目标主动接近产生的真实扩张？

横向经过是 C2 的关键反事实，报告 bearing rate、shear、closest-pass distance 与
closing intensity，但不定义报警类别。route truth、intended route、event onset、
clearance、lifecycle、token、consume timestamp 和 alert threshold 全部不属于 R1。

## 二、准入从“数据集全通过”改为“声明级”

每个 `source / session / unit / claim` 独立记录：

```text
claim_id
source_family
capture_cluster_id
session_id
unit_id
eligible
evaluated
abstained
abstention_reason
evidence_grade
truth_provenance
interpolated_fraction
time_sync_status
transform_chain_status
```

一个来源只需为明确 claim 提供足够证据；不支持的 claim/unit abstain，不否决整库。
核心 claim 必须至少由两个独立 source/truth family 同方向支持。不同来源不需要各自
覆盖所有反事实。

## 三、冻结来源拼图

| 来源 | 主要用途 | 明确不用来证明 |
| --- | --- | --- |
| 新受控刚性目标采集 | C1 纯旋转、C2 静态表面接近/主动目标接近/横向经过；主要确认 | 自然场景泛化、人体或助盲安全 |
| Bonn RGB-D Dynamic | C1 纯旋转与静态几何；C2 相机接近静态表面；rigid-flow oracle | 独立动态人体真值 |
| REveL | C2 人体主动接近、横向经过与相机自运动；只使用权威可闭合子集 | 同源 bbox/LiDAR/centroid 的独立精度证明 |
| JRDB | 近场 `0–20m` 遮挡、人群、支持率与迁移压力诊断 | confirmatory independent truth、远距泛化 |
| THÖR/MAGNI（可选） | 外部迁移敏感性 | 把恢复/插值轨迹升级为 direct truth |

从本目标冻结日起至 `2026-08-08`（含）不新增数据集。只有已运行的三来源证据明确显示某个核心
claim 完全没有第二个独立 family，才可另立版本化 source amendment；不得在本 R1
内漫游式扩表。

已完成的 HOT3D metadata audit 只保留为未选来源记录；不下载 HOT3D tar/RGB。
ADT、AV2、CODa 不再扩 prescreen。

当前本地 REveL `dynamic` 和 Bonn `moving_obstructing_box` 已在旧 detector/source
研究中被读取，不能冒充新的 validation 或 sealed holdout。它们可作为 identity-bound
engineering/discovery 单元，但旧 outcome、窗口、阈值和结果不得进入 R1 producer。
Bonn 的后续确认必须从未读 sequence metadata-only 冻结；REveL 若没有第二个独立
capture session，则 C2 动态 claim 最多得到 discovery/迁移证据，不能通过切分同一
371.805 秒 bag 伪造 session 复制。

## 四、证据等级

- `A`：直接 mocap/OptiTrack、独立激光或同等级外部测量；
- `B`：独立多相机/几何重建，带同步、外参和不确定度；
- `C`：source-native annotation 或短间隔插值；
- `D`：与 signal 输入共享派生祖先的内部 proxy。

A/B 可进入主要确认；C 只进入自然场景迁移或敏感性分析；D 只调试。短插值不删除
整个 episode：direct 区间可评价，插值区间 abstain 或单列 C 级。signal 和主要
truth 不得共享会直接诱导结果的派生链。

## 五、分声明最低证据

### C1

必需：RGB、内参、可靠旋转真值或高质量 gyro、静态区域。
不需要：人体 ID、route/event truth、LiDAR 人体支持或远距几何。

确认组合：受控采集 + Bonn。每个 family 必须有独立 session，纯 yaw/pitch/roll
均至少有可评价单元；rolling shutter、同步或旋转真值不够的单元 abstain。

### C2

必需：RGB、旋转真值、独立相机到表面或相机/目标相对轨迹；association 只负责
绑定目标，不提供被评价距离。
可选：depth/rigid-flow oracle。
不需要：报警阈值、lifecycle 或 intended route。

静态表面接近由受控采集 + Bonn 支持；主动接近和横向经过由受控采集 + REveL 的
权威可闭合子集支持。若 REveL 的同步、稳定 ID 或 marker-to-surface 外参仍未闭合，
其相关单元 abstain，不能用中心距离或同源 bbox 回救。

## 六、受控采集边界

具体机械矩阵、同步/外参、84 trial、producer 隔离与停止条件以
[受控采集与来源子集协议](USTRF_EGOMOTION_COMPENSATED_LOOMING_SIGNAL_R1_CONTROLLED_CAPTURE_AND_SOURCE_SUBSET_PROTOCOL_2026-07-25.md)
为唯一执行合同。

第一阶段只允许刚性目标、小车/滑轨、固定尺寸表面、标定相机/IMU 与独立外部测距/
定位；不采集盲人、自由行走或人体安全 outcome。录制前必须冻结：

- 设备、时钟、同步 offset/jitter、内外参和 rolling-shutter/readout；
- marker-to-object-surface 外参与不确定度；
- session/capture cluster、C1/C2 cell 和 clean split；
- 文件 identity/hash、许可与停止条件；
- 纯旋转、相机接近静态表面、目标主动接近、横向经过四种机械动作。

## 七、比较与判定

### R1-A：先检验物理上界

第一轮冻结并比较：

- `RAW_FLOW_ENERGY`；
- `BBOX_LOG_AREA_GROWTH`（detector miss 记 abstain）；
- `UNCOMPENSATED_LOCAL_RADIAL_EXPANSION`；
- `ORACLE_ROTATION_COMPENSATION`；
- `FULL_6DOF_RESIDUAL_DIAGNOSTIC`。

完整 6DoF residual 只作机制诊断，不是 collision-cue 性能上界。R1-A 不运行或调节
部署型 rotation estimator，只回答“在真实旋转 truth 可用时，物理信号是否存在”。

### R1-B：再检验部署差距

只有 R1-A 的 oracle 在某 claim 的两个冻结 family 上同方向成立，才另行 hash-bind
causal `ROTATION_COMPENSATED_LOCAL_EXPANSION`。R1-A outcome 不得用于同轮调 flow、
ROI、网格、窗口或质量门；实现变化必须版本化并使用未打开的 validation session。

先比较共同 support 上的连续量，不选阈值。source/session 等权；缺失不填零。
报告 pure-rotation suppression、true-closing retention、concordance、worst-source、
support/abstention、证据等级和 leave-one-source/session-out。

三类解释：

1. oracle 也失败：关闭当前 Looming 观测/问题定义；
2. oracle 成功而部署方法失败：物理信号存在，但补偿/flow/ROI 实现未闭合；
3. 只有一个来源成功：强 domain dependency，不作通用声明。

强确认要求两个 A/B family；一个 A/B 加一个 C 只能写成“确认 + 迁移复现”。
任何单元的短插值区间单独降为 C 或 abstain，direct 区间仍可评价；不得把
source-interpolated 改写为 direct observation。

R1 通过最多得到 `CONTINUOUS_GEOMETRIC_SIGNAL_SUPPORTED_ACROSS_CLAIMS`，仍不授权
TTC、碰撞概率、报警、App、route/lifecycle、人体、安全或生产。
