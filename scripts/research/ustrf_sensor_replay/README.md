# USTRF sensor replay

状态：bounded discovery closed / R3 admitted 2-of-3 / formal data limitation / evaluator not run / benchmark-only / production-isolated

## 稳定 Interface

通过 `scripts/run_research_tool.py ustrf-sensor-replay <tool.py>` 调用。`normalize_sources.py` 只接收冻结的来源清单与预注册，输出统一、hash-bound 的 RGB + metric depth + camera-to-world pose bundle；`run_replay.py` 重新校验所有文件后计算同步、深度重投影、clearance proxy 与跨来源最差项，并在提供独立 pose estimate、route truth/prediction、event truth/candidate alert 时计算 pose drift、路线投影误差、event recall、false alerts/min 与 alert clearance。来源 Adapter 内显式封装单位、时间和坐标约定。

R3 使用 `prepare_estimator_inputs.py -> estimate_rgbd_pose.py -> derive_r3_route_candidate.py -> prepare_r3_review_bundle.py -> finalize_r3_reviews.py -> run_replay.py`。estimator 输入账本物理删除 GT pose；candidate trace 必须在 review 前冻结且对两位 reviewer 隐藏；五项事件门逐来源 AND，并分别记录 worst source。任何 source review 拒绝、pose/route 不可评或单源阈值失败都保持 `DO_NOT_SELECT_HARDWARE`。

R3 来源替换先运行 `prescreen_openloris_sources.py`。它根据普通公开可下载性、D435i RGB-D、独立 ground-truth trajectory 与轨迹运动统计生成候选；许可和隐私元数据缺失不阻止下载或隔离研究。即使发现三条以上轨迹，`three_source_count_credit` 仍固定为 false。只有下载完整连续 RGB-D 片段、按原流程冻结 candidate，并由两位隔离 reviewer 都准入后，轨迹才可计入三源。来源替换不得改 `configs/ustrf_sensor_replay_r3_prereg_v1.json`，审核 anchor 容差继续为 15 帧。

OpenLORIS 穷尽后，LILocBench 来源替换使用 `prescreen_lilocbench_sources.py`。该工具读取已哈希的 `dynamics_0` / `lt_changes_dynamics_0` `base_link` GT；因 GT 为 20 Hz、RGB-D 为 15 Hz，工具先建立 nominal 15 Hz RGB 时间线，再按冻结的最大 pose 时间差关联 GT，从而保持 24/12 是 RGB 帧而不是原始 GT 行。官方直链已经足以开放完整 RGB-D 下载和隔离内部研究，不再等待数据权利、同意或隐私收据；两条轨迹在完整 RGB-D、`base_link -> D455 color optical` 外参链和完整片段双模型/裁决闭环后累计准入 `2/3`，第三条前 evaluator 仍关闭。

第三来源检索由 `configs/ustrf_sensor_replay_r3_third_source_discovery_v1.json` fail-closed 冻结。`prescreen_third_source_gt.py` 只读取选中 ROS 2 bag 的实际 RGB header 时间、独立 `world_T_body` mocap pose 与静态 body-to-color-optical 链，运行相同 24/12/0.03/0.50 reject-only 门；不解码 RGB/depth 像素、不生成 candidate、不授予来源计数，也不运行 evaluator。只有报告字段 `gt_route_prescreen_passed=true` 才能进入完整连续 RGB-D 适配。

该有界检索已按 `s9 -> s12 -> s13 -> s14` 顺序闭合：s9 在 GT-only 阶段因 pose 对齐率和 truth/causal unknown 门失败，未做完整适配；s12、s13 通过 GT-only 与完整 RGB-D 几何门，但两位隔离 reviewer 对路线有效性不一致，按双模型 AND 门拒绝；s14 通过 GT-only，完整适配后最低有效深度率 `.440792 < .50`，且两位 reviewer 均拒绝路线有效性。最终仍为 `2/3` 数据局限，不降低门、不复用 Bonn 负样本、不运行 evaluator，因而不输出五项事件指标或 worst-source 结果。

`idsia_msmpt_package` Adapter 消费 IDSIA MSMPT 的 camera-1 完整准备包。`prepare_idsia_msmpt_rgbd.py` 从 ROS 2 bag 提取 RGB/depth，按 20 ms 上限关联，以静态 `chair -> camera_1_color_optical_frame` / depth optical 外参执行 Brown 模型 pure registration 和 nearest-z 冲突处理；`0/65535`、非有限、负值和无法以 `uint16` 毫米表示的投影深度全部保持 unknown，不填洞。准备收据绑定 bag、metadata、GT、标定与 RGB/depth hash chain；单源 normalize 或 candidate 生成均不等于准入。

`lilocbench_calibration.py` 实现无 I/O authority 的 fail-closed 数学核心：解析官方 intrinsics/transform 列表，按显式 `parent_T_child` 方向组合 `base_link -> camera_front_color_optical_frame`，核验 optical `+Z` 确为 base `+X` 前向，并用 `T_color_depth`、plumb-bob color distortion 和 nearest-z buffer 将 raw depth 注册到 color raster。它本身不负责下载、不绕过认证、付费或访问控制、不填洞，也不自行生成 source bundle；完整归档到位后仍必须先复核成员哈希、同步和 registration receipt。

`prepare_lilocbench_rgbd.py` 消费已校验的官方解包目录与 GT，先复核四个标定成员和 GT 哈希，再把 `camera_front` 原始 `uint16` 深度注册到 color raster。原始 `0/65535` 均保留为未知，输出仍为毫米 `uint16` PNG，不填洞；RGB 使用同卷 hardlink，收据记录 2397 帧完整关联、同步分位数、有效深度率、外参方向和 raw/aligned hash chain。随后用 `configs/ustrf_sensor_replay_r3_lilocbench_dynamics_0_source_v1.json` 的 `lilocbench_package` Adapter 生成统一 bundle；单源 normalize 完成不等于审核准入或三源门通过。

`openloris_package` Adapter 使用 `color.txt` 与 `aligned_depth.txt` 做一对一时间关联，并从 OpenCV YAML 读取 D435i 内外参。office 的 OptiTrack 真值在 marker frame，必须转换为 `world_T_marker × inverse(base_T_marker) × base_T_color`；cafe 的 LiDAR-SLAM 真值在 base_link frame，转换为 `world_T_base × base_T_color`。二者都不能把原始 groundtruth 直接冒充 camera pose。

## 输出

只写调用者指定的 `artifacts.local/evidence/ustrf-sensor-replay-r3/`。下载保存在 `artifacts.local/downloads/ustrf-sensor-replay-r3/`；不写 App assets、训练集或仓库根目录。

## 安全边界

公开数据许可和设备测量是来源事实，不由模型生成；许可、同意或隐私元数据缺失不阻止普通公开数据的下载和隔离内部研究，也不得被写成已确认。路线与事件 review 必须使用 `ustrf_event_review_v1` 的隔离模型回执，且 reviewer 不可见候选提醒；缺 pose estimate、路线真值或事件真值时相应指标必须是 `not_evaluable`，不得记为 0。公开 replay 只形成离线研究证据，不开启 U0、Android、硬件或生产权限。

## 停止条件

少于 3 个可通过普通公开渠道下载且通过技术合同的来源、任一必需闭环指标缺失、或 worst-source 失败时，结论保持 `DO_NOT_SELECT_HARDWARE`。新硬件仍必须独立通过 `>=100 pair / >=0.95 source-aligned / INTER_FRAME_STABLE`；本 Module 不得放宽该门。
