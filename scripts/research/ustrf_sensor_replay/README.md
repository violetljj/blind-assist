# USTRF sensor replay

状态：active / R3 source admission failed / benchmark-only / production-isolated

## 稳定 Interface

通过 `scripts/run_research_tool.py ustrf-sensor-replay <tool.py>` 调用。`normalize_sources.py` 只接收冻结的来源清单与预注册，输出统一、hash-bound 的 RGB + metric depth + camera-to-world pose bundle；`run_replay.py` 重新校验所有文件后计算同步、深度重投影、clearance proxy 与跨来源最差项，并在提供独立 pose estimate、route truth/prediction、event truth/candidate alert 时计算 pose drift、路线投影误差、event recall、false alerts/min 与 alert clearance。来源 Adapter 内显式封装单位、时间和坐标约定。

R3 使用 `prepare_estimator_inputs.py -> estimate_rgbd_pose.py -> derive_r3_route_candidate.py -> prepare_r3_review_bundle.py -> finalize_r3_reviews.py -> run_replay.py`。estimator 输入账本物理删除 GT pose；candidate trace 必须在 review 前冻结且对两位 reviewer 隐藏；五项事件门逐来源 AND，并分别记录 worst source。任何 source review 拒绝、pose/route 不可评或单源阈值失败都保持 `DO_NOT_SELECT_HARDWARE`。

R3 来源替换先运行 `prescreen_openloris_sources.py`。它根据普通公开可下载性、D435i RGB-D、独立 ground-truth trajectory 与轨迹运动统计生成候选；许可和隐私元数据缺失不阻止下载或隔离研究。即使发现三条以上轨迹，`three_source_count_credit` 仍固定为 false。只有下载完整连续 RGB-D 片段、按原流程冻结 candidate，并由两位隔离 reviewer 都准入后，轨迹才可计入三源。来源替换不得改 `configs/ustrf_sensor_replay_r3_prereg_v1.json`，审核 anchor 容差继续为 15 帧。

`openloris_package` Adapter 使用 `color.txt` 与 `aligned_depth.txt` 做一对一时间关联，并从 OpenCV YAML 读取 D435i 内外参。office 的 OptiTrack 真值在 marker frame，必须转换为 `world_T_marker × inverse(base_T_marker) × base_T_color`；cafe 的 LiDAR-SLAM 真值在 base_link frame，转换为 `world_T_base × base_T_color`。二者都不能把原始 groundtruth 直接冒充 camera pose。

## 输出

只写调用者指定的 `artifacts.local/evidence/ustrf-sensor-replay-r2/`。下载保存在 `artifacts.local/downloads/ustrf-sensor-replay-r2/`；不写 App assets、训练集或仓库根目录。

## 安全边界

公开数据许可和设备测量是来源事实，不由模型生成；许可、同意或隐私元数据缺失不阻止普通公开数据的下载和隔离内部研究，也不得被写成已确认。路线与事件 review 必须使用 `ustrf_event_review_v1` 的隔离模型回执，且 reviewer 不可见候选提醒；缺 pose estimate、路线真值或事件真值时相应指标必须是 `not_evaluable`，不得记为 0。公开 replay 只形成离线研究证据，不开启 U0、Android、硬件或生产权限。

## 停止条件

少于 3 个可通过普通公开渠道下载且通过技术合同的来源、任一必需闭环指标缺失、或 worst-source 失败时，结论保持 `DO_NOT_SELECT_HARDWARE`。新硬件仍必须独立通过 `>=100 pair / >=0.95 source-aligned / INTER_FRAME_STABLE`；本 Module 不得放宽该门。
