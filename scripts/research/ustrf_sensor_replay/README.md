# USTRF sensor replay

状态：active / benchmark-only / production-isolated

## 稳定 Interface

通过 `scripts/run_research_tool.py ustrf-sensor-replay <tool.py>` 调用。`normalize_sources.py` 只接收冻结的来源清单与预注册，输出统一、hash-bound 的 RGB + metric depth + camera-to-world pose bundle；`run_replay.py` 重新校验所有文件后计算同步、深度重投影、clearance proxy 与跨来源最差项，并在提供独立 pose estimate、route truth/prediction、event truth/candidate alert 时计算 pose drift、路线投影误差、event recall、false alerts/min 与 alert clearance。来源 Adapter 内显式封装单位、时间和坐标约定。

## 输出

只写调用者指定的 `artifacts.local/evidence/ustrf-sensor-replay-r2/`。下载保存在 `artifacts.local/downloads/ustrf-sensor-replay-r2/`；不写 App assets、训练集或仓库根目录。

## 安全边界

公开数据许可和设备测量是来源事实，不由模型生成。路线与事件 review 必须使用 `ustrf_event_review_v1` 的隔离模型回执，且 reviewer 不可见候选提醒；缺 pose estimate、路线真值或事件真值时相应指标必须是 `not_evaluable`，不得记为 0。公开 replay 只形成离线研究证据，不开启 U0、Android、硬件或生产权限。

## 停止条件

少于 3 个许可清晰且通过合同的来源、任一必需闭环指标缺失、或 worst-source 失败时，结论保持 `DO_NOT_SELECT_HARDWARE`。新硬件仍必须独立通过 `>=100 pair / >=0.95 source-aligned / INTER_FRAME_STABLE`；本 Module 不得放宽该门。
