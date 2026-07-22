# USTRF R1.2d 后续工作计划（2026-07-22）

状态：execution plan / production-isolated / authority unchanged

## 结论先行

R1.2d 之后不应立即启动另一轮 detector 架构竞赛。P2 在离线 17 来源上把 small/London-like recall 提高 `+2.20pp/+2.54pp`，但三个 seed 的 London 都是 `0/22`，正事件仍为 `4/6`，Bridge bollard 也稳定漏检；YOLOE 外部参考仍以 `5/6` 和更好的关联覆盖领先。这说明当前最高价值缺口不是再加一个检测头，而是：

1. 把真实 route-conditioned event truth 从 `0` 推进到可审计采集链；
2. 判断 SM-S9280 是否能产生同一 frame 绑定的 metric depth + stable pose，而不是重复旧 ARCore 窗口；
3. 只有正式 truth 与 device metric geometry 都闭合后，才执行 U0 效果矩阵。

因此后续顺序固定为 `首个真实 matched pair → 10-episode pipeline pilot → 独立 ARCore frame-bound 几何 canary → 正式 120-episode truth / geometry admission → U0`。detector 数据诊断保留为冻结 backlog，不进入主动队列。

## 为什么不继续训练 detector

| 证据 | 含义 | 决策 |
| --- | --- | --- |
| P2 三 seed 的离线 London-like recall 均高于 P3 | stride-4 对一般小框有作用 | 保留负结果和实现，不宣称结构无效 |
| P2/P3/YOLOE 的 London 均为 `0/22` | London 失败不是单纯输出 stride 或同权重分辨率问题 | 关闭 P2/分辨率搜索 |
| P2/P3 都漏 Bridge，bollard 仅 40 个唯一训练图像且 39 组标签版本分歧 | bollard 数据覆盖/质量不足以支持架构结论 | 未来先做数据准入，不先换头 |
| P2 假检测/图增加 `+0.236`，压力范围 `154`，P3 仅 `18` | 离线 recall 增益伴随不稳定检测压力 | 不选 lucky seed，不调阈值回救 |
| eligible human truth=`0`，device metric geometry admission=`false` | 现有模型比较不能回答真实助盲效果 | 主动队列转回 truth 与 geometry |

现有 YOLOE 继续作为 benchmark-only 外部参考；不替换 App 默认模型，也不把 `5/6` 当作生产通过。

## 工作面 A：真实路线事件采集链（第一优先）

### A1. 首个 matched pair canary

唯一立即执行项是已经物化的 `route_session_01__route_obstacle__pair_01`：

- 同一安全地点、光照、SM-S9280、mount、相机配置、route plan/provider/choice；
- positive：轻质静态障碍物进入所选路线；matched negative：同类障碍物保持在路线外；
- 各 `10–20s`，现场观察员、随时停止通道，操作者不依据 App 输出行动；
- 依次生成原始视频、source/consent/privacy receipt、capture clock、frame ledger、非未来 explicit route/projection receipt；
- 机械 validator 通过后才进入两份互不可见的人类 review；有分歧时才进入独立裁决。

现成入口：`artifacts.local/evidence/ustrf-sc/real-route-event-pilot-v1-20260721-r1/START_HERE.md`。

停止条件：任一视频时长、clock、frame identity、route causality、投影哈希或隐私/许可收据失败，则只修采集链并重采该 pair；禁止补写摘要数字或绕过 validator。

### A2. 扩到 10-episode pilot

只有 A1 的正/负两集都通过原子证据校验后，才采其余 8 集。完整 pilot 必须精确覆盖冻结的 1 session × 5 scene × 正/负 matched pair，并运行：

```powershell
E:\codex-tools\bin\blindassist-python.cmd scripts\validate_ustrf_sc_route_conditioned_event_pilot.py `
  --config configs\ustrf_sc_route_conditioned_event_collection_v1.json `
  --manifest artifacts.local\evidence\ustrf-sc\real-route-event-pilot-v1-20260721-r1\manifest.json `
  --output artifacts.local\evidence\ustrf-sc\real-route-event-pilot-v1-20260721-r1\pilot_audit_report.json
```

通过含义仅为 `collection pipeline audit passed`；`route_conditioned_truth_eligible`、U0、训练、Android runtime 和 production authority 仍必须为 false。

### A3. 正式 truth 矩阵

10-episode pilot 通过后，先复核操作负担、隐私、review 一致性和失败重采率，再由用户明确决定是否扩到正式 `6 session × 5 scene × 2 episode = 120 episode / 60 matched pair`。正式矩阵完成和独立双审裁决闭合前，不运行有效 U0。

## 工作面 B：SM-S9280 frame-bound 米制几何 canary（第二优先）

### B1. 改变采集来源，而不是重复旧窗口

旧证据为 raw-depth source-aligned `1/861`、`0/843`，pose 为 `EPHEMERAL_PER_FRAME`；因此禁止再跑同一种 900-update 审计。新 canary 必须采用独占 ARCore Session 的单一 `Frame` 数据源，在同一次 `Session.update()` 返回的 frame 上绑定：

- `Frame.getTimestamp()` 与 `getAndroidCameraTimestamp()`；
- `acquireCameraImage()`；
- `acquireRawDepthImage16Bits()` 及 raw-depth confidence；
- `Camera.getPose()`、`getTrackingState()`、image/depth intrinsics 和坐标变换；
- 每帧唯一 identity、设备/SDK/build fingerprint、捕获参数和原始哈希。

ARCore 官方 `Frame` API 提供当前 frame 的 camera image、raw depth、camera timestamp 和 pose，且 pose 只有在 `TRACKING` 时可用：[Frame reference](https://developers.google.com/ar/reference/java/com/google/ar/core/Frame)、[Camera reference](https://developers.google.com/ar/reference/java/com/google/ar/core/Camera)。Depth API 也明确 depth 在跟踪特征不足或失跟时可能暂不可用：[Depth developer guide](https://developers.google.com/ar/develop/java/depth/developer-guide)。

不把 SharedCamera 作为 metric-depth 解法。官方说明 SharedCamera 虽可让 Camera2 与 ARCore 共用相机，但 ARCore 在该模式下不能使用硬件 depth sensor：[SharedCamera reference](https://developers.google.com/ar/reference/java/com/google/ar/core/SharedCamera)。

### B2. 可停止的小规模设备门

先实现 benchmark-only recorder 与收据，不接 App runtime。只跑一个受控 `20–30s` 移动场景，并沿用现有 admission 硬门：

- 至少 `100` 个不同 camera timestamp 的有效 RGB-depth 对；
- source-aligned fraction `>=0.95`；
- pose receipt 必须为 `INTER_FRAME_STABLE`，且非 `TRACKING` frame 不得进入分母；
- 每个输入绑定同一 frame/coordinate system/intrinsics/transform，缺 depth 或错时序 fail closed；
- 原始包和摘要均由 validator 重算，不能信任自报计数。

ARCore Recording API 可把视频、IMU 和自定义 metadata 保存进同一 MP4，适合保留可重放审计包，但官方也警告录制有性能开销、回放的 pose/trackable 结果可能与实时不同。因此 recording 只保存输入，不替代实时准入收据：[Recording overview](https://developers.google.com/ar/develop/recording-and-playback)、[Android recording guide](https://developers.google.com/ar/develop/java/recording-and-playback/developer-guide)。

停止条件：若 SM-S9280 在新数据源上仍达不到 `100 / 0.95 / INTER_FRAME_STABLE`，将手机米制几何标为本阶段不可行，停止重试；下一选择是未来眼镜/独立深度设备的全新证据链，而不是放宽门槛。

### B3. admission 仍是独立阶段

B2 通过只说明输入可绑定。之后仍须补齐 `validate_ustrf_sc_device_metric_geometry.py` 要求的 depth registration、camera/body frame、外参、body-local ground truth、route-event truth 和 target-device benchmark 五类 typed artifact。全部通过也只授权 geometry shadow，不授权 App 提示或生产。

## 工作面 C：U0 的开启条件

仅当以下两项同时成立时，才进入 U0：

1. 正式 120-episode/60-pair human truth gate 通过，而非 10-episode pilot；
2. device metric geometry admission 通过，而非 B2 输入 canary。

届时使用现成六臂 evaluator、LOSO、正确路线/错误路线负控、最差 session/scene/matched-pair、critical recall、false alerts、clearance 与因果不退化门。U0 结果先进入 shadow；student 训练、App feedback 和 production promotion 仍需独立授权。

## 冻结 backlog：未来 detector 数据路线

在 A/B 取得新证据或用户再次明确授权前，不执行下列工作，只保留设计：

1. 对 London/Bridge 做 expected-class-blind top-k、pre/post-NMS、IoU/中心距离和目标尺度诊断，区分“无候选”“错类”“几何不匹配”“关联失败”；不得据诊断调本轮阈值。
2. 先发现并审查多个独立来源的 bollard/delineator 数据，要求许可、exact geometry、图像 SHA、来源互斥验证和零未决标签冲突；事件帧仍禁止入训。
3. 数据准入后先用固定 P3 baseline 做 data-only 配对消融，确认 London/Bridge 事件收益，再决定是否提出新架构；P2 不自动复活。
4. 任何新 detector 轮次继续要求三 seed、最差来源、固定事件、目标条件假告警和未分配路线压力；单一 detector AP 不能晋级。

## 推荐执行顺序与决策点

| 顺序 | 动作 | 通过后 | 失败后 |
| ---: | --- | --- | --- |
| 1 | 采 `route_obstacle_pair_01` 两集 | 扩 10-episode pilot | 只修采集链并重采该 pair |
| 2 | 完成 10-episode audit | 决定是否承担正式 120 集 | 保持 truth=0，不进 U0 |
| 3 | 独占 ARCore frame-bound canary | 补五类 geometry artifacts | 冻结手机 metric geometry，转未来硬件 |
| 4 | 正式 truth + geometry 双门 | 执行 U0 shadow evaluation | 保持现有 benchmark-only 状态 |
| 5 | U0 通过后重新评估 detector/student | 新建独立预注册 | 不回调旧 R1.2d |

当前单一推荐下一动作：按 `START_HERE.md` 采集首个 `route_obstacle` matched pair。除此之外不新增训练、模型、阈值或 R1.3 来源。
