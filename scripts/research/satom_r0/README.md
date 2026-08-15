# SATOM-R0

状态：`ACTIVE_REVERSIBLE_EXPLORATION / IMPLEMENTED_SYNTHETIC_MECHANICS_CANARY_ONLY / REAL_DEPTHART_PRIOR_DATASET_NOT_RUN`

## 科学问题

在完全 past-only 条件下，`metric pose + simulated ToF ROI + frozen DepthART prior`
的多帧任务空间占据记忆，能否在 parent-macro 和 worst-parent 上稳定超过单帧
DepthART、ToF-only、随机/round-robin 扫描与均匀多帧融合？

SATOM-R0 是新路线，不是 TARO、Assistive Geometry、Q-Plane 或 DepthART D3R6 的
continuation。旧路线 terminal、数据角色和禁止动作保持不变。

## 稳定 Interface

- 三带、米制 range-bin 的 deterministic evidential memory；每帧用 metric pose warp，
  随后 decay，再融合 frozen prior 与稀疏 ToF ray evidence；
- VL53L1X/ToF4M 风格的可配置 range、ROI、first-return quantile、noise 和 missing 模拟；
- ToF cone 到 prior surface 的显式隐变量近似关联，不把标量 range 写满整个 ROI；
- `center-only / random / round-robin / max-entropy / task-weighted information-gain`
  五种因果策略；策略只读过去 memory、当前 frozen prior summary、frame index 和 seeded RNG；
- 单帧 prior、ToF-only、uniform multi-frame fusion 基线；
- shuffled timestamp、wrong extrinsic、wrong ROI 负控；
- pooled、parent-macro、worst-parent、false-clear、false-block、coverage、clearance MAE、ECE。

当前 canary 的 prior 是由合成 truth 人工扰动得到的
`SYNTHETIC_DEPTHART_LIKE_PRIOR`，只验证 mechanics、因果性和 evaluator，绝不算
DepthART 或算法 utility 证据。真实入口会拒绝未声明 `family=DepthART`、`frozen=true`、
`truth_derived=false` 的 manifest。

## 输出

每次运行只写一个版本化 JSON result，包含 arm 配置、causality 声明、scan trace、
pooled、逐 parent、parent-macro 与 worst-parent 统计。大体积数据、dense prior 和运行
结果只能留在忽略的 `artifacts.local/`；不得提交数据、模型或 benchmark payload。

## 运行

```powershell
python -m scripts.research.satom_r0.run_satom_r0 `
  --synthetic-canary `
  --output artifacts.local/evidence/satom-r0/synthetic-canary/result.json
```

真实数据 manifest schema 为 `blindassist.satom_r0.dataset_manifest.v1`。每个 parent
绑定一个带 SHA-256 的 NPZ，必须含：

```text
timestamp_s[N]
truth_depth_m[N,H,W]
prior_depth_m[N,H,W]
intrinsics[N,4]                 # fx, fy, cx, cy
world_from_camera[N,4,4]
candidate_camera_height_m[N]    # frozen-prior + gravity only
truth_camera_height_m[N]        # evaluator-only registered-depth geometry
gravity_down_camera[N,3]        # unit vector in camera x-right/y-down/z-forward
prior_confidence[N,H,W]         # optional
```

注册 RGB-D 只进入 evaluator 和 simulated ToF sensor；scan policy、memory 和 candidate
输出都不能读取 `truth_depth_m`。真实首轮必须使用 Development/consumed parent，并
保留数据集、DepthART checkpoint/output、相机与 pose provenance；不得把 canary 或
Development 包装成 fresh Confirmation。

## 安全边界

注册 RGB-D truth 只允许进入 evaluator 与 simulated ToF sensor；candidate policy 和
memory 不得读取 truth、未来帧或完整 parent 分布。所有输出均为研究模拟，不可替代盲杖、
导盲犬或人工判断，也不产生 Android、默认 App、产品或安全权限。

## 停止条件

唯一 successor：在现有 Bonn（优先）或其他完整 RGB-D+pose Development parent 上，
物化与 RGB/pose 同时间绑定的 frozen DepthART dense prior，建立 real manifest，先跑
最小多-parent E0。TUM 原始主包当前本机已清理，不把历史 clearance report 伪装成 dense prior。

若 deterministic SATOM 在 parent-macro 与 worst-parent 上不能同时相对
`single_frame_depthart / tof_only_round_robin / satom_random / satom_round_robin /
uniform_multiframe_fusion` 显示可解释增量，则关闭 SATOM-A，不训练 refiner、memory
网络或 scheduler。当前禁止训练、Android/设备接入、论文 claim、默认 App 或安全结论。
