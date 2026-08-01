# HFTF H1 geometry teacher canary protocol R0

日期：2026-08-01

workflow：`DEVELOPMENT_STANDARD`

状态：`FROZEN_RESULT_NOT_RUN`

上游门：`HFTF_H0_2_INDEPENDENT_SESSION_REPLICATION_ADMITTED`

## 1. 唯一问题

在四个已通过 source-specific authority 的 SANPO-Synthetic source sessions 上，冻结的
metric geometry 是否能稳定生成 action-agnostic 的
`theta × distance × horizon × height` field，并同时证明：

1. `foot/body/head` 相对 single-height 不是机械重复；
2. `0.4/0.8 s` future 相对 current 不是机械重复；
3. unknown、遮挡和 source failure 不会被写成 safe。

成功上限只有 `GEOMETRY_PROXY_MECHANISM_SUPPORTED`。本实验不训练 RGB student，也不
读取人类 event/collision/safety truth。

## 2. 冻结来源

parent independent unit 是 source session，不是 frame/cell：

- discovery/mechanics session：`e1ae36e0…de856`；
- frozen-transform replication sessions：
  `001217c6…910a`、`0099b54c…864c`、`00bdf8ce…5896`。

每个 session 使用现有 hash-bound 25-frame replay 与相应 H0.1/H0.2 authority report。
任何 source authority、canonical transform、`+Z` local-ground proxy 或 report hash
不通过即整体 `H1_SOURCE_AUTHORITY_NOT_EVALUABLE`。

## 3. 冻结 field

### 3.1 轴

- theta：24 个等宽 bins，覆盖 `[-180°, 180°)`，每 bin `15°`；
- radial distance edges：`[0, 1, 2, 3, 4, 6, 8] m`，共 6 bins；
- horizon：`current=0.0 s`、`near=0.4 s`、`far=0.8 s`；
- height bands，相对每帧 source-derived local ground plane：
  - `foot = [0.05, 0.35) m`
  - `body = [0.35, 1.35) m`
  - `head = [1.35, 2.05] m`

这些是 synthetic standard-body proxy 数值，不是 participant 尺寸或物理标定。

### 3.2 horizon binding

以 manifest nominal relative time 选择每个 anchor 的未来 observation：

- near：最接近 `anchor + 400 ms` 的同 session frame，允许误差 `<=100 ms`；
- far：最接近 `anchor + 800 ms` 的同 session frame，允许误差 `<=100 ms`；
- 一个 anchor 缺任一 future observation 时不进入可用 anchor 数，但仍记录为
  missing，不得计为 known/safe；
- 每 session 至少需要 12 个同时具有 current/near/far 的 anchors。

nominal time 来自 `source_frame_num / session description fps`，不是实测 capture time。

## 4. 冻结 teacher 算法

1. 使用 H0 冻结的 `p_world = R_xyzw @ p_opencv_camera + translation_m`；
2. 每个 observation 用 semantic-ground + metric-depth 的 H0 确定性 local-plane
   fitter；camera ground proxy 为 camera position 到该 plane 的正交投影；
3. field 坐标原点固定为 anchor 的 ground proxy，vertical 使用 anchor local-plane
   normal；theta zero 是 anchor camera forward 在 local plane 上的投影；
4. obstacle points 排除 ground proxy classes、sky 和无效深度；dynamic semantic classes
   保留并在 atlas 单列；
   - point cloud 固定从原分辨率每 8 pixels 在 x/y 各采样一次，起点为 `(4,4)`；
   - 排除 class IDs `0,1,3,5,6,17,27,30`；dynamic 单列 IDs
     `12,13,14,15,16,21`；
5. 每个 cell 的 `known_score`：
   - 9 probes 精确定义为 cell 中心，以及
     `theta lower/upper × distance lower/upper × height lower/upper` 的八个角点；
   - cell 中心和八个角点至少 5 个投影在 observation image 内；
   - 对在图像内的 probe，metric depth 必须到达该 probe 前缘，容差 `0.20 m`；
   - `known_score = passing probes / 9`；
6. cell `known_score >= 5/9` 才可评价 risk，否则 tri-state 为 `UNKNOWN`；
7. `risk_score` 为该 cell 体积内 obstacle points 的 clipped support：
   `min(1, point_count / 8)`；无 obstacle points 只有在 known 过门时才可为 `SAFE`；
8. single-height current 使用 `[0.05, 2.05] m`，其 risk 明确定义为三个
   multi-height risk 的 max；执行器仍必须复核差值 `<=1e-12`；
9. 不使用 RGB 类别预测、路线意图、future teacher 信息选择 anchor/阈值或任何事件
   标签。

## 5. 固定指标与门

所有比例以预定义 denominator 计算；missing/invalid/UNKNOWN 不移出 required cell
denominator。

### 5.1 source/mechanics validity

- 4/4 source sessions authority 通过；
- 每 session usable anchors `>=12`；
- single-height vs multi-height max consistency error `<=1e-12`；
- 每 session current required-cell known coverage `>=0.15`；
- 每 session near 与 far required-cell known coverage各 `>=0.10`。

### 5.2 multi-height 非冗余

对 current 中三个 height cells 都 known 的同一 `anchor × theta × distance`：

- `height_disagreement`：`max(risk)-min(risk) >=0.25`；
- 每 session disagreement fraction `>=0.02`；
- 4/4 sessions 均须通过。

### 5.3 future 非冗余

对 current 与对应 future height cell 都 known 的 required cells：

- `future_change`：`abs(risk_future-risk_current) >=0.25`；
- near 或 far 的 union change fraction每 session `>=0.02`；
- 4/4 sessions 均须通过。

这只说明 layered/future proxy 有非零结构内容，不说明它能被 causal RGB student 预测，
也不说明变化对应真实风险。

## 6. 必须输出

- 每 session authority、anchors、required/known/unknown cell counts；
- current/near/far known coverage；
- single/multi consistency；
- height disagreement 与 future change numerator/denominator/fraction；
- dynamic-support、occlusion/unknown 和 largest-change atlas 索引；
- worst-session 指标；
- source report、manifest、teacher implementation hash。

atlas 只做 failure localization，不参与 threshold 或终点选择。

## 7. 终点

按顺序：

1. source/anchor/known/consistency 任一失败：
   `H1_GEOMETRY_TEACHER_NOT_EVALUABLE`；
2. multi-height 4/4 未过：
   `H1_MULTI_HEIGHT_PROXY_NOT_SUPPORTED_STOP`；
3. future 4/4 未过：
   `H1_FUTURE_PROXY_NOT_SUPPORTED_STOP`；
4. 全部门通过：
   `GEOMETRY_PROXY_MECHANISM_SUPPORTED`。

任何终点都不自动授权 H2。只有成功终点才允许另行冻结 H2 causal student protocol。

## 8. 禁止解释

- standard-body proxy 是真实身体尺寸或 camera-to-person 标定；
- nominal horizon 是精确 capture-time dynamics；
- teacher change 是真实 collision/event truth；
- teacher mechanics 证明 history RGB 可预测；
- frame/cell 数量增加 source-session 独立性；
- H1 成功触发研究主线、Android、提醒、默认 App、生产或安全权限。
