# HFTF H1 forward-sector geometry teacher canary protocol R1

日期：2026-08-01

workflow：`DEVELOPMENT_STANDARD`

状态：`FROZEN_RESULT_NOT_RUN`

## 1. 改写理由

H1 R0 的 360° anchor-centric field 在 source、usable anchor 与 consistency 门均通过，
但在 9-probe known coverage 第一顺序门关闭为
`H1_GEOMETRY_TEACHER_NOT_EVALUABLE`。这说明单个单目 observation 不能支持完整周向场，
不能解释为 geometry teacher 已失败，也不能在 burned sessions 上降低门或删掉
UNKNOWN 来救援。

R1 检验一个不同且更贴近单目助盲移动输入的表示合同：只输出 camera-forward
`[-45°, +45°]` locomotion sector，仍同时覆盖所有候选前向方向，不读取路线意图或动作
标签。该范围在读取任何 R1 teacher outcome 前冻结，不是从 R0 的最好 cells 反推；R0
四个 sessions 永久排除于 fresh evidence。

## 2. 唯一问题与上限

在四个全新、独立且 source authority 通过的 SANPO-Synthetic train sessions 上，
forward-sector metric geometry 是否能稳定生成 action-agnostic 的
`theta × distance × horizon × height` field，并同时证明：

1. `foot/body/head` 相对 single-height 不是机械重复；
2. `0.4/0.8 s` future 相对 current 不是机械重复；
3. UNKNOWN、遮挡和 source failure 不会被写成 safe。

成功上限只有 `SYNTHETIC_FORWARD_SECTOR_GEOMETRY_PROXY_MECHANICS_ONLY`。不训练 RGB
student，不读取人类 event/collision/safety truth，不证明 rear/full-azimuth support。

## 3. Outcome-blind 新来源

选择规则固定为：排除 R0 burned sessions 后，取 official SANPO-Synthetic train 中按
完整 session ID 字典序最小的前四个合格 sessions。parent independent unit 是 source
session，不是 frame/cell：

- `00c2a1cd…d4e3`
- `013e2db5…9d3a`
- `01c00b13…1a70`
- `026d78f9…e610`

四个 authority 均已在 teacher outcome 前通过
`HFTF_H0_2_SANPO_CANONICAL_PROXY_REPLICATED`。机器协议绑定完整 session ID、authority
report、manifest、dataset spec 与 pose SHA-256；替换、重复或增加 session 均 fail
closed。source authority 只证明 source-specific geometry proxy 可用，不是 H1 outcome。

## 4. R1 唯一表示改动

- theta：6 个等宽 bins，覆盖 `[-45°, +45°]`，每 bin `15°`；
- point 的 theta interval 左闭右开，仅 `+45°` 上界右闭；
- probes 使用同一组冻结 theta edges，边界外 points 不 wrap、不进入 field；
- radial distance、height、horizon、teacher、known/UNKNOWN、denominator 与所有数值门
  均保持 R0 不变。

保留的轴为：

- distance edges：`[0, 1, 2, 3, 4, 6, 8] m`；
- horizons：`current=0`、`near=0.4 s`、`far=0.8 s`，tolerance `100 ms`；
- height：`foot=[0.05,0.35)`、`body=[0.35,1.35)`、
  `head=[1.35,2.05] m`。

future field 继续使用 anchor origin/normal/forward/right。nominal time 继续由
`source_frame_num / source fps` 复算，不冒充精确 capture time。

## 5. Teacher 与 UNKNOWN

R1 沿用 R0 的冻结实现合同：

1. `p_world = R_xyzw @ p_opencv_camera + translation_m` 与 per-frame local-ground
   proxy；
2. x/y stride 8、offset 4 的 obstacle points，固定 semantic exclusion/dynamic IDs；
3. 每 cell 为中心加八角点共 9 probes，至少 5/9 通过 image、camera-z、depth-front
   `0.20 m` 与 semantic nonzero 才 known；
4. `risk=min(1, obstacle_point_count/8)`；零点只有在 known 时才可为 safe；
5. single-height risk 是三个 height risks 的 max，并要求 consistency
   `<=1e-12`；
6. UNKNOWN/invalid 永远保留在冻结 denominator 中。

## 6. 固定 denominator 与门

对 usable anchor set `U=current+near+far all bound`：

- 每 horizon coverage denominator：`|U| × 6 × 6 × 3`；
- height disagreement denominator：`|U| × 6 × 6`；
- future near-or-far union denominator：`|U| × 6 × 6 × 3`。

门保持 R0 数值不变：

- 4/4 source authority；
- usable anchors 每 session `>=12`；
- current/near/far known coverage 每 session分别 `>=.15/.10/.10`；
- height disagreement：全 height known 且 `max(risk)-min(risk)>=.25`，固定分母
  fraction `>=.02`，4/4；
- future union change：jointly known 且任一 near/far 与 current 差值 `>=.25`，
  固定分母 fraction `>=.02`，4/4。

## 7. 顺序终点

1. source/anchor/known/consistency 任一失败：
   `H1_GEOMETRY_TEACHER_NOT_EVALUABLE`；
2. multi-height 4/4 未过：
   `H1_MULTI_HEIGHT_PROXY_NOT_SUPPORTED_STOP`；
3. future 4/4 未过：
   `H1_FUTURE_PROXY_NOT_SUPPORTED_STOP`；
4. 全部门通过：
   `GEOMETRY_PROXY_MECHANISM_SUPPORTED`。

任何终点都不自动授权 H2。成功只允许另行冻结 H2 causal-student protocol；不触发
研究主线、Android、提醒、默认 App、生产或安全权限。

## 8. Burn 与停止规则

正式 runner 提交后只执行一次。执行后四个 R1 sessions 即 burned：

- 不在这些 sessions 上修改 sector、阈值、probe、semantic 或 denominator；
- coverage 失败则停止该 evidence version，只能在全新 sessions 上提出另一个预冻结
  support hypothesis；
- height/future 若在更早顺序门失败时只作 diagnostic，不形成支持或否定；
- atlas 只定位失败，不参与门或终点选择。
