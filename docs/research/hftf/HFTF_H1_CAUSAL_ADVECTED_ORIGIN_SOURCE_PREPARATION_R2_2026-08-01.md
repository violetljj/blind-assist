# HFTF H1 causal-advected-origin source preparation R2

日期：2026-08-01

状态：`FROZEN_SOURCE_PREPARATION_ONLY_TEACHER_NOT_AUTHORIZED`

## 1. R2 机制假设

R1 的 fast-egomotion source 在 future coverage 门失败：固定 anchor-centric 0–8 m
field 到 `0.8 s` 时已有大量 cells 落在 future camera 后方。R2 不缩短 horizon、不
删除该运动机制，也不降低 known 门；它把 field 改成 history-causal rolling origin：

1. 对每个 anchor 选择最接近 `anchor-400 ms` 的严格历史 frame，容差 `100 ms`；
2. 仅由 history-to-anchor camera translation 估计 constant velocity；
3. 把 velocity 投影到 anchor local-ground plane；
4. horizon `h` 的 origin 固定为
   `anchor_ground_origin + tangent_velocity * h`；
5. forward/right/up 始终使用 anchor basis；future pose 不参与 origin、方向或样本
   选择，只提供冻结 target observation。

因此 H2 将来仍可从历史 RGB 学习该目标；teacher 不读取 future route intent 或 action
label。它预测的是沿因果自运动外推而滚动的局部空间，不声称预测环境主体的真实运动。

## 2. 保持不变

R2 保留 R1 的：

- 6-bin `[-45°,45°]` forward sector；
- `[0,1,2,3,4,6,8] m`、`0/.4/.8 s` 与
  `foot/body/head`；
- 9 probes、5/9 known、semantic zero UNKNOWN、camera-z 与 `0.20 m` depth
  tolerance；
- UNKNOWN 固定 denominator；
- current/near/far coverage `.15/.10/.10`；
- height/future nonredundancy `.02/.02`；
- 4/4 source sessions 全过。

usable anchor 必须同时具有 history/current/near/far。缺 history 不可消失为 known，
但不进入预定义 usable set；正式协议须复核每 session 仍至少 12 个 usable anchors。

overall future change 可包含滚动 origin 穿过静态空间所产生的差异，不能解释为
environmental dynamics。dynamic semantic support 必须单列；任何动态运动 claim 需要
另行冻结 opportunity-qualified protocol。

## 3. Outcome-blind fresh sources

排除 R0/R1 八个 burned sessions 后，按 official SANPO-Synthetic train 完整 ID
字典序选择具备 `camera_chest/left`、从 frame 0 有至少 25 个 RGB/mask/depth 对齐帧并
通过 frozen source authority 的前四个：

1. `036943049ded9e1a6356de88a2b80b27938dd4386bcad341c57e1145b198e34a`
2. `03b6dc99c1ac44b77d6c3ac36c17a5db748eca862e0008ac7928d7a82a5de68c`
3. `03c8727961f01047f54c3c4e38c48f4128738d100c00bb54145454bc4df9e3b0`
4. `03d7059322ffd8300f2eb85fd612e635242cbe08363e85e0ff74124e916b39e1`

每个 replay 从 frame 0 获取 25 帧，target fps 为 `min(10, source fps)`。若某 session
缺模态或 authority 不通过，只能按同一字典序规则顺延，不能根据 teacher field 表现
替换。

## 4. 权限边界

本合同只授权 source acquisition 与 frozen-canonical authority verifier。四个 exact
authority/report/manifest/spec/pose hashes 尚未绑定前：

- 不得执行 R2 teacher；
- 不得读取任何 R2 field coverage、height 或 future outcome；
- 不得把 source authority 当作 R2 支持证据；
- H2、主线、Android、提醒、默认 App、生产与安全均未授权。
