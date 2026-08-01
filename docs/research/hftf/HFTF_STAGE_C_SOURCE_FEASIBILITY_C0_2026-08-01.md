# HFTF Stage C source feasibility C0

日期：2026-08-01

状态：`FROZEN_BEFORE_C0_MEDIA_CONTENT_OR_GEOMETRY_OUTCOME`

## 1. 目的

R4 已支持 split-source 的 teacher mechanics，但解析 terrain 没有 RGB，SANPO 当前
semantic-ground representation 又没有产生自然台阶/落差机会。因此不能从 R4 直接跳到
student training。

C0 只回答一个更早的问题：是否存在精确绑定的 RGB、metric depth、pose 与 future
timeline transport，使下一轮能冻结 session-disjoint 的 history-RGB student canary，
同时保留 `UNKNOWN != SAFE`。C0 不生成 student 效果、自然危险真值或人类事件结论。

## 2. 固定 source roles

SANPO-Synthetic 继续承担 causal obstacle 与 layered-future teacher role。Stage C
后续 source pool 必须排除 R0–R4 outcome-open sessions，以完整 parent session
分 train/dev/held-out，不能用 frame、pair 或 window 冒充独立样本。

EgoWalk 只承担 natural RGB/depth/pose transport 与 semantic-independent surface
observability canary。其官方 trajectory source 固定为 dataset revision
`8a167f27e7cf1018c24de1b3fee5265425d145d2`；官方 API 固定 commit
`37016060b91cae42c0633b0add9fc985ca11e945`。官方 decoder 把 depth 解释为
`gray16le` 毫米，除以 1000 得米，0 变为缺失。

实际走过的 EgoWalk 轨迹最多是弱 traversability support，绝不是 hazard/safe truth。
它的官方说明明确提示 odometry 会出现 null 与中途回零重初始化，所以 C0 fail closed。

## 3. 冻结的 metadata-only cohort

冻结前只读取了 239 个 pose parquet、四个 meta JSON 与远端文件尺寸/LFS hash；没有下载
或打开 RGB/depth media，也没有计算 teacher/student outcome。

每条候选必须满足：

- 至少 320 帧，`frame=0..n-1`，trajectory 字段与文件名一致；
- timestamp 严格递增，中位间隔 `195–205 ms`，每步 `150–250 ms`；
- 七个 pose 字段 0 null，四元数最大 norm error `<=.001`；
- 位移至少 5 m、单步不超过 1.5 m，且无“已离开 3 m 后一步跳回起点 0.25 m 内”的
  重初始化迹象；
- pose/RGB/depth 三件套、LFS SHA-256 与 camera height metadata 均存在。

健康条目按 pose+RGB+depth 总字节升序、trajectory ID 破同分；依次选取两个不同录制
日期后停止。冻结结果为：

1. `2024_08_15__19_45_11`
2. `2024_07_11__12_33_57`

这两个 trajectory 在 C0 后即为 consumed Development transport canary，不能再包装成
fresh student train/dev/held-out effect data。

## 4. 时间和泄漏合同

统一 timeline 是 5 Hz。student 输入 offsets 固定为
`[-0.6,-0.4,-0.2,0] s` 的 history RGB；teacher horizons 固定为
`[0,.4,.8] s`。future depth/pose 只可在 anchor 坐标系构造 geometry proxy label；
student 不得读取 future RGB、depth、pose、semantic mask。

任何媒体或 label outcome 都不得参与 source selection。后续 student 比较必须在同一
label/UNKNOWN denominator 下同时保留：

- single-frame RGB；
- history RGB current-only field；
- history RGB current+future layered field。

## 5. C0 顺序门

先用独立 runner 复算全部 239 条 metadata inventory 和精确 cohort；再只下载这两个
trajectory 的 RGB/depth，核对 LFS/local SHA，完整解码并要求 RGB/depth/pose 帧数、
5 Hz timeline 与 frame-index seek 全部一致。

每个 source 还固定抽 32 个均匀 frame，检查 RGB、16-bit depth、正有限深度和
bottom-half depth support。这里只评价 surface observability；不使用 semantic class
或 annotation，也不从深度外观人工挑选“好地形”。缺失深度一律 `UNKNOWN`。

顺序终态为：

1. `C0_EGOWALK_METADATA_SELECTION_NOT_EVALUABLE`
2. `C0_EGOWALK_MEDIA_TRANSPORT_NOT_EVALUABLE`
3. `C0_NATURAL_SURFACE_OBSERVABILITY_NOT_EVALUABLE`
4. `C0_STAGE_C_SOURCE_TRANSPORT_FEASIBILITY_SUPPORTED`

## 6. 权限上限

C0 成功只允许冻结下一份 Stage C label-and-student canary protocol。它不授权标签生成
正式执行、student training/effect、研究主线切换、Android/App 修改或安全/产品 claim。

机器可读真源：
[HFTF_STAGE_C_SOURCE_FEASIBILITY_C0_2026-08-01.json](HFTF_STAGE_C_SOURCE_FEASIBILITY_C0_2026-08-01.json)
