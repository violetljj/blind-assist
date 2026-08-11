# DepthART task-preserving D2 task-evidence head protocol

状态：`PRE_OUTCOME / MECHANICS_PASS / METADATA_POOL_LOCKED / SOURCE_USE_NOT_ACTIVATED`

D1 的 FAIL 保持不变。已消费结果显示候选相对 canonical reference 净修正了更多 occupancy
cells，但同时产生 false-block，且清距误差不是稳定的一阶尺度偏差：全局 affine diagnostic
仍为 `0.27231 m` MAE，leave-one-parent 只有 2/6 折达到 `0.20 m`。因此 D2 明确拒绝
post-hoc affine rescue，也不在 D1 上拟合参数。

D2 只冻结一个新假设：保留 exact `608×448` DepthART/HTP depth 主干，在其几何 evidence
上训练一个 277-parameter task head。输入只含 candidate clearance/validity、ground support、
band support/intrusion、observed forward 与 band identity；运行时禁止读取 canonical reference。
head 对三 horizon occupancy 做 cumulative-max 单调投影，clearance residual 被限制在
`±0.5 m`；缺少 ground、有效深度或 band support 时，强制输出 `UNKNOWN_GROUND`。

纯 CPU mechanics canary 已 PASS，只证明 horizon 单调、hard UNKNOWN 与 residual bound，
不证明准确率。新数据采用 32 个与既有 research、D1 和 R2 均不重叠的 Training-fold
metadata pool。数据 admission 分三阶段：

1. 先只取 intrinsics 索引与 trajectory，按 pose 冻结 300-frame portrait continuity，取前 16 个合格身份；
2. 再只取 depth/confidence，在不运行模型的情况下要求每身份至少 1800 known、180 clear、
   900 occupied cells，九个 grid 各至少 100 known，且至少 450 个 valid band clearances；
3. 只给前 8 个支持合格身份补 RGB；前 4 个固定为 TRAIN，后 4 个封存为 DEVELOPMENT。

训练 recipe 固定为单 seed、单 step-500 checkpoint，Development 在 checkpoint hash 冻结前
不得打开。D2 Development 继续使用 D1 的 metric semantics 与绝对门，不降低 false-clear、
false-block、MAE、transition 或 coverage 阈值；任一 parent denominator 缺失仍 fail-closed。

即使 D2 PASS，也只证明四身份 identity-disjoint feasibility，不授权 R2 candidate lock。
R2 cohort 继续 sealed，设备性能、默认 App、production 与 safety 均不授权。当前唯一门是先
建立精确 source-scope receipt，再做 Phase-A HEAD；receipt 前不得请求任何媒体。
