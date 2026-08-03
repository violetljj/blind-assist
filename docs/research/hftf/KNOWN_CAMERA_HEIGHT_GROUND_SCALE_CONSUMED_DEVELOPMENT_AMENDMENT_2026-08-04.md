# Known camera height ground scale consumed Development amendment

日期：2026-08-04

状态：`USER_AUTHORIZED_CONSUMED_DATA / FROZEN_BEFORE_DA_AND_EFFECT_EXECUTION`

用户明确授权消费过的数据也可用于本目标。本 amendment 因此不再要求当前首次运行必须
fresh；它先用已物化 TartanGround base corpus 做 `CONSUMED DEVELOPMENT`，尽快判断
known-height physical operator 是否有信号并允许后续另立优化协议。

该授权改变数据角色，不改变证据标签：结果不得称为 fresh、held-out、泛化、产品或安全
证据。旧 outcome 可以用于 Development 诊断；任何优化必须在看到本次结果后另写 revision，
不能静默改本轮算子或覆盖输出。

## 冻结 consumed cohort

输入固定为
`artifacts.local/evidence/hftf/stage-c-d5-tartanground-development-corpus-v0`。其 8 个
source 中，按既有 operator 的 `[0.8,2.2] m` height receipt 门，在读取 DA 或本次效果前
机械保留 5 个 parents：

- `CoalMine/Data_diff/P1000`，`H=1.7341949989 m`；
- `Gascola/Data_diff/P1000`，`H=1.7627710550 m`；
- `OldScandinavia/Data_diff/P1000`，`H=1.6767242957 m`；
- `SeasonalForestWinterNight/Data_diff/P1000`，`H=1.0753522962 m`；
- `MiddleEast/Data_diff/P1002`，`H=1.3392212549 m`。

每个 parent 使用既有 samples 中 33 个 current anchor RGB/depth，共 165 帧；不按内容、
ground coverage 或结果换帧。AbandonedCable、Rome、WaterMillNight 仅因 frozen height
range 不合格而不进入，不作算法失败。

候选算子、三带 clearance、UNKNOWN 与效果门继承原 R0，不改阈值。raw DA 是 comparator，
source metric depth + dynamic pose + metadata robot height 生成 truth field。Tartan local NED
固定重排为 optical right/down/forward。

本轮允许输出：coverage、MAE、agreement、false-clear、temporal delta、UNKNOWN reasons、
parent-macro 和相对 raw DA；并报告相对逐帧 median aligned scale 的 diagnostic error。后者
不是独立真值，不能单独授权成功。

终态必须带 `CONSUMED_DEVELOPMENT`。本轮完成后，若效果差，可基于明确 failure atlas
另立 R1 优化；若效果好，也只能作为继续投资的开发信号。
