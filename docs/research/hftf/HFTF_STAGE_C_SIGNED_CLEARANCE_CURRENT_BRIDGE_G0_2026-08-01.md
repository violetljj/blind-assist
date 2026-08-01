# HFTF-G0：support-equivalent clearance current bridge

状态：
`FROZEN_AFTER_F0_1_STOP_BEFORE_G0_CLEARANCE_OR_SOURCE_SCAN_OUTCOME`

## 决策

F0.1 的负终态保持永久不变。后继不换更强时序 backbone，也不在已开封 official-test
上调参；它把研究问题改成：

> 单帧 RGB 能否先学习以米为单位、与 reference lattice support 等价的 current
> clearance proxy，再由冻结的人体包络算子稳定导出 body/head risk？

旧机制直接学习稀疏的二值 risk 边界。G0 改为预测 obstacle points 相对 swept
human prism 的连续净空 proxy；第二小 proxy 的严格负值恰好对应原 teacher 的
“support count 至少 2”。这是监督与物理分解机制变化，不是 F0.1 rescue。

这里不把 proxy 误称为真正连续 SDF。半开 box 的排除边界与包含边界几何距离都可能
为零，无法同时满足连续 SDF 和 membership 完全等价。冻结定义因此先计算 closed-box
SDF 的绝对大小，再用原 teacher membership 强制符号；精确零值只用 float64
`nextafter` 朝正/负方向打破平局。原 teacher 的 `searchsorted`、最后 `8 m`
`isclose(atol=1e-12)`、height 边界、reference stride-4/offset-2 point set、
semantic filter 和 anchor basis 都逐项复用。少于两个点时 pre-clip 为正无穷，
之后 clip 到 `+1 m`。

## 为什么先做 current

F0.1 的 `SF_CURRENT` median-seed F1 只有 `0.173267`，远低于 `0.6`；history future
delta 也只在一个 seed 为正。继续增加时序容量无法区分“历史无信息”和“current
物理状态根本没学会”。G0 因此完全不评价 future，只先证伪连续物理中间量是否能解决
current cross-source learnability。

## 两级门

G0-D0 只使用已经 consumed 的 12 个 F0.1 sources，验证 clearance-proxy mechanics：

- box 几何量、原 membership、符号强制和第二阶统计量必须结构正确；
- `clearance < 0` 必须与原 `support_count >= 2` 在所有 known cells 完全一致；
- unknown 始终为 null，绝不能导出 safe；
- 每个 `source × height` 都必须有正负 known target、至少 20 个 clipped
  毫米量化 bin、至少 5 个 `|target| <= 0.2 m` 的近边界格；
- risk 和 safe 各自都至少有一个未在 `-0.5/+1.0 m` 饱和的 target，并逐类报告
  clip saturation。

来源规划必须校验全部 11 个 parent 的精确 hash，并以 F0/F0.1 plan、acquisition、
authority、teacher-opportunity、effect-result 与历史 burn 的闭合链证明 freshness。
9 个 outcome-open 旧来源仅用于开发：原 6 train 做探索训练，原 3 dev 做模型与超参
选择；冻结所有模型、loss、activation、clip、阈值、gates 后，才可一次性打开 F0
早先 outcome-blind 预排且从未 acquired/opened 的 3 个 sessions 做
**one-shot fresh evaluation**。它们不能再参与选择，也不能在 outcome 后替换。

SANPO official-test 剩余 pool 只做 metadata scan，预留 3 个 future heldout；G0
期间不得获取或打开它们。

## G0-D1 预声明方向

后续训练合同必须同预算比较：

- `DIRECT_RISK_CURRENT`；
- `SIGNED_CLEARANCE_CURRENT`。

三个 seeds 全部报告。每个 fresh `source × height` 在 opening 后必须至少有 5 个
positive known、20 个 negative known 且 known coverage 至少 `0.1`，否则固定为
`NOT_EVALUABLE`，不得换来源。clearance arm derived-risk median F1 必须至少
`0.6`，相对 direct-risk median delta 至少 `+0.05` 且每 seed 为正；body/head、
worst fresh source、绝对 worst-source F1、recall、FPR、risk/safe/near-boundary
分层 clipped-target MAE 和 UNKNOWN 防火墙也必须同时过门。

若失败，终止为
`SIGNED_CLEARANCE_CURRENT_CROSS_SOURCE_LEARNABILITY_NOT_SUPPORTED_STOP`，不在同一
fresh evaluation 上换 loss、阈值、margin、backbone 或 clearance 定义。若通过，也只允许
另行冻结使用预留 fresh heldout 的 causal-transport canary；不直接支持 temporal
HFTF、主线替换、Android、生产或安全主张。
