# DA V2 QNN 逐算子与 HTP linting profile R0 结果

日期：2026-08-05

## 结论

同一 `518x686 FP16` cached DLC 的冷态 profile 表明，QNN 图的主要成本在 Transformer
attention 主干，不在 JNI 或 layout transform。`detailed` 级别把 88.24% 周期归到
Transformer encoder，其中 attention Softmax 复合事件 56.15%；HTP `linting` 级别重新按
硬件关键资源归因后，Transformer 仍为 87.76%，MatMul 为 70.81%。两种 profiling 语义不同，
不得把 detailed 的 56.15% 解读为一个独立 Softmax primitive 的裸硬件占比；共同结论是
attention 计算主干支配。

## Detailed per-op

冷态 thermal 0，10 秒运行共 24 个 profiled executions，每次 470 个 operator events。
算子周期和 root accelerator 周期闭合误差 `5.8e-14%`。

| 高层类别 | root cycle share |
|---|---:|
| Softmax-attributed attention composite | 56.15% |
| MatMul | 16.17% |
| LayerNorm | 6.58% |
| Resize | 6.93% |
| Reshape | 2.95% |
| Transpose | 0.73% |

region 汇总：Transformer encoder 88.24%，depth head 8.93%，patch embed 0.30%，其余
2.53%。reshape+transpose 合计仅 3.68%，不支持“layout transform 是首要瓶颈”的假设。

12 个 block attention Softmax 事件各占 root 约 4.67-4.70%；depth-head 最重的两个 Resize
分别约 3.61% 和 2.45%。profiled execute 时间受详细采样开销显著放大，不是 App latency。

## HTP linting

冷态 thermal 0，5 秒运行共 11 个 executions，每次 470 ops。critical path mean/P95
`117.14M/117.35M cycles`，单次算子周期和 critical path 比值约 0.996。

| 资源组合 | summed-op cycle share |
|---|---:|
| HVX + HMX + DMA | 73.57% |
| HMX + DMA | 12.89% |
| HVX + DMA | 9.78% |
| HVX only | 3.60% |

inclusive resource coverage：DMA 96.24%、HVX 86.96%、HMX 86.62%。initial VTCM acquire
mean/P95 `2.33/3.29 ms`。日志没有 `DramToTcm`、`TcmToDram`、`SystemService` 或
`BlockZapOp` 明细，所以：

- 可以确认绝大多数关键算子同时使用 DMA；
- 不能由此量化 DDR bytes、证明 VTCM pressure/spill，或声称 shared buffer 会降低图内主干成本；
- overlap/wait 字段是并发调度的非加性量，不能逐 op 直接求和当作 latency。

linting 把关键工作主要归到 attention `MatMul_1`，而 detailed 把 attention 复合成本主要
归到 Softmax。这是 profiling attribution 层级差异，不是两个结果互相否定。

## 路由决定

继续打磨 JNI/layout 的 QNN 图内收益上限较小；若未来进入模型侧，优先问题应是 attention
token/softmax/matmul 结构、输入尺寸或经独立几何/false-clear 门验证的混合精度，而不是盲目
全图 INT8。当前独立准确率与安全 evidence gate 未建立，因此 R6/R7 不启动。

在模型变化前，仍有一项低风险系统工作：把 QNN FP16 output 直接 decode/resize 到 Native
owned aligned-depth buffer，再由 Native geometry 消费，只向 Kotlin 返回小结构。该工作必须
先通过逐元素 resize parity 和几何状态/字段 parity。

## 证据

- detailed bundle result SHA-256：`C535C9503550B6033CE911DD7FCC892B228C21FE5A45DBF9346A114D4105B61B`
- detailed profile CSV SHA-256：`2A6D232104BCEDAB665D96AD2EB626F3DE28FC68167388B3438C32655C374F40`
- detailed operator analysis SHA-256：`BADDDEC710B41E0D14F809592B3D5297BA7FEEECB90A8BABF22AF79858045B0D`
- HTP linting bundle result SHA-256：`F3DD3368D33B5AE390E7A6CC033362AAB5C71FC4C3DB5B3204D963CB24146E48`
- HTP linting raw text SHA-256：`6D4CE82661A09F0F806F7FB24D1E3682DB5C4F7FFAEC7DDFAD57B140D3863245`
- HTP linting analysis SHA-256：`190771BA9878C06AAF50367BA9A85671AE70C81982D744E8AC1E25CCEB639FC4`

所有结论均为单设备性能/资源诊断，不建立 depth accuracy、traversability、产品或 safety authority。
