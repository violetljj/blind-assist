# USTRF JRDB RGB continuity / ego-motion availability R0 结果（2026-07-25）

状态：`EGOMOTION_QUALITY_AVAILABILITY_INSUFFICIENT / VALID`

权限：`PRE_G3_SOURCE_AVAILABILITY_ONLY / G3_CLOSED / G4_CLOSED / SIGNAL_CLOSED / ROUTE_TRUTH_CLOSED / ANDROID_CLOSED / HUMAN_CLOSED / PRODUCTION_CLOSED`

## 结论

32 帧 stitched RGB transport、capture timestamp、person exclusion mask 与 sparse-LK 特征均完整可用，但冻结的单一 global RANSAC 2D affine 仅有 `11/31` pair 通过，低于预注册 `28/31` 门。唯一反复失败项是 inlier ratio：20 个 pair `<0.65`。

因此终态是：

`EGOMOTION_QUALITY_AVAILABILITY_INSUFFICIENT / VALID`

不能因为通过子集的 residual 很低，就把未解释的 35%–60% background track 默认为 ego-motion=0 或降低 inlier 门。该 JRDB 短窗不支持扩大到整 sequence，也不开放 G3/G4。

## 冻结窗口与方法

- sequence：`cubberly-auditorium-2019-04-22_1`
- frames：`000000.jpg`–`000031.jpg`
- pairs：31
- person bbox expansion：16 px
- sparse LK：1000 corners、0.01 quality、12 px spacing、21×21、3 levels
- full affine RANSAC：2.0 px、2000 iterations、0.99 confidence
- spatial gate：4×3 grid 至少 8 cells

labels 只用于 person mask；未读取 route/event/outcome。

## 结果分布

| 指标 | min | median | max | 冻结门 |
| --- | ---: | ---: | ---: | ---: |
| timestamp gap (s) | 0.04573 | 0.07149 | 0.08998 | `(0, 0.2]` |
| detected features | 657 | 712 | 803 | ≥120 |
| valid tracks | 649 | 706 | 792 | ≥80 |
| occupied grid cells | 11 | 12 | 12 | ≥8 |
| inlier ratio | 0.4033 | 0.6293 | 0.7511 | ≥0.65 |
| median residual (px) | 0.4470 | 0.6647 | 0.9399 | ≤1.5 |
| p95 residual (px) | 1.3983 | 1.6633 | 1.9818 | ≤3.0 |
| affine condition | 1.0002 | 1.0027 | 1.0074 | ≤10 |
| determinant | 0.9955 | 1.0001 | 1.0041 | `[0.8,1.25]` |

纹理、track 数、空间覆盖、时间连续、模型数值条件与 inlier residual 均充足；失败集中在 global affine 对全部背景 motion 的解释比例。这更像 cylindrical panorama + 室内深度层次/视差对单一 affine 的模型限制，而不是数据运输缺失；该解释是诊断，不改变冻结 terminal。

## 资源与验证

- producer/validator 各读取 32,941,877 network bytes，低于 128 MiB；
- 只读取一次 21,915,466-byte central directory 与 32 个 JPEG；
- full archive download：false；
- 32/32 JPEG 已持久化并逐帧 SHA-bound；
- config SHA：`78f4d3a35eeace24aba8729437821c4e820295e8003ab9951957ebf2e26c562e`
- producer PID：`25056`
- receipt SHA：`3a9c5c334ce388b7cf770a3858252758cccae9781070755c3361aff5460cd8e1`
- validator PID：`57808`
- validation SHA：`efd53d51bdd67e89d02cd0f46a473860aa711fcfa2bab35ff3e79b435239fce5`
- validator：deterministic recomputation、PID isolation、32 frame hash、network budget、terminal 与全部高权限关闭均通过

首次 producer 在 32 帧读取完成后因 pair iterator 长度断言错误 fail closed，未生成 receipt；只修复 `frames[:-1]` 与 `frames[1:]` 配对并更新实现摘要，方法和阈值未改变。

## 决策

1. 不降低 0.65 inlier ratio，不改 RANSAC threshold，不加 homography/dense flow/source-specific fallback 回救；
2. 不扩整条 JRDB sequence；
3. 不运行 ego-aware expansion、signal 或 G4；
4. 保留单帧 source-authority success 与本短窗 quality failure，两者不互相覆盖；
5. 后续只有 metric depth、VIO/IMU、真实 route provider 或新 route-authoritative data pack 才构成信息增量。

JRDB 当前没有可证明的 intended-route truth；3D person labels 或 robot-forward axis 不能自动替代导航路线意图。

