# SAGE-LM V1-B-R3/R4 Dense Boundary Result

日期：2026-08-25（Asia/Hong_Kong）

状态：`DEVELOPMENT / DENSE_FIELD_PARTIAL_RESCUE_BELOW_GATE / JOINT_SUPPORT_ACCUMULATION_REJECTED / B2_NOT_RUN`

## 冻结问题

R2 已在正确 source pose 的 24 episode 上建立 B0=`24/24` 与 B1 candidate extraction failure。R3 复用同一
Development cohort，只把独立 RGB LSD segments 替换为 [DeepLSD](https://github.com/cvg/DeepLSD) 的连续 distance /
orientation field 加 fragment fusion；pose、oracle association、9 px localization gate、interpretation-plane geometry、
confidence、arrival、policy 与八条 criteria 不变。R3 不运行 LK、Depth Anything metric range 或 B2。

使用官方 MegaDepth checkpoint，DeepLSD repository commit=`d873fd3619d6e44a9f625bc437ab4786057677e5`，checkpoint
SHA-256=`56E9BAE263977CAA289AE49802987A67D4F00F376AE7E542176C8AE0A6CC2083`。实现只运行官方网络的
distance/orientation field，不使用因 Windows 旧 CMake/pybind11 contract 无法直接构建的 optional `pytlsd` discrete
extractor。field gate 在 outcome 前固定为 distance `<=2.5 px`、vertical orientation `<=20 deg`；局部 4 px fragments
可先共线融合，但最终 fitted line 的逐行有效 support 仍须 `>=9 px`。现有 oracle localization `<=9 px` 门另行保留。

## R3：DeepLSD dense field + fragment fusion

| 指标 | R2 sparse LSD | R3 dense field | R3 目标 | 裁决 |
|---|---:|---:|---:|---|
| true boundary pair available | 2/24 | **15/24** | >=18/24 | fail |
| B1 geometry output | 2/24 | **13/24** | >=18/24 | fail |
| B1 confident geometry | 0/24 | **0/24** | >=12/24 | fail |
| `BOUNDARY_CANDIDATE_MISSING` | 21/24 | **9/24** | <=6/24 | fail |

R3 显著提高真实 boundary 进入候选池的比例，支持 `SPARSE_LINE_REPRESENTATION_IS_A_MATERIAL_BOTTLENECK`；但四个预设
目标全部未过，不能写成 dense boundary route pass，也没有资格启动 B2。13 个 geometry output 的 median center error
`0.0349 m`、median range error `0.1820 m`，说明候选一旦足够接近，既有 geometry 仍保持准确；confidence 仍为 `0/24`
只说明这些输出没有通过冻结 confidence chain。R3 的 `support_length_px` 已是 dense field supported-row count，而该 chain
仍按旧 LSD segment support 的归一化合同计算，因此不能把 `0/24` 直接解释成 13 个 geometry 都质量差，也不授权降低门。

## R4：pose-conditioned multi-view accumulation

按 R3 失败前预定的 successor，R4 不再要求两帧各自形成完整 segment。它在 Frame A 的 image x 与 `0.6–6.0 m`
depth grid 上枚举共享 vertical 3D boundary hypothesis，投影到 Frame B，按两帧 DeepLSD field support 的几何均值排序；
固定 x stride=`2 px`、55 depth samples、top-96 joint hypotheses、每视图 final support `>=9 px`。B1 oracle 只在联合
hypothesis 形成后选择真实 boundary。

| 指标 | R3 | R4 joint accumulation |
|---|---:|---:|
| true boundary pair available | 15/24 | **9/24** |
| B1 geometry output | 13/24 | **8/24** |
| B1 confident geometry | 0/24 | **1/24** |
| missing | 9/24 | **15/24** |
| target-front arrival | 0/24 | **1/24** |

R4 比 R3 退化。当前 `maximize joint field support -> retain top-96` objective 会让高支撑 clutter boundaries 占据
hypothesis budget，不能保证真实低对比 aperture boundary 的 proposal coverage。因此本轮拒绝的是这个具体 joint-support
proposal objective，不是所有 geometry-conditioned detection。不得在已打开 R3/R4 outcome 上调 distance/orientation、grid、
top-k、support、9 px 或 confidence 门来追目标。

## 结论与边界

当前因果链为：正确 pose 与 line-plane geometry 已通过；DeepLSD dense field 证明 sparse segment representation 确实损失
大量 boundary recall，但单次替换仍低于充分性门；naive joint-support 3D accumulation 又损失 proposal coverage。由于 B1
仍未建立 `>=18/24` candidate/geometry，B2 automatic association 没有运行，也没有被裁决。dense-native confidence
需要在 coverage 先过门后另立 R6；不能用当前 representation-contract mismatch 反推 geometry failure。

新的 successor 若启动，必须改变 proposal objective，例如有明确 aperture-boundary coverage 目标的 task-specific
supervision 或带 diversity/coverage 的 geometry-conditioned proposal；不得扫 detector zoo、降低门或直接调 B2。本结果只属于
curated R2 Development cohort，不建立自然 identity、导航、安全、Android、P1 或默认 App authority。

本机证据：

- `artifacts.local/evidence/sage-lm-v1b-r3/deeplsd-dense-boundary-b1-r1/report.json`
  (SHA-256 `5F25BB4FFD7FFA587C4A4BD8A8DE4F416B1785157CEA5109E03528CFDA77DC27`)；
- `artifacts.local/evidence/sage-lm-v1b-r4/pose-conditioned-accumulation-b1-r1/report.json`
  (SHA-256 `92FA0ADAA10F22428DC0B46C2AE76B254FB382155DB8A0AEB4B746C92ED939FE`)。
