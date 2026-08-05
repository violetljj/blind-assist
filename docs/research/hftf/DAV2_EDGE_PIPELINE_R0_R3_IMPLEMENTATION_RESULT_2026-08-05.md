# DA V2 端侧流水化与等价 Native 优化 R0-R4 实现结果

日期：2026-08-05

## 结论

在冻结 `518x686 FP16` cached DLC、前处理、5000 点、240 次 RANSAC、seed 1729、
阈值和几何输出语义的条件下，CameraX/QNN 与 CPU 几何已实现一图一任务的两级重叠；
FP16 解码、aligned-depth bridge 和地面几何均建立了独立 parity 门并提供默认关闭的 Native 路径。当前结果只支持
`SM-S9280 / SM8650 / Android 16` 上的部署和性能诊断，不新增准确率、产品或安全 authority。

## R0：流水化重叠

实现约束：

- QNN executor 与 geometry executor 各最多一个运行任务；
- 三槽 owned YUV pool 与三槽 aligned-depth pool；
- 每级 one-running/one-replaceable-pending，过期或关闭结果释放 owned slot；
- phase-locked cadence 使用固定 deadline grid，错过 deadline 时跳过而不追帧；
- canonical 串行路径继续保留。

45 秒饱和 A/B：

| 路径 | 前 40 秒完成 | 吞吐 | full P95 |
|---|---:|---:|---:|
| serial | 228 | 5.700 Hz | 181.39 ms |
| pipeline | 367 | 9.175 Hz | 185.43 ms |

流水化吞吐提高 61.0%，但单帧延迟没有同比下降，符合重叠机制预期。

## R4 提前执行：频率矩阵与持续运行

首轮相对上次接纳帧节流在 5 Hz 仅达到 4.04 Hz，因此 R0 矩阵诚实失败并保留。修复为
phase-locked deadline 后，2/3/4/5 Hz 实测为 `2.00/3.00/4.00/4.9818 Hz`，R1 矩阵通过；
5 Hz 时 316/316 geometry `VALID`、thermal 0。

canonical Kotlin decode 的 10 分钟 R1：

- 5.00 Hz，3018 processed，全部 `VALID`；
- QNN P95 95.22 ms，decode+align P95 44.53 ms，geometry P95 121.25 ms；
- full P50/P95/max 223.06/257.69/301.71 ms；
- thermal before/max/after 0/2/2；11 GC / 278 ms；
- PSS endpoint +42,518 KiB，native heap endpoint +49,905,104 bytes；endpoint 不证明 leak slope。

这组结果推翻了目标文件中 120-170 ms 的流水化工程估计：5 Hz 持续 CPU 竞争会抬高
canonical Kotlin geometry 延迟。

## R1：bit-exact Native FP16 decode

前两次失败均保留：一次未设置 quiet NaN bit（1022 mismatches），一次错误 canonicalize NaN
（2045 mismatches）。最终实现保持 sign/payload 并强制 quiet bit，对全部 65,536 个 half bit
pattern 与 Android `Half.toFloat` raw bits 完全一致，mismatch 0。

10 分钟 R2（Native decode、canonical Kotlin geometry）：

- 5.00 Hz，3025 processed，全部 `VALID`；
- decode P95 2.28 ms，decode+align P95 13.94 ms；
- geometry P95 124.14 ms；
- full P50/P95/max 219.31/231.67/274.35 ms；
- thermal max 2；11 GC / 282 ms；无 material memory/GC regression。

相对 R1，decode+align P95 改善 68.7%，full P95 改善约 10.1%；长期瓶颈转为 geometry。

## R2：严格等价 Native C++ geometry

Native 端逐项复刻 lower ROI、stride、确定性 cap、Java `Random` 48-bit LCG、240 次 RANSAC、
归一化残差、Jacobi 最小特征向量、quantile 和所有拒绝原因。canonical Kotlin 仍为 authority，
Native 只通过显式开关启用。

R1 真机 parity 覆盖真实 DA V2 深度、0.75/1.25 缩放、周期 NaN、ROI 条纹缺失、确定性
微扰和两个不足候选拒绝样例。6 个有效样例最大字段误差最坏为 `2.22e-16`，两个拒绝样例
reason 完全一致。热状态 2 下的 100 次诊断基准：Kotlin P50 83.50 ms、Native P50
12.23 ms，约 6.83x；不与冷态 R0 基准混作严格 A/B。

30 秒完整相机短跑：

- 5.00 Hz，175/175 `VALID`；
- QNN P95 93.51 ms，decode+align P95 14.05 ms；
- Native geometry P50/P95 14.08/17.56 ms；
- full P50/P95 112.38/119.88 ms，fresh age P95 162.86 ms；
- thermal max 0，无 owned resource leak 或 runtime failure。

第一次 10 分钟 R3b 虽完成 5.00 Hz 和 3025/3025 `VALID`，但运行中本地 test APK 被重建，
旧 runner 在结束时误绑定了不同 APK 哈希，因此证据目录已写入 `INVALID_RECEIPT.md`，不得
作为正式持续门。修复 runner 为安装前锁定 APK 哈希后，固定 APK 的正式 R3c 已通过：

- 5.00 Hz，3024/3024 `VALID`，`UNKNOWN=0`；
- QNN P50/P95 91.47/96.01 ms；
- decode/resize/decode+align P95 2.25/11.50/13.69 ms；
- Native geometry P50/P95 15.40/17.80 ms；
- full P50/P95/max 116.39/123.18/134.27 ms，fresh age P95 135.74 ms；
- thermal before/max/after 0/2/2；11 GC / 275 ms；
- YUV/aligned-depth pool 均 3/3 归还，runtime failures 0；
- device-installed test APK 与收据均为
  `2FC970AAEF67A993ED201775D32D6B28389CE5E5AED2BDE35AD1CE1A5340355E`。

相对 R2（Native decode + Kotlin geometry），full P95 从 231.67 ms 降至 123.18 ms，改善
46.8%；geometry P95 从 124.14 ms 降至 17.80 ms，改善 85.7%。这些比较来自同一设备的
独立 10 分钟运行，不是交错 paired A/B，因此只作 device performance evidence。

## R4：QNN output 到 Native geometry 的 direct-depth bridge

R4 保持 QNN output、align-corners 语义和几何合同不变，把 QNN FP16 direct output 在 Native
线程内解码到 thread-local workspace，并使用预计算映射写入 owned direct aligned-depth slot；
geometry executor 随后直接读取该 slot。它消除了 Java raw-depth `FloatArray` 和 Java aligned-depth
`FloatArray`，但仍保留 Native decoded workspace 与 owned direct aligned buffer，不能称为全链路
zero-copy。

R2 parity 使用真实 QNN output 加覆盖全部 65,536 half bit pattern 的 tiled fixture，共检查
614,400 个 aligned outputs：finite raw-bit mismatch 0、non-finite class mismatch 0、最大数值误差 0；
geometry 字段误差 0，拒绝 reason 完全一致。paired microbenchmark 中 staged/direct P50 为
`9.924/9.547 ms`、P95 为 `10.015/9.680 ms`，direct 约快 3.95%；该微基准只解释 bridge。

固定 APK 的正式 10 分钟 R4：

- 5.00 Hz，3026/3026 `VALID`，`UNKNOWN=0`；
- QNN P50/P95 `90.11/94.82 ms`；direct bridge P50/P95 `6.82/12.83 ms`；
- Native geometry P50/P95 `15.51/17.88 ms`；
- full P50/P95/max `114.01/120.16/130.66 ms`，fresh age P95 `132.67 ms`；
- thermal before/max/after `0/2/2`；12 GC / 274 ms；
- YUV/direct aligned-depth pool 均 3/3 归还，runtime failures 0；
- device-installed app/test APK SHA-256 与安装前收据分别一致为
  `9FAD90EFFE90BD2416DA8AB894125836BDDEB06FE2492F5B2D112281CB67B04F` /
  `DD47063C08657F970BD437AB406CB08C90067526E3F57EBE3BCEC12A5D40FEBD`。

相对独立 R3c，full P95 从 123.18 ms 降到 120.16 ms（约 2.5%）；这不是交错 paired A/B，
不能把差值全归因于 direct bridge。R4 的强结论是等价路径已在固定 APK 下持续 10 分钟通过，
并消除了两份 Java 大数组，而不是建立产品或安全收益。

## 证据

- pipeline saturation：
  `artifacts.local/evidence/hftf/dav2-pipeline-r0-saturation-{pipeline,serial}-20260805/result.json`
- R0 failed matrix SHA-256：`0CD04764FC5556D0AA10A16B41CCF2FADB39B534E6D86BE8ADD09C81A3B6BE26`
- R1 passed matrix SHA-256：`B996BE1B9CFFC4C23F8747699CB81FAC75507AB76978CC6E50C7C1F46E2A1AA6`
- R1 sustained result SHA-256：`54F79418FB9038094BCA3501AD6ED40B2437D666B41E2F211D04252844495111`
- FP16 full-domain parity SHA-256：`46AEE2D0B818499481EB81699CC1678B41EA3B5CC03E48AD836688A68389BE30`
- R2 sustained result SHA-256：`AE68EAB0AB7A16CCA81C239D12DE8393077789D662229D4FF1A9F096DE4CE013`
- Native geometry R1 parity SHA-256：`D8DA3CC856B6D9540FB03F622A2516AD5A8DDB7C567C83A1CEA52E4F0026EF4D`
- Native geometry short camera SHA-256：`020DF4AFFA1C15DE5F7C64B988BD311AA6220BB6C612130E9248D5A4A535679E`
- Native geometry R3c sustained result SHA-256：`3F9FFCE6B424E44356F0A16D312DE37715CAA3161D346D26373A12C4D0E87311`
- Native geometry R3c sustained gate SHA-256：`33225988C60C0F45CE90A3F384FD9473EE7A3A0A036C90D448289516B9535DBF`
- direct-depth R2 parity SHA-256：`E9A0F4132DEB8A16020A782524785DDB0650AEC44238ED240D45F835A51EEE65`
- direct-depth R4 sustained result SHA-256：`F04760F3F3F7970DEA729D88B714D357FFDC21102C79D7FBB33A8C2198EB37FD`
- direct-depth R4 sustained gate SHA-256：`D0E8C3CB330C1F4F5F5F85AB841B430822BBFC4BD22E651CD2DF44847FA601A4`

## 边界与下一步

R3 已对同一 cached DLC 完成 QNN `detailed` per-op 与 HTP `linting` profiling；Transformer
attention 主干占约 88%，reshape+transpose 仅 3.68%。详见
`DAV2_QNN_OPERATOR_PROFILE_R0_RESULT_2026-08-05.md`。profiling overhead 不解释为 App
latency。模型变体仍因独立准确率/false-clear gate 缺失而 HOLD；canonical FP16 路径始终保留。
