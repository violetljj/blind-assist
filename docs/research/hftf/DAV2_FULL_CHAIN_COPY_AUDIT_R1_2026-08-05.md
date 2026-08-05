# DA V2 全链路 copy / materialization 审计 R1

日期：2026-08-05

## 结论

已对三条可严格证明且可独立做 parity 的路径建立 Native/direct 实现：FP16 output 的 Kotlin half
decode 改为 bit-exact Native decode；RGB bridge 可选 direct buffer，消除明确的
921,600-byte/frame Java byte array 到 Native 的再次复制；direct-depth bridge 消除 Java raw-depth
与 aligned-depth 两份 `FloatArray`。direct RGB 没有稳定延迟收益，继续默认关闭；direct-depth 已通过
固定 APK 10 分钟门。其余大对象不能仅因体积大就称为冗余 copy，其中多项是算法必需 materialization。

## 每帧数据路径

| 阶段 | 大小 | 分类 | 当前结论 |
|---|---:|---|---|
| Camera YUV tight ownership | 460,800 B | lifecycle ownership copy | CameraX 关闭前取得 owned frame；保留 |
| tight YUV -> Native I420 | 460,800 B | converter workspace copy | 已确认，尚未消除 |
| Native RGB output | 921,600 B | required RGB materialization | direct bridge 可避免 Java heap byte array |
| legacy RGB Java -> Native | 921,600 B | definite redundant bridge copy | direct route 已消除 |
| canonical resized HWC FP32 | 4,264,176 B | strict canonical intermediate | 保留，用于严格前处理语义 |
| FP16 QNN input | 2,132,088 B | required tensor | direct buffer，QNN JNI 无显式 memcpy |
| FP16 QNN output | 710,696 B | required tensor | direct buffer，QNN JNI 无显式 memcpy |
| decoded raw FP32 depth | 1,421,392 B | geometry input materialization | direct-depth 路径移出 Java heap，保留 thread-local Native workspace |
| aligned 640x480 depth slot | 1,228,800 B | geometry coordinate contract | direct-depth 路径使用 owned direct slot，跨 executor handoff 无整图复制 |

QAIRT direct input/output buffer 是否在 backend 内部发生隐式 copy 不能由 JNI 源码否定；需要 shared-buffer
注册或 backend profiling 另行建立证据。

## 已通过的门

### FP16 decode

全部 65,536 个 half bit pattern raw-bit parity mismatch 0。短跑 decode P95 从约 35.68 ms 降至
2.30 ms；10 分钟链路中 decode+align P95 从 44.53 ms 降至 13.94 ms。

### direct RGB bridge

四个 rotation 共比较 3,686,400 个 RGB bytes 和 4,264,176 个 FP16 tensor elements，mismatch 0。
100 次 paired benchmark：legacy/direct P50 `8.005/7.977 ms`，P95 `9.005/9.107 ms`；没有可重复
latency gain，因此只保留为 copy-only development candidate。

### direct-depth bridge

QNN FP16 direct output 在 Native 中解码一次，按冻结 align-corners 语义写入 owned direct aligned-depth
slot，再由独立 geometry executor 直接读取。真实 QNN output 与覆盖全部 65,536 half pattern 的 tiled
fixture 共比较 614,400 个 aligned outputs，finite raw-bit/non-finite class mismatch 均为 0，最大误差 0；
geometry 输出也严格一致。paired microbenchmark staged/direct P50 `9.924/9.547 ms`、P95
`10.015/9.680 ms`。

固定 APK 的 10 分钟 R4 达到 5.00 Hz、3026/3026 `VALID`；direct bridge P50/P95
`6.82/12.83 ms`，full P50/P95 `114.01/120.16 ms`，最大热状态 2，pool 3/3 归还、runtime
failure 0。该路径消除了 Java 两份深度数组，但 Native decoded workspace、owned direct aligned buffer
和 backend 内部未知搬运仍存在。

证据 SHA-256：

- direct RGB parity + benchmark：`058F25C00F4421E5C14842F0DDFE07E0034F432A86B5F86B9DC50976CE7A8327`
- direct RGB + Native decode short camera：`34663765C4FBF0ED39FCBAC9E704984D8E40BF8BEDEFC5091BAE99F9ABEF946A`
- direct-depth R2 parity：`E9A0F4132DEB8A16020A782524785DDB0650AEC44238ED240D45F835A51EEE65`
- direct-depth R4 sustained result/gate：`F04760F3F3F7970DEA729D88B714D357FFDC21102C79D7FBB33A8C2198EB37FD` / `D0E8C3CB330C1F4F5F5F85AB841B430822BBFC4BD22E651CD2DF44847FA601A4`

## 未授权推断

- endpoint PSS/native-heap delta 不是 leak slope；
- direct buffer 不自动等于 backend zero-copy；
- parity 与性能改善不证明 depth accuracy、traversability、false-clear 或 safety；
- 不因工程优化结果修改模型、阈值、候选数、RANSAC 次数或数据 authority。
