# TARO O0R ARKitScenes source-and-adapter contract lock

状态：`TARO_O0R_SOURCE_AND_ADAPTER_CONTRACT_LOCK_PASS / SCIENTIFIC_STATUS_NOT_RUN`

## 结论

TARO 已不再停留在“候选数据源字段看起来可能有用”的泛化映射。本合同冻结一条可证伪、
outcome-blind 的真实 O0R 前门：使用 ARKitScenes `Training` 中全新且 visit-disjoint 的
`8 ADAPTER_FIT + 16 O0R_EVAL_CANDIDATE` parent，FARO 注册高分辨率深度只负责
scale/support/boundary 与 body/path query truth，DepthART-S 只作为后续冻结 baseline，模型输出不得
参与 source、frame、truth 或 query 选择。

这次只锁协议，没有下载 24 个 source payload、没有打开这些 identity 的 RGB/depth/FARO body、
没有运行 DepthART、没有物化 truth、没有执行八臂 factorial，也没有建立真实 causal headroom。

## 为什么选择这条源路线

ARKitScenes 官方数据格式提供逐 capture 的 RGB、毫米深度、confidence、pinhole intrinsics、
timestamped trajectory、ARKit mesh 与可选 FARO laser geometry；depth-upsample 子集又给出同 timestamp
的 AppleDepth/FARO 对照。仓库已有 hash-bound license、下载语义、truth reader、registration receipt
和 deterministic reducer，因此本版本只补 TARO 缺失的完整 factor/query truth、连续 boundary、
uncertainty calibration、fresh roles 与 deterministic injection contract。

Aria Digital Twin 当前仍缺可靠 parent/site 独立性与 body/query truth；ScanNet++ 虽有高精度 mesh，
但需要新的许可/下载链且默认体量远大于本路线。二者继续保留候选，不作为本合同的组合通过条件。

## 冻结角色与防火墙

- `ADAPTER_FIT`：8 个独立 visit，只允许拟合 source residual/uncertainty cells；模型输出和 task metric
  均禁止。
- `O0R_EVAL_CANDIDATE`：16 个独立 visit，必须先完成 truth-only admission。至少 12 个 parent、
  truth-clear/truth-occupied 各至少 6 个 parent、每 parent 至少 12 个 exact-timestamp frame 才可继续。
- 全部 source-eligible parent 保留；看到任何模型输出后不得丢弃、替换或重分配。
- B1 consumed Selection、Calibration、Confirmation 和历史 O0M artifact 均不复用。

## Truth 与 injection

- source-frame receipt 绑定 exact decimal timestamp、逐帧 K、bounded pose interpolation、gravity、
  visit/video identity；`max_source_timestamp_ns` 必须包含实际右侧 pose bracket，任何缺口都变为
  `UNKNOWN / NOT_ADMITTED`。
- query 固定为 P0 capsule profile、2 m horizon、三个 lateral offset × 三个 yaw，共 9 个 query/frame；
  每个物理帧先生成一个 immutable base receipt，再生成 9 个 query-bound receipt，缺一即整帧不准入。
- scale 的 truth-only 值是 model-free FARO absolute metric reference；只有 truth-only result 签署后，
  才能在预冻结 common support 上计算 candidate-relative `median log(FARO/candidate)`。support、boundary
  value 同样只由 FARO geometry 建立；uncertainty 只由 8 个 fit parent 的
  confidence/range-conditioned real residual distribution 建立，禁止 constant sigma 和 teacher truth。
- 八臂只允许按 factor block 做确定性 copy-and-patch；K corruption 继续是独立负控。
- primary 是 `SCALE_SUPPORT_BOUNDARY versus NONE` 的 parent-macro paired interval-score improvement；
  95% LCB 必须 `> 0.02 m`，同时 false-clear UCB 与 known-coverage LCB 通过 non-inferiority。

## Implementation seam 冻结

本合同在提交前补齐以下接口缝，避免由 implementation 事后猜测：

- O0R source characterization 输出独立的 `source_frame_receipt.v1` 与 `query_receipt.v1`，不冒充完整
  P0 `TaroFrameReceipt`。ARKitScenes 没有 source-native pose/IMU covariance、sparse tracks 或真实
  camera-body transform，禁止用零值/默认值补造；因此本阶段也不建立 P0 metric-anchor independence。
- 手持 iPad 只定义 virtual ground-frame query：原点为相机中心在 FARO support plane 的投影，forward
  为光轴在平面上的投影，lateral 由 forward×gravity 得到。这是 source-characterization corridor，
  不是佩戴者 body truth 或穿戴式 mount claim。
- FARO/RGB 使用 `1920×1440`，AppleDepth/confidence 使用 `256×192`；intrinsics 按 pixel-center
  公式精确放大 `7.5×`。AppleDepth 只用于 ADAPTER_FIT residual，9-query truth 始终在 high-res FARO
  frame 上计算。
- signed boundary 固定为 obstacle 外正、内负、边界零；support removal、valid depth、local completeness、
  virtual capsule、query knownness 与三态 interval rule 均在机器合同中给出数值。
- uncertainty cell 的边界归属、parent-macro q95、range→confidence→global fallback、support bootstrap、
  12 位 canonical float/hash 规则均已固定；全局仍不足即 uncertainty invalid/query UNKNOWN。
- 旧 `arkitscenes_truth_reader.py` 只允许复用 pose/K/unprojection/ground primitives；其 float timestamp、
  `load_manifest_frame()` 和三带 `derive_assistive_truth()` 不得成为 TARO 高层入口。旧 GeometryR2 reducer
  只是 reference，TARO 必须实现新的 `taro_query_reducer_p0_contract_v1`。
- denominator 依次固定为 exact-timestamp frame、source-eligible frame、9/9 complete admitted frame；
  任一分母 undefined 都是 `FAIL_NOT_DROP`。clear/occupied parent 各要求至少 6 个不同 admitted frame
  命中相应状态，同一 parent 可同时提供两类支撑。

## 当前权限与唯一 successor

当前 execution authority 仍为 false。唯一 successor 是：

`TARO_O0R_ARKITSCENES_SOURCE_ADAPTER_IMPLEMENTATION_LOCK`

它只允许在新 TARO Module 中实现并静态测试 receipt、truth、uncertainty 与 injection adapter；在实现
hash 和独立 truth-only one-shot preflight lock 提交前，不得下载 source payload、物化 truth、运行
DepthART 或执行 O0R。

## Claim ceiling

本结果只建立 pre-outcome source/role/adapter 合同。没有真实 O0R 结果，不证明 GaugeFix、PARA、
穿戴式主动观察、设备、产品或安全有效性；默认 App 与其他研究路线不变。
