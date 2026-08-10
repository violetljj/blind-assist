# TARO O0R source-adapter runtime

状态：`IMPLEMENTATION_LOCK_PASS / SYNTHETIC_ONLY / SOURCE_IO_FALSE / SCIENTIFIC_STATUS_NOT_RUN`

该 Module 只实现
`TARO_O0R_ARKITSCENES_SOURCE_AND_ADAPTER_CONTRACT_LOCK` 冻结的纯内存 mechanics；没有
downloader、archive/path reader、materializer、DepthART runner、scientific evaluator 或 artifact writer。

## 稳定 Interface

- `decimal_timestamp_ns`、`interpolate_camera_to_world_exact`：以 `Decimal` 选择 exact-nanosecond
  pose bracket，再执行 official inverse-trajectory interpolation；future right bracket 进入因果水位；
- `build_source_frame_receipt`、`build_query_receipts`：source receipt 必须命中冻结的
  `8 ADAPTER_FIT + 16 O0R_EVAL_CANDIDATE` role/visit/video roster，并绑定 exact member/stem、bytes、
  SHA-256、CRC32、decoded-payload content、K、pose 与水位；9 个 query receipt 只能由同帧、只读、
  whole-hash-bound FARO support geometry 生成，且不冒充完整 P0 receipt；
- `fit_uncertainty_model`：唯一输入是 roster-bound `ADAPTER_FIT` source frame 及已加载 FARO、
  AppleDepth、confidence arrays；scale/boundary pixel residual 与 support frame residual 均在 Module 内推导，
  不接受 caller-computed residual。support observation 每唯一物理帧只计一次，exact 8-parent roster、重复
  receipt/frame、128 actual-observation 与 4-parent gate 均 fail closed；返回模型是私有 factory-bound
  identity，cells/counts 深冻结且 resolve 时重验 roster、SHA、cell schema/value/count；
- `derive_faro_geometry`、`build_truth_query_factor_frame`：从绑定 FARO/K/gravity 构造 model-free
  `SCALE=0` truth、support 与 sparse boundary evidence，并把 common point IDs/FARO evidence 放入独立、
  sealed、immutable base geometry；FARO truth 只接受冻结 eval role，query frame 会从 FARO support plane
  重算，S/P/B value hashes 进入 base evidence；
- `build_candidate_depth_output_receipt`、`build_candidate_query_factor_frame`：从不同 candidate-depth
  raster 在同一 immutable base/common support 上运行同一 TARO-specific S/P/B extractor。完整 output receipt 固定
  `depthart-s-metric-indoor-448-official-fp32`、checkpoint、RGB/K/source/output hash 与
  `baseline_log_metric_scale=0`；不接受裸 scale、FARO alignment 或 caller-signed correction；
- `reduce_query_factor_frame`、`reduce_complete_query_bundle`：只使用
  `taro_query_reducer_p0_contract_v1`，强制 9/9 receipt/frame binding。knownness 读取 base coverage 与
  SUPPORT validity，不从 BOUNDARY 重建 support；BOUNDARY 只贡献 obstacle/boundary localization；
- `inject_factor_blocks`：覆盖 8 arms × 2 modes。只接受 genuine candidate-extractor baseline 与
  FARO-truth oracle；VALUE_ONLY 只换 `value`，FULL_BLOCK 才换 value/validity/uncertainty，未命名 block、
  immutable base geometry 与逐组件 parent-frame lineage 都必须保持 hash-bound；factorial reducer 还必须
  同时收到真实 baseline/oracle parent context 并重建 exact injection hash。

公开 API 不提供 factor-frame reseal、caller residual builder 或 self-signed truth-only candidate-scale
correction。真实 candidate-relative SCALE 必须等独立、已提交的 truth-only result/one-shot lock 后，
在后续 runner 中绑定 actual truth bundle、candidate/FARO arrays 与 frozen common support；当前能力不存在。

## 数值与失败语义

- scalar JSON 固定 round-12、`-0` 归零、NaN/Infinity 拒绝；arrays 以 dtype/shape/content receipt 进入
  canonical hash；nested factor schema、provenance、uncertainty scope 与 content seal 均重算；
- query segment 为 0.20–2.00 m；signed clearance 是 sparse obstacle/boundary point 到有限 path segment
  的 support-plane distance 减 0.30 m；`lower > 0.05` 才 CLEAR，`upper <= 0` 才 OCCUPIED；
- receipt/roster/asset/hash/base geometry/knownness/uncertainty 任一失败均 fail closed；UNKNOWN 从不当 negative。

## 输出

稳定接口只返回进程内 receipt、geometry、uncertainty model、factor frame 与 reducer result；不写文件、
不创建 scientific artifact，也不把 synthetic test object 登记成 source/truth evidence。

## 安全边界

- Module 没有 downloader、archive/path reader、materializer、DepthART runner、scientific evaluator、trainer
  或 artifact writer；
- 24 个 selected source body 未打开；`artifacts.local/` 下冻结的 future dataset/work/truth/O0R roots 必须
  在 implementation lock 时全部不存在；
- synthetic PASS 只支持纯内存 interface/mechanics，不认证真实 decoder、inference receipt、truth、模型、
  wearable mount、active observation、device、product 或 safety。

## Synthetic canary

`test_source_adapter.py` 的 44 项 focused tests 全部只构造内存 synthetic arrays/identities，覆盖 exact
timestamp、right-bracket watermark、roster/asset/decoded-content binding、FARO/eval role 与 query-plane
binding、内部 residual 推导、frame-level support observation 计数、parent-macro uncertainty/fallback、
FARO 与不同 candidate depth 的真实 extractor、deep-read-only whole geometry hash、immutable base/sparse
boundary、固定 candidate output receipt、极端 scale fail-closed、CLEAR/OCCUPIED/UNKNOWN、genuine 9-query
truth bundle、8×2 injection、逐组件 lineage/parent context、nested mutation、canonical replay，以及移除
caller residual、caller-constructed/duck-typed uncertainty、public reseal 与 self-signed scale capability。

测试没有读取合同中的 24 个 source body，没有物化真实 truth bundle，也没有运行 DepthART 或 O0R。
Implementation PASS 只关闭纯内存 interface/mechanics blocker；唯一 successor 是另行提交的
`TARO_O0R_ARKITSCENES_TRUTH_ONLY_ONE_SHOT_PREFLIGHT_LOCK`，该锁本身仍不得下载、打开或执行。

## 停止条件

任一 contract/hash/role/decoded-payload/query/geometry/uncertainty/factor-lineage 绑定漂移，任一 focused test
失败，或任一冻结 future root 已存在，implementation lock 必须 fail closed；不得用自签 receipt、降 gate、
删 UNKNOWN 或先看 DepthART/source outcome 绕过。
