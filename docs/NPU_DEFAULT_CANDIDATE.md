# NPU 正式设备能力路由

状态：`current / GATE_POLICY_V3 / PROMOTED_WITH_CPU_FALLBACK`

策略 ID：`blindassist_detector_backend_policy_20260727_v3`

## 当前决定

- 正式 `com.linnan.blindassist` 已采用设备能力路由：
  `SM8650 + arm64-v8a + live QNN HTP FP16 capability` 选择
  `QUALCOMM_QNN_HTP`，其他设备和能力/委托初始化失败路径选择
  `CPU_XNNPACK`。
- `QUALCOMM_QNN_HTP` 是受支持设备的生产首选；`CPU_XNNPACK` 是正式兜底，
  当前没有下一默认候选。
- `GPU_DELEGATE` 只保留为 benchmark comparator，不再与 NPU 并列作为默认候选。
- 只有“阻断门”参与 `candidatePromotionReady`。没有预先冻结阈值的发布测量和
  诊断指标不得在看过结果后升级为一票否决项。
- 独立 `:npu-candidate` App 继续保留为 fail-closed 复现实验包；正式 App 通过
  manifest provider 接入 `ProductionQnnRoutingObjectDetectorProvider`，不会把
  CPU fallback 记成 NPU 成功。
- 候选使用 Qualcomm QNN runtime/LiteRT delegate `2.47.0`、arm64-only；
  不允许初始化失败后回落 CPU。

代码真源：

- `core/vision/src/main/java/com/linnan/blindassist/vision/DetectorBackendPolicy.kt`
- `core/vision/src/main/java/com/linnan/blindassist/vision/TfliteYoloDetector.kt`
- `core/vision/src/main/java/com/linnan/blindassist/vision/RuntimeObjectDetectorFactory.kt`
- `core/vision/src/main/java/com/linnan/blindassist/vision/ProductionDetectorRoutePolicy.kt`
- `app/src/main/java/com/linnan/blindassist/vision/ProductionQnnRoutingObjectDetectorProvider.kt`
- `npu-candidate/`

## 为什么选择 NPU

设备为 Samsung SM-S9280 / Snapdragon 8 Gen 3（SM8650）。三后端使用相同
YOLO11n FP16 320 模型、相同输入和生产解码/NMS/风险路径。

| 协议 | CPU | GPU | QNN HTP NPU |
| --- | ---: | ---: | ---: |
| 100图完整检测 P50/P95 | 53/55 ms | 26/31 ms | **12/15 ms** |
| 10分钟完整检测 P50/P95 | 53/57 ms | 33/58 ms | **16/21 ms** |
| 90帧 SANPO 完整检测 P50 | 53 ms | 27 ms | **12 ms** |
| 10分钟失败/热保护 | 0/0 | 0/0 | **0/0** |

正式 `PRODUCTION_ROUTE` 复验中，NPU 在100图风险/稳定风险/反馈/事件状态上与
CPU `100/100` 一致，
在90帧连续事件的风险、反馈、事件状态和运行时事件 ID 上与 CPU `90/90`
一致。正式 App 日志显示 QNN graph finalize `status 0x0`，随后记录
`Detector ready backend=qualcomm_qnn_htp` 和
`ProductionDetectorRoute: route=qualcomm_qnn_htp`；没有 CPU fallback。

这些结果证明同设备上的延迟和持续性能优势，不证明能耗优势。持续测试期间设备
连接 USB 且接近满电，没有外部功耗仪数据。

## 晋升门

### 门的分类规则

- `BLOCKING`：只允许直接关系到运行完整性、关键风险、提醒行为、持续稳定、
  设备路由和回滚的项目。任一项非 `PASS` 才能阻止默认切换。
- `RELEASE_CONSTRAINT`：包体、启动时间和延迟等发布测量。只有在测量前已经冻结
  明确上限时才有否决权；当前结果只能要求披露和后续优化。
- `DIAGNOSTIC`：逐框数值差异、能效观察等归因/监控项，不自动决定生产后端。
- `NOT_EVALUATED` 只表示未知。只有缺少某项阻断门必需证据时才会阻塞，不能把
  任意未知项默认为失败。

### 阻断门

| 门 | 状态 | 依据或阻塞 |
| --- | --- | --- |
| NPU runtime 完整性 | `PASS` | QNN 2.47 graph finalize 成功，runtime marker 为 `qualcomm_qnn_htp`，无 CPU fallback |
| 关键风险不回退 | `PASS` | 有界100图风险/反馈100/100对齐 CPU，未观察到关键事件漏报增加 |
| 提醒生命周期不回退 | `PASS` | 90帧 recall=1、重复提醒=0、身份重建=0、PASSED最终退出2/2 |
| 持续运行稳定 | `PASS` | 正式路由10分钟5938帧、0失败、无 thermal throttle |
| 目标设备与 CPU 路由 | `PASS` | SM8650/arm64 且 live QNN HTP FP16 能力成立才走 NPU；不支持、API 26–30、能力检查失败或 delegate/graph 初始化失败均记录原因并走 CPU |
| 回滚完整性 | `PASS` | CPU 基线 APK 可按精确 SHA-256 恢复并冷启动，随后可恢复 NPU 正式包；当前用户自有状态文件为0，数据保留子项 `NOT_EVALUABLE`。两项会随版本启动变化的 Android/ProfileInstaller 标记已单独披露 |

全部阻断门通过，`candidatePromotionReady=true`。这不把逐框完全一致、事后
发明的包体阈值或无外部功耗仪的能效结论升级为晋升门。

### 发布约束与诊断

| 项目 | 分类/状态 | 当前观测 |
| --- | --- | --- |
| 完整链路延迟 | `RELEASE_CONSTRAINT / OBSERVED` | NPU 12/15 ms，CPU 53/55 ms |
| 包体 | `RELEASE_CONSTRAINT / OBSERVED` | 最终正式集成 debug 113,293,303 bytes；拉取的 CPU 基线56,143,576 bytes；此前无冻结上限 |
| 冷启动 | `RELEASE_CONSTRAINT / OBSERVED` | SM8650 最终正式集成冷启动1,141 ms；此前无冻结上限 |
| 逐框检测差异 | `DIAGNOSTIC / OBSERVED` | 严格等价86/100；14图均已归因且风险/反馈100/100不变 |
| 能效 | `DIAGNOSTIC / NOT_EVALUATED` | USB测试不能建立功耗结论；记录未知但不自动否决 |

## 证据绑定

- `artifacts.local/evidence/npu-production-route-acceptance/20260727-232243/summary.json`
  - SHA-256：
    `47A3A19A65402D74B43826B6B2144CE4534B44128C7E1B1171CA2F487C014D7B`
  - 最终正式 APK：
    `A1BD48CBDC41000477183BB8579725A9039818F4489563A9F7DE3643B966FDD5`
  - 100图正式路由报告：
    `861733CB278C46E963D760A9F17ADEC16BF8BF0D4B5FC225CDCE001D60044537`
  - 90帧事件报告：
    `4E74D8E08DA40002169762712E19E6F0F002D2AE8968375880EF103C4D51429A`
  - 600秒稳定性报告：
    `961C33C1B285625F73B5782D6BD8BC366B22FD70C7E65383C18B79B17C0024BC`
  - 正式风险/事件报告绑定最终 APK；稳定性报告绑定仅在回滚收据说明字符串上
    不同、路由与 provider 代码相同的前一正式 APK，差异已在 summary 中披露。
- `artifacts.local/evidence/cpu-gpu-npu-full-pipeline/20260727-100-images/full-pipeline.json`
  - SHA-256：
    `5105B4D3CCC2CE437CBC4D4BA3A7093E38469B65D67FC000BD5265D892BF86BA`
- `artifacts.local/evidence/cpu-gpu-npu-full-pipeline/20260727-qnn247-100-images/full-pipeline.json`
  - SHA-256：
    `5B59AB2E21AADE9B38CF0A174F5A31DD82AFED2678F5B85F91BA4B5C7A02404E`
  - NPU差异归因：7个 CPU 检测在 NPU 侧缺失（CPU 置信度
    `0.353–0.371`，集中在阈值附近）、6个框 IoU 低于严格 `0.95` 门、
    3个同类框置信度差超过 `0.03`；14图风险/反馈均未改变。
- `artifacts.local/evidence/cpu-gpu-npu-full-pipeline/20260727-100-images/soak-npu.json`
  - SHA-256：
    `E88E0280E0E4E54356E7D1C51F680DBCFB610E6A5A097435CD7FC0255541BB4C`
- `artifacts.local/evidence/cpu-gpu-npu-sanpo-event-lifecycle/20260727-90f/sanpo-event-lifecycle.json`
  - SHA-256：
    `9A6DC20EC1E443CBF08E8D93E11A935E2B5D4A2E48F51D41A83DF6AF58A64CF8`
- `artifacts.local/evidence/cpu-gpu-npu-sanpo-event-lifecycle/20260727-qnn247-90f/sanpo-event-lifecycle.json`
  - SHA-256：
    `A426F8AA765806E97DD5256DFC40C0B5FCBF9F7BDC7AB3ED075996F188403BCD`
- `artifacts.local/evidence/npu-candidate-acceptance/20260727-230057/summary.json`
  - 候选 APK SHA-256：
    `FA0D132817924559C80C2C3C80EE3714DAE9C8DB93750A7FEB5898850D8596B6`
  - 终态：`PASS_CANDIDATE_INSTALL_COEXISTENCE_AND_ROLLBACK`

本地 evidence 不进入 Git；复现实验必须重新绑定模型、数据、设备和报告哈希。

## 后续规则

1. 新增 SoC/ABI 必须先进入独立候选验证，再扩展
   `ProductionDetectorRoutePolicy`；设备名字或营销型号本身不授权 NPU。
2. 正式 provider 的 CPU fallback 日志属于正常、可审计路由，不得改成静默回退。
3. 逐框差异、包体、冷启动和能效继续作为发布说明与优化指标跟踪。
4. 若要新增硬阈值，必须在新测量前冻结阈值、适用设备和失败处置，不能根据
   已看到的结果反向设置门槛。

本策略授权上述设备能力后端路由，不授权安全、真实用户或独立助行结论。
