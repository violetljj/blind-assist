# NPU 下一默认候选策略

状态：`current / NEXT_DEFAULT_CANDIDATE / PRODUCTION_DEFAULT_UNCHANGED`

策略 ID：`blindassist_detector_backend_policy_20260727_v1`

## 当前决定

- 生产默认后端继续使用 `CPU_XNNPACK`。
- `QUALCOMM_QNN_HTP` 是唯一的下一默认候选。
- `GPU_DELEGATE` 只保留为 benchmark comparator，不再与 NPU 并列作为默认候选。
- NPU 只有在全部晋升门为 `PASS` 后才能成为生产默认；`HOLD` 和
  `NOT_EVALUATED` 均按未授权处理。
- 外部 delegate 注入仅允许 `com.linnan.blindassist.benchmark` 基准包；App 包
  即使显式请求候选后端也会 fail closed。

代码真源：

- `core/vision/src/main/java/com/linnan/blindassist/vision/DetectorBackendPolicy.kt`
- `core/vision/src/main/java/com/linnan/blindassist/vision/TfliteYoloDetector.kt`

## 为什么选择 NPU

设备为 Samsung SM-S9280 / Snapdragon 8 Gen 3（SM8650）。三后端使用相同
YOLO11n FP16 320 模型、相同输入和生产解码/NMS/风险路径。

| 协议 | CPU | GPU | QNN HTP NPU |
| --- | ---: | ---: | ---: |
| 100图完整检测 P50 | 53 ms | 21 ms | **12 ms** |
| 10分钟完整检测 P50/P95 | 53/57 ms | 33/58 ms | **15/19 ms** |
| 90帧 SANPO 完整检测 P50 | 53 ms | 27 ms | **12 ms** |
| 10分钟失败/热保护 | 0/0 | 0/0 | **0/0** |

NPU 在100图风险/反馈上与 CPU `100/100` 一致，在90帧连续事件的风险、反馈、
事件状态和运行时事件 ID 上与 CPU `90/90` 一致。设备日志显示 QNN HTP
`548/548` 节点完整委托。

这些结果证明同设备上的延迟和持续性能优势，不证明能耗优势。持续测试期间设备
连接 USB 且接近满电，没有外部功耗仪数据。

## 晋升门

| 门 | 当前状态 | 依据或阻塞 |
| --- | --- | --- |
| QNN HTP 全图委托 | `PASS` | 548/548 节点，单分区 |
| 完整链路延迟 | `PASS` | 100图、10分钟和90帧均领先 |
| 风险与反馈一致性 | `PASS` | 100/100；事件序列90/90 |
| 10分钟稳定性 | `PASS` | 5957帧、0失败、无 thermal throttle |
| 检测集合一致性 | `HOLD` | 100图中检测集合仅86/100达到严格等价 |
| 冷启动与包体 | `HOLD` | 初始化约1.3秒；完整 QNN benchmark APK 不可直接用于发布 |
| 能效 | `NOT_EVALUATED` | USB测试不能建立功耗结论 |
| 共享事件生命周期 | `HOLD` | 2次事件身份重建；PASSED事件运行时退出0/2 |

只要任一门不是 `PASS`，`candidatePromotionReady` 必须保持 `false`。

## 证据绑定

- `artifacts.local/evidence/cpu-gpu-npu-full-pipeline/20260727-100-images/full-pipeline.json`
  - SHA-256：
    `5105B4D3CCC2CE437CBC4D4BA3A7093E38469B65D67FC000BD5265D892BF86BA`
- `artifacts.local/evidence/cpu-gpu-npu-full-pipeline/20260727-100-images/soak-npu.json`
  - SHA-256：
    `E88E0280E0E4E54356E7D1C51F680DBCFB610E6A5A097435CD7FC0255541BB4C`
- `artifacts.local/evidence/cpu-gpu-npu-sanpo-event-lifecycle/20260727-90f/sanpo-event-lifecycle.json`
  - SHA-256：
    `9A6DC20EC1E443CBF08E8D93E11A935E2B5D4A2E48F51D41A83DF6AF58A64CF8`

本地 evidence 不进入 Git；复现实验必须重新绑定模型、数据、设备和报告哈希。

## 落地顺序

1. 保持 App 默认 CPU，QNN 依赖只存在于 `device-benchmark`。
2. 修复共享事件身份重建和 PASSED 退出问题，并在同一90帧序列复验。
3. 对置信度阈值附近的检测差异做逐框归因，不以风险结果相同替代检测等价。
4. 建立 arm64-only 候选打包方案，测量冷/暖初始化、缓存和安装体积。
5. 使用外部功耗仪或可靠设备能耗接口完成同工作量能效比较。
6. 全部门通过后，另行发起生产默认切换；该切换必须有独立代码审查、release
   APK验证和回滚路径。

本策略只授权候选工程和验证，不授权安全、真实用户或独立助行结论。
