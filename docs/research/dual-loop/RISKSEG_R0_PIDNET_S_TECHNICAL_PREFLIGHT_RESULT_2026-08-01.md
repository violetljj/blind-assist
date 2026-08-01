# RISKSEG-R0 PIDNet-S 技术预检结果

状态：`COMPLETE / VALID / PIDNET_S_TECHNICAL_PREFLIGHT_PASS /
TRAINING_IMPLEMENTATION_LOCKED / DEFAULT_APP_UNCHANGED`

日期：2026-08-01（Asia/Hong_Kong）

训练实现锁：
[RISKSEG_R0_PIDNET_S_TRAINING_IMPLEMENTATION_LOCK_2026-08-01.json](RISKSEG_R0_PIDNET_S_TRAINING_IMPLEMENTATION_LOCK_2026-08-01.json)

## 结论

唯一候选 PIDNet-S 通过冻结的 `512x288 / four-class / full W8A8` 技术预检，
可以按已授权顺序进入三 seed 正式训练。这个结果只证明模型图和冻结 App 决策链在
SM-S9280 上具备部署起点；未读取 event-eval 模型 outcome，不构成事件效果、默认替换
或独立助行/安全证据。

## Host 与 artifact

- 官方源码 commit：`4c158cf24ce432f0a8cb43364fae38d93cee0dc3`，MIT；
- ImageNet 权重：`PIDNet_S_ImageNet.pth.tar`，38,061,375 bytes，
  SHA-256 `f96e2c96...1a5f359`。官方原 Google Drive URL 已失效，本次从 Zenodo
  record 14606189 恢复同名 byte artifact，并匹配其 size/MD5；不把镜像误写成官方
  当前直链；
- ONNX checker 和 ONNX Runtime parity 通过，最大绝对误差
  `2.7120113e-05`；
- full integer TFLite：7,911,768 bytes，SHA-256
  `d492d050...0c2ddb`；input/output 均为 INT8 NHWC，
  `[1,288,512,3] -> [1,288,512,4]`，float activation tensor 为 0；
- synthetic 与 train-only 非评价 RGB canary 输出有限，argmax 仅为 `0..3`。

## SM-S9280 / QNN

正式 10 分钟运行使用 Samsung `SM-S9280 / SM8650 / API 36`、QNN delegate
2.47.0 artifact（runtime version `[0,24,0]`）和量化 HTP capability。logcat 显示两次：

```text
163 nodes delegated out of 163 nodes with 1 partitions
caching in RESTORE MODE
```

QNN 生成 7,962,624-byte serialized context 并在第二次加载复用。7,619 个 timed
sample 的结果：

| 指标 | 结果 | 门 |
|---|---:|---:|
| inference P95 | 5.244 ms | 诊断 |
| 全链路 P50 | 74.332 ms | 诊断 |
| 全链路 P95 | 75.739 ms | <=100 ms |
| 全链路最大值 | 88.405 ms | 诊断 |
| 初始 2 分钟 P95 | 75.812 ms | 基准 |
| 末 2 分钟 P95 | 76.004 ms | 比值门 |
| 末/初 P95 | 1.00255x | <=1.20x |
| failure | 0 | =0 |
| maximum thermal status | 0 | < severe |

全链路计时包括 RGB resize/ImageNet normalize/INT8 quantize、QNN inference、
INT8 dequantize/finite/argmax、冻结 `TraversabilitySegmentationAnalyzer` 与冻结
`AssistDecisionKernel`/事件/feedback planner。benchmark-only taxonomy 只把新 class
ID `1/2` 绑定到冻结 adapter 的 obstacle/boundary 接口，没有修改冻结 adapter 或规则。

## 装载修复披露

第一次 device attempt 在模型图转换前失败：target debug APK 使用
`extractNativeLibs=false`，HTP V75 skel 位于 APK 内而非 delegate 指定的文件目录。
runner 随后从同一已安装 APK 原样提取 QNN skel 到 App 私有 code cache，再重新运行。
模型、输入尺寸、INT8 精度、算子图、taxonomy 和冻结规则均未变化；修复发生在任何
event-eval outcome 访问和正式训练 step 前。成功运行确认 `163/163` 节点由 HTP
接管，因此不是 CPU fallback 掩盖结果。

## 下一步

训练 implementation lock 已绑定实现 commit、脚本/数据/权重/预检证据哈希和完整
recipe。只允许依次运行 `20260801 / 20260802 / 20260803`；checkpoint 只按 dev
mean IoU 选择，event-eval 继续保持输出防火墙，直至三 seed checkpoint 与最终 INT8
artifact 固定。
