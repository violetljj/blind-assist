# USTRF 跨相机 marker held-out R1.2 结果（2026-07-21）

状态：**六来源 oracle、冻结离线 detector、Android parser canary 与 SM-S9280 同设备重放均已完成**。结果已经解封，禁止换源、改 bbox/contact、移动 polygon、调 `.05/.30` 或用本轮来源调 export/parser。

## 结果先行

| 来源 | 预期 | Oracle | 离线目标 | Android 目标 | Android 关系 |
| --- | --- | --- | --- | --- | --- |
| Thailand border | inside | inside，通过 | traffic cone，IoU `.612` | 匹配，IoU `.594` | inside |
| Bridge bollard | inside | inside，通过 | bollard，IoU `.511` | 匹配，IoU `.514` | inside |
| Urban roadwork | inside | inside，通过 | traffic cone，IoU `.672` | 匹配，IoU `.642` | inside |
| Tokyo driveway | outside | outside，通过 | traffic cone，IoU `.353` | 匹配，IoU `.334` | outside |
| Sidewalk closed | outside | outside，通过 | traffic cone，IoU `.379` | 匹配，IoU `.382` | outside |
| Vancouver work zone | outside | outside，通过 | 未匹配 | 未匹配 | 无目标关系；假告警 false |

- Oracle：`6/6` 来源在 1%/2%/3% frame-width 不确定性下均与预注册关系一致。
- 冻结 offline PyTorch YOLOE：正例事件召回 `3/3`；负例目标假告警 `0/3`；目标实例匹配 `5/6`。
- 离线/Android 的目标级六来源结论逐项一致：正例事件召回均为 `3/3`，负例目标假告警均为 `0/3`，实例匹配均为 `5/6`。
- 共现 route-inside detection 在离线/Android 分别为 `7/5` 个；数值与 NMS 后端漂移被单列，不得替代目标召回或假告警。Vancouver 未匹配说明 taxonomy 覆盖不等于每个实例都能检出。

## 静态导出与 Android 边界

- YOLOE 三类 FP16 TFLite 已静态导出，SHA-256 `f726ea0f5cea1cdf0a4e1a473f37ccbadc8b9bdf4b9ea71d02311556c0690163`。
- Host LiteRT 张量：输入 `[1,640,640,3] float32`；主输出 `[1,39,8400] float32`；mask prototype 输出 `[1,160,160,32] float32`。主输出前 `4+3` 通道与现有 bbox parser 的类别布局一致。
- SM-S9280 / Android 16 API 36 的 R1.1 九帧 parser canary 为 `OK (1 test)`：共解析 41 个检测、6 帧目标匹配，三类标签均在真实输出中出现；canary 明确记录 `r12_sources_read=false`。
- 同设备 R1.2 六来源 target-aware replay 为 `OK (1 test)`，目标事件结论与离线臂六来源逐项一致；匹配来源的 Android/oracle 几何也全部一致。
- 最终 benchmark-only APK SHA-256 为 `08e3b3012937d8b63d80e4ab0756f8db98213c0a28e10b6ad9e1fcf7b8d3af98`。该资产只存在于 benchmark，不改变 App 默认模型。

## 证据强弱与结论

本轮证明的是：在六个全新公开视频、每来源一个预注册评分帧、视觉路线 proxy 下，冻结三类 YOLOE 离线臂能够闭合三个正例，并在三个负例上不把指定目标判成路线内；它明显优于原 COCO 模型的 taxonomy 全阻塞。

它尚不构成生产晋级证据：样本只有六来源/六帧，polygon 与 bbox 是 Codex 视觉代理，不是 human event truth 或设备米制几何；Vancouver 仍有实例漏匹配；本轮只验证单帧 parser 与同设备输出，没有延迟分布、持续热稳定或真实连续事件闭环。默认模型与 App 路径保持不变。

## 证据入口

- `target_instance_ledger.json`：`0f320ccd159f783698bae79bb0cf1efb0f6877ce736af6d4f03a06096b28a6d7`
- `frame_projection_receipt.json`：`ca7e6b415c99844a273778d04b08686d9b04e58602335d1bcc6346c5129a747a`
- `oracle_geometry_output.json`：`2bdf23d4845a579749976d7d9447a151de12ded2263b548740c4389c21a02c5c`
- `offline_detector_output.json`：`b5369e8e26433e2a07dc3d75eef780621efd1b26807fbf53e5700eb83ea1fec9`
- `android-arm/android_r12_output.json`：`075abf429a4405e68a084d28667bba4c237c79849eb0798005a0df282e9428a9`
- `android-arm/device_run_receipt.json`：`591f3e2bdc5c757a33a5e358a0f68a3f046dc8043bba65fb77b10b518e0872b4`
- `host_device_consistency_report.json`：`9e78ad8c9f4e368cab7537fd1a6572515113b1cd0d1eaa7e67adfcb090ad3e5a`
- `evidence_index.json`：`3155c1125e38c95b040f90dedb24d3252c02f91bc61a09b525a5ba0802e35aae`
- 证据根：`artifacts.local/evidence/ustrf-crosscam-codex/multisource-r12-heldout-v1/`

下一轮应扩大独立来源与连续事件数据，并引入 human-reviewed event truth、设备米制几何、延迟和热稳定门。Vancouver 只能作为后续新诊断集的问题线索，不能回调本轮阈值或 prompt。
