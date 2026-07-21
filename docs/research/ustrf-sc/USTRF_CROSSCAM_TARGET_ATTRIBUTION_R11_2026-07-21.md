# USTRF 跨相机目标归因诊断 R1.1（2026-07-21）

状态：**六来源诊断已完成**。这是已解封来源上的 `seen_diagnostic_not_held_out` 结果，不是新盲门；不训练、不调阈值，不授权 App 或生产集成。

## 冻结合同

- 每事件只冻结一个物理目标。相似锥桶/标杆不得在目标出画后替换；不可见帧标为 `absent/occluded`，不伪造接触点。
- 目标账本 SHA-256：`c6800fe25d25942b95ccc1fd7f3a273402123cfc9b78904cf7e4e899caf38b4b`。
- 逐帧/稳定窗口投影收据 SHA-256：`d92576001127c086860afbe17758f565e4eb94a90447f214e466ab2ff786770c`。
- Edmonton 不再使用 671–735 秒静态 polygon；仅冻结 702750–703500ms 稳定短窗内、703000ms 可见的唯一标杆。671000/734750ms 的独立 polygon 只保存在审计附录，不进入目标评分。
- oracle 只读取冻结接触点和精确帧 polygon，自行重算画面宽度 1%/2%/3%；不读取 Android detector。
- Android 先检查 label inventory，再以 label allowlist + 唯一最大 IoU `.30` 匹配目标。其他 detection 全部计为 cooccurrence，不能代替事件召回。

## Oracle 与 Android 结果

| 来源 | 可评分目标帧 | Oracle robust | Oracle | Android taxonomy | 目标匹配 | 零检测帧 | 共现 inside / 共现总数 | 首因归类 |
| --- | ---: | --- | --- | --- | ---: | ---: | ---: | --- |
| Japan | 2 | outside, outside（预期 inside） | 失败 | unsupported | 0 | 2 | 0 / 0 | polygon/事件路线合同 |
| Edmonton | 1 | inside | 通过 | unsupported | 0 | 1 | 0 / 0 | detector 类不覆盖 |
| London | 2 | inside, inside | 通过 | unsupported | 0 | 0 | 1 / 8 | detector 类不覆盖；另有旧式共现误归因 |
| Ulm | 2 | outside, outside | 通过 | unsupported | 0 | 0 | 0 / 3 | detector 类不覆盖 |
| Jakarta | 1 | outside | 通过 | unsupported | 0 | 0 | 1 / 1 | detector 类不覆盖；另有旧式共现误归因 |
| Cape Town | 1 | outside | 通过 | unsupported | 0 | 0 | 1 / 4 | detector 类不覆盖；另有旧式共现误归因 |

Japan 的唯一锥桶底部接触点在两帧距旧 polygon 边界约 `52.73px / 59.65px`，三档均为 outside。此前教师摘要的 inside 关系没有 bbox/contact 可复算支撑；R1.1 不移动 polygon 回救，按冻结顺序将首因改判为 polygon/事件路线合同不一致。

Edmonton 的新 current-frame polygon 使目标接触点距边界 `48.38px`，高于 1%/2%/3% 的 `6.4/12.8/19.2px`，三档均为 inside。原 64 秒静态投影合同因此被短窗合同取代，而不是继续 unresolved。

SM-S9280 / Android 16 API 36 上 instrumentation `OK (1 test)`，运行 `2.209s`。shipped COCO 80 类不含 traffic cone、delineator 或 bollard，因此六来源均为 `unsupported_taxonomy`，`event_recall/false_alarm=null`；Japan/Edmonton 的全零 detection 只是这个覆盖结论的观测表现，不能再称普通 COCO 类漏检。London/Ulm 的原视频在设备解码为 427×240、教师帧为 426×240；normalized 几何在固定 `≤0.5%` aspect-ratio drift 合同下映射，双方尺寸逐帧记录。

## 结论

1. 目标关联错误得到直接证实：London/Jakarta/Cape Town 在目标不受 taxonomy 支持时仍有 route-inside 的 person/car 等共现检测；旧“任意 inside 即事件”会产生 3 个 legacy inside 帧。
2. Edmonton 投影合同已在目标绑定短窗内闭合；这不等于整个 64 秒动态路线都已恢复。
3. Detector 类覆盖是五个 oracle 通过来源的首要阻塞。当前模型无法对冻结的锥桶/标杆目标计算有效召回或假告警。
4. Japan 不是 detector 优先问题：oracle 已先失败，应先重审该事件的路线/正例定义。
5. 风险语义臂尚不能被称为“目标与 Android 都通过后的最终失败”，因为目标 taxonomy 从未通过；现有共现统计只证明旧 source-level 归因规则错误。

## 证据

- Oracle output：`oracle_geometry_output.json`，SHA-256 `aa0e84c06fb7b118d37d342ddf207314a01e27faecf6937711f68bdfd9a5393c`。
- Android output：`android-arm/android_r11_output.json`，SHA-256 `1b09e8e68ff90bb4e398cd08f31ab33472e36be4015c10b2852a48f3408c9fce`。
- Device receipt：`android-arm/device_run_receipt.json`，SHA-256 `d667757969407a02ad5f8a0d56b9710b7a790785c1de685e68c1251d59502f9e`。
- Attribution report：`r11_attribution_report.json`，SHA-256 `757068cd2e7c2716afcd6c5dc048d8aad5e982b6dbde8eebf2a34ea2ad632139`。
- 证据根目录：`artifacts.local/evidence/ustrf-crosscam-codex/multisource-r11-diagnostic-v1/`。

下一轮必须换六个未查看结果的新来源并重新预注册。进入新 held-out 前，应先决定目标 taxonomy 路线（可检测目标类或独立 marker detector），且不得用本六来源调 IoU、polygon 或 detector 阈值。
