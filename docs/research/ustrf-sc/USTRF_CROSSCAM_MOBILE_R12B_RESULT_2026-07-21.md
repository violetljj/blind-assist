# USTRF 跨相机移动端连续事件 R1.2b 结果（2026-07-21）

## 结论

R1.2b **性能门通过、事件门失败、总体失败**，继续保持 `benchmark-only / do_not_replace_default_model`。

- SM-S9280 上，冻结的同一 FP16-640 模型仅切到 benchmark APK 内 GPU delegate 后，600 秒共完成 4,795 次检测，inference p50/p95 为 `40/54ms`，总检测 p50/p95 为 `84/105ms`；0 解码失败、0 推理失败，电池温度 `31.1→35.1°C`，最大 Android thermal status `0`。移动端算力与稳定性门通过。
- 连续事件正例只命中 `4/6 = 66.7%`，低于冻结门 `>=90%`；Japan 与 London 漏警，London 还使出画清除门失败。负例假告警、重复交付、共现接管、身份切换均为 `0`，关联歧义帧率 `5.34%` 通过。
- 没有打开 R1.3，没有导出/运行 320 候选，没有训练，也没有改变 prompt、三类静态 taxonomy、`.05/.30/.45`、bbox 或 polygon。Vancouver 仍只作不参与门禁的漏检线索。

## 冻结范围与候选选择

主协议 `configs/ustrf_crosscam_mobile_r12b_prereg_v1.json` 先冻结单变量顺序：

1. C1：同一模型、同一 640 输入，只将 CPU 改为 GPU delegate；
2. C2：仅当 C1 未通过时才运行同权重 320 CPU；
3. C3：仅当前两者未通过时才运行同权重 320 GPU。

C1 九帧 canary 在 SM-S9280 上通过：27 次测量、0 失败、稳定目标匹配 `6/9`、三类标签均输出；inference p50/p95 `35/44ms`，总检测 p50/p95 `75/88ms`。按“首个全门通过即停止”规则，C1 被选中，C2/C3 未导出、未运行。

冻结模型 SHA-256 为 `f726ea0f5cea1cdf0a4e1a473f37ccbadc8b9bdf4b9ea71d02311556c0690163`；C1 canary 收据 SHA-256 为 `37d7112e641bc3451ce21bcb576867ebd6d9da05438a4c7e767874db6ca2944c`。

## 先排除取帧偏差

原 Android `MediaMetadataRetriever` 在九个相同时间戳上只通过 `4/9` 静态 canary 等价门，其中 London/Ulm 还出现 `426→427px` 宽度漂移。该失败不能归因于 detector。

第一次冻结的 ffmpeg 精确帧传输虽然达到 `9/9` 尺寸一致，但像素仅 `2/9` 完全一致，因此保留为失败收据。随后只修订 decoder 语义，复用原始 canary 的 OpenCV 4.13.0 `VideoCapture + CAP_PROP_POS_MSEC + cv2.imwrite`，候选、模型和所有阈值均不变：

- 主机：`9/9` 尺寸一致、`9/9` RGB 像素一致；
- SM-S9280：`9/9` 目标状态一致、`9/9` 匹配标签一致、0 失败；
- 12 段输入共物化 223 张逐帧 SHA-256 绑定 PNG，设备连续重放不再解码原视频。

有效 transport v2 SHA-256 为 `86945f31f98e576a042af090d58330b99270b7adc5f3ab3ce4632fa45aa03009`；设备准入收据 SHA-256 为 `7310a3f6547ddade4eb13d7136860997af4218aa2066299834bb68d2d27b56ff`。

## 连续事件结果

| 指标 | 冻结门 | 结果 | 判定 |
| --- | ---: | ---: | --- |
| 正事件召回 | `>=90%` | `4/6 = 66.7%` | 失败 |
| 负例假告警 | `0` | `0` | 通过 |
| 首次正确告警 | `<=5,000ms` | 已命中正例均通过 | 通过 |
| 重复告警交付 | `0` | `0` | 通过 |
| 出画清除 | `<=1,000ms` | London `10,000ms` fail-closed | 失败 |
| 共现物接管事件 | `0` | `0` | 通过 |
| 身份切换 | `0` | `0` | 通过 |
| 关联歧义帧率 | `<=10%` | `5.34%` | 通过 |

失败分成两类，不能再统称为“手机太慢”：

- Japan：主 anchor 已匹配、两个可见 anchor 均重获，12 帧中关联 9 帧，但没有触发路线内事件。既有 R1.1 oracle 已指出同一目标接触点对当前 polygon 为 robust outside；这暴露的是“正事件标签与冻结路线几何代理不相容”，不是 detector 漏检。
- London：primary anchor 未匹配、两个可见 anchor 均未重获、22 帧关联为 0，是冻结 detector 对远小标杆的连续召回失败；从未建立目标轨迹后，出画门按协议 fail-closed 为 10 秒。

其余四个正例中，Edmonton 首警延迟 `250ms`，Thailand/Bridge/Roadwork 均为 `0ms`。Vancouver 为冻结负例漏检线索且 `gate_eligible=false`，没有参与候选选择或阈值回调。

## 设备结果

| 指标 | 门槛 | 结果 | 判定 |
| --- | ---: | ---: | --- |
| inference p50 / p95 | 报告 / `<=120ms` | `40 / 54ms` | 通过 |
| 总检测 p50 / p95 | 报告 / `<=160ms` | `84 / 105ms` | 通过 |
| 连续运行 | `600s` | `600s` | 通过 |
| 解码 / 推理失败 | `0 / 0` | `0 / 0` | 通过 |
| 温升 | `<=8°C` | `4.0°C` | 通过 |
| 最终电池温度 | `<=45°C` | `35.1°C` | 通过 |
| 最大 thermal status | `<=2` | `0` | 通过 |

完整设备输出 SHA-256 为 `8828a2131b2ee513d1389729052ea99872622d43cc7b7807a9e0071674cc4ff4`。

## 决策与下一步

R1.2b 不授权替换默认模型，也不解封 R1.3。下一轮仍只能使用 seen diagnostic：

1. 先冻结 Japan 的“事件 truth 与 route polygon/接触点关系”审计，决定应修正事件定义还是路线几何来源；不得通过移动 polygon 回救当前结果。
2. 对 London 做不改 `.05/.30` 的候选/logit 与分辨率因果诊断；若另开 detector 候选，必须新预注册并保持单变量可归因。
3. 只有连续事件协议内部一致、seen gate 达标且同设备性能门继续通过后，才按已冻结的 12 槽位、6 正/6 负与双 VLM provisional event truth 协议解封 R1.3。

本轮所有生成证据位于 `artifacts.local/evidence/ustrf-crosscam-codex/mobile-r12b-seen-diagnostic-v1/`，不进入生产资产。
