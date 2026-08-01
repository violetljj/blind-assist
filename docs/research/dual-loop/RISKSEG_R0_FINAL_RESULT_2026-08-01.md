# RISKSEG-R0 最终结果

状态：`COMPLETE / VALID_NEGATIVE_DEVELOPMENT_RESULT /
RISKSEG_R0_EVENT_QUALITY_OR_STABILITY_FAIL /
PIDNET_S_TRAINED_FINAL_DEVICE_PASS /
RISKSEG_R0_TRAINED_NOT_PROMOTABLE_KEEP_YOLO /
NO_APP_SWITCH / NO_RULE_RESCUE`

日期：2026-08-01（Asia/Hong_Kong）

机器可读结果：
[RISKSEG_R0_FINAL_RESULT_2026-08-01.json](RISKSEG_R0_FINAL_RESULT_2026-08-01.json)

## 结论

`RISKSEG-R0` 已按完整顺序执行到终点，但不能晋级。保留当前 YOLO 默认 App，不接入
learned segmentation，不执行发布晋级，也不在已消费的 30-event 回归集上调阈值、
改规则、挑 seed 或增加 gate 救结果。

这不是部署性能失败。固定决策 seed 的 PIDNet-S INT8 在 SM-S9280 上通过了最终 10 分钟
QNN/HTP 门；关闭候选的直接原因是事件质量与跨 seed 稳定性失败：

- 三个 seed 中 `0/3` 通过冻结事件质量门；
- 决策 seed 召回没有高于 YOLO，关键漏报没有减少；
- learned false-alert event 从 YOLO 的 `6/14` 恶化到 `13/14、13/14、14/14`；
- 共同命中事件普遍更晚；
- clearance 虽改善，但不能抵消 recall、critical miss、false alert 和 timing 的失败。

## 1. 数据与训练工件

520 帧按 source session 固定为 `320 train / 200 dev`，train/dev session 零重叠。
训练器固定使用官方 PIDNet-S、官方 ImageNet 预训练权重、四类 head 和
`512x288` 输入；训练期间没有访问 event-eval outcome。

| seed | 角色 | 完成轮数 | best epoch | dev mIoU | boundary F1 | worst-session mIoU | checkpoint SHA-256 |
|---:|---|---:|---:|---:|---:|---:|---|
| 20260801 | decision | 46 | 16 | 0.262028 | 0.233523 | 0.108696 | `e33c824b…9d5850` |
| 20260802 | stability | 83 | 53 | 0.266485 | 0.239839 | 0.110119 | `081dd212…953d2d` |
| 20260803 | stability | 40 | 6 | 0.359005 | 0.232578 | 0.078172 | `591c2a29…a8c6b0` |

三 seed 的像素指标波动明显，尤其 seed 20260803 的 dev mIoU 最高，但它并没有得到更好
的事件晋级结果。因此这些像素指标只保留为描述性稳定性证据，不反向选择 seed。

三份 checkpoint 均导出静态简化 ONNX，并通过 PyTorch/ONNX Runtime 数值一致性；随后
使用同一冻结 train-only calibration 转换为 full W8A8 TFLite。三份模型均为
7,927,976 bytes、INT8 输入/输出、零 float tensor，四类尺寸与有限值 canary 通过：

| seed | TFLite SHA-256 |
|---:|---|
| 20260801 | `cf7a4d11ec72b281fe62381d32601af106871215a5e0cc286bc04c2027cec0dc` |
| 20260802 | `ba3cd804d0da0f5bb06081884a09c65f9b238331c8b2f117cf160a9c3918c601` |
| 20260803 | `68ac01ce8d1926e2d3e0a06a4be7179bef1f5868c5b1d538e8b6f7aa79e4b1c7` |

训练冻结收据位于
`artifacts.local/evidence/riskseg-r0/training-v1/training_freeze_receipt.json`，
SHA-256 为
`8c771feec2064f6f2b4388198d2b4f7935a62b9c8938a6598d9af196311c095d`。

## 2. 固定事件评价

event-eval 是在任何模型/YOLO/oracle outcome 打开前冻结的 output-blind cohort：

- 30 parent events / 30 source sessions / 1,920 frames；
- `8 blocking obstacle positive / 8 boundary level-change positive /
  7 parallel-curb negative / 7 normal-walkable negative`；
- 与 520 train/dev 和固定 90-frame regression 按 source session 零重叠；
- manifest SHA-256：
  `6fe64391a9557fd9d80ad8f861cd4b043293955e2ff3e5979ffe7fb33d33b7ff`。

每个 seed 在 SM-S9280 上运行同一冻结三臂链：

```text
A_CURRENT_YOLO_ONLY
B_LEARNED_SEGMENTATION_ONLY
C_TRUTH_MASK_ORACLE_REFERENCE
```

每次都产生 `1,920 frames × 3 arms = 5,760` 条 trace，设备测试为
`1 test / 0 failures`。主机评分器从 raw trace 独立重算 parent-event summary，并核对
设备 summary、manifest 窗口和 trace SHA。

### 2.1 汇总

| arm / seed | hit / 16 | recall | critical miss | false alert / 14 | clearance / 16 |
|---|---:|---:|---:|---:|---:|
| YOLO（共同基线） | 13 | 0.8125 | 3 | 6 | 5 |
| learned 20260801 | 13 | 0.8125 | 3 | 13 | 14 |
| learned 20260802 | 14 | 0.8750 | 2 | 13 | 10 |
| learned 20260803 | 13 | 0.8125 | 3 | 14 | 6 |
| truth-mask oracle（共同参考） | 13 | 0.8125 | 3 | 9 | 8 |

冻结 recall 门要求相对 YOLO 至少 `+0.15` 且 `+3 events`。三个 learned seed 的增量为
`0 / +1 / 0 events`，全部失败。关键漏报减少量为 `0 / 1 / 0`，也全部低于至少减少
2 个的门。

false-alert guardrail 要求不高于 YOLO 的 `6/14`，而 learned 为
`13/14、13/14、14/14`，三 seed 全失败。clearance 是唯一全部通过的方向，但它不能覆盖
其他 trade-off。

共同命中时序也全部失败：

| seed | common hits | median delay | 晚于 2 帧比例 |
|---:|---:|---:|---:|
| 20260801 | 10 | +3 帧 | 0.6000 |
| 20260802 | 11 | +5 帧 | 0.6364 |
| 20260803 | 11 | +3 帧 | 0.5455 |

决策 seed 不通过、通过 seed 数为 `0/3`、false-alert/clearance guardrail 的跨 seed 组合
不通过，因此稳定性门失败。独立评分结果为
`RISKSEG_R0_EVENT_QUALITY_OR_STABILITY_FAIL`，SHA-256：
`28d6476d61e3e9bc2d0f6bff69a39ede413478ad089f165ae55b9351e52ec776`。

truth-mask oracle 经过同一冻结 adapter/事件链后也只有 `13/16` recall、`9/14`
false-alert 和 `8/16` clearance。它只是链路参考，不是完美上限；但 learned candidate
即使只和 YOLO 比较也已明确失败，因此不需要依赖 oracle 解释来关闭候选。

## 3. 最终真机性能

固定决策 seed `20260801` 的 exact trained INT8 模型在 SM-S9280 / SM8650 上完成最终
10 分钟 QNN/HTP 运行：

| 指标 | 结果 | 门 |
|---|---:|---:|
| observed duration | 600,064.646 ms | ≥600,000 ms |
| samples | 7,727 | >0 |
| failure count | 0 | 0 |
| total pipeline P95 | 77.374 ms | ≤100 ms |
| inference P95 | 5.198 ms | 记录 |
| initial-window P95 | 72.984 ms | 记录 |
| final-window P95 | 78.548 ms | 记录 |
| final / initial P95 | 1.07624x | ≤1.20x |
| maximum thermal status | 0 | < severe |

流式 logcat 记录两次 `173/173 nodes / 1 partition` 完整委派。hash-scoped clean cache
先进入 SAVE MODE，第二 runtime 进入 RESTORE MODE；没有 QNN error。独立终态为
`PIDNET_S_TRAINED_FINAL_DEVICE_PASS`：

- receipt SHA-256：
  `e6413dbbed26e1735650f6b8b1fb21249e3da3793a1ed06f19ed4720d5a7886b`；
- streamed logcat SHA-256：
  `559e7c7117bda6a80b4e9271b2fbb6ae3ad1be03267a8163f25cf2b73ce5380f`；
- validation SHA-256：
  `cc92b2fbeb531c4ed5aaee322268ecf8668864fb6129c5465f4c8b6f6ce526c5`。

这个性能 PASS 不具有覆盖事件质量否决的权限。

## 4. 执行异常与证据修复

所有异常都保留，未用于改变模型、数据、阈值或评分门：

1. seed 20260801 的第一次 `connectedDebugAndroidTest` 计算成功，但 Gradle 随后卸载
   target App，连同 external-files 输出一起删除。改为手动安装 APK、直接
   `adb instrument` 并在卸载前拉取；输入、模型、链和评分均未改变。
2. 第一次直接 instrument 在打开 manifest 时因 Android scoped-storage 所有权失败，
   0 帧执行。随后把同一 hash-closed view 复制为 target App UID，App 自身复核 manifest
   SHA 后运行。
3. 第一次 trained-final 10 分钟收据通过数值门，但事后 logcat dump 已覆盖初始化段，
   独立证据判 `INVALID`。保留该收据后，以相同模型、相同 600 秒门、clean hash-scoped
   cache 和流式 logcat 完成一次复核。
4. 旧 validator 把预检模型的 `163/163` 节点和 cache 已存在时的两次 RESTORE 写死。
   trained 模型实际为 `173/173`，clean cache 正确生命周期是 SAVE→RESTORE。validator
   只修正为“delegated==total、1 partition”和“与 receipt cache-before 一致的生命周期”；
   所有数值门不变，旧预检回归仍 PASS，partial delegation 单测仍 FAIL。

第 4 项是在 raw log 固定后进行的 compatibility repair，因此必须保留此披露。它只改变
技术日志解析，不影响已经明确失败的事件评价，也不产生 App 晋级。

## 5. 决策与后继边界

最终动作固定为：

1. `RISKSEG_R0_TRAINED_NOT_PROMOTABLE_KEEP_YOLO`；
2. 不修改默认 App，不接 learned segmentation，不跑 release promotion；
3. 不在这 30 个已消费 parent events 上调参、挑 seed、改 taxonomy、加 gate 或修改
   事件链；
4. PIDNet-S 的训练工件、三臂 trace 和设备性能只作为 Development 负结果与部署
   可行性证据保留；
5. 若未来继续风险分割研究，必须另立新的因果假设、重新冻结实现，并取得新的
   session-disjoint parent-event cohort；本结果不自动授权 successor。

证据强度只到单架构、单事件 cohort、单设备的 Development 判断，不支持独立助行、
安全、产品或跨设备结论。
