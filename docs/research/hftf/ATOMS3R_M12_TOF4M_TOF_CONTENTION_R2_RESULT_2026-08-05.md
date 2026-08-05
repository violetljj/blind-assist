# AtomS3R-M12 + ToF4M 单变量争用对照 R2

## 结论

`TOF_NOT_PRIMARY_CAMERA_WAIT_CAUSE / RETAIN_TOF_SAMPLING / NO_PARAMETER_OPTIMIZATION_AUTHORITY`

仅关闭 ToF 连续读取、保持 XGA `1024×768`、JPEG quality 10、自动曝光、网络、
MJPEG 主机接收和参考 pipeline 不变的五分钟对照，没有消除相机等待型慢帧，也没有
改变典型的 camera capture→JPEG ready 耗时。因此本轮不支持把 ToF 读取认定为设备
慢帧主因，正式固件继续开启 ToF。慢帧比例的下降主要落在前一帧网络写出较慢的诊断
桶中；单次顺序 A/B 不足以把该变化归因于 ToF。

## 冻结对照

| 项目 | ToF 开启 R1 | ToF 关闭 R2 |
| --- | --- | --- |
| 固件 | `slow_frame_r4` | `slow_frame_r5_tof_off` |
| ToF 连续读取 | 开启 | 关闭 |
| 分辨率/quality | XGA / 10 | XGA / 10 |
| 曝光 | 自动，实际值 490 | 自动，实际值 490 |
| 时长 | 300.578 s | 300.468 s |
| 处理帧/fps | 7,071 / 23.52 | 7,125 / 23.71 |
| 重连/错误 | 0 / 0 | 0 / 0 |

R5 将 `sampling_enabled` 加入 `/api/status`，并将
`X-ToF-Sampling-Enabled` 加入每个 MJPEG part。R2 全部 7,125 个处理帧均记录为
`false`，ToF update count 全部为 0。旧 R1 固件尚无该 header，其开启状态由固件
身份和该轮有效 ToF/update 账本绑定。

## 结果

| 指标 | ToF 开启 | ToF 关闭 | 解释 |
| --- | ---: | ---: | --- |
| 冻结规则 slow fraction | 17.71% | 15.31% | 下降 2.39 个百分点 |
| 用 R1 的 36.320 ms 固定阈值复算 | 17.71% | 15.68% | 避免各轮 MAD 阈值差异 |
| 全部 acquire P50/P95 | 14.42/51.86 ms | 17.05/54.43 ms | 未改善相机等待 |
| 全部 preceding write P50/P95 | 21.63/32.68 ms | 19.04/29.10 ms | 本轮网络写出尾部较轻 |
| capture→JPEG ready P50/P95/P99 | 36.58/72.60/76.62 ms | 36.56/72.59/72.87 ms | 典型相机阶段基本相同 |
| capture→完整 JPEG P50/P95/P99 | 65.01/111.19/140.05 ms | 62.20/110.79/131.20 ms | P50/P95 小幅变化，P99 改善 |
| host latest queue P50/P95/P99 | 7.90/28.67/35.39 ms | 7.79/28.58/35.35 ms | 主机排队等价 |
| latest overwrite | 105 | 104 | 等价 |

按 R1 的同一诊断分层：

- camera wait 桶（`acquire>=30 ms && preceding write<40 ms`）从
  `981/7,070 = 13.88%` 变为 `1,022/7,124 = 14.35%`，并未减少；
- network write 桶（preceding write `>=40 ms`）从
  `130/7,070 = 1.84%` 变为 `20/7,124 = 0.28%`；
- 所以 slow fraction 的净下降不能支持“ToF 造成 camera framebuffer 等待”。它与
  本轮更轻的 Wi-Fi/socket write 尾部同时出现，但没有重复交错 A/B/A/B，不能进一步
  声称关闭 ToF 改善了网络。

## 证据与边界

ToF-off summary：
`artifacts.local/evidence/atoms3r-e2e/20260805T103813.003945Z/summary.json`，
SHA-256 `96c1cdd50cca088dc2489938c5ad2b76cab0a49aee7c0c9e8c5ffaa6c3078dc5`。
该轮原始 host 解析器把缺失的 ToF timestamp 0 算成负 capture timestamp；这些
ToF skew 字段明确不可评估，也未用于上述结论。R5 最终代码已将无时间戳 skew 记为
不可评估，防止后续误读。

实验后已恢复并刷入 `atoms3r_m12_tof4m_slow_frame_r5`：
`sampling_enabled=true`、ToF `ready=true/valid=true`，最终 3 帧协议验收 0 重连、0 错误，
退出后 `stream_clients=0`。最终固件 program/RAM 为
`1,077,703 B (32%) / 62,608 B (19%)`；最终 app bin SHA-256 为
`1114cab4d6f4352484c8a32d91d6826eb4b3f7ccab79f8c5e0d1c920b5b3c5c5`。

这是单台设备、单一场景、顺序单次五分钟 Development 机制证据，不是 ToF 精度、
图像质量、模型准确率、物理反馈、人体、产品或安全证据。下一性能优化若继续，应针对
camera framebuffer/帧节拍机制；不要通过关闭 ToF 换取未经重复验证的微小尾延迟变化。
