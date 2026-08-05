# AtomS3R-M12 + ToF4M 固定曝光对照 R4

## 结论

`FIXED_EXPOSURE_REDUCES_BUT_DOES_NOT_ELIMINATE_TWO_CYCLE_DELIVERY / RETAIN_AUTO_DEFAULT_PENDING_LIGHTING_VALIDATION`

仅关闭自动曝光并固定到自动模式长期报告的实际值 `490` 后，五分钟处理吞吐从
`23.52 fps` 增至 `25.13 fps`。用 R1 固定阈值 `36.320 ms` 复算，slow fraction
从 `17.71%` 降至 `8.84%`；capture→framebuffer return 超过 54 ms 的双周期帧从
`15.08%` 降至 `7.58%`。自动曝光控制是 36/72 ms 交付双峰的重要影响因素，但固定
曝光没有消除双峰，因此不是唯一机制。

本轮没有验证明暗切换、逆光、室内外或移动场景的图像可用性。性能改善不授权牺牲
环境适应性，正式配置继续使用自动曝光；待人在设备附近时再做画质与照度阶跃验证。

## 冻结合同

| 项目 | 自动曝光 R1 | 固定曝光 R4 |
| --- | --- | --- |
| 分辨率/quality | XGA / 10 | XGA / 10 |
| brightness | 1 | 1 |
| 曝光模式 | auto | manual |
| 实际/手动值 | 490 | 490 |
| framebuffer/grab | 2 / LATEST | 2 / LATEST |
| PSRAM DMA | false | false |
| ToF sampling | true | true |
| 时长 | 300.578 s | 300.719 s |

R4 通过现有 `/api/camera` session-only 控制应用，不改写开机默认值。逐帧 header
确认本轮 7,557 帧全部 `auto_exposure=false`、`exposure_value=490`、
`camera_psram_dma_enabled=false`；ToF 全程有效。

## 结果

| 指标 | 自动曝光 R1 | 固定 490 R4 | 变化 |
| --- | ---: | ---: | ---: |
| processed frames | 7,071 | 7,557 | +486 |
| effective fps | 23.52 | 25.13 | +6.8% |
| reconnect/error | 0/0 | 0/0 | 均通过 |
| 各轮自身 MAD slow fraction | 17.71% | 14.65% | -3.06 pp |
| R1 固定 36.320 ms slow fraction | 17.71% | 8.84% | -8.87 pp |
| capture→return >54 ms | 15.08% | 7.58% | -7.50 pp |
| capture→return >90 ms | 0.86% | 0.25% | -0.61 pp |
| capture→return P50/P95/P99 | 36.58/72.60/76.62 ms | 36.53/72.44/72.65 ms | P95 双峰仍存在 |
| camera-wait 桶/全部 interval | 13.88% | 7.48% | -6.40 pp |
| network-write 桶/全部 interval | 1.84% | 0.32% | -1.52 pp |
| capture→完整 JPEG P50/P95/P99 | 65.01/111.19/140.05 ms | 62.61/97.81/111.68 ms | 尾延迟改善 |
| capture→反馈记录 P50/P95/P99 | 114.48/149.38/178.37 ms | 114.01/139.85/153.40 ms | P95/P99 改善 |

各轮自身 `median+3×MAD` 阈值分别为 `36.320 ms` 和 `36.148 ms`。固定曝光轮阈值
更严格，会把更多 36.15–36.32 ms 的微小抖动标为 slow，因此机制比较以 R1 的固定
阈值和 >54 ms 双周期定义为主，不用 14.65% 单独下结论。

固定曝光轮 latest queue 覆盖 186 帧，高于 R1 的 105 帧；这是设备到帧更快且网络
仍会突发时的 host 新鲜度取舍。处理吞吐与端到端 P95/P99 同时改善，不把覆盖帧隐藏
为成功，也不把它解释为模型准确率证据。

## 证据、恢复与边界

R4 summary：
`artifacts.local/evidence/atoms3r-e2e/20260805T110907.635589Z/summary.json`，SHA-256
`3e93f038163cf6ad38e0523d342e9264fdb76ba1a709bae23e5265098acf6a66`。
`run_accepted=true`，0 reconnect、0 error，60 个状态样本。

测试后已通过 session API 恢复 XGA/quality 10/brightness 1/自动曝光/补偿 0；3 帧
协议验收确认 `auto_exposure=true`、实际 exposure 仍为 490、ToF valid，0 reconnect/
error，退出后 `stream_clients=0`。正式固件仍为
`atoms3r_m12_tof4m_slow_frame_r6`，无需重新刷写。

这是单场景、单台设备、顺序单次 A/B 的 Development 性能机制证据。它支持自动曝光
控制与慢交付频率有关，但不证明固定 490 适合真实环境，更不构成图像质量、检测准确率、
风险、人体、产品或安全证据。
