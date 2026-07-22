# USTRF 六来源 held-out 几何验证 R1（2026-07-21）

状态：**未通过**。不得进入动态路线投影、微调、App 或生产集成。

## 冻结与执行顺序

- 预注册：`configs/ustrf_crosscam_geometry_multisource_r1_v1.json`，SHA-256 `a97697da30660f21c29145c36fad5b092ed20a4ca31b1c4c92e48b4be13a2c0b`。
- 六来源 polygon/投影收据在读取本轮 detector/risk 输出前独立冻结。geometry freeze SHA-256 `eb4ec47806132ac1d33459e03e4b664930ef64244d1ce5f9b4ff79b3d4c8090a`；freeze index SHA-256 `9fb86aff979ba839bf906908aa924e69fcf4008604e2521b0c6b263fbe37054f`。
- Edmonton 在解封前已因 671–735 秒期间相机转向和路线横移标为 `fail_closed_projection_instability_unresolved`，设备臂没有 staging、解码或检测该视频。
- 固定误差档始终为画面宽度 `1% / 2% / 3%`；`threshold_fit=false`、`parameter_search=false`、`thresholds_changed=false`、`training_performed=false`。
- Pexels 未参与 held-out 门。

Codex 教师臂的三个关键帧参考为：Japan `inside=3/outside=0/uncertain=0`，Edmonton `0/0/3`，London `3/0/0`，Ulm、Jakarta、Cape Town 均为 `0/3/0`。教师参考门为正来源 `2`、负来源 `0`、unresolved `1`，单臂通过；它是 provisional 事件目标参考，不是客观传感器事实、真人用户效果或 Android 全帧计数。

## SM-S9280 Android 固定重放

表内三档与 robust 均为 detection 计数；召回和假告警按来源判定，不跨来源或跨帧平均。

| 来源 | 1% inside/outside/uncertain | 2% inside/outside/uncertain | 3% inside/outside/uncertain | robust inside/outside/uncertain | 事件召回 | 假告警 | 来源结论 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Japan（正） | 0/0/0 | 0/0/0 | 0/0/0 | 0/0/0 | 0 | — | resolved；8 帧无任何 detection |
| Edmonton（正） | 0/0/0 | 0/0/0 | 0/0/0 | 0/0/0 | — | — | unresolved；投影不稳定，未解码 |
| London（正） | 22/54/10 | 15/54/17 | 15/46/25 | 15/46/25 | 1 | — | robust inside；15 个 inside 均为 person |
| Ulm（负） | 5/24/13 | 5/24/13 | 5/24/13 | 5/24/13 | — | 是 | 5 个 inside 均为 person |
| Jakarta（负） | 12/5/2 | 12/4/3 | 12/4/3 | 12/4/3 | — | 是 | 12 个 inside 均为 person |
| Cape Town（负） | 58/65/13 | 53/63/20 | 50/61/25 | 50/61/25 | — | 是 | inside 为 47 person + 3 car |

最终 source-level 门：正来源 robust inside `1/3`（要求至少 2），负来源 robust inside `3/3`（要求 0），unresolved `1/6`（允许最多 1）。因此总门未通过。

## 失败归因

1. **投影不稳定**：Edmonton 的单一静态 current-frame polygon 不能覆盖整个 64 秒窗口；这是风险解封前冻结的 unresolved，不是结果后回避。
2. **detector 漏检**：Japan 教师目标参考为 route-inside，但 shipped Android YOLO 在 8 个固定重放帧中 detection 总数为 0，故事件召回为 0。现有证据首先指向 detector 漏检，而不是 polygon 边界误差。
3. **负来源假告警不是边界 polygon 被轻微穿越**：Ulm、Jakarta、Cape Town 的 robust-inside 全部来自 polygon 内真实 person（Cape Town 另有 3 个 car），不是目标 delineator/cone 的边界抖动；这些关系在 1%–3% 下保持 inside。教师目标参考仍把预注册的边界物体判为 outside，因此当前主要问题是 source-level “任意 inside detection 即事件命中”的目标关联/语义混淆。没有证据支持通过移动 polygon 或调整误差阈值回救。
4. **London 召回需保留限定**：Android 的 15 个 robust-inside 全为 person；它满足冻结的 source-level 门定义，但不是对 `center_marker_intrusion` 目标身份的独立证明。

下一步仍不是微调，也不是动态路线投影。应先保持全部阈值冻结，补目标关联诊断：核对目标实例与 co-occurring route user，确认 Japan 的 detector miss，并把 Edmonton 改为可信的逐帧/动态投影证据后另行预注册。只有重新满足同类 source-level 门，才讨论接入眼镜内参、安装外参、IMU/位姿和世界路线。

## 可复现证据

- 教师 summary：`artifacts.local/evidence/ustrf-crosscam-codex/multisource-r1-v1/teacher/teacher_multisource_summary.json`，SHA-256 `763ebbeac1b8aa41876faa9c17c24d77991f1406249684061db52559b6d6aaf1`。
- Android output：`artifacts.local/evidence/ustrf-crosscam-codex/multisource-r1-v1/android-arm/android_multisource_output.json`，SHA-256 `3120f7a9a4bfc48871398143b31753a94a6bcfdfdc44d61d59c4f3fc7c0d5f6f`。
- 设备收据：`artifacts.local/evidence/ustrf-crosscam-codex/multisource-r1-v1/android-arm/device_run_receipt.json`，SHA-256 `96474a5d34f1652aa7936425a1abd16daed3c3bc1692b317421df9453fa3bace`。
- 最终门报告：`artifacts.local/evidence/ustrf-crosscam-codex/multisource-r1-v1/multisource_geometry_gate_report.json`，SHA-256 `bcfb40c1e4ae1e2d5997a368ef929e86f2cff944b6adbe3766eb6f6571b9db64`。
- 真机：Samsung SM-S9280 / Android 16 API 36；instrumentation `OK (1 test)`，37.285 秒；shipped `yolo11n_fp16_320.tflite` SHA-256 `00edb41a528b0a7e709c4af8ce3e685491492c4539274804e5cfc17a1a867cd2`。
