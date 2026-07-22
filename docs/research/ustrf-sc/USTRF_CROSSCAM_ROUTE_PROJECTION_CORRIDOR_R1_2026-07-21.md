# USTRF 跨相机路线投影与走廊几何 R1（2026-07-21）

状态：研究 R1 的结构性修复与真机测试完成；跨来源正样本召回尚未验证，不具备 U0、App 或生产权限。

## 结论

R0 的边界假阳性不是 detector 置信度问题，而是旧 gate 把“底部中心出发的固定宽折线”当成了路线走廊，并用 bbox 底部 25% 面积与之相交。Pexels 道路车辆只以 `2.7364px` 余量落入该固定走廊，却被提升为确定的路线内 `HIGH`。

R1 把问题拆为两个独立合同：路线投影必须先给出当前相机帧中的显式凸多边形和来源收据；物体与走廊的关系只用 bbox 底边中心作为无类别地面接触代理，并显式保留投影误差带。只有名义位置和边界距离在窄/中/宽三档下全部一致，才可称为确定 inside/outside；靠近边界一律是 `UNCERTAIN_BOUNDARY`，不得升级为路线内风险。

这解决了“把固定画面中心当路线”和“边界轻微相交即确定侵入”两个结构问题，但没有凭空产生眼镜相机的标定、位姿或米制路线。公开视频使用的仍是逐样本大模型可见人行道 proxy。

## 投影合同

当前相机帧必须携带 `route-projection-receipt.json`，至少绑定：

- 原视频投影模式和 forward-axis 权威来源；
- 路线来源、来源权限和投影置信度来源；
- 归一化相机图像坐标中的凸路线 polygon；
- 是否存在 world route、动态 projection 和 camera-pose receipt；
- 自动多模型事件参考、metric geometry、U0、训练、Android runtime 与 production authority。

Pexels receipt 明确为 `rectilinear_identity_v1 + manual_visible_sidewalk_proxy_v1`，置信度使用披露的 proxy 默认值 `0.5`；`dynamic_projection_present/world_route_present/camera_pose_receipt_present` 全为 false。因此该 polygon 只在本样本当前画面中有效。

未来眼镜相机不复用手机画面坐标。已有 benchmark-only 链条 `AndroidCameraPoseComposer → WorldRouteCameraProjector → CameraAnalysisGeometryMapper → ExplicitRouteGeometryFusion` 应接收该设备自己的内参、mount 外参、逐帧 pose/时钟和 world-route receipt，再生成同一 polygon 合同。任何缺失或无可信 forward axis 的输入返回 `route_unknown_or_invalid`，不得回退固定中心走廊。Sparse-LK 以后只能传播已经获授权的路线投影，不能生成路线意图。

## 走廊几何

冻结配置 `configs/ustrf_crosscam_projected_corridor_geometry_r1_v1.json`：

- route：当前相机帧显式凸 polygon；
- object contact：bbox bottom-center；
- projection uncertainty：画面宽度的 `0.01 / 0.02 / 0.03`；
- robust rule：三档关系全部相同才给出确定关系，否则为边界不确定；
- label 不参与 gate，unknown/invalid 输入 fail closed。

Python 审计通过稳定入口 `scripts/run_research_tool.py ustrf-crosscam-codex audit_projected_corridor_geometry.py` 调用。Android 等价实现只加入 `device-benchmark` 测试 APK；旧 U0 v1 gate、正式 App 和默认模型没有改动。

## 同设备 Pexels 重放

输入沿用 R0 的 12 个 500ms 帧和 SM-S9280 shipped YOLO 输出，共 8 个 detection。三档固定重放结果：

| 误差带 | inside | uncertain | outside | 旧 gate 保留的车辆 |
| ---: | ---: | ---: | ---: | --- |
| 1% frame width | 0 | 0 | 8 | outside |
| 2% frame width | 0 | 1 | 7 | uncertain |
| 3% frame width | 0 | 1 | 7 | uncertain |

鲁棒汇总为 `inside=0 / uncertain=1 / outside=7`。旧 gate 保留的唯一车辆不再是确定 inside：1% 下 outside，2%/3% 下 uncertain，综合结果为 `UNCERTAIN_BOUNDARY`。这表示 R1 消除了该样本的“确定路线内”假断言；它不等于已经测得长期 false-alert rate，也不能替代正样本召回。

正式审计：`artifacts.local/evidence/ustrf-crosscam-codex/pexels-3874684-negative-r0/projected_corridor_geometry_r1_audit_v3.json`，SHA-256 `4cdef98a96ff15788379f572581f4922293cef768c01ef4671d1174205f652ff`。

## 验证与后续门

- Python 合同/流水线 4 tests 通过，覆盖 end-to-end、hash/unknown fail closed、Pexels 三档关系、中心正例与非法 polygon。
- `:device-benchmark:compileDebugKotlin` 和 `assembleDebug` 通过。
- SM-S9280 / Android 16 API 36 上新 Kotlin gate 3/3 instrumentation tests 通过；receipt 保存在同一 evidence 目录。
- 六来源 held-out 预注册已冻结在 `configs/ustrf_crosscam_geometry_multisource_r1_v1.json`：Japan/Edmonton/London 正例与 Ulm/Jakarta/Cape Town 负例，Pexels 只作 development sentinel，不能回救 held-out failure。

下一步只执行这组六来源的独立路线 polygon/投影收据和正负重放。至少两个正来源出现 robust inside、所有负来源均不得 robust inside、未决来源不超过一个，才允许讨论进一步几何集成；在此之前不微调模型，也不接入 App。
