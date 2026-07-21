# USTRF cross-camera Codex proxy benchmark

状态：active

## 稳定 Interface

```powershell
E:\codex-tools\bin\blindassist-python.cmd scripts/run_research_tool.py ustrf-crosscam-codex prepare_review_bundle.py --help
E:\codex-tools\bin\blindassist-python.cmd scripts/run_research_tool.py ustrf-crosscam-codex validate_teacher_reviews.py --help
E:\codex-tools\bin\blindassist-python.cmd scripts/run_research_tool.py ustrf-crosscam-codex evaluate_proxy_benchmark.py --help
E:\codex-tools\bin\blindassist-python.cmd scripts/run_research_tool.py ustrf-crosscam-codex prepare_android_bbox_route_proxy.py --help
E:\codex-tools\bin\blindassist-python.cmd scripts/run_research_tool.py ustrf-crosscam-codex convert_android_bbox_route_candidate.py --help
E:\codex-tools\bin\blindassist-python.cmd scripts/run_research_tool.py ustrf-crosscam-codex audit_projected_corridor_geometry.py --help
E:\codex-tools\bin\blindassist-python.cmd scripts/run_research_tool.py ustrf-crosscam-codex evaluate_target_oracle_geometry.py --help
E:\codex-tools\bin\blindassist-python.cmd scripts/run_research_tool.py ustrf-crosscam-codex materialize_r11_android_input.py --help
E:\codex-tools\bin\blindassist-python.cmd scripts/run_research_tool.py ustrf-crosscam-codex evaluate_r11_attribution.py --help
E:\codex-tools\bin\blindassist-python.cmd scripts/run_research_tool.py ustrf-crosscam-codex materialize_r12a_continuous_input.py --help
E:\codex-tools\bin\blindassist-python.cmd scripts/run_research_tool.py ustrf-crosscam-codex materialize_r12b_exact_frame_input.py --help
```

输入必须是有来源、许可和 SHA-256 收据的公开第一视角视频。准备器只向 Codex 教师暴露当前/过去图像、时间戳和预注册路线覆盖；隐藏代理标签、未来帧、深度、review、adjudication 或 blind 字段不得进入评审材料。

360° 等距柱状视频必须先按配置中的冻结 yaw/pitch/FOV 投影为普通透视视野；投影方向仍属于近似相机几何，不能宣称眼镜真实光轴。原始全景直用造成的失败 run 必须保留，不能覆盖。

## 输出

所有抽帧、路线覆盖图、教师收据、共识和代理评测只写入调用者指定的 `artifacts.local/evidence/ustrf-crosscam-codex/` 子目录；模块目录不接收生成物。

## 安全边界

- `codex_visual_teacher_provisional_v1` 是交互式视觉银标，不是人工事件真值、设备米制几何或用户提醒权威。
- 几何只能声明为 `assumed_geometry_v1` / `pseudo_metric`；正式 U0 120/60 人类真值合同、Android、训练和生产权限保持 false。
- 本通道不修改 App、默认 YOLO、正式 U0 evaluator 或设备 admission。

## 停止条件

若跨来源最差值没有方向一致的 route-specific 增益，结果依赖未来帧泄漏、单一教师自评、单一相机假设或 `unknown` 膨胀，则停止 R0，不调阈值回救；改写数据/评审假设后另开一轮。

R1.1 诊断另有硬停止：目标实例账本必须先冻结唯一实例、可见性、bbox 与接触点；投影必须逐帧精确覆盖，或来自预声明的稳定窗口。普通 person/car 只能计为共现，不能代替目标召回。当前六来源已解封，只能作为 `seen_diagnostic_not_held_out`；最终门必须换新来源。

R1.2a 只允许把已经解封的 R1.1/R1.2 降级为 5–15 秒连续工程诊断。连续关联只能由冻结 target anchor 启动并在后续冻结 anchor 上复核；未标注的中间帧只是 association trace，不升级为连续人工真值。Vancouver 固定为 `miss_lead_only`，不得回调 prompt、`.05/.30`、bbox、polygon 或门槛。R1.3 仅冻结 12 个未打开槽位与双 VLM 独立复核合同，R1.2a 期间禁止来源发现、下载、解码或打分。

R1.2b 只在同一 seen diagnostic 上做移动端工程隔离：先验证冻结 canary 与设备输入帧等价，再按预注册顺序选择第一个同时通过 parser/延迟门的 backend/size 候选。第一个候选通过后必须停止后续候选；连续重放使用逐帧 SHA-256 绑定 PNG，不能让 Android 原视频解码差异污染 detector 归因。它不授权 R1.3 解封、训练、App 默认 backend 或生产模型替换。
