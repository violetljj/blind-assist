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
E:\codex-tools\bin\blindassist-python.cmd scripts/run_research_tool.py ustrf-crosscam-codex evaluate_r12c_truth_geometry_consistency.py --help
E:\codex-tools\bin\blindassist-python.cmd scripts/run_research_tool.py ustrf-crosscam-codex evaluate_r12c_v2_truth_geometry_consistency.py --help
E:\codex-tools\bin\blindassist-python.cmd scripts/run_research_tool.py ustrf-crosscam-codex materialize_r12c_v2_exact_frame_input.py --help
E:\codex-tools\bin\blindassist-python.cmd scripts/run_research_tool.py ustrf-crosscam-codex download_r12d_roadwork_images.py --help
E:\codex-tools\bin\blindassist-python.cmd scripts/run_research_tool.py ustrf-crosscam-codex prepare_r12d_training_dataset.py --help
E:\codex-tools\bin\blindassist-python.cmd scripts/run_research_tool.py ustrf-crosscam-codex train_r12d_detector_matrix.py --help
E:\codex-tools\bin\blindassist-python.cmd scripts/run_research_tool.py ustrf-crosscam-codex evaluate_r12d_model.py --help
E:\codex-tools\bin\blindassist-python.cmd scripts/run_research_tool.py ustrf-crosscam-codex summarize_r12d_multiseed.py --help
```

输入必须是有来源、许可和 SHA-256 收据的公开第一视角视频。准备器只向 Codex 教师暴露当前/过去图像、时间戳和预注册路线覆盖；隐藏代理标签、未来帧、深度、review、adjudication 或 blind 字段不得进入评审材料。

360° 等距柱状视频必须先按配置中的冻结 yaw/pitch/FOV 投影为普通透视视野；投影方向仍属于近似相机几何，不能宣称眼镜真实光轴。原始全景直用造成的失败 run 必须保留，不能覆盖。

## 输出

所有抽帧、路线覆盖图、教师收据、共识和代理评测只写入调用者指定的 `artifacts.local/evidence/ustrf-crosscam-codex/` 子目录；模块目录不接收生成物。

## 安全边界

- `codex_visual_teacher_provisional_v1` 是交互式视觉银标，不是人工事件真值、设备米制几何或用户提醒权威。
- 几何只能声明为 `assumed_geometry_v1` / `pseudo_metric`；正式 U0 120/60 自动多模型事件参考合同、Android、训练和生产权限保持 false，直到各自自动门禁闭合。
- 本通道不修改 App、默认 YOLO、正式 U0 evaluator 或设备 admission。

## 停止条件

若跨来源最差值没有方向一致的 route-specific 增益，结果依赖未来帧泄漏、单一教师自评、单一相机假设或 `unknown` 膨胀，则停止 R0，不调阈值回救；改写数据/评审假设后另开一轮。

R1.1 诊断另有硬停止：目标实例账本必须先冻结唯一实例、可见性、bbox 与接触点；投影必须逐帧精确覆盖，或来自预声明的稳定窗口。普通 person/car 只能计为共现，不能代替目标召回。当前六来源已解封，只能作为 `seen_diagnostic_not_held_out`；最终门必须换新来源。

R1.2a 只允许把已经解封的 R1.1/R1.2 降级为 5–15 秒连续工程诊断。连续关联只能由冻结 target anchor 启动并在后续冻结 anchor 上复核；未标注的中间帧只是 association trace，不升级为连续人工真值。Vancouver 固定为 `miss_lead_only`，不得回调 prompt、`.05/.30`、bbox、polygon 或门槛。R1.3 仅冻结 12 个未打开槽位与双 VLM 独立复核合同，R1.2a 期间禁止来源发现、下载、解码或打分。

R1.2b 只在同一 seen diagnostic 上做移动端工程隔离：先验证冻结 canary 与设备输入帧等价，再按预注册顺序选择第一个同时通过 parser/延迟门的 backend/size 候选。第一个候选通过后必须停止后续候选；连续重放使用逐帧 SHA-256 绑定 PNG，不能让 Android 原视频解码差异污染 detector 归因。它不授权 R1.3 解封、训练、App 默认 backend 或生产模型替换。

R1.2c 先把六个正事件的 alertable anchor 与独立 route oracle 对齐；每个正例至少需要一个在 `.01/.02/.03` 三档不确定性下均为 inside 的 anchor。缺失时固定标为 `truth_geometry_conflict`，由两个 fresh-context 模型独立复核，再由第三模型仲裁；这里的“人工角色”由模型承担，不等待真人，但也不授予真人或生产真值权限。禁止拖动旧 polygon 回救历史失败。只有六例全部一致才允许唯一的 London FP16-768 GPU 单变量候选；机械 canary 后直接跑完整连续事件，须正例 `6/6` 且负例、重复交付、共现接管、身份切换均为 `0`，才可运行 600 秒 soak。768 仍漏 London 时停止分辨率搜索，转向新预注册的小目标 detector 假设；R1.3 在全门通过前保持封存。

Japan 经模型仲裁排除后，非 R1.3 补位只允许使用在 R1.2c 前已经打开的 seen source。`validate_r12c_seen_positive_prereg.py` 会验证来源/许可/视频/关键帧哈希、两份独立模型复核、唯一目标、`.01/.02/.03` robust geometry、清除代理和 R1.3 封存状态。补位合同本身不授权 768；必须先把补位事件物化到新的 R1.2c 连续清单并重跑完整六正例 oracle。

R1.2c v2 已以 Bangkok 完成六正例 oracle `6/6`，但 768 真机连续事件仍漏 London，故 600 秒 soak 被顺序门跳过，分辨率搜索关闭。R1.2d 只允许前瞻冻结的 stride-4/P2 小目标 detector 假设；候选权重、训练 manifest 与数据审查/许可/精确几何收据未齐前，candidate execution、R1.3、训练和生产权限均为 false。

R1.2d 只把显式 stride-4/P2 与同家族 stride-8/P3 control 做配对因果比较：两臂共享固定数据、三组 seed、backbone 张量、训练配方、`.05/.45/.30` 阈值和 R1.2c v2 的 12 事件输入。公开 ROADWork 的 Pittsburgh train 与其他城市 validation 必须来源隔离；Mendeley bollard 先按精确图像 SHA-256 去重，只进训练。事件告警生成不得读取 `expected_class`，但 target anchor 仍用于事后目标条件关联，因此“目标假告警为零”不等于全局无误报；必须同时报告未关联 route-inside detection pressure。未激活目标的 10 秒值是右删失哨兵，不可混入实际清除延迟；歧义必须和关联覆盖率按来源共同报告。不得挑 lucky seed、事后改阈值、用 aggregate 掩盖最差来源，或从正向研究结果直接授权 R1.3、INT8、设备、App/生产替换。
