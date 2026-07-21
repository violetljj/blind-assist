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
