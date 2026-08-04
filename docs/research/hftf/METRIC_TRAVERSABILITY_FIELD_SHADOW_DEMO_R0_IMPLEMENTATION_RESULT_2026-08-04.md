# MetricTraversabilityField shadow/demo R0 实现结果

日期：2026-08-04（Asia/Hong_Kong）
终态：`IMPLEMENTATION_COMPLETE / CONTRACT_AND_RENDER_TESTS_PASS / DEVELOPMENT_ONLY_SHADOW_DEMO / MAINLINE_UNCHANGED / DEFAULT_APP_UNCHANGED`

## 结论

目标文件要求的软件分层已经落地：HFTF 侧车现在能保留连续方向的米制观测场、三档身体扫掠包络、侵入区域、相邻帧趋势、质量/支持/provenance 和 `UNKNOWN`；左/中/右只由独立 `AlertMapper` 产生，且输出固定为 non-actuating shadow/demo。四联屏渲染器可同屏显示 RGB、米制深度热图、BEV 身体包络和方向净空曲线，并能输出 PNG 序列或 MP4。

这是一项软件机制和可视化能力结果，不是新模型效果结果。没有运行 fresh/held-out 评价，没有建立安全方向、绕行、训练增量、正式提醒或生产权限。

## 已实现

- Python producer：`metric_traversability_field.py`。
- 既有 external RGB sparse-scale sidecar 追加 rich field、shadow alert 和可选显示资产；旧 `raw_clearance/scaled_clearance` 字段保留。
- 侧车记录冻结画质诊断；模糊/极端曝光为 `UNKNOWN_IMAGE_QUALITY`。可选压缩 NPZ 保留原始/校正深度并写入 SHA-256、尺度和内参收据。
- 动态四联屏：`render_metric_traversability_field_demo.py`。
- Kotlin 隔离合同与 mapper：`:hftf-metric-depth-canary-core`；不在默认 App runtime graph。
- JSON Schema：`schemas/metric_traversability_field_r0.schema.json`。
- Python 与 Kotlin 聚焦测试。

## 使用

侧车原命令追加：

```text
--visualization-dir artifacts.local/evidence/hftf/metric-field-r0/assets
--research-depth-dir artifacts.local/evidence/hftf/metric-field-r0/depth
--demo-alert-horizon-m 1.5
```

然后渲染：

```powershell
& E:\codex-tools\bin\blindassist-python.cmd render_metric_traversability_field_demo.py `
  --input artifacts.local/evidence/hftf/metric-field-r0/sidecar.jsonl `
  --frames-dir artifacts.local/evidence/hftf/metric-field-r0/frames `
  --video artifacts.local/evidence/hftf/metric-field-r0/demo.mp4 `
  --summary artifacts.local/evidence/hftf/metric-field-r0/render_summary.json
```

上述命令应在 HFTF 研究脚本目录内执行；该 renderer 是研究实现入口，不是跨模块稳定接口。

显示资产和运行证据继续放在 ignored `artifacts.local/`，不把大 payload 写进仓库。
在没有合法真实米制输入时，可用
`generate_metric_traversability_synthetic_demo.py` 复核四联屏和动态包络软件机制；
该脚本会在画面、JSONL 和 summary 三处标记 `SYNTHETIC / evidence_authority=false`，
不得进入算法效果材料。

## 验证

- `test_metric_traversability_field.py`：6/6 pass。
- `test_metric_scale_anchor.py`：4/4 pass；丰富主输出可独立解析尺度收据，不依赖 legacy 三带非空。
- `test_run_external_rgb_clearance_sidecar.py`：4/4 pass。
- `test_render_metric_traversability_field_demo.py`：1/1 pass。
- 新实现与既有 clearance/occupancy/sparse-scale 聚焦回归合计：28/28 pass。
- JDK 17 下 `:hftf-metric-depth-canary-core:test`：32/32 pass。
- 既有聚焦回归、schema/协议一致性、hygiene 和 diff 检查见本次最终交付记录。

## 仍然不成立的结论

- scale-free 最低分三带不是米制净空或安全方向。
- Known-height 已消费手机结果不产生合法米制效果。
- `CLEAR_OBSERVED` 不等于真实环境通行，更不等于安全。
- 候选方向不触发用户提醒；未来接入需要独立数据、设备、human-factors 和 promotion 合同。
