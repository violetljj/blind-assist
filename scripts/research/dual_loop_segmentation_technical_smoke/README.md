# Dual-loop segmentation technical smoke

状态：`canary` / `TECHNICAL_ONLY`

## 研究问题与版本

`DUAL_LOOP_SEGMENTATION_TECHNICAL_SMOKE_R0` 只回答一个接口问题：一个已声明的
轻量 model-B reference 是否能在固定 RGB 输入上加载、产生预期空间输出、返回有限值，
并给出可描述的 argmax 类别分布与主机运行时间。

它是中央图像阻塞 D0-A successor 之后的独立 technical audit，不是 D0-A、D0-B
效果评价或模型选型结论。输入可以复用 D0-A successor 的排除式 RGB slot，但脚本不
读取 Agent 标签、中央阻塞状态、YOLO 输出、风险、反馈或融合结果。

## 稳定 Interface

从仓库根目录运行：

```powershell
& E:\codex-tools\tools\venvs\blindassist-venv-export312\Scripts\python.exe `
  scripts\research\dual_loop_segmentation_technical_smoke\technical_smoke.py `
  --manifest artifacts.local\evidence\central-obstruction-agent-label-readiness-d0-a-successor-r0\calibration-input-manifest.json `
  --model artifacts.local\evidence\segmentation-candidate\gpu-smoke-20260713-int8.tflite `
  --output artifacts.local\evidence\dual-loop-segmentation-technical-smoke-r0\report.json `
  --visualization-dir artifacts.local\evidence\dual-loop-segmentation-technical-smoke-r0\visualizations
```

输入 manifest 必须含 `fixed_units[].observations[]`，每个 observation 必须绑定
`unit_id`、`session_id`、`slot_ordinal` 和 `review_image_path`。模型只能声明一个；
该 runner 没有候选比较、阈值、类别重映射、拓扑算子或融合参数接口。

## 输出

所有产物只写入 `artifacts.local/evidence/dual-loop-segmentation-technical-smoke-r0/`：

- `report.json`：模型/manifest SHA256、输入输出 tensor 合同、有限值检查、类别像素分布、
  每样本主类和主机端 P50/P95/MAX；
- `visualizations/clip-*.png`：每个 fixed clip 的预测 overlay contact sheet，仅作输出
  可视化，不含风险标签或真值。

`PASS_INTERFACE_ONLY` 只能证明 technical interface 可运行。类别塌缩、全零/全一分布或
缺少非 walkable 输出必须如实记录为诊断警告，不能转写成分割有效性、可通行性、风险或
YOLO 互补增量。

## 安全边界

- 不访问 D0-A Agent label、中央阻塞结论、YOLO trace、风险/反馈或融合输出。
- 不进入 D0-A readiness、不改变 D0-A terminal、不自动授权 D0-B。
- 不选择或比较多个模型；不调模型、类别、阈值、连通域、路线或主阻塞算子。
- 不写 `app/src/main/assets/`，不修改默认模型、Android、生产行为或提醒。
- 不把主机耗时当成手机/Snapdragon 延迟；没有设备测量时设备性能保持 `NOT_MEASURED`。

## 停止条件

- 接口/有限值失败只关闭本 technical-smoke evidence instance；不会扩大为整个双环问题关闭。

## 失败资产复用

报告和 overlay 可作为 diagnostic、可视化和后续候选的 regression fixture。失败的
reference 不得包装成 unseen confirmation、A/B/C 增量或生产证据；后续若要进入模型
选择或融合，必须另立明确的 Development 实验并冻结客观互补单位。
