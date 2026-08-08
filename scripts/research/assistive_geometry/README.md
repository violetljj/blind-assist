# Assistive Geometry research scripts

状态：`B0_CONTRACT_AND_PREFLIGHT_ONLY / NO_TRAINING_AUTHORITY`

本目录只包含 BlindAssist Assistive Geometry B0 的冻结合同验证与 synthetic rectangular
shape/export preflight：

## 稳定 Interface

- `validate_b0_task_contract.py`：对 B0 JSON 合同执行 fail-closed schema/不变量检查；
- `test_validate_b0_task_contract.py`：覆盖有效合同和关键违规合同；
- `preflight_depthart_rectangular_shape.py`：用真实 DepthART-S metric checkpoint 验证
  `1×3×608×448` PyTorch shape、dynamic camera prompt 与 ONNX graph/checker。

## 输出

输出只允许写入 `artifacts.local/evidence/hftf/`。这些脚本不选择数据 roster、不读取独立
task outcome。当前合同和结果真源位于 `docs/research/assistive-geometry/`。

## 安全边界

本模块不训练 student、不运行 teacher matrix，也不授权 QNN/HTP、默认 App、产品或 safety。
`UNKNOWN` 不得当作负例；synthetic shape 与 benchmark geometry 不得冒充任务质量。

## 停止条件

合同违规、checkpoint/shape 不匹配、非 finite 输出、camera prompt drift 或 ONNX checker
失败均立即 fail closed。B0 数据 roster 未冻结前不得从本目录启动训练或打开独立 outcome。

验证：

```powershell
E:\codex-tools\bin\blindassist-python.cmd -m unittest `
  scripts.research.assistive_geometry.test_validate_b0_task_contract
```
