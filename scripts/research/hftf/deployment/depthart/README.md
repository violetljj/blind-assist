# DepthART deployment module

状态：`current / deployment-only / adapter-first`

## 稳定 Interface

本目录是 DepthART QAIRT/QNN/HTP/ONNX 与 SelectiveScan 部署工作的规范入口。
P0-A 实现已物理迁入本目录；旧路径由 shim 和 `legacy_adapter.py` 提供兼容导入。
本目录同时保存 P0-A 的定向回归测试；旧测试路径保留兼容 shim。

## 输出

- 候选 ONNX/QAIRT 图
- lowering、operator、parity 和 backend 诊断记录
- 只写入 `artifacts.local/` 的 receipt 与日志

## 安全边界

部署脚本只能证明导出、lowering、数值 parity 或设备可行性，不能单独证明算法准入、
默认 App、产品安全或生产授权。

## 停止条件

- parity 或 lowering 不通过时停止进入 HTP/backend 评估
- 缺少冻结输入、receipt 或调用方清单时停止物理迁移
- 不移动并行任务产生的 SelectiveScan `.cpp/.xml/.exp/.lib`

产物目录：`artifacts.local/`
