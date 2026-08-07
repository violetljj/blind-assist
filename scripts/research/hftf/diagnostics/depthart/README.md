# DepthART diagnostics module

状态：`diagnostic / deployment-observability-only`

## 稳定 Interface

本目录是 QNN/HTP operator profile、linting 和资源重叠诊断的规范入口。
Python 分析器、linting 配置和设备运行脚本已物理迁入本目录；仓库内部旧 shim 已退役。
`run_qnn_native_cached_context_r0.ps1` 也属于本目录的设备边界诊断。
设备脚本只产生部署诊断证据，不改变默认 App。

## 输出

- operator profile CSV 聚合
- HTP linting 文本聚合
- 只写入 `artifacts.local/evidence/` 的诊断 receipt

## 安全边界

这些结果只说明 operator、HTP 资源和 profiling 观察，不是 App 端到端延迟、
算法准确率、产品安全或默认 App 授权。

## 停止条件

- ADB 设备不在线、QNN/QAIRT 工具缺失时停止，不伪造结果
- profiling 输出不完整或运行间 operator 数变化时停止聚合
- 不把诊断结果登记为新的算法 successor

产物目录：`artifacts.local/`
