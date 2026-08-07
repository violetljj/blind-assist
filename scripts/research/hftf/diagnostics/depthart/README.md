# DepthART diagnostics module

状态：`diagnostic / deployment-observability-only`

## 稳定 Interface

本目录是 QNN/HTP operator profile、linting 和资源重叠诊断的规范入口。
Python 分析器和 linting 配置已物理迁入本目录；旧路径保留兼容 shim。
设备运行脚本仍暂留在 `scripts/research/hftf/`，待设备命令路径回归后再迁移。

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
