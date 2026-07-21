# USTRF-SC research implementations

状态：active

## 稳定 Interface

领域外调用只经过 `scripts/` 根目录的 U0 teacher artifact/Android adapter。实现只接收去标签视频、frame ledger、显式或预注册 control route、hash-bound LOSO artifact 和冻结配置；任何 event/review/adjudication/blind 字段均拒绝。

## 输出

只写调用者明确指定的 `artifacts.local/` 路径。模型、下载、临时场与设备证据不得写入仓库根目录。

## 安全边界

Depth Anything V2 Small 仅是 Apache-2.0 的离线相对深度辅助 teacher，不是米制深度、人体事件真值、Android 运行时模型或生产授权。最终 decision 必须由设备内 shared `AssistDecisionKernel` 从 object-agnostic evidence 产生。

## 停止条件

若正式 U0 中 dense route 相对 detector/uniform/shuffled 无预注册增益，或收益依赖 future/blind/同源泄漏、unknown 扩张、手工事后解释，则停止该 teacher 路线，不扫描阈值回救。
