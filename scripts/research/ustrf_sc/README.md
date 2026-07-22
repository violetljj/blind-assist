# USTRF-SC research implementations

状态：active

## 稳定 Interface

领域外调用只经过 `scripts/` 根目录的稳定 Adapter。U0 实现只接收去标签视频、frame ledger、显式或预注册 control route、hash-bound LOSO artifact 和冻结配置；model-proxy pilot 与 ARCore canary 使用各自独立 schema/validator，不向 U0 candidate runner 传 review、adjudication 或 blind 字段。

## 输出

只写调用者明确指定的 `artifacts.local/` 路径。模型、下载、临时场与设备证据不得写入仓库根目录。当前 model-proxy pilot 输出 10 episode、双模型回执、生成 lineage 与逐帧哈希；ARCore canary 输出原始 JSONL、设备收据和 host audit。

## 安全边界

Depth Anything V2 Small 仅是 Apache-2.0 的离线相对深度辅助 teacher，不是米制深度、人体事件真值、Android 运行时模型或生产授权。模型生成/模型审核的 pilot 同样不是 human truth；它只能授权扩大代理矩阵。最终 decision 必须由设备内 shared `AssistDecisionKernel` 从 object-agnostic evidence 产生。

## 停止条件

若 ARCore canary 达不到 `100 / 0.95 / INTER_FRAME_STABLE`，冻结手机 metric geometry，不重试同一窗口、不扩代理矩阵或开启 U0。只有 geometry gate 通过后，若正式 U0 中 dense route 相对 detector/uniform/shuffled 无预注册增益，或收益依赖 future/blind/同源泄漏、unknown 扩张、事后解释，才停止 teacher 路线且不扫描阈值回救。
