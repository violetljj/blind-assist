# RCLE 证据访问与传输标准 R1

日期：2026-07-27

状态：`ADOPTED_FORWARD_ONLY / RISK_BASED`

## 目的

本标准纠正把“字节经过工具”机械等同于“读取并使用该模态”的做法。访问限制应保护
候选独立性、避免 outcome 泄漏并阻止未授权算法运行，而不应因为归档格式、压缩布局
或等价传输实现制造虚假的 `NOT_EVALUABLE`。

历史合同、receipt 和 terminal 不回写；与本标准不一致的旧访问条款由明确引用本标准
的后继协议取代。

## 分层访问语义

后继协议必须分别记录以下状态，不再用单个 `accessed=true/false` 代替：

| 层级 | 含义 | 是否自动构成 RGB 使用 |
| --- | --- | --- |
| `transport_presence` | 压缩对象、网络 range 或 archive block 到达本地/内存 | 否 |
| `transient_decode` | 解码器因 solid/co-pack 格式经过非目标成员并立即丢弃 | 否 |
| `materialized_content` | 成员被写盘、缓存、导出或交给可复用消费者 | 是 |
| `human_or_model_inspection` | 人或通用视觉模型查看 RGB 内容 | 是 |
| `claim_algorithm_consumption` | RCLE RGB 算法读取 RGB | 是 |
| `selection_or_tuning_influence` | 内容或 outcome 影响候选、窗口、参数或停止规则 | 是 |

“不运行 RGB”默认表示：

- 不把 RGB 输入 RCLE 或其他用于当前 claim 的视觉算法；
- 不查看、渲染、导出或持久化 RGB；
- 不用 RGB 内容或 outcome 选源、选窗、调参或解释角色。

它不禁止为取得已授权 geometry 成员而下载含 RGB 的压缩对象，也不禁止 solid
decoder 在内存中经过并立即丢弃不可分离的 RGB 字节。

## 传输与实现规则

1. 在 identity、许可、预算和目标 member allowlist 已冻结后，可选择 range、
   member、solid block、完整 archive 或等价缓存中实际可行且资源合理的获取方式。
2. “选择性下载”是优先策略，不是科学门槛。完整 archive 只有在超出预算、违反许可、
   无法验证身份或会造成未受控持久化时才禁止。
3. 非目标成员允许 `transient_decode`，但必须在同一处理步骤丢弃；不得写盘、生成
   thumbnail、交给图像库、人工查看或算法消费。
4. 可恢复的超时、range 不兼容、solid co-pack、工具缺陷或缓存失败属于工程事件。
   可在候选、窗口、公式和门槛不变时切换等价 transport，不需要另立科学协议。
5. transport 变更记入 execution ledger。只有改变候选身份、数据角色、窗口规则、
   科学门槛、公式/算法，或新增语义内容访问时，才需要新的 preaccess contract 和
   独立 review。
6. 不因 candidate order 中较早来源出现可恢复工程问题而跳过其他已锁定来源。
   各来源独立执行；在所有合理且授权的 transport 路径耗尽后，才可把该来源判为
   transport `NOT_EVALUABLE`。

## 保留的硬边界

本标准不放松以下科学与权限边界：

- 不在结果后扩候选、替换来源、增补窗口或降低门槛；
- 不从名称、描述或 metadata 推断 motion role；
- 不把 RGB 输入 RCLE 算法，不查看 RGB，不用 RGB 影响 selection/tuning；
- 不把失败 pair 移出固定 denominator，也不跨来源 pooled rescue；
- 不启动 Android、设备、产品或安全外推；
- 不绕过许可、身份、完整性、隐私和总资源预算。

## 终态原则

`NOT_EVALUABLE` 应表示证据在合理且授权的路径耗尽后仍无法取得或验证，不能表示
“首选工具不支持”“压缩流包含未使用模态”或“为了保持记录整齐而拒绝等价实现”。
