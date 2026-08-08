# HFTF support migration queue

状态：`current / bounded-fallback`

`support` 只表示文件名仍不足以判断职责，不是执行权限。角色统计只计算 Git 可见文件，
忽略 build/cache 产物。每次只处理一个可验证的主题簇，
保留兼容 Adapter、manifest 和回归结果。

## 队列

| 优先级 | 主题簇 | 目标角色 | 完成条件 |
|---|---|---|---|
| P0 | DepthART QAIRT/QNN/HTP/SelectiveScan | `deployment` | 文件、converter、协议和输出路径可由一个 Adapter 重放 |
| P1 | metric-depth / clearance / student / teacher | `current` 或 `archive` | current 真源声明 successor；关闭路线不得挂 active 下一步 |
| P1 | parity / export / Android bridge | `deployment` 或 `diagnostics` | parity 与部署证据分开 |
| P2 | stage-C / D-series 历史实验 | `archive` 或 `diagnostics` | 确认无当前调用方后迁移完整主题簇 |
| P2 | ToF / camera / capture helper | `diagnostics` | 明确输入、输出和 artifacts.local 根目录 |

## 每批合同

1. `scripts/research/audit_research_structure.ps1 -Json` 生成迁移前报告。
2. 用 `rg` 查找调用方、protocol、receipt 和旧路径引用。
3. 建立新路径、兼容 Adapter 和最小回归测试。
4. 迁移后确认 support 数量下降且无调用方断裂。
5. 只提交该主题簇；并行工作树文件不得混入。

当前 Git 可见 `support = 0`，没有主动迁移批次。新增文件若只能落入 `support`，结构门禁会
直接失败；先明确职责并更新 `roles.json`，不能把含混分类留给下一窗口。
