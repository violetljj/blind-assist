# HFTF / DepthART Module Index

状态：`current / role-index`

这是 HFTF 目录的短职责入口；详细状态以
[`docs/research/hftf/README.md`](../../../docs/research/hftf/README.md) 为准。
机器匹配规则见 [`roles.json`](roles.json)；`support` 是迁移缓冲区，不代表当前执行权限。
DepthART P0 部署簇的最短读取路径见
[`DEPTHART_P0_DEPLOYMENT_INDEX.md`](DEPTHART_P0_DEPLOYMENT_INDEX.md)。

| 分区 | 识别规则 | 典型文件/目录 | 权限 |
|---|---|---|---|
| `current` | DepthART、metric-depth、clearance、student、teacher | `train_*student*`, `evaluate_*clearance*` | 仅按 current successor 执行 |
| `deployment` | `deployment/` 或 `qnn`、`qairt`、`onnx`、`htp`、`selective_scan`、`converter`、`package` | `deployment/depthart/*`, `rewrite_depthart_qairt_*` | 只证明导出/部署可行性，不证明算法或安全 |
| `diagnostics` | `diagnostics/` 或 `test_*`、`inspect_*`、`render_*`、`*_audit*`、`*_parity*`、`*_probe*` | `diagnostics/depthart/*`, `analyze_*` | diagnostic-only，不能自动产生 successor |
| `archive` | 已关闭 round、旧 teacher/student、历史 replay | 由对应 README 标明 `closed/paused/archive` | 不得作为当前下一步 |

## 维护合同

- 新文件先按职责命名，再登记到本页；不要把部署 converter 放在算法脚本旁而不标记。
- 不为美化目录重命名已被 protocol、receipt 或外部命令引用的历史文件。
- 需要真实迁移时，先建立 manifest、兼容 Adapter 和回归测试，再分批移动。
- 新文件若命中 `support`，必须在下一次结构治理中补充更具体的角色或标记为 archive。
