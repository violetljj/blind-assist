# HFTF / DepthART research Module

状态：`current / module-contract / dynamic-authority-delegated`

本页只定义稳定调用面。DepthART/HFTF 的动态状态、唯一 successor 和执行权限只由
[路线 current](../../../docs/research/hftf/README.md)持有；完整旧 Module 叙事保存在
[2026-08-13 历史快照](ARCHIVE_FULL_MODULE_HISTORY_2026-08-13.md)，不能据此恢复旧实验权限。

职责分区和文件匹配规则见 [INDEX.md](INDEX.md) 与 [roles.json](roles.json)。部署、诊断、
平台和已关闭 round 必须保持角色分离；`support` 文件预算为 0。

## 稳定 Interface

从仓库根目录调用稳定 Adapter：

```powershell
E:\codex-tools\bin\blindassist-python.cmd scripts/run_research_tool.py hftf <tool.py> [args...]
```

长任务、正式 one-shot 或资源风险较高的执行，按 owning contract 通过：

```powershell
pwsh -NoProfile -File scripts/run_guarded_host_research.ps1 <owning arguments>
```

工具选择顺序：

1. 先从 [DepthART route current](../../../docs/research/hftf/README.md)确认唯一 successor 与 authority；
2. 再从 [HFTF role index](INDEX.md)定位 `current / deployment / diagnostics / platform / archive`；
3. 只读取 successor 直接绑定的 runner、validator、protocol 和 tests；
4. 不从本目录文件名、mtime、历史 README 或旧 execution lock 推导权限。

## 输出

输出只能写入 owning protocol 声明的 `artifacts.local/work/` 或
`artifacts.local/evidence/hftf/<run-id>/` 新目录。正式 evidence root 不得覆盖、原地 repair
或用新进程续写，除非 owning current 明确声明可恢复协议和 exact checkpoint 前缀。

## 安全边界

- 本 Module 是 host research surface，不自动影响 Android、默认模型、反馈策略或产品权限；
- teacher、synthetic、source-derived、model output 和 device measurement 必须分开标记；
- `UNKNOWN`、缺 source、缺 truth 和不可观测状态不得补零或当作 negative；
- 导出、QNN/HTP、性能或可视化通过不证明算法精度、用户效果或安全。

## 停止条件

遇到 authority/hash/schema/data-role/root-collision 漂移、owning terminal、超预算、缺失必需
source，或 validator 无法独立复算时立即停止，并按 owning contract 只关闭受影响的最小
evidence version。不得为获得“完整结果”而降低门、替换 identity、读取未授权 outcome 或重跑
consumed one-shot。

## 维护合同

- 新文件必须命中 [roles.json](roles.json) 的具体职责，不能落入 `support`；
- 新稳定入口登记到 [INDEX.md](INDEX.md) 或根 [scripts index](../../README.md)，不把研究轮次重新平铺到根目录；
- 详细 round 说明写入日期化 protocol/result 或 archive，不再追加到本 README。
