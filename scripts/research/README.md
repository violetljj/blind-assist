# Research script Modules

## 30 秒定位

先读 [`REGISTRY.md`](REGISTRY.md)，再按职责进入一个 Module；HFTF/DepthART 的细分入口是
[`hftf/INDEX.md`](hftf/INDEX.md)。这些索引只维护路径、职责和权限边界，不复制动态研究结论。
结构审计使用 [`audit_research_structure.ps1`](audit_research_structure.ps1)，只读输出 Module
合同和 HFTF support 迁移清单。

新的研究路线必须创建 `scripts/research/<domain>/`，不得把轮次脚本重新平铺到 `scripts/` 根目录。每个 Module 的 `README.md` 至少包含以下合同：

```markdown
# <domain>

状态：proposal | discovery | canary | development | confirmation | frozen | archive

## 研究问题与版本
声明 scientific question、protocol version、evidence instance、当前 stage 和
允许 claim。遵循 `docs/RESEARCH_GOVERNANCE.md`。

## 稳定 Interface
调用方式、输入不变量和失败模式。

## 输出
只允许写入 artifacts.local/ 下的明确目录。

## 安全边界
数据权威、训练、设备、Android 和生产授权边界。

## 停止条件
定义信息增益/资源预算和最小 failure scope。失败必须写 learning record；
停止具体候选、实现、来源或 evidence version，不因固定次数自动关闭整个问题。

## 假设与规则质疑
候选写明 causal difference、expected information gain、falsifier、cost 和
selection reason。允许质疑 AGENTS/current/threshold，但必须版本化修改，不静默绕过。

## 失败资产复用
声明失败数据/代码能否作为 negative evidence、diagnostic、regression fixture、
canary、counterexample、stress case 或 source characterization；不得重新包装为
unseen confirmation。
```

领域外调用只能经过 `scripts/` 根目录的稳定 Adapter。共享 Implementation 只有在至少两个调用域证明真实复用后才能进入 `research/common/`；不要预先制造万能框架。
