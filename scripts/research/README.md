# Research script Modules

新的研究路线必须创建 `scripts/research/<domain>/`，不得把轮次脚本重新平铺到 `scripts/` 根目录。每个 Module 的 `README.md` 至少包含以下合同：

```markdown
# <domain>

状态：proposal | active | frozen | archive

## 稳定 Interface
调用方式、输入不变量和失败模式。

## 输出
只允许写入 artifacts.local/ 下的明确目录。

## 安全边界
数据权威、训练、设备、Android 和生产授权边界。

## 停止条件
冻结验收指标；连续两轮同一假设失败后停止参数回救，改写假设或关闭路线。
```

领域外调用只能经过 `scripts/` 根目录的稳定 Adapter。共享 Implementation 只有在至少两个调用域证明真实复用后才能进入 `research/common/`；不要预先制造万能框架。
