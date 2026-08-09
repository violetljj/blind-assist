# Research Module Registry

状态：`current / navigation-only`

本页是 `scripts/research/` 的职责索引，不复制实验结果。先按“当前入口、部署、诊断、
历史”定位，再打开对应 Module README。轮次名（`r0/r1/r2/p0/d0`）只表示 evidence
identity，不代表当前 authority。

全部 Module 的逐项入口和机器校验数量见 [`MODULE_INDEX.md`](MODULE_INDEX.md)，机器分类规则见
[`module_families.json`](module_families.json)。新增 Module 若未被唯一分类，结构门禁失败。

## 当前入口

| 工作域 | 稳定入口 | 当前职责 |
|---|---|---|
| DepthART/HFTF | [`hftf/README.md`](hftf/README.md) | 算法候选、几何学生和部署前置；动态状态以 `docs/research/hftf/README.md` 为准 |
| 双环 | [`dual_loop/README.md`](dual_loop/README.md) | 论文次线与 risk-seg/temporal 研究；动态状态以 `docs/research/dual-loop/README.md` 为准 |
| 候选事件 | [`candidate_event_mining/README.md`](candidate_event_mining/README.md) | discovery-only 候选池与复核 bundle |
| 公共实现 | [`common/README.md`](common/README.md) | 至少两个域证明复用后的共享实现 |

## HFTF 职责分层

`hftf/` 暂不大规模移动历史文件；以下是稳定的语义分区，按文件名和 README 归属定位：

- `current`：DepthART、当前 metric-depth 候选和仍有明确 successor 的实现。
- `deployment`：ONNX、QAIRT、QNN、HTP、SelectiveScan、converter 和 device preflight。
- `diagnostics`：parity、质量屏、可视化、failure atlas 和不产生 promotion authority 的探针。
- `platform`：ToF、CameraX、端到端延迟、持续性能和设备适配。
- `archive`：已关闭的 HFTF campaign、旧学生、旧 teacher 和历史 replay。
- `governance`：README、索引、角色合同和迁移规则。

如果一个文件同时命中多个分区，以 `roles.json` 的 `role_order` 为唯一优先级；不要从
文件名推导当前权限，必须回到对应 current README。`support` 配额为 0，不能留待以后分类。

## 统一定位规则

1. 先读本页，再读目标 Module 的 README。
2. 只调用 `scripts/` 根稳定 Adapter；不要让外部调用方依赖 Module 内部文件路径。
3. 输出只能写入 `artifacts.local/`，源码、协议、收据和生成物不混放。
4. 新 Module 必须先登记本页，并声明 `status/authority/successor/default_app_impact/artifact_root`。
5. 历史 Module 不删除；若不再承担当前执行职责，标记 `archive`、`paused` 或 `diagnostic`。
