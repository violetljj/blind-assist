# Codex 高效执行与上下文控制

状态：current

适用范围：BlindAssist 日常工程、研究实施、长任务续作与共享工作树协作。

本页承接不需要在每次启动时完整加载的执行细节。权限、Git 硬边界、研究
authority 摘要和机械验证入口仍以根 `AGENTS.md` 为准。

## 六项执行合同

非平凡任务在开始执行前，将请求归一化为下列六项。能从用户请求、当前代码和
current 文档安全推断的字段直接填写，不为形式完整反复询问。

```text
目标：
范围：
已知入口：
禁止事项：
完成标准：
验证命令：
```

合同用于限制探索半径，不是额外审批门。小型、明确、单文件任务可以只在工作计划
中隐式维护；跨窗口或高风险任务写入 handoff。

## 最小读取与搜索

1. 从 `docs/PROJECT_STATE.md` 选择任务类型。
2. 普通工程只读直接相关代码、测试和 current 文档；不默认读取研究、设备、发布、
   archive、snapshot、全量日志或 `artifacts.local/`。
3. 优先使用 `rg --files` 找候选文件，再用 `rg -n` 定位命中，最后读取必要区段。
4. 只有当前证据不足时才扩大搜索范围。不要用全仓库全文读取代替入口定位。
5. 历史用于解释来源，实时 Git、代码、配置、设备和 current 文档用于确认当前状态。

## 工具输出预算

- 大型 Gradle、ADB、测试、转换、训练和 benchmark 输出写入
  `artifacts.local/work/<task>/`；正式证据写入
  `artifacts.local/evidence/<domain>/`。
- 聊天默认只返回结论、退出状态、关键指标、失败附近最多 100 行和证据路径。
- 不重复贴出完整日志。需要比较时，生成小型摘要、差异或机器可读 receipt。
- 长命令应有超时、进度和输出路径。判断健康状态时同时检查子进程 CPU/I/O、产物
  增长和进度记录；单有 PID 或没有异常不代表任务健康。
- 临时诊断输出使用任务专属目录，不把仓库根目录或整个 `tmp/` 当作可随意清理的
  空间。

## 同任务、换任务和 handoff

- 同一目标的实现、验证、修复和交付留在同一任务，避免重复恢复上下文。
- 目标、authority、工作树所有权或交付物发生实质变化时，使用新任务；不要让旧
  对话历史承担新任务的状态管理。
- 仅在跨窗口、不可逆外部动作、昂贵验证、真机/外部依赖或共享工作树风险较高时，
  创建 `artifacts.local/work/codex-handoffs/<TASK-ID>.md`，并登记到同目录
  `INDEX.md`。
- handoff 使用 `docs/CODEX_TASK_HANDOFF_TEMPLATE.md`，正常检查点控制在 20 个
  非空行内，只保留：目标/非目标、允许路径、分支和现场、已完成工作、验证结果、
  blocker/authority boundary、下一条命令。
- 恢复 handoff 时先验证 `git status --short`、分支、允许路径和已声明的验证事实；
  旧 handoff 不能替代当前检查。
- 持久技术决定进入 owning current 文档或 `DEVELOPMENT_LOG.md`；聊天历史与 handoff
  都不是长期真源。

## 共享工作树与提交

- 开始和提交前都检查 `git status --short`。
- 为任务声明允许修改的路径；未声明的改动默认属于其他任务。
- 两个任务需要同一文件时应合并所有权或使用独立 worktree/branch，不能互相吸收。
- 使用显式路径或 hunk 暂存。脏索引中提交时使用
  `git commit --only -m "<message>" -- <task-owned-paths>`。
- 复用仍与当前文件一致的验证 receipt，只补跑 receipt 之后实际改变所需要的最小
  gate。
- 推送后分别核对本地 `HEAD`、upstream 和远端目标 ref；不要只把 `git push`
  返回成功当作完整交付证明。

## 验证选择

- 文案/链接：内容审阅、链接/index 检查、`git diff --check`。
- 单模块实现：相关单测或 lint。
- 公共接口、CameraX、vision、risk、feedback、权限或 assets：相关测试加 Android
  build。
- 项目结构、脚本入口和文档治理：运行根 `AGENTS.md` 中列出的结构、hygiene 和
  docs-index 机械门。
- 设备、研究、host compute 和 release 不在本页复制命令；按根 `AGENTS.md` 路由到
  owning current 文档。

验证失败时先判断它属于本任务回归、环境阻塞还是并发工作影响。不得修改无关文件来
“让全仓库变绿”；应保留准确失败边界、证据路径和剩余风险。
