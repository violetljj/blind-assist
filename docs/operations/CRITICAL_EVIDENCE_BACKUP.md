# 关键证据备份与恢复

备份以真实文件和恢复后的 SHA-256 为准。清单、Git 中的结论或一次成功上传都不等于可恢复备份。
副机备份可降低主机单盘损坏风险；同地点的两台机器不构成异地灾难备份。

## 本轮范围（2026-09-24）

本轮覆盖打包时的非忽略源码工作快照（含尚未提交的工作，逐文件哈希固定其身份）、
App assets，以及 `artifacts.local/evidence/` 下这三个明确目录前缀：

- `ba-inherit-spatial-`：继承模型、选择、封存及源代码快照；
- `ba-query-occupancy-`：基准原始 capture、观测、标签、预测和结果；
- `ba-local-`：support、transfer、stability、rescue 的原始 capture、观测、标签及审计结果。

这不是论文依赖的完整闭包。其他研究线、历史模型和数据集、UE 引擎及场景资源、
Python 环境和完整 Git 历史未纳入；本轮也不重新执行科学评估，不声称跨机算法复现。
没有复制 ignored 主机连接配置或凭据。打包快照不代表随后文档整理的最终提交。

实际异机恢复已通过：主包 **13,667 文件 / 5,689,482,974 原始字节**，补包
**65 文件 / 1,467,553 字节**。补包是上述实验的 resource-fabric run receipts 加备份工具和文档快照；
两个包有少量源码重叠，不能将相加文件数解读为不重复资产数。主包 ZIP 为 2,788,832,117 字节。
逐文件恢复大小及 SHA-256 全部一致，两个 archive 和最终恢复 helper 的两端 SHA-256 也一致。

- 主包 SHA-256：`cf8d327f7d842e40d98fdb1a18ebfc192af1f837791734cdea7c42dc334ac843`
- 补包 SHA-256：`091556ddfc7a308ebb978dfcb160b006f86c4442625a49813c7a5b34a3cfd543`

主包快照时间为 2026-09-24 02:19:58–02:22:22（香港），基础 Git revision 为
`2893137734524a0035839c844b1d6cb00bc25187`，实际 WIP 身份以逐文件 manifest 为准。
首次恢复遇到 Windows 长路径限制，工具增加 extended-path 支持后在全新 `*-restored-v2`
目录完整恢复通过。失败的部分恢复目录已核查边界后删除，失败说明仍保留；任务 Python 进程已退出。

控制端记录在 `artifacts.local/evidence/project-maintenance-20260924/`：
`backup-selection.json` 固定选择，`backup-manifest.json` 逐文件记录字节数与 SHA-256，
`worker-restore-receipt.json` 记录异机解包及逐文件校验结果（只有该文件 PASS 才表示恢复通过）。
`backup-transfer-receipt.json` 及 `worker-backup-completion.json` 保留两端 archive/helper 哈希和清理结果。
副机通过 `BLINDASSIST_ARTIFACTS/evidence/project-maintenance-20260924-backup/` 定位，
持久保留 archive、restored 文件树、capacity 和 restore receipt，所有者为本次项目维护任务。
这些是耐久备份，不属于完成后自动删除的临时文件。

## 重复使用

工具仅使用 Python 标准库。选择 JSON 至少包含 `paths` 数组，各项是仓库内明确相对文件或目录。
目录会包含所有子文件；先核查选择范围及凭据排除。只允许仓库 `artifacts.local` 这个既有
入口 junction，拒绝其他嵌套链接。所有目标和 receipt 必须不存在，绝不覆盖旧备份。

```powershell
python tools/backup_critical_evidence.py create --root . --selection artifacts.local/evidence/MY_BACKUP/selection.json --archive artifacts.local/evidence/MY_BACKUP/payload.zip --receipt artifacts.local/evidence/MY_BACKUP/manifest.json
python tools/backup_critical_evidence.py restore --archive PATH_TO_BACKUP/payload.zip --destination NEW_EMPTY_PARENT/restored --receipt NEW_EMPTY_PARENT/restore-receipt.json
```

先按[副机流程](WORKER_HANDOFF.md)检查实际 artifact 卷容量，再传输到独立新目录；比较两端 archive
SHA-256，并在目的机解包、校验所有文件。记录文件数、原始字节数、archive 哈希、恢复结果和未覆盖范围。
工具失败会保留部分文件供诊断，且不写成功 receipt；不要将它视为完成，也不要直接覆盖重跑。
恢复校验不执行归档中的代码或反序列化 pickle 模型。

新增论文核心结果或更换 App/保留模型时更新明确选择，形成新的备份身份；不能把本次快照当成持续同步。
范围有重大变化时优先检查真实依赖，而不是增加流程表单。
