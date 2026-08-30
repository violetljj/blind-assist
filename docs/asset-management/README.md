# BlindAssist 资产总账与生命周期

BlindAssist 的数据、模型、特征、证据和实验输出统一进入一个权威资产总账。
目标不是把现有文件再复制一遍，而是让每项长期资产都具备稳定身份、明确用途、
可追踪派生关系、证据权限和存储生命周期。

## 系统边界

```text
artifacts.local 下的稳定资产
        ↓ 零拷贝发现
master-assets.sqlite3 权威总账
        ↓ resolve 时记录 consumer / purpose
实验消费与派生事件
        ↓
共享 resource/cache、难例库、薄实验结果
        ↓
保留 / 冷归档 / 可重建 / 精确清理决策
```

体系由两层组成：

- `tools/data/asset_catalog.py` 管理所有现存稳定资产，是发现、查询、解析、消费、
  派生和生命周期的权威入口。
- `tools/data/resource_fabric.py` 管理已经进入内容寻址仓的不可变资源、共享缓存、
  难例和薄实验。它的资源、缓存和引用会自动同步进总账。

跟踪的分类策略位于 `data/asset-management-policy.json`；生成的数据库与报告位于：

```text
artifacts.local/evidence/resource-fabric/catalog/master-assets.sqlite3
artifacts.local/evidence/resource-fabric/reports/assets/current/
```

## 两级身份

全量建账不能要求先复制或读取数百 GiB 的全部内容，因此身份分两级：

| 身份强度 | 含义 | 用途 |
| --- | --- | --- |
| `metadata` | 稳定 locator、文件清单、字节数、mtime 指纹 | 立即完成零拷贝接入、解析、消费和生命周期管理 |
| `content` | 每个文件 SHA-256 与目录树 SHA-256 | 不可变资产校验、跨路径去重和 sealed 引用 |

`metadata` 不是内容相同的证明。重要资产通过 `hash` 增量升级到 `content`；已进入
resource fabric 的对象和缓存会直接以内容身份导入。

## 管理范围

默认全量登记稳定根的每个直接子项，包括 `datasets`、`models`、`evidence`、
`downloads`、`knowledge`、`sources`、`external`、`synthetic`，以及其他历史根。
策略未识别的稳定根不会被丢弃，而是进入 `legacy_unclassified`，等待补齐 owner、
authority 和 rebuild 信息。

以下动态根只登记排除原因，不扫描为长期资产：

```text
cache/ runtime/ runtimes/ work/ worktrees/ tmp/
gradle-home/ tools/ vendor/ logs/ failed_runs/ crash-diagnosis/ transfer/
```

顶层 junction/reparse alias 不跟随，避免重复计数或越过 `artifacts.local` 权限边界。
扫描期间消失的活跃文件会计数为 `vanished_entries`，不会伪装成稳定资产。

## 建账和报告

从仓库根运行：

```powershell
E:\codex-tools\bin\python.cmd tools\data\asset_catalog.py discover
E:\codex-tools\bin\python.cmd tools\data\asset_catalog.py report
E:\codex-tools\bin\python.cmd tools\data\asset_catalog.py verify
```

`discover` 是零拷贝操作：不移动、不删除、不重写资产，只更新本地 SQLite 总账。
它会同步 resource fabric 的资源、缓存、薄实验和难例引用，并从 Git 跟踪的代码、
配置和文档中提取现存 `artifacts.local/...` 依赖。报告中的 bytes 是逻辑引用大小，
不等同于 NTFS 去重或 hardlink 后的物理占用。

单独刷新代码引用或查询资产：

```powershell
E:\codex-tools\bin\python.cmd tools\data\asset_catalog.py references
E:\codex-tools\bin\python.cmd tools\data\asset_catalog.py list `
  --root datasets --state present --limit 50
E:\codex-tools\bin\python.cmd tools\data\asset_catalog.py list `
  --evidence-status unknown --limit 50
```

引用被区分为精确资产、根级、模板、已知根内缺失 locator 和未知旧根。后两类是调用
迁移或历史恢复待办；对 excluded 动态根的根级引用不会被错误当成长期数据资产。

## 统一解析与自动消费记录

新实验不得继续把资产路径写死在实现里。运行前通过 locator、asset id、唯一名称
或 content id 解析：

```powershell
E:\codex-tools\bin\python.cmd tools\data\asset_catalog.py resolve `
  datasets/my-dataset-v1 `
  --consumer l10-example-v1 `
  --purpose observation-input `
  --experiment-id l10-example-v1
```

返回值包含绝对路径、资产身份、证据状态、存储状态和本次 usage event id。传入
`--consumer` 与 `--purpose` 时会在同一事务中自动记录消费；两者必须同时提供。

读取公开输入通常使用默认 `--evidence-effect none`。打开 Development truth、结果
或其他会消费证据权限的内容时必须显式声明：

```powershell
E:\codex-tools\bin\python.cmd tools\data\asset_catalog.py resolve `
  asset:<id> `
  --consumer dtr-example-v1 `
  --purpose development-truth `
  --evidence-effect development_consumed
```

一旦任意生命周期事件把资产标记为 `development_consumed`，后续扫描、重新登记或
手工 transition 都不能把它恢复为 `fresh`、`reserved` 或 `sealed_final`。存储状态
仍可在 `active`、`shared`、`sealed_cold`、`rebuildable` 之间独立变化。

无法通过 resolver 的旧调用方可先显式记账：

```powershell
E:\codex-tools\bin\python.cmd tools\data\asset_catalog.py consume `
  datasets/my-dataset-v1 `
  --consumer legacy-evaluator `
  --purpose disclosed-development-replay
```

## 派生关系

输出落盘并被 `discover` 或 `register` 登记后，记录它由哪些输入产生：

```powershell
E:\codex-tools\bin\python.cmd tools\data\asset_catalog.py derive `
  --output datasets/my-normalized-v1 `
  --input datasets/my-raw-v1 `
  --transform normalize-observations `
  --transform-version v1 `
  --producer research/active/l10-r0/normalize.py `
  --parameters-json '{"size":518}'
```

`derive` 会原子写入 derivation 和输入 usage events。可同时绑定 code/config SHA-256、
实验 id 和 evidence effect。resource fabric 的共享 cache 会自动生成同类派生记录。

## 内容校验与重复识别

对已经静止且需要跨路径去重或 sealed 引用的资产执行：

```powershell
E:\codex-tools\bin\python.cmd tools\data\asset_catalog.py hash `
  datasets/my-dataset-v1
```

文件资产得到 `sha256:<digest>`；目录得到与 resource fabric 兼容的
`tree-sha256:<digest>`。相同 content id 只说明字节和相对文件树相同，不自动授权
删除任一路径。清理仍必须检查活跃 consumer、证据角色、重建来源和明确白名单。

## 单项零拷贝登记

新资产不必等待下一次全盘扫描：

```powershell
E:\codex-tools\bin\python.cmd tools\data\asset_catalog.py register `
  artifacts.local\datasets\my-dataset-v1 `
  --kind dataset --asset-class data `
  --evidence-status reserved --storage-status active `
  --owner l10-r0 `
  --retention-reason "fresh source for the declared gap" `
  --claim-ceiling "Development admission pending"
```

登记只读取元数据并写总账，不复制 payload。

## 每项资产必须最终回答的问题

- `asset_id`、locator、content id 和身份强度是什么？
- owner、来源 URL、license、schema/version 和 retention reason 是什么？
- 哪些 consumer 在什么目的下访问过？是否打开了 outcome/truth？
- 由哪些输入、模型、代码、配置和参数派生？
- 当前 evidence status、claim ceiling 和 storage status 是什么？
- 能否重建、重建命令和成本是什么？
- 是否仍有活跃 consumer，是否存在相同 content id 的其他 locator？

报告会持续暴露 `legacy_unclassified`、unknown evidence/storage、未消费资产、缺失
资产和内容重复组。这些是治理待办，不是自动删除清单。

## 新实验的最低接入规则

1. 稳定输入先进入总账；不可变关键输入升级到 content identity。
2. 代码通过 `resolve --consumer --purpose` 获取路径并自动记账。
3. 重复规范化或特征进入 resource fabric cache，不放进实验目录。
4. 失败、难例和证据缺口保存 selector 与引用，不复制媒体。
5. 派生输出记录 inputs、transform、代码/配置/模型身份。
6. 实验目录只保存 manifest、参数、小结果和 evidence boundary。
7. 删除、移动和去重是单独的精确授权操作；总账报告本身不执行清理。
