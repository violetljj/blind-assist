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

`managed_assets` 是精确例外：它只扫描 policy 明确列出的稳定 locator，不会打开该
动态根的其他兄弟目录。目前 `runtime/carla-asset-library` 以一个复合 CARLA 资产库
纳管；CARLA 进程、临时缓存和其他 runtime 仍然排除。复合资产通过 Git 跟踪的
`semantic_profile` 展开成可单独解析的语义组件，而不复制或拆散原目录。

顶层 junction/reparse alias 不跟随，避免重复计数或越过 `artifacts.local` 权限边界。
扫描期间消失的活跃文件会计数为 `vanished_entries`，不会伪装成稳定资产。

### CARLA 与 SEVN 的真实语义资产

- `datasets/sevn` 是 `F:\ba-data\SEVN` 六个上游文件的同卷 NTFS hardlink 规范入口，
  共 `29,684,140,135` 逻辑字节；旧 caller 继续可用，但没有复制第二份 payload。
- 这六个文件不是一个模糊“数据集”：`coord.hdf5` 是 4,988 帧本地 ORB-SLAM2 位姿，
  `graph.pkl` 是同一批 frame id 的 4,988 节点/6,366 边连通人行道图，
  `images.hdf5` 是 `4,988 x 84 x 224 x 3` 低分辨率全景，
  `high-res-panos.zip` 是 4,988 张 `3840 x 1280` RGB 全景，`label.hdf5` 是门、门牌号、
  街牌的 evaluator 标注，Zenodo JSON 是来源与许可凭据。前四者的 frame id 已核验
  完全对齐；标注只覆盖其中 1,570 帧，不能伪装成 runtime observation。
- SEVN V1 与地址/全景不相交 V2 面板均已打开，因此 policy 将该资产固定为
  `development_consumed`。它仍可用于特征缓存、难例、回归和 Development 实验，
  但不能重新解释成 fresh confirmation 数据。
- `runtime/carla-asset-library` 不是单一数据集。它包含 CARLA 0.9.16 executable 与
  Unreal world 资产、Town 地图/OpenDRIVE/HD point cloud、车辆/行人/道具/天气蓝图，
  Python client 环境，场景/布局/actor 注册表，以及 21 个实际存在的场景与实验包。
  采集数据按 exact CARLA frame id 对齐 wearable RGB、metric depth、instance
  segmentation、witness RGB、相机位姿/标定、actor state、visibility/contact/occlusion
  truth；后几项属于 evaluator-only 特权信息。
- C4 多地图场景包单独登记了 6 张地图、8 类场景、8 个 layout、16 个 episode、
  24 个 fresh-server sensor shard、40 类注册资产和 119 个 actor placement。场景类是
  `narrow_alley / mall_exit / parking_lot / bus_stop / construction_zone / rainy_night /
  backlight / crowded_pedestrians`，不是从目录名猜出来的泛化标签。
- CARLA 复合库同样是 `development_consumed`。模拟器运行文件本身标为
  `not_applicable`，场景/传感器/evaluator 包分别带自己的证据角色和 claim ceiling；
  具体实验结果的上限继续由 manifest 与 canonical evidence 决定。
- Git 中的 CARLA 场景协议/生成器由 Git 版本控制；生成的场景、传感器 payload、
  结果与证据继续位于 `evidence/...` 或上述复合资产库。运行时二进制可复用，但不
  因此获得证据权限。

## 建账和报告

从仓库根运行：

```powershell
E:\codex-tools\bin\python.cmd tools\data\asset_catalog.py discover
E:\codex-tools\bin\python.cmd tools\data\asset_catalog.py classify-authority
E:\codex-tools\bin\python.cmd tools\data\asset_catalog.py classify-authority --apply
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

E:\codex-tools\bin\python.cmd tools\data\asset_catalog.py components datasets/sevn
E:\codex-tools\bin\python.cmd tools\data\asset_catalog.py components `
  runtime/carla-asset-library

E:\codex-tools\bin\python.cmd tools\data\asset_catalog.py resolve `
  datasets/sevn#high-resolution-panoramas `
  --consumer <run-id> --purpose observation-input

E:\codex-tools\bin\python.cmd tools\data\asset_catalog.py resolve `
  runtime/carla-asset-library#dtr-carla-c4-multimap `
  --consumer <run-id> --purpose multimap-scene-input
```

`asset#component-key` 返回组件的真实文件/目录、组件内容或 metadata 身份、数据角色、
证据状态和 claim ceiling，并把组件键写入 usage event。这样实验复用的是“SEVN 高分辨率
全景”或“CARLA C4 多地图场景包”，不再只是消费一个无法解释的大目录。

引用被区分为精确资产、根级、模板、已知根内缺失 locator 和未知旧根。后两类是调用
迁移或历史恢复待办；对 excluded 动态根的根级引用不会被错误当成长期数据资产。
2026-08-31 的 106 条 unresolved 基线已在
[`data/asset-reference-migration-20260831.json`](../../data/asset-reference-migration-20260831.json)
逐 caller 登记：物理存在但被策略排除、兼容 alias、历史/封存记录、计划输出、测试
fixture、已迁移活动调用方与真正缺失依赖分别计数；历史记录不为降低报告数字而改写。

## 证据权限分类与旧总账迁移

`evidence_status` 回答资产自身能否承载某种证据权限；`usage_events` 另行回答某次
运行由谁、为何、以什么 effect 消费了它。两者不得合并：模型或外部依赖被使用过，
不等于它们是 fresh/consumed cohort；原始数据是 source material，也不等于它已获
fresh confirmation authority。

| 状态 | 语义 |
| --- | --- |
| `not_applicable` | 模型、代码型外部依赖等不承载 cohort/outcome 证据权限 |
| `source_material` | 原始/观测来源；没有 fresh、confirmation 或 final 权限，待具体运行另行 admission/consume |
| `reserved` | 经明确协议保留，尚未获 fresh 使用权限 |
| `fresh` | 由明确协议和身份约束授予的未开封权限；不能由自动分类产生 |
| `development_consumed` | outcome/truth 已用于 Development；不可恢复 fresh/reserved/sealed_final |
| `sealed_final` | 明确 final 协议下的不可逆状态；不能由自动分类产生 |
| `diagnostic` | 明确只允许诊断/机制用途，不具有 fresh confirmation 权限 |
| `unknown` | 缺少足够可审计事实；保留并进入待裁决队列 |

`discover` 会保留已有 lifecycle，因此修改根策略不会静默覆盖旧 `unknown`。旧总账
必须通过显式迁移：默认 dry-run 的 `classify-authority` 只给出 projected count；
加 `--apply` 后，每项变化同时写入 lifecycle event 和
`authority_classifications`，记录 `rule_id / source / reason / evidence / policy SHA-256`。

自动分类只接受以下高置信依据：

1. asset class 明确表明权限轴不适用 (`not_applicable`) 或仅是来源材料
   (`source_material`)；
2. 当前 `PROJECT_STATE / CURRENT_DECISION / DTR / L10` authority 文档直接引用
   outcome-bearing locator；
3. 小型 result/report/receipt 等实际文件内容明确声明 `consumed`、
   `Development-only` 或 `diagnostic-only`，并记录该断言文件的 SHA-256；
4. policy 中 exact locator 规则同时通过 authority 文档的多个文本 anchor。

自动规则被禁止产生 `reserved`、`fresh` 或 `sealed_final`，也不会仅凭目录名中出现
`fresh`、`sealed`、`final` 等字样升级权限。已有 `development_consumed` 和
`sealed_final` 仍由 lifecycle invariant 保护，不会被迁移改写。

`report` 额外生成：

```text
artifacts.local/evidence/resource-fabric/reports/assets/current/evidence-authority-queue.json
artifacts.local/evidence/resource-fabric/reports/assets/current/evidence-authority-queue.md
```

它们按“空资产无可检查断言 / evidence 未被当前 authority 引用且无显式断言 /
legacy 需人工裁决 / 无匹配规则”分组，并列出每个仍为 `unknown` 的 locator。队列是
待裁决清单，不是删除、归档或自动降权清单。

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
