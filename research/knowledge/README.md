# BlindAssist knowledge reserve

这里存放可跨路线复用的论文、算法、优秀项目、数据集和工具知识。
它不是新的路线决策面，也不是实验结果总账。

最新定向增量：[2026-08-30 L10 / DTR 文献增量](LITERATURE_DELTA_2026-08-30.md)。

## 两层模型

- `items/*.json` 记录来源本身：规范链接、摘要、可复用机制、输入输出、
  适用限制和旧编号别名。来源事实尽量稳定、保持中性。
- `uses/*.json` 记录某条 BlindAssist 路线怎样使用其中的具体机制：
  项目落点、相对原作的修改、预期效果、复现状态、实际指标、证据引用和
  claim boundary。它可以随新实验实时更新，并由 Git 保留修改历史。

同一 item 可以拥有多条相互独立的 use。一个 use 的 `falsified` 只表示
“该机制在这条路线、这个实现和证据范围内的假设被证伪”，绝不等于论文、
算法或项目被普遍否定。`UNKNOWN` 和 `NOT_EVALUABLE` 也不是负结果。

## 目录

```text
 research/knowledge/
  README.md
  items/   # 一项外部知识一个 JSON 文件
  uses/    # 一条 route × item 使用关系一个 JSON 文件
  decision/ # 故障分类、当前 terminal、golden cases 和编译索引
  migrations/ # 一次性迁移覆盖清单；不是第二份知识正文
tools/knowledge.py
tools/migrate_scattered_knowledge.py
```

PDF、源码快照、模型和原始复现实验输出不进 Git；放在
`artifacts.local/knowledge/<item-id>/`，use 记录只保存来源、哈希、
manifest 或结果引用。

## 读取

```powershell
python tools/knowledge.py validate
python tools/knowledge.py context --route obstacle-avoidance --limit 4
python tools/knowledge.py context --route ten-meter-copilot --limit 4 --query "spatial layout"
python tools/knowledge.py list
python tools/knowledge.py list --route ten-meter-copilot
python tools/knowledge.py list --verdict falsified
python tools/knowledge.py search "occlusion"
python tools/knowledge.py show paper-deepsolo-2023
python tools/knowledge.py show use-grail-r1c-p-orient-anything-v2
```

`search` 和 `context --query` 支持自然多关键词：标点与连字符会归一化，所有
有效关键词都必须命中；同一状态/证据优先级内，题名、别名、机制名和完整短语
命中更靠前。例如 `search "z-buffer visible door"` 与
`context --route obstacle-avoidance --query "existence observability"` 不要求原文中
出现完全相同的连续字符串。

## 研究决策引擎

日常故障入口是 `diagnose`，不是先搜索更多论文。它从一个预编译 JSON 索引读取
全部机制、route use、`experiments/index.jsonl` 和当前高权威 terminal；查询过程不
逐个打开 `items/uses`，也不调用模型或网络：

```powershell
python tools/knowledge.py diagnose --route obstacle-avoidance --symptom "静态占据在 dropout 窗口仍为 0/9，matcher 没有 closing speed"
python tools/knowledge.py diagnose --route ten-meter-copilot --symptom "遮挡后无法 reference-reacquire exact instance" --observed "有 public anchor frame 和 box" --json
python tools/knowledge.py diagnose --route goal-copilot-p0 --symptom "near-identical instances cause wrong-target identity drift" --write-plan artifacts.local/knowledge/decision/near-id-plan.json
```

输出固定为四段：

1. 最可能的 1–3 个故障层和仍需核对的证据；
2. 2–4 个最相关机制、已有路线状态和 contraindication；
3. 已尝试的 use/experiment/current terminal，以及必须保留的停止边界；
4. 一个只改变单一信息因子的最小实验，含 baseline、cohort、primary metric、
   stop、`NOT_EVALUABLE` 和 claim ceiling。

`decision/config.json` 持有故障层、双语签名、机制 override 和实验模板；
`decision/terminals.json` 只保存当前决策所需的窄 terminal 摘要及仓库证据锚点，
不替代 owning route README；`decision/golden_cases.json` 冻结真实故障回归集。
编译索引 schema v2 的 `associations` 将同一 run 的重复 ledger 行合并，并稳定保存
`run_id / use_ids / protocol_id / code_revision / input_fingerprint /
artifact_refs / decision_id`。只从显式字段、experiment evidence 或完全一致的
terminal decision 建边；没有权威来源的关联保持 `null`，不靠文本相似度猜测。
新实验行可显式写 `use_ids`、`protocol_id`、SHA-256 `input_fingerprint`、
`artifact_refs` 和 `decision_id`；编译器会拒绝未知 use、冲突的稳定字段或一项
decision 被多个 run 占用。

当 item、use、experiment、配置或 terminal 变化时，重建索引并运行冻结回归：

```powershell
python tools/knowledge.py build-decision-index
python tools/knowledge.py evaluate-decision-engine
python tools/knowledge.py validate
```

`validate` 会比较源文件 fingerprint；索引过期时明确失败，不允许悄悄使用旧判断。
新增知识不是决策引擎的 KPI。只有真实故障暴露召回盲点或 successor 缺少机制时，
才增加 item；普通升级优先修改故障签名、机制映射、terminal 或 golden case。

`validate` 也会核对迁移清单中的每个旧编号/派生键：它必须能解析到一个
canonical item，而且清单声明的 use 必须存在并真正属于该 item。这样批量迁移
不会只凭“生成了很多文件”就被视为完成。

`list/search --json` 可向路线代码或 agent 返回完整结构化结果。item 的
canonical id、aliases 和全文字段都能被 `search/show` 解析。

日常算法研究优先使用 `context --route <route>`。V2 不会把全库直接塞进上下文，
而是在同一个总预算里先给出最新 current terminal，再按以下顺序压缩 use：已有
实验结论或 `rejected`，`active / planned`，`adopted`，`candidate`，最后是
`retired`。默认总共返回 4 条 terminal + use 记录，先确保每个实际存在的 use
优先级层至少出现一个代表，再按顺序补满；同时报告各状态、判定和省略数量。用
`--query` 同时收窄 terminal 与 use，用 `--all` 获取完整路线，用 `--json` 直接
交给路线脚本或 agent。每条精简 use 仍保留 applicability、机制、项目接法、修改、
预期/实际效果、claim boundary、证据和最近一次更新。

冷启动建议使用文本输出和 `--limit 4`；只有自动化消费者需要完整字段时才加
`--json`。当前路线目录别名 `dtr-r0`、`l10-r0` 也可用于 `context`、`list`、
`search` 和 `diagnose`，分别解析为稳定主线，并合并历史上两种名称下的 use。

## 状态不是一个混合标签

use 用三个正交字段描述当前判断：

| 维度 | 示例 | 回答的问题 |
| --- | --- | --- |
| `use_state` | `candidate / active / adopted / rejected / retired` | 项目现在还打算怎样使用它？ |
| `reproduction_status` | `not_attempted / partial / reproduced / failed` | 原作机制或目标效果复现到了什么程度？ |
| `verdict` | `positive / negative / mixed / falsified / not_evaluable / unknown` | 当前证据对路线假设意味着什么？ |

因此“代码跑通但功能门槛失败”可以写成
`mechanics_only + negative`；“来源真值不够”应写成
`not_attempted + not_evaluable`，不能写成 falsified。

## 新增知识

可直接复制一个现有 item 修改，也可以使用命令创建首个机制：

```powershell
python tools/knowledge.py new-item --id paper-example-2026 --kind paper --title "Example" --canonical-ref "https://example.org/paper" --year 2026 --venue "ExampleConf" --summary "一句话说明来源贡献。" --mechanism-id useful-mechanism --mechanism-name "Useful mechanism" --mechanism-description "机制如何工作。" --mechanism-input "input" --mechanism-output "output" --mechanism-limitations "它不能证明什么。" --tag route-topic
```

一篇来源有多个机制时，直接在 item 的 `mechanisms` 数组中增加对象，然后
运行 `validate`。规范 URL、DOI 和 aliases 会做全库重复检查。

## 建立或修改路线使用

```powershell
python tools/knowledge.py new-use --id use-example-route-mechanism --item paper-example-2026 --route example-route --mechanism useful-mechanism --source-scope "原作中真正借用的部分。" --project-application "在本路线接到哪里。" --modifications "与原作相比改了什么。" --expected-effect "预期改变哪个可观察量。" --claim-boundary "即使成功也不能推出什么。"
python tools/knowledge.py update-use use-example-route-mechanism --state active --applicability "只用于已冻结输入和命名 cohort 的机制实验。" --reproduction partial --verdict mixed --effect "实际观察到的效果。" --metric "metric_name=value on named cohort" --evidence repo research/active/example-route/result.json "可复核结果" --note "完成第一轮复现并更新当前判断。"
```

`update-use` 只改给出的字段，并强制追加一条 history note。证据引用支持：

- `repo`：当前仓库内存在的文件；
- `experiment`：`experiments/index.jsonl` 中的实验 id；
- `git`：`REVISION:path/in/repo` 的历史锚点；
- `artifact`：`artifacts.local/...` 下的本地证据；
- `external`：公开 HTTP 链接。

完整性采用选择性门槛：`candidate / adopted / rejected / retired` 可以保留历史上的
稀疏机制描述；一旦 use 进入 `planned` 或 `active`，`usage.applicability` 必须明确
适用条件，而且它引用的每个 mechanism 都必须有非空 `inputs` 和 `outputs`。这只
约束真正准备投入实验的知识，不批量膨胀旧候选。

## 路线使用规则

`route` 使用不会随 SC、R0、F1 等实验或协议名变化的稳定主线 id。当前两条主线固定为
`obstacle-avoidance`（避障）和 `ten-meter-copilot`（十米副驾）。具体实验名只写入
use id、`evaluation.setup`、evidence、history 或 `experiments/index.jsonl`，不得作为新
use 的 route。这样实验改名或换代后，同一主线仍能检索全部历史尝试与停止边界。
读取命令接受 `dtr-r0` / `l10-r0` 作为便捷别名；`new-use` 会自动写成对应稳定主线
id，避免继续产生分叉。

1. 主线冷启动先读取 `context --route <route>`；开新机制前再按 tag 或问题关键词
   搜索，避免重复探索已消费路线。
2. 真正采用、复现或证伪时新增/更新 use；当前路线结论仍由
   `docs/CURRENT_DECISION.md` 和 owning route README 持有。
3. use 必须写清“借了什么、改了什么、观察到什么、不能推出什么”。
4. 历史失败和 consumed evidence 可以作为知识与诊断，但不会因入库恢复
   fresh/confirmation authority。

## 2026-08-28 散落知识迁移

迁移收据位于
`migrations/migration-scattered-knowledge-2026-08-28.json`。它冻结了 15 组
来源、204 条旧知识映射及各来源内容哈希，覆盖：

- DTR `DR01–DR60`、L10 `1–20`、USTRF `P01–P13`、RCLE `E1–E5`；
- Goal Copilot prior-art 表和两轮共 40 篇逐篇深读；
- Frontier Upgrade 14 篇、HFTF 9 个相邻工作、TARO 16 个相邻工作；
- Project Guideline 八项组件吸收决定；
- IDEA archive 中可唯一规范化的来源，以及无法可靠拆回单篇来源的联合综述。

跨清单同一来源只保留一个 item，例如 Project Guideline、GuideTouch、
Closing the Gap、Depth Anything V2 和 AI Guide Dog；各路线的判断仍是独立 use。
迁移后库内共有 191 个 canonical items、206 个 route uses、14 个 routes。

迁移同时恢复了已有项目结果，而没有把“阅读过”误写成“复现过”：NearID-style
冻结适配臂记录为 `partial + negative`，analytic spatial-layout 臂记录为
`partial + mixed + rejected`，ABotN official waypoint 记录为 metric-arrival
成立但 visual handoff 未成立，Orient Anything V2 的 GRAIL 使用继续保留原有
`falsified` 边界。

本机仍保留两份 hash 绑定的 `artifacts.local` 深读原文时，可复核迁移是否与冻结
来源完全一致：

```powershell
python tools/migrate_scattered_knowledge.py --check
```

该命令只重读冻结来源并核对 `source_ref`、SHA-256、source-group 数量和 legacy-id
覆盖收据；它不会重新序列化已经在迁移后继续演进的 item/use。当前 item/use 与
迁移映射是否仍可解析由 `python tools/knowledge.py validate` 独立负责。
该命令是迁移审计工具，不是日常新增入口。其他机器没有这两份本地深读报告时，
知识库和 `knowledge.py validate/list/search/show` 仍可正常使用；不要为通过
`--check` 而伪造报告。后续新知识直接新增 item/use，不追加到旧迁移收据。
