# BlindAssist knowledge reserve

这里存放可跨路线复用的论文、算法、优秀项目、数据集和工具知识。
它不是新的路线决策面，也不是实验结果总账。

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
tools/knowledge.py
```

PDF、源码快照、模型和原始复现实验输出不进 Git；放在
`artifacts.local/knowledge/<item-id>/`，use 记录只保存来源、哈希、
manifest 或结果引用。

## 读取

```powershell
python tools/knowledge.py validate
python tools/knowledge.py list
python tools/knowledge.py list --route l10-r0
python tools/knowledge.py list --verdict falsified
python tools/knowledge.py search "occlusion"
python tools/knowledge.py show paper-deepsolo-2023
python tools/knowledge.py show use-grail-r1c-p-orient-anything-v2
```

`list/search --json` 可向路线代码或 agent 返回完整结构化结果。item 的
canonical id、aliases 和全文字段都能被 `search/show` 解析。

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
python tools/knowledge.py update-use use-example-route-mechanism --state active --reproduction partial --verdict mixed --effect "实际观察到的效果。" --metric "metric_name=value on named cohort" --evidence repo research/active/example-route/result.json "可复核结果" --note "完成第一轮复现并更新当前判断。"
```

`update-use` 只改给出的字段，并强制追加一条 history note。证据引用支持：

- `repo`：当前仓库内存在的文件；
- `experiment`：`experiments/index.jsonl` 中的实验 id；
- `git`：`REVISION:path/in/repo` 的历史锚点；
- `artifact`：`artifacts.local/...` 下的本地证据；
- `external`：公开 HTTP 链接。

## 路线使用规则

1. 开新机制前先按 route、tag 或问题关键词搜索，避免重复探索已消费路线。
2. 真正采用、复现或证伪时新增/更新 use；当前路线结论仍由
   `docs/CURRENT_DECISION.md` 和 owning route README 持有。
3. use 必须写清“借了什么、改了什么、观察到什么、不能推出什么”。
4. 历史失败和 consumed evidence 可以作为知识与诊断，但不会因入库恢复
   fresh/confirmation authority。

当前三个种子覆盖候选机制和项目内证伪两类情形。原有 L10 与 DTR 文献储备
仍作为 provenance 保留；已结构化条目用 `l10:1`、`dtr:DR45` 等 aliases
承接旧编号，后续可逐条迁入而不复制成第二套身份。
