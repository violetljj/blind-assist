# 候选事件自动挖掘

状态：`current / discovery-only / THESIS_DEVELOPMENT`

当前真源是 [scripts/research/candidate_event_mining/README.md](../../../scripts/research/candidate_event_mining/README.md) 与 [candidate event mining contract](../../../configs/candidate_event_mining_contract_v1.json)。本页是文档索引入口，记录为什么建立这个独立 Module，以及它和当前 dual-loop/HFTF 的关系。

## 目标和流程

候选事件自动挖掘的价值在于把长视频/公开数据从“人工猜窗口”变成可复现的发现流水线：

```text
模型专属批量推理 adapters
  -> canonical frame trace
  -> 窗口触发与 context expansion
  -> 同 session 去重
  -> 跨 session 证据聚类
  -> candidate-blind review bundle
  -> Luna 独立复核
  -> candidate pool / quarantine
```

当前覆盖 12 类搜索目标：正前方障碍接近、横穿、用户接近静态障碍、台阶/落差、平行路沿、门框/桌角/树枝、正常通行负例、转头/抖动负例、动态人群、YOLO 漏检而 segmentation/depth 有响应、segmentation 高频响应、HFTF future-field 变化明显。

## 当前边界

这是 `THESIS_DEVELOPMENT` 下的 discovery 工具，不是新算法主线，也不是对当前双环或 HFTF 终态的改写。candidate trigger、Luna model review 和 candidate pool 都不是客观真值；它们不能直接授权训练、事件效果、Android、默认 App、生产或安全结论。模型输出对 review 默认隐藏，最终候选池同时保留 keep/quarantine 计数和证据 hash。

当前首批已将 4 个公开 source 的 byte-verified 副本物化到 `F:\ba-data\blindassist-candidate-event-mining\`，并在实际 project index 中登记 URL、admission time、内容 hash、source/session、许可证和本地路径；它们都保持 `THESIS_DEVELOPMENT_CONSUMED_DISCOVERY`，不被重新称为 fresh holdout 或 confirmation。后续下载和媒体仍只进入该目录；仓库的 `artifacts.local/` 只保存运行收据、JSONL、bundle 和候选池。

## 项目索引

初始化空索引：

```powershell
E:\codex-tools\bin\blindassist-python.cmd `
  scripts/run_research_tool.py candidate-event-mining init_project_index.py `
  --output F:\ba-data\blindassist-candidate-event-mining\project_index.json
```

当前首批 project index 已登记 4 个 source/session；后续任何实际来源在 batch trace 前仍必须注册。`init_project_index.py --source-record <record.json>` 可在重新校验完整字段后追加来源，禁止重复 `source_id`，并拒绝越出 `F:\ba-data` 的媒体路径。没有必要时不下载数据。

## 运行和证据

完整命令、canonical frame schema、review receipt 字段、停止条件和失败资产角色见 Module README。稳定入口是：

```text
scripts/run_research_tool.py candidate-event-mining <tool.py> [args...]
```

核心实现和纯标准库回归测试位于 `scripts/research/candidate_event_mining/`；合同测试不依赖 GPU、Android 或网络，真实 host adapter 则需要本地 OpenCV/Ultralytics/PyTorch/Depth Anything 运行时。当前最小验证是 12 类合成信号、去重/聚类、candidate-blind bundle、Luna receipt 和 fail-closed 缺失复核测试。
