# Public-video research campaign

状态：archive

本目录是已冻结的 public-video / public-silver 历史研究 Module，集中保存原先平铺在 `scripts/` 根目录的 315 个 CLI、合同、测试和网络候选 Adapter。它是研究证据的可复现 Implementation，不是当前 App、默认模型或生产授权路径。

## 稳定 Interface

```powershell
E:\codex-tools\bin\blindassist-python.cmd scripts/run_research_tool.py public-video <tool.py> [args...]
E:\codex-tools\bin\blindassist-python.cmd scripts/run_public_video_campaign_tests.py
```

`run_research_tool.py` 负责兼容 campaign 内的 sibling imports 和仍位于 `scripts/` 根目录的 SANPO helper；调用方不要自行注入 `PYTHONPATH`。设备闭环统一调用根 Adapter `scripts/run_public_video_edge_inference.ps1`。

## 输出

历史命令的下载、数据、benchmark 和证据继续写入 `artifacts.local/`；本目录不接收生成物。

## 安全边界

- [迁移时的完整语义索引](MIGRATION_INDEX_2026-07-21.md) 保留每个历史脚本的职责与安全边界。
- 历史失败结论保持有效；目录迁移不改变任何训练、校准、blind、Android 或 production authorization。

## 停止条件

- 本 campaign 已冻结，不再追加 r7/r8 实验轮次。
- 新的 route-conditioned、object-agnostic risk-field 主线必须建立新的领域 Module。
- detector 若继续，只能建立独立 crop-view FP 抑制 Module；不得在失败的 r1 上继续调 NMS、overlap 或 score。
