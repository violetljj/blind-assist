# Public-video research campaign

本目录是已冻结的 public-video / public-silver 历史研究 Module，集中保存原先平铺在 `scripts/` 根目录的 315 个 CLI、合同、测试和网络候选 Adapter。它是研究证据的可复现 Implementation，不是当前 App、默认模型或生产授权路径。

稳定 Interface：

```powershell
E:\codex-tools\bin\blindassist-python.cmd scripts/run_research_tool.py public-video <tool.py> [args...]
E:\codex-tools\bin\blindassist-python.cmd scripts/run_public_video_campaign_tests.py
```

`run_research_tool.py` 负责兼容 campaign 内的 sibling imports 和仍位于 `scripts/` 根目录的 SANPO helper；调用方不要自行注入 `PYTHONPATH`。四个 PowerShell 网络/设备 Adapter 可直接从本目录调用。

- [迁移时的完整语义索引](MIGRATION_INDEX_2026-07-21.md) 保留每个历史脚本的职责与安全边界。
- 新的 route-conditioned、object-agnostic risk-field 主线应建立新的领域目录，不在这里继续追加 r7/r8 实验轮次。
- 历史失败结论保持有效；目录迁移不改变任何训练、校准、blind、Android 或 production authorization。
