# Hardware support run log

| Date | Code version | Configuration changes | Metrics (units/denominators) | Conclusion / result link |
| --- | --- | --- | --- | --- |

| 2026-10-11 | `host/cnh_sim2real_*.py` (source hashes in payload) | EXPLORE，既有13真实CNH片段；far-only模板/差分floor，冻结三模型两路径及既定两格诊断 | 12922decoded/12726满窗；旧M3 H/B7875/1951、5格10795/6661、S10799/6726；新各12726/12726；bias340/1024、floorR².0024；CPU保守≤400/1800command-wall、GPU≤150/300s | 适配原型完成，物理合同/迁移未修复；独立raw/forward PASS；仅真实回放，重训/标注采集仅草案。[报告](SIM2REAL_INPUT_CONTRACT_DEV_20261011.md) |
