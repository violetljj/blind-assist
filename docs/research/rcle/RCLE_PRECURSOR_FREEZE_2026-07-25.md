# RCLE 前序 Looming 现场冻结

状态：frozen snapshot

日期：2026-07-25

## 结论

用户中止本项目其他工作后，原 `EGOMOTION_COMPENSATED_LOOMING_SIGNAL_R0/R1` 不再续跑。遗留代码和日期化文档具备可复算价值，保留为 RCLE 的前序机制、来源权威与失败边界证据；它们不构成 RCLE-Minimal Phase A 完成，不开放 Bonn、受控采集、Android、人体或生产后继。

## 保留范围

| 范围 | 数量 | 处理 |
| --- | ---: | --- |
| `scripts/research/egomotion_compensated_looming/` | 41 个代码、测试、README 与依赖文件 | 保留为 frozen precursor Module；后续 Phase A 只在独立 `rcle_minimal` 子模块中新增工作 |
| `docs/research/ustrf-sc/USTRF_EGOMOTION_COMPENSATED_LOOMING_*.md` | 12 份目标、协议与结果文档 | 保留为日期化历史证据，由 USTRF-SC 索引和 RCLE current 入口降级管理 |
| `artifacts.local/datasets/egomotion_compensated_looming_*` | 本地数据 | 继续忽略，不进入 Git |
| `artifacts.local/evidence/ustrf/egomotion_compensated_looming_*` | receipts、traces 与评价证据 | 继续忽略，不进入 Git；只用于复算冻结结论 |

Module 的 41-file manifest SHA-256 为 `bf4bf4c7617a1530a5999015b4b4bc9d7fc78ea9b311eec2b522bbd54eaaed0a`；12-document manifest SHA-256 为 `6398f2a7e89a774cab5673ac0ab8be8ed7aed777f2d4802c4a4bb15b7995cb3a`。

聚合算法为：按仓库相对路径排序，对每个文件写入 `SHA256␠␠normalized/path`，使用 LF 连接并保留末尾 LF，再计算 UTF-8 manifest 的 SHA-256。

## 冻结终态

- R0：`FAIL_CLOSED_NEW_DATA_OR_TRUTH_AUTHORITY_BLOCKED / VALID`。公共来源和反事实 cell 分母未闭合，六个正式信号臂未形成 R0 算法结论。
- R1：`R1_CLAIM_SCOPED_SOURCE_PROGRAM_NONAUTHORITATIVE_EVALUATION_QUARANTINED_INPUT_AUTHORITY_BLOCKED / VALID`。
- Bonn 的 base/oracle/full-6DoF traces 是实际计算产物，但 canonical 3×3/500ms truth 仍为 `18/18` abstain。
- 503-pair 评价使用全局 q90 signal 与中央 ROI q05 truth proxy，空间单元和 authority 不匹配，只能作为 diagnostic；其自报 stop 没有接受或停止 RCLE 的权限。
- 原受控硬件、84-trial、Bonn/REveL 后继和旧 R1-A 路线全部停止，不再形成自动任务。

## 验证

从仓库根目录执行：

```powershell
E:\codex-tools\bin\blindassist-python.cmd -m unittest discover `
  -s scripts/research/egomotion_compensated_looming `
  -p 'test_*.py' -v
```

结果：43 项 focused tests 全部通过。

全部 39 个 Python 文件通过 `py_compile`。冻结第三方依赖见 Module 内的 `requirements-frozen.txt`。三个只读 validator 复算通过：

- `validate_source_authority_inventory_r0.py`；
- `validate_r1_claim_scoped_source_program_r0.py`；
- `validate_claim_scoped_r1_freeze.py`。

这些验证只证明冻结代码、receipt 和终态一致，不证明算法有效。

## 根目录模型清理

根目录的 `mobileclip_blt.ts` 实际是 599,764,649 字节的 PyTorch TorchScript ZIP 模型，不是 TypeScript 源码，也不属于 RCLE runtime。其 SHA-256 为 `A67804D1B0F07B8B9A20C1761EC0847F34660F5FA338EC70E8F3FCE68ED95E54`。

已核验规范副本 `E:\codex-tools\models\ultralytics\mobileclip_blt.ts` 的大小和 SHA-256 完全一致，因此移除仓库根目录误放副本；规范副本继续保留，可供历史 annotation proposal 工具使用。

## RCLE 复用边界

Phase A 可以复用：

- rotation/flow/local-affine 的纯函数与单元测试思路；
- manifest、receipt、abstention 和输入防火墙模式；
- 已发现的单位、空间对齐和 truth-authority 失败案例。

Phase A 不得复用为主结果：

- 旧 R0/R1 的数据角色、阈值、stop 或评价结论；
- 503-pair diagnostic 分数；
- Bonn central-ROI truth proxy；
- route-conditioned USTRF 的窗口、truth、candidate output 或 lifecycle。

Phase A 必须重新从程序生成输入、明确的 trial 真值和 R1.1 Kill Gate A 开始。
