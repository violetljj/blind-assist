# 规范符合性审查

## 审查结论

**PASS**

- 审查报告 SHA256：`19e7a2a5eb7394ee3c7496ff6096c25690de76f46f787e432027ff7176d06212`
- 审查时 evidence map SHA256：`e92abd6609e8ba3c5b9299a12ca0fd4240ba7d370d6f2c24aa3797af3a552887`

最终报告满足用户要求的主体交付：14 篇 PDF 已下载并完成全文精读，报告提供了机制级详细分析、建设性系统路线、分阶段实验、停止条件，并明确区分论文证据等级、研究建议与生产晋级。审查中发现的 P1 状态/引用槽漂移和 `sigmoid/no-BN` 术语问题已集中修复；复核后的 evidence map 与冻结报告一致。当前没有必须修复项。

---

## 逐项符合性

| 审查项 | 结果 | 证据 | 判定 |
|---|---|---|---|
| 下载论文全文 | PASS | `.downloads/papers/2026-07-frontier-upgrade/` 中 14/14 PDF 存在；14/14 SHA256 与 `refs/paper-inventory.json` 一致 | 产物和追溯元数据齐全 |
| 全文精读 | PASS | `notes/segmentation.md` 记录 19/19 页，`notes/robustness.md` 记录 77/77 页，`notes/temporal-human.md` 记录 66/66 页，合计 162/162 页 | 不是摘要级罗列 |
| 论文数量和页数 | PASS | JSON inventory 为 14 篇、页数求和 162；报告第 7 行写 14 篇、162 页；`paper-inventory.md` 同为 162 页 | 数字闭合 |
| 详细分析 | PASS | 三份精读笔记覆盖 2+5+7=14 篇；每篇包含机制、定量证据/页码、局限、迁移点、最小实验和停止条件 | 满足“精读、详细分析” |
| 综合而非堆摘要 | PASS | 报告 §3 按边界、鲁棒性/噪声、不确定性、时序、轨迹、人因六条机制线综合；§4–§8 转成架构、合同、实验矩阵和近期顺序 | 形成建设性路线 |
| 建设性与可执行性 | PASS | §5 给出已完成 P1，以及 D0/T0/P2/I0/E2/R1–R3/U0–U1/T1–T3/H1–H2 的状态、产物、进入条件与停止条件；§8 收敛为近期 P2/D0/T0/U0 四项 | 路线可预注册、可证伪、可回退 |
| “已建议”与“已验证”区分 | PASS | §2 用 P1 审计支持“P1 已完成但未稳定/未晋级”；§5 把已完成、待执行、当前主线和条件候选分开；§9 明确剩余路线尚未证明提高 | 未把建议写成实验结果 |
| 论文等级区分 | PASS | Mobile-Seed 按预印本降级；Escalator 明确为 position paper；AI Guide Dog 明确为 AAAI Spring Symposium；STEPP 的 accepted-manuscript 状态由 inventory 给出 | 没有把弱证据升级为主会算法证据 |
| 生产边界 | PASS | §3–§9 多处禁止 GPU FPS 外推手机、禁止双向未来帧/VLM直接告警；结尾固定 `benchmark-only` 和 `do_not_replace_default_model` | 没有暗示生产晋级 |
| 本地链接 | PASS | 8/8 相对链接存在：inventory JSON/MD、evidence map、三份笔记、P1 审计及其余本地引用 | 无断链 |
| 论文原始链接 | PASS | 报告 14 个外部论文链接与 inventory 中 14 个来源 URL 精确匹配 | 来源可追溯 |
| P1 最新事实 | PASS | 报告中的 best mIoU/boundary `0.4642/0.5235`、range `0.2951`、OS4/OS16 失败值与 `docs/SANPO_P1_LRASPP_ALIGNMENT_2026-07-13.md` 一致 | 已建议/已验证边界基本正确 |
| Evidence-claim map 当前性 | PASS | `LOCAL-06` 已更新为 P1-A range `0.2951`、OS4/OS16 拒绝及 P2/I0/E2 条件路线；所有 slot 已改到报告 §2/§3/§5 的真实位置 | 与冻结报告和 P1 审计一致 |
| P1 归一化术语 | PASS | 报告统一写 `sigmoid/no-pooled-BN`；未再出现 `sigmoid/no-BN` | 与代码/P1 审计一致 |

---

## 事实与数字核对

### 已闭合

1. **论文总量**：14 篇；PDF、JSON inventory、报告参考文献列表一致。
2. **总页数**：11+8+11+11+11+35+9+7+11+11+8+9+9+11 = **162**；三份笔记的 19+77+66 也等于 162。
3. **当前数据**：报告所述 real-only r3 为 400 train、200 dev、120 blind；训练/dev 12 个 session，加 2 个 blind session，共 14 个。与 robustness 笔记一致。
4. **标注质量回连**：train 38 HUMAN/362 MACHINE，dev 39 HUMAN/161 MACHINE，MACHINE 合计 523/600=87.2%；报告与 robustness 笔记一致。
5. **seed 因子**：model-seed selection 范围 0.2685、sampler 范围 0.0112、约 24.1×，且五组 worst scene 均为 `step_curb`；报告与本地审计/笔记一致。
6. **P1 结果**：P1-A 的 best mIoU/boundary 为 0.4642/0.5235、最差 selection 0.1970、range 0.2951；OS4/OS32 的两个 boundary 值为 0.0271/0.0130；OS4/OS16 与 OS8/OS16 最佳 selection 为 0.0968/0.1549。报告与 P1 审计一致。
7. **离线和 INT8 门**：报告 §5.1–§5.2 的数值与 `scripts/sanpo_candidate_quality_gate.py` 的不可变 dataclass 一致。
8. **设备门**：报告采用代码中的当前默认值 0.90/0.05/0.50/0.90/0.10/100 ms，与 `DeviceEventThresholds` 一致。`docs/SANPO_CANDIDATE_PROMOTION_GATES.md` 的 JSON 示例更严格，但该文档明确说明代码 dataclass 才是当前权威阈值，因此不构成报告冲突。

### 未发现的冲突

- 没有发现用论文 GPU FPS 推断 Android 延迟。
- 没有发现把 geometric boundary 与 `boundary_step_curb` 偷换。
- 没有发现把 entropy/MI、semantic unknown 和 extra abstain 合并。
- 没有发现把连续帧冒充独立 conformal calibration 样本。
- 没有发现把戴黑眼镜模拟参与者写成真实 BLV 用户证据。

---

## 必须修复项

**无。**

审查过程中发现并已复核关闭两项问题：

1. `refs/evidence-map.md` 已把 P1 从“未验证”更新为完成四组×五短跑后的真实结论，并把 PID/MSEED、MRFP/UPC/SWSEG、ValUES/KAND、时序与人因论文的 slot 改到报告 §3/§5 的实际章节和阶段。
2. 报告 P1 行已从 `sigmoid/no-BN` 更正为 `sigmoid/no-pooled-BN`；代码中的 `lraspp_high_bn`、`lraspp_low_bn` 没有被错误描述为删除。

---

## 非阻塞建议

1. `plan/project-overview.md` 仍写“当前阶段：S1 Evidence”。报告已经进入 review，建议在主代理完成两阶段审查后更新计划状态；这不影响报告内容正确性。
2. `Watch Your STEPP` 已在冻结报告补入 §3.5，覆盖缺口已经关闭；无需把它强行加入近期四项，只需保证 evidence-map 不再指向不存在的阶段。

---

## 验证命令与结果

### V1：PDF、哈希、页数

执行 inventory 驱动的 `Test-Path + Get-FileHash SHA256 + pdfinfo` 全量检查。

```text
paper_count=14
inventory_pages=162
pdf_exists=14
hash_match=14
page_match=14
pdf_pages=162
```

### V2：报告标题、链接和来源

```text
report_lines=246
headings=23
top_sections=10
report_pages_claim=162
inventory_pages=162
markdown_links=22
external_links=14
exact_inventory_urls_in_report=14
local_links=8
local_links_ok=8
Watch Your STEPP section=True
```

### V3：格式与占位符

```powershell
git diff --check -- docs/research/frontier-upgrade-2026-07/BLINDASSIST_FRONTIER_PAPER_UPGRADE_REPORT_2026-07.md
rg -n "TODO|TBD|PLACEHOLDER" docs/research/frontier-upgrade-2026-07/BLINDASSIST_FRONTIER_PAPER_UPGRADE_REPORT_2026-07.md
```

结果：`git diff --check` 无输出；未发现 TODO/TBD/PLACEHOLDER。

### V4：当前审查文件

本文件写入后必须再执行：

```powershell
git diff --check -- docs/research/frontier-upgrade-2026-07/plan/review/spec-compliance.md
```

复跑结果支持整体状态 **PASS**。
