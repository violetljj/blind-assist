# USTRF 跨相机 truth—路线几何一致性 R1.2c（2026-07-21）

## 结论

R1.2c 已完成协议冻结与独立 oracle 执行，但当前在 Japan 上 **fail-closed**：六个正事件中 `5/6` 至少存在一个 alertable `robust-inside` anchor；Japan 的两个 alertable anchor 均为 `robust-outside`，状态固定为 `truth_geometry_conflict`。

因此，London FP16-768 GPU 候选虽然已按单变量预注册，**当前不允许导出或执行**；完整连续事件、600 秒 soak 与 R1.3 也都未获授权。旧 polygon、旧 R1.2b 失败结果、阈值和 detector/tracker 均未改变。

## 冻结的 oracle

独立 oracle 只读取冻结的事件 `alertable_start_ms`、目标实例接触点和 hash-bound route polygon，不读取 detector、tracker 或 R1.2b 关联输出。每个正事件必须至少有一个可见 target anchor 满足：

- anchor 时间不早于冻结的 `alertable_start_ms`；
- 在相对画宽 `.01/.02/.03` 三档不确定性下都判为 `inside`；
- 缺失时直接标记 `truth_geometry_conflict`，不把 detector 漏检当作首因。

| 正事件 | alertable anchors | robust-inside anchors | 状态 |
| --- | ---: | ---: | --- |
| Japan | 2 | 0 | `truth_geometry_conflict` |
| Edmonton | 1 | 1 | consistent |
| London | 2 | 2 | consistent |
| Thailand | 1 | 1 | consistent |
| Bridge | 1 | 1 | consistent |
| Roadwork | 1 | 1 | consistent |

Japan 在 `10000ms` 与 `12000ms` 的接触点都对旧 polygon 为 robust outside。项目中的“人工角色”由模型承担：两个 fresh-context 模型独立复核后，由第三模型处理分歧，不等待真人。禁止拖动或重拟合旧 polygon 回救 R1.2b；若独立路线来源被判无效，只能在新版本中前瞻冻结新来源并完整重跑，旧失败保持不变。

本轮模型仲裁已经完成。A/B 都否定 Japan 是已证实的 strict positive；A 建议 unknown/exclude，B 倾向 strict negative。第三模型最终裁定为 `event_truth_unknown / route_relation_inconsistent / EXCLUDE_FROM_SCORE`：strict contact alertable interval 为空，原静态 polygon 只保留 provisional visualization 权限，不能作为真实全窗路线；独立 r797a future-route oracle 也给出 `context_only / explicit_route_intervention=false`，只作辅助一致性证据，不能单独升级正负真值。Japan 因此不得补作第六正例，当前仍缺一个合格正事件。

最终 adjudicated oracle 收据位于 `artifacts.local/evidence/ustrf-crosscam-codex/truth-geometry-r12c-seen-diagnostic-v1/truth_geometry_consistency_adjudicated_v2.json`，SHA-256 为 `77de4ca68c0f1724cb4ceeb768d87fea816f81e68f27052a5b126a9956469740`；模型仲裁收据 SHA-256 为 `e4cc4b45187a2ee41a3bb711679f4efe92b15cb76800eb99fd447f37633794e6`。旧的 pre-adjudication 收据保留为不可改写历史。

## London 单变量与后续门

2026-07-22 后续已预注册一个不消费 R1.3 的补位正例：Bangkok Modern Center `328000–340000ms` 红白交通锥事件。两份独立模型复核一致，333s/336s 在 `.01/.02/.03` 下均 robust inside；328s 因边界余量不足降为非计分 uncertain，339s 只作 clear proxy。详见 [seen positive 预注册](USTRF_CROSSCAM_SEEN_POSITIVE_R12C_PREREG_2026-07-22.md)。该合同将资格计数补到六个，但仍须物化 R1.2c v2 并重跑完整六正例 oracle，本页 v1/Japan 失败历史不改写，768 仍未授权。

唯一候选为 `r12c_c1_sameweights_fp16_768_gpu_london_only`：复用 `yoloe-11s-seg.pt` 同一 SHA-256、静态三类、`.05/.30/.45`、FP16 与 GPU delegate，仅将输入从 `640` 改为 `768`。协议顺序固定为：

1. 六个正事件 truth—geometry 全部一致；
2. 机械 parser/标签/失败计数/延迟 canary；
3. 直接跑完整 12 事件连续重放，不再选择别的候选；
4. 正例必须 `6/6`，负例假告警、重复交付、共现接管、身份切换均为 `0`，并通过首警、出画与歧义率门；
5. 只有完整事件门通过，才运行 SM-S9280 600 秒 soak；
6. oracle、事件门与 soak 全通过后，才允许生成 R1.3 unlock receipt。

若 768 仍漏 London，停止分辨率搜索，另开预注册的小目标 detector 假设。FP16-320、INT8、tracker 优化与提前消费 R1.3 均保持关闭。

## R1.3 复核规则

R1.3 v2 仍只保存 12 个未打开槽位（6 正/6 负）。未来解封后，每个来源由两个彼此不可见结果、fresh context 的 VLM 独立复核；detector 输出对两者隐藏。两者分歧或任一 `truth_geometry_conflict` 均进入第三个 fresh-context 模型仲裁，在解决事件定义、路线来源/有效性、目标身份和 alertable 区间前不计分。模型可承担项目的“人工角色”并推动研究继续，但该 provisional truth 不等于真人或生产级真值。

## 验证与权限

- R1.2c focused 合同测试：`4 tests` 通过；
- 独立 oracle：成功生成 hash-bound 收据，并按预期只在 Japan 上 fail-closed；
- 未导出/运行 768，未运行连续重放或 soak，未读取 R1.3 来源；
- 保持 `benchmark-only / do_not_replace_default_model`，不修改 App、默认模型或生产路径。
