# SANPO P3 split / session 重构审计（2026-07-13）

## 结论

P3 已进入数据扩充阶段，但尚未完成，也未授权训练。当前 real-only canonical 只有 12 个可用于 train/dev 的独立 SANPO official-train session，四个场景均为 `2 train + 1 dev`；它在 P3 的 `4–6 train + 2–3 dev` 门下必须失败，不能靠重排、扩窗或把同一 session 切段来伪造独立性。

当前 canonical 全图四类像素分布为：

| split | walkable | boundary | obstacle | unknown |
|---|---:|---:|---:|---:|
| train（400 帧） | 53.065% | 0.857% | 22.336% | 23.742% |
| dev（200 帧） | 37.647% | 16.976% | 19.476% | 25.902% |

boundary 的 train/dev 占比相差约 `19.8×`；dev 的 step/curb 单 session 仍主导 boundary 真值。因此当前 dev 不足以支持稳定跨 session 结论。

## 已落地的 P3 规划门

新增 `scripts/plan_sanpo_p3_session_split.py`。它只接受 SANPO official-train 候选，在打开任何 manifest 前拒绝 official-test/blind 条目，然后从原分辨率 panoptic mask 通过固定 `SANPO_MAP` 统计四类像素。求解以 native session 为原子，不允许 raw-mask SHA 跨 train/dev 复用。

固定硬门如下：

- 每场景 train `4–6` 个独立 session、dev `2–3` 个独立 session；
- 四类 train/dev pixel share 比率均不超过 `2.0×`；
- train/dev 每类至少由 2 个 session 贡献，dev boundary 至少由 3 个 session 贡献；
- dev boundary 单 session 最大贡献不超过 `50%`，其余 split/class 不超过 `60%`；
- official split、manifest/mask SHA、连续帧、未知 native class、路径逃逸、重复底层 mask 或搜索空间超限任一异常均 fail closed；
- 无 gate-green 精确组合时不写 plan/report；成功报告必须记录输入哈希、session inventory、原始像素数/占比、集中度、effective session count、assignment SHA 和 `blind_access=not_accessed`。

定向单测 `9/9` 通过，覆盖候选顺序不变性、阈值边界、极细 boundary 原图像素、official-test sentinel 预读取拒绝、底层 mask 泄漏和无可行分布组合时零输出。

## 本地库存与扩充缺口

本地不存在额外、完整且可合法加入 train/dev 的 official-train session。`rjY60`、`utp60` 只是现有 native session 的扩窗；旧 stairs draft 也复用现有 session；其他已下载的候选属于 official test，禁止进入 train/dev。

达到最低 `4 train + 2 dev` 需要每场景新增 3 个 independent official-train session，即至少新增 12 个。候选发现遵循两阶段流程：

1. 只读取官方 train 的稀疏原始 mask，生成 candidate-only 清单，不下载 RGB；
2. 对候选执行连续 50 帧原始 mask 几何门，随后才下载 RGB、人工复核 scene，并进入 P3 planner。

首轮 180-session 通用实例扫描因 20 分钟硬超时只留下元数据，没有写出最终 mask 报告，结果不采信。随后完成两段互不重叠的 lateral 定向扫描：official-train 索引 `0–99` 产生 5 条新增稀疏候选，索引 `100–219` 再产生 8 条。第一段候选共检查 19 个连续窗口，`0/19` 通过；拒绝主因是 pedestrian/rider 同时进入中心走廊或存在其他中心风险污染，另有窗口帧数不足。第二段 8 条候选的连续窗口读取因权限审核链路断开而未执行，不能计为接受或拒绝。

候选工具同时完成两项治理修正：发现器新增 zero-based `--start-session-index`，使批次覆盖范围可复核且不重复；连续窗口筛选器的 `--retries` 现在同时覆盖 description 与单张 mask 下载，修复 TLS EOF 会丢失整条报告的问题。新增 4 项定向测试均通过。

## 当前状态

- 数据质量严重度：High；置信度：High。
- P3 planner：已实现、测试全绿。
- P3 session coverage：红；至少缺 12 个已复核独立 session，当前新增 accepted session 仍为 0。
- canonical / training：未重建、未启动。
- blind：未读取；official test 只在 recipe 元数据层被识别并由 planner 预读取拒绝。
- App / 模型资产：未修改；结论保持 `do_not_replace_default_model`。
