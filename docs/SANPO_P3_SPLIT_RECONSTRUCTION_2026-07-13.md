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
2. 对候选执行连续 50 帧原始 mask 几何门，随后才下载 RGB、由隔离的 GPT/Codex 流程复核 scene，并进入 P3 planner。

首轮 180-session 通用实例扫描因 20 分钟硬超时只留下元数据，没有写出最终 mask 报告，结果不采信。随后完成两段互不重叠的 lateral 定向扫描：official-train 索引 `0–99` 产生 5 条新增稀疏候选，索引 `100–219` 再产生 8 条。第一段候选共检查 19 个连续窗口，`0/19` 通过；拒绝主因是 pedestrian/rider 同时进入中心走廊或存在其他中心风险污染，另有窗口帧数不足。第二段 8 个推荐窗口也已全部完成，`0/8` 通过。曾因 TLS EOF 中断的 `SPoj3wSxZDDu1zDcE8RYTRxtxrRP_AMK` 在授权重试后得到完整证据：侧向目标 46 帧、最长连续段 46 帧、可行走走廊 50 帧，但触发 `center_target_contaminates_lateral_negative`，故同样拒绝。

候选工具同时完成两项治理修正：发现器新增 zero-based `--start-session-index`，使批次覆盖范围可复核且不重复；连续窗口筛选器的 `--retries` 现在同时覆盖 description 与单张 mask 下载，修复 TLS EOF 会丢失整条报告的问题。新增 4 项定向测试均通过。

## P3-A：lateral 局部时窗预筛

`0/27` 的 exact-window 结果表明，原先“整 session 的 6 帧稀疏命中”对 50 帧 clean-lateral 合同的假阳性过高。发现器现新增可选的 `--local-lateral-frame-count 16` 预筛：仅在 sparse lateral 命中后，以推荐起点读取 16 个 10-FPS 原始 mask，要求 lateral target 至少 8 帧且有 8 帧连续段、可行走走廊至少 13 帧，并且零 center hazard / center lateral target 污染。

该预筛只能拒绝候选或授权调用既有 exact 50-frame gate，绝不表示接受；50 帧 geometry gate、RGB 下载、人工场景复核和 P3 planner 仍为必经步骤。预筛的逐帧证据、阈值和拒绝原因写入候选报告。P3-A 的四项单测与既有窗口三项重试测试均通过。official-train 索引 `220–339` 的 120 条只读 GCS 清单扫描完成，`0` candidate、`0` network/data failure、`0` local-prefilter rejection；全部条目在 `camera_chest/left` 首个 mask 清单中不足 6 帧，因而未打开任何 mask 像素。报告 SHA256：`be0a79899e135518aff021920f8671b980aed2e87e2c74d5f000c62cff21a61`。抽样核验 index 219/220/279/339 的 chest/head 也无公开 left segmentation mask。随后 `340–559` 的 220 条清单扫描同样完成，`0` candidate、`0` failure、`0` prefilter rejection，且 220 条全都不足 6 帧；报告 SHA256：`89bcec839f0e1187fa6537a36d04e43825a3aaad671952618344993257a50286`。至此 official-train `0–559` 已按不重叠范围完成 **chest-left** 候选发现：前 220 条仅产生 13 条 sparse lateral 候选，且 27 个 exact 50-frame windows 全部拒绝；后 340 条不具备可抽样的 chest-left segmentation sequence。这不等价于 head-left 已穷尽。P3 视角/来源合同现改为 chest 优先、head-left 回退；见 [P3 视角与新增来源契约](SANPO_P3_VIEW_SOURCE_CONTRACT_2026-07-13.md)。下一步是按该合同重新扫描 official train 的 head 候选，而不是继续将空 chest 清单误当成全数据集负例。

## P3-B：中心障碍候选与边界发现器补齐

既有 P3-A 只审计 clean-lateral，不应被误读为所有 scene 都没有可用候选。对 index `100–219` 已保存的 15 条 sparse 记录中 7 条 `center_obstacle`，逐条执行同一官方 train、50 帧、mask-only 几何门：4/7 通过，分别为 `gie8DH9dneyPGOyjs1un2ymDP0xWrWNH`、`Nta3Pe6qqxaApoI2CuagFKvoj1mzwpCO`、`SK1d5RB_ktBZhLvPs9-Q9hXeHojD49u-`、`wXwTZnJHaWsMeYvf-hR0a3ZHrNXH3I0G`；其余两条因不足 50 个对齐 mask 拒绝，一条因 15 帧/7 连续帧不足而拒绝。前三条均已下载为隔离的 50 帧 RGB+mask draft，并通过下载清单、GCS MD5、本地 SHA、连续帧和本地重放几何门；它们保持 `pending_review`，尚未人工/RGB 场景复核、未进入 recipe、canonical 或训练。前三份本地 selection evidence SHA256 为 `2f327c78522e08e4df5c3a75a2607e77e054e2f8aec56d4397c68903c302b65f`、`943b9fa2c4e1419e704c9480ee4512b02ddd1fbc3db8191fec4d6824895964f`、`1a6b998035af366a33e4358cd8021f15bdbd549843a0f1495ead019fc01e020b`。

同时，发现器补入 `step_curb` 稀疏候选：只要求 curb/stairs 到达下视野、存在可走走廊，输出的只是 boundary candidate；严格 50 帧门和 RGB review 才会区分平行边界与 step/curb，不能以 source mask 自动写安全语义。定向测试现为 18 项通过。

首个可恢复的 10-session `step_curb` 小批次（official-train index `0–9`）产生 5 条 sparse candidate；4 条属于既有 native session，只有 `wZ9KabBA03cxMpwWeYekeTGt502Ddp6h` 新增。其 50 帧 mask-only gate 通过，且 50 帧 RGB+mask draft、本地验证和回放门均通过；但该 session 只有 `camera_head/left` 标注，目视抽查显示高仰视角和不确定的当前路径关系。它仅保留为跨视角 boundary candidate，禁止与现有 `camera_chest/left` canonical 静默混合；应在 P3 recipe 加入相机同质性或跨视角独立评估门。其 sparse / remote screen / local replay SHA256 分别为 `3ce5844f1ff3d019d0d3f5e1c59ee4967a5fc05f600d0936044396c97ebf7bcc`、`4515b8d8019ca100e97bfe716472233f8d30177dff0a3df493eb7891b3c6064f`、`b618cbc8616e817caf9d279edf364757be2ee93bf576fcda9abeb5eeccb43ad0`。后续 scan 改为按新合同 chest 优先、head-left 回退，但任一 head 数据只能走独立 cross-view gate。

按该视角合同，official-train index `0–69` 的 auto-view scan 在 70 个 session 中选中 chest `42`、head `28`，`0` 网络失败，扫描报告 SHA256 `34621f8621d62152cdcb3c35410d638d1912dde27f2dcbe1df4904f22ef62b69`。head 产生 5 条 center-obstacle 和 12 条 step-curb 稀疏候选；5 条 center 的 exact 50-frame mask-only gate 为 `3/5` 通过：`GEwWDMMe8FFo-wHxZ4Pg1chw15P2nEEz`、`wZ9KabBA03cxMpwWeYekeTGt502Ddp6h`、`y9FMchN_WFT0sKOUar9WcrUoiKHQwD4U`，其余两条按中心侵入持续性拒绝。GEw/wZ9 已各导入 50 帧 official-train head-left draft，连续 frame、50 个唯一 RGB SHA 和 manifest validation 均通过，仍全部 `pending_review`；不进入 recipe、canonical 或训练。GEw manifest/inventory SHA256 为 `b6283156b5d059bfdfe97230227baa73ca9e8b64800ebdc92c5911f4e393d01f` / `c18922fafc1efdb1ab1b829b01ced79ef13839548432bfb6eb21a4d1502bdeb8`，wZ9 为 `8b90f8afd9bf99e96834e322d8824d62c2d4095d894fea57186e1358880c37b4` / `d0d957464e394da5a002565d6469b48332b81d4ab48a363d592beac895806873`。y9F 下载在 11/50 帧超时，缺 final manifest/inventory，已终止并排除。

## 当前状态

- 数据质量严重度：High；置信度：High。
- P3 planner：已实现、测试全绿。
- P3-A lateral prefilter：已实现、7 项定向测试全绿；official-train chest-left 0–559 的 lateral 候选发现已穷尽，未新增 clean-lateral accepted session。
- P3-B center / boundary：4 个 center mask-only 候选通过，2 个已下载为待复核 draft；step/curb sparse discovery 已补齐但尚未远程执行。P3 总 coverage 仍为红，不能重建 canonical 或训练。
- canonical / training：未重建、未启动。
- blind：未读取；official test 只在 recipe 元数据层被识别并由 planner 预读取拒绝。
- App / 模型资产：未修改；结论保持 `do_not_replace_default_model`。
# 2026-07-13 P3-B continuation addendum

## Chest-view step/curb discovery, indices 20-39

The `step_curb` detector remains a **candidate-only** source-mask search: it finds persistent curb/stair pixels in the lower field together with a walkable corridor. It cannot determine whether the real scene is a traversability hazard, a parallel edge, or an unrelated boundary. Every candidate therefore remains `pending_review` until independent RGB semantic review; none is added to a P3 recipe, canonical dataset, training run, or production model.

For official-train indices `20-29`, the chest-left sparse scan produced six previously unseen candidates with no network/data failures. Five completed the 50-frame source-mask geometry screen: `JrFB7oRxppNref18SJEl_t72ADmhGWlo`, `-PqSDmiEe2pXjmYHgxh4YEBsj0T5LU10`, `JtMYI6rJ4wiDsEVffAkee0kR5Zmrf8vM`, `1sftEbnzzIfYBdODrjDO9TbhcnsyKjAv`, and `W1ZpmAq74J8xfKg23BcwLx0QHfxM6W_w`; `C-g5n2_S5wUCaj75l2JdzeHyjxMBDxmv` was rejected for insufficient aligned source masks. The sparse report SHA256 is `fae4f27f91590559f6a369ae69e1f07ceeac74d9c7d6aea19910cbd3a8e3f717`.

`JtMYI6rJ4wiDsEVffAkee0kR5Zmrf8vM` and `W1ZpmAq74J8xfKg23BcwLx0QHfxM6W_w` have each been downloaded into isolated 50-frame RGB+mask drafts, passed manifest validation and local geometry replay, and remain `pending_review`. The W1 middle-frame inspection is a chest-mounted continuous descent past stair/railing geometry, so it is a plausible `step_curb` review candidate rather than a clean parallel-boundary negative. This inspection is not human approval and does not change labels. W1 local replay reports `target=50`, `longest_run=50`, `path=50`, `median_path=0.9215`, and `max_center_block=0.06891`.

The next chest-only official-train batch, indices `30-39`, completed with `0` candidates and `0` failures. It attempted ten independent sessions; seven did not expose a sufficient first chest mask page and the other three had no qualifying sparse hit. Its report SHA256 is `659f553fdb418efd083d54a68365cccd03a1a4d02bdfa99771a9292b80121824`. This is negative discovery evidence, not a data-quality pass: P3 coverage is still red and model training remains forbidden.

For chest-only official-train indices `40-49`, ten independent sessions were attempted (eight with chest inventory; two with insufficient first mask pages), producing seven new sparse candidates and zero network/data failures. All seven passed the mandatory remote 50-frame source-mask geometry gate: `DD9W-6F3D126azdsR_Usvu6zkNqkP8XG`, `75G7mMDsi4csa74ehbypzxLfs3ZJshsX`, `zY36WhUkV1AOl_aqxxBofB0_ash1nr7p`, `r7O3U6QanV_aqafpjva98r4ataD9BY50`, `reigDTj3Eqg5IraZfRGRHJTCxvVqMtHh`, `c4Mh8piOiKI4EEHHQOLTbFUfCDLlq29P`, and `XOjHrx-neyD2GJuPsd5mgT9vNvytEzyL`. The sparse report SHA256 is `219aaa995139d74dd5285f4e8b43b3810d5fdb7c1c1eb1285d273cc5e2f13287`.

`DD9W-6F3D126azdsR_Usvu6zkNqkP8XG` was selected first due to the strongest exact geometry signal (`target=40`, `longest_run=24`, `path=50`, `median_path=0.9983`, `max_block=0.28791`). Its isolated 50-frame RGB+mask download and manifest validation passed, and its local replay passed with evidence SHA256 `c74c49fbc1cbbfd7dc11a2fa2d115519e54c7937692c7bc88519fcd50a2f34cb`. Its middle RGB frame appears to be an ordinary urban sidewalk with vehicles, snow/curb edge and street furniture, not an unambiguous step. Therefore it remains `pending_review`; it is neither a `step_curb` positive nor an alert label. This confirms the intended guardrail: persistent source boundary geometry alone cannot establish risk semantics.

For chest-only official-train indices `50-59`, ten sessions were attempted, seven had insufficient first mask inventories, and two candidates were discovered with zero terminal network/data failures (a transient TLS EOF recovered under the configured retry contract). The sparse report SHA256 is `c25351c0d3ac8499dad2293f3e714c69eea918a2137cf75055c3a704b88e070b`. Both `qFDP9gJDz4MyXNxjl6mIEPFCUb8n1guU` and `gcyWXbvJVi0vjf4lLgIkR_0txpX8LAx9` passed their remote 50-frame geometry gates. qFDP then completed an isolated 50-frame RGB+mask draft, manifest validation and local replay (`target=50`, `longest_run=50`, `path=50`, `median_path=0.8152`, `max_block=0.3212`; selection SHA256 `74ce730413f5ab6827e5b257b3c59a1661d21be6c5bdc66e5b36f87dcaed8a70`). Its middle RGB depicts a curb-cut/crosswalk transition with pedestrians. It is a credible `step_curb` review candidate, but source geometry and inspection alone cannot decide its alert semantics; it remains `pending_review` and does not enter recipe, canonical, or training.

For chest-only official-train indices `70-79`, ten sessions were attempted (seven chest inventories; three insufficient first mask pages), yielding five sparse candidates and zero failures. The sparse report SHA256 is `a13c05a45fd750276c0c323930429be485c38d96245eded3c115421d08c71d3a`. All five passed the remote 50-frame geometry gate. `vczXAwthxnadTYUS_TiR7IHiqEQrdSJx` was selected for isolated download because it had 50 target/50 run/50 path frames and `max_block=0.24441`. Its RGB+mask draft, manifest validation, and local replay passed; local selection SHA256 is `2cf176bae0239f763f2473654fef82a05b538993a7e7aa90c96f52c1e0d0aad6`. Its middle RGB shows entrance stairs along the right side outside the forward sidewalk corridor. This is a strong `parallel_boundary`/matched-negative review seed, not an automatic step warning or training label; it remains `pending_review`.

For chest-only official-train indices `80-89`, ten sessions were attempted (eight chest inventories; two insufficient first mask pages), yielding three sparse candidates and zero failures. The sparse report SHA256 is `6bbeda410b728d423a3161a68d428e4fd032ee7f21343ee4024a5326ff1a4720`. All three passed the remote 50-frame geometry gate, but their maximum target corridor-blocking ratios were `0.02550`, `0`, and `0`; they were retained as mask-only review candidates rather than downloading more likely ordinary parallel-boundary RGB. This is a deliberate data-quality decision, not a rejection or a positive-label assignment.

For chest-only official-train indices `90-99`, ten sessions were attempted (eight chest inventories; two insufficient first mask pages), yielding five sparse candidates and zero failures. The sparse report SHA256 is `c9600534e7129f68d5112e42070602420bd88672589352eb75720864818ca422`. All five passed the remote 50-frame geometry gate. The highest maximum corridor-blocking ratio was `0.13272`, below the already downloaded high-information qFDP/vczX windows and without a new scene signal, so the batch remains mask-only evidence. No RGB was downloaded and no semantic label or training decision was made.

For chest-only official-train indices `100-109`, ten sessions were attempted (six chest inventories; four insufficient first mask pages), yielding two sparse candidates and zero failures. The sparse report SHA256 is `4130f8cb1059805461d88b75b3ec4694c56b50f4a76061a4f1cdd63186c74d67`. `gie8DH9dneyPGOyjs1un2ymDP0xWrWNH` was already an isolated center-obstacle draft, so it was not duplicated under this profile. The new `yGcmWEL64jMkK-ic2NWpFK4cpsCoJ8pN` passed remote geometry (`target=50`, `run=50`, `path=50`, `median_path=0.9811`, `max_block=0.07497`) but remains mask-only due to low incremental information. No RGB, semantic label or training decision was created.

## Semantic-review audit of boundary drafts

Three 50-frame chest-view drafts now have hash-bound model-review responses. This is a routing aid for a future dense-annotation queue only; it is not human review, a risk label, benchmark truth, training evidence, or promotion authority. Each validation result is explicitly `promotion=not_promoted`.

- `qFDP9gJDz4MyXNxjl6mIEPFCUb8n1guU`: `needs_recapture`, expected `no_alert`, confidence `0.78`. The five-second curb-cut/crosswalk window and pedestrian context cannot establish an event-level alert decision.
- `vczXAwthxnadTYUS_TiR7IHiqEQrdSJx`: `reject`, `parallel_boundary`, expected `no_alert`, confidence `0.91`. The entrance stairs are outside the forward walking corridor; retain only as an unlabelled matched-negative review seed.
- `W1ZpmAq74J8xfKg23BcwLx0QHfxM6W_w`: `needs_recapture`, expected `no_alert`, confidence `0.74`. The forward scene is a bounded descent/ramp beside stair/railing geometry, not enough to establish a discrete unsafe elevation change or alert threshold.

The repeated disagreement between persistent source-mask boundary geometry and operational alert semantics is direct evidence against using this profile as a pseudo-label source. The next admissible path is longer-context (`10–20 s`) consented, GPT/Codex-reviewed event collection with same-session matched counterparts and anchors, while public drafts remain isolated and `pending_model_review`. P3 coverage remains red; no canonical reconstruction, training, calibration, benchmark decision, or default-model replacement is authorized.

## Center-obstacle continuation, indices 110-129

The non-overlapping official-train `110-129` auto-view sparse scan completed with seven `center_obstacle` candidate rows and zero failures; report SHA256 is `10e4dab53e6aecf385f35671977071b846c1dfc29ae6a9a023463d70d5ef3476`. One row (`Nta3Pe6qqxaApoI2CuagFKvoj1mzwpCO`) duplicates an earlier isolated candidate and is not re-screened. The other six are new discovery-only rows: chest `We-WV-8Vm4iIQjrqprxcTUBltze15m5i`, `HEmFiCyijsIoIt_sehLfZ4xyCUq06mss`, and `J4PblHagnYNO09IdrDjQShle2Z0Dk-Yu`; head-only `LpjeyfDdXRDw03hbRDdm4e6MPcR0Ypua`, `FeJgoz6tfxhlDqiesDGIZdbM3VoE_HMZ`, and `VjpjnvAcYfG88ud095O2fv9dFUj4OAOM`.

No exact 50-frame result, RGB download, semantic label, P3 recipe row, canonical entry, training decision, or promotion decision exists for these six rows. Chest candidates must pass the exact source-mask gate before any isolated RGB draft; head-only candidates must additionally remain in the independent cross-view lane.

The resumed chest exact-screening pass produced one accepted remote-mask window for `We-WV-8Vm4iIQjrqprxcTUBltze15m5i` (`target=26`, `longest_run=17`, `path=50`, `median_path=0.8065`, `max_block=0.21535`; report SHA256 `eedf3ff1ff54289141b524e1d68c387c561fbe473c67124b109b7f35606941c0`) and one accepted window for `J4PblHagnYNO09IdrDjQShle2Z0Dk-Yu` (`target=41`, `longest_run=35`, `path=50`, `median_path=0.9140`, `max_block=0.10590`; report SHA256 `bd0acd7645bc3e2e1c111bf2c21691024e43dff29f8558656034b76982f68b04`). `HEmFiCyijsIoIt_sehLfZ4xyCUq06mss` was rejected because only 30 aligned source masks exist for the requested 50-frame window (report SHA256 `f479f9cfcd6abff5b0035bbd430b3dce456316e6ac971e0a52b93e210c36a70fc`). The accepted decisions are remote-mask-only queue candidates, not labels or P3 admissions.

The isolated 50-frame RGB+mask download for the higher-information J4P window reached the enforced 10-minute timeout without writing a final `manifest.draft.jsonl`; its empty/incomplete local directory is retained for forensic evidence and excluded. It must be re-downloaded into a fresh directory with a complete manifest before local replay, PII review, model review, or annotation-queue routing. No RGB-based semantic conclusion was drawn.

A fresh, isolated J4P retry completed after a longer transfer window with 50 RGB/mask frame pairs and `validation_ok=True`, still `benchmark_ready=False`. The local geometry replay exactly reproduced the remote result (`target=41`, `run=35`, `path=50`, `median_path=0.9140`, `max_block=0.10590`; selection SHA256 `7d86dccd97a365c4aa9efb1765fdef8751e586e457cab3d2fe8e241babfbf88e`). Hash-bound model review inspected frames 0/25/49 and returned `needs_recapture`, primary bucket `center_obstacle`, `corridor_event_present=true`, expected `no_alert`, confidence `0.76`, and `selection_evidence_agrees=false`; verifier result is `promotion=not_promoted`. The large fixed steel structure/rail/bench geometry is present in a conservative mask corridor, but the five-second visual context shows a plausible open passage and changing camera orientation, so it cannot establish an actual route-blocking event. It is added only as an unclassified static-structure counterfactual seed requiring longer same-session context, a matched traversal and independent GPT/Codex event adjudication.

The other accepted chest window, We-WV, also completed an isolated 50-frame RGB/mask draft with `validation_ok=True` and local replay (`target=26`, `run=17`, `path=50`, `median_path=0.8065`, `max_block=0.21535`; selection SHA256 `bed76cb0b28dfecc21e1469813b90f14bc145bb6f354cf0aabdd63151760b06e`). Hash-bound visual review of frames 0/25/49 returned `reject`, primary bucket `parallel_boundary`, `corridor_event_present=false`, expected `no_alert`, confidence `0.95`, and `selection_evidence_agrees=false`; verifier result is `promotion=not_promoted`. The open plaza/walkway remains navigable; benches, planters and entrance fence explain the mask signal, while distant pedestrians/dog do not establish a near-forward blocking event. It is an unclassified false-positive counterfactual seed only. The two completed center-profile RGB reviews therefore add direct evidence that geometry-only center-obstacle selection is currently dominated by static-boundary false positives; no draft from this batch enters dense annotation, P3 canonical, training, calibration or promotion.

Two earlier downloaded center-profile drafts completed the same hash-bound review closure. `3ok1zz3n49UtV6i2AGMLv-2gBifNtc56` had a strong 50/50 mask signal but visual review showed static plaza seating, fence, planting and pigeons with an open paved route; it was rejected as `parallel_boundary`, no corridor event, expected `no_alert`, confidence `0.97`, selection disagreement and `promotion=not_promoted` (selection SHA256 `93a9a34ebe8cb3b7363c520200f9639c05d82d3af409c695026e093ce764e659`). `cBVSUvSNGsl6C4kHrsaG6RwL9u0L0Ln2` showed a crowded market passage (selection `target=43`, `run=31`, `path=50`, `max_block=0.08826`; SHA256 `6b1128d27ba201178bc314043f695971e2580a621e0db0820eb2d7aa2d834ff8`), but five seconds cannot distinguish normal crowd flow from a route-blocking cut-in. It remains `needs_recapture`, `center_obstacle`, corridor event present, expected `no_alert`, confidence `0.72`, selection disagreement and `promotion=not_promoted`. Neither single model review is a label; both become unclassified counterfactual seeds requiring 10–20 second same-session context, matched negatives and independent GPT/Codex lifecycle/event adjudication.
