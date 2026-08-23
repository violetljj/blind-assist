# Public Identifiable Referent C2 small-roster protocol V1 (2026-08-24)

状态：`PROTOCOL_LOCK / ONE_METADATA_FREEZE / ONE_MATERIALIZATION / 5_TO_7_EPISODES / REFERENCE_IMAGE_INSTANCE_UNIQUE_ONLY / NO_BASELINE / NO_ALGORITHM`

## 唯一问题与终态

C2 只回答：能否从公开 multi-view source 实际构造 5--7 个满足
`PUBLIC_IDENTIFIABLE_REFERENT_CONTRACT_V1` 的 source-disjoint `REFERENCE_IMAGE_INSTANCE + UNIQUE` episodes？

只允许两个科学终态：

- `SMALL_ROSTER_MATERIALIZABLE`；
- `C2_NOT_EVALUABLE_*`，并记录 metadata、transport、identity、truth 或 interface 的具体失败。

本轮不运行 passive baseline、detector、matcher、provider、teacher、Active Referent Search、FSM、tracker、RL、control、
P1 或默认 App。成功只证明合法小 roster 可构造，不证明 identity、grounding、navigation 或产品能力。

## 固定 source 与 roster 规则

Source 固定为 SUN3D 官方 `Fully Annotated Sequences with Pose Correction` registry。它只提供 native object ID、polygon、
polygon camera XYZ、file list 与 corrected extrinsics；不再使用 generic category goal。已消费
`hotel_umd/maryland_hotel3` 永久排除。剩余官方顺序的 7 个 sequences 全部读取 metadata，每个 sequence 最多一个
episode；不得因 yield、object label、later pixels 或任何模型输出换源、补源或从同一 sequence 切第二个 episode。

每个 sequence 的 target 由 annotation/extrinsics-only 固定规则选出：

1. 排除 `wall / floor / ceiling` structural labels；label 只在 private selector 中使用，不进入 public goal；
2. target 至少有一个 reference frame，native polygon clipped bbox 占 640×480 的 `2%--80%`；
3. 至少 3 个 later-visible frames，bbox 至少 1%，source-frame gap `>=30`；
4. 每个 later frame 相对 reference 必须满足 camera translation `>=0.30 m` 或 target viewing-ray angle `>=15°`；
5. target ranking 固定为：selected later frames 中 same-normalized-class distractor count 降序、全部 qualifying later
   views 数降序、normalized label、native object ID；distractor 只作 metadata-priority，不设成功配额；
6. reference 是该 target 最早能支持至少 3 个 qualifying later views 的 frame；later observations 固定取 qualifying
   list 的 first/middle/last；不得看 RGB 后换 frame。

若 metadata freeze 后合格 source 少于 5，立即签署 `C2_NOT_EVALUABLE_METADATA_ROSTER_LT_5`，不下载任何图像。若为
5--7，则 roster、private physical instance ID、map-derived world anchor、reference/later filenames、native regions、
viewpoint metrics、source hashes、protocol/runner/contract/schema hashes 与预算全部冻结。

## 一次 materialization 顺序

Materializer 必须严格按以下 barrier：

1. 验证 roster file/body SHA 与 implementation lock；
2. 只下载 reference images，验证 SHA、640×480 和 frozen public target region；
3. 对每个 episode 写 provider-public contract 与 evaluator-private identity lock；
4. 写 `identity-lock-barrier.json`，证明 later-image GET count 仍为 0；
5. barrier 后才下载 3 个 frozen later images/episode；
6. 写 evaluator-private `VISIBLE + native region` truth，绑定 public/private receipts，并运行 V1 truth validator；
7. 写 public manifest、private evidence manifest 与 final report。

不得删除/覆盖 formal root，不重选、不补帧、不重试模型。HTTP 只允许普通 GET；没有 provider/model budget。网络或文件
失败保留 partial journal 并签署 `C2_NOT_EVALUABLE_TRANSPORT_OR_MATERIALIZATION`，不得另开 replacement roster。

## 预算与成功门

- metadata sources：固定 7；metadata GET 上限 `15`（registry 1 + annotation/extrinsics 14）；
- episodes：`5--7`；reference images：每 episode 1；later images：每 episode 3；
- image GET 上限 `28`，payload 总上限 `20,000,000 bytes`；
- provider/teacher/detector/matcher/baseline calls：全部 `0`；
- source IDs、reference SHA、later SHA 全部跨 episode/role 唯一；
- 100% contracts 为 `REFERENCE_IMAGE_INSTANCE + UNIQUE`；
- 100% identity locks 在 later image GET 前存在并 hash-bound；
- 100% later observations 通过 frozen real-viewpoint gate、native region truth 与 V1 truth binding；
- episode sources 100% disjoint；same-class distractor episode count只报告，不作为成败门。

只有全部门成立才输出 `SMALL_ROSTER_MATERIALIZABLE`。C3 passive baseline 仍需另立版本，本协议不自动授权。

Claim ceiling：
`PUBLIC_IDENTIFIABLE_REFERENCE_IMAGE_SMALL_ROSTER_MATERIALIZATION_ONLY_NO_IDENTITY_BASELINE_ALGORITHM_NAVIGATION_CONTROL_SAFETY_OR_PRODUCT_CLAIM`。
