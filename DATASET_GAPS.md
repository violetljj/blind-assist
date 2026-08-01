# DATASET_GAPS

生成时间：`2026-08-02T01:14:04+08:00`
扫描器 schema：`dataset-master-ledger-v1`

## 结论

本次扫描发现 `5872` 条 session/package 记录，其中 `2832` 条有可定位媒体，`3040` 条为 manifest/structured-only；物理文件 `136317` 个，累计字节 `83,184,764,516`。`consumed/burned` 只表示历史使用证据，不等于全局封存；`fresh/reserved` 只有本地路径或 metadata 明确出现时才标记，未知不被解释为不存在。

## 扫描根与重复别名

- `checkout_artifacts.local`：`E:\linnan\linnan\artifacts.local`
- `outer_legacy_artifacts.local`：`E:\linnan\artifacts.local`
- checkout 内 `.downloads` 与 `test-artifacts.local` 是 junction alias，未作为独立 root 重复扫描。
- `E:\linnan\artifacts.local` 是存在的 legacy outer root，本次作为独立可访问 root 扫描；它与 checkout root 的内容重复由 SHA-256 duplicate groups 标出。
- 仓库 source/config/docs/scripts/build 等工程文件不作为数据 payload 计入；其中的资产说明只有在 artifacts.local 扫描根内物化为 manifest/metadata 时才进入 ledger。
- 文件发现快照：`2026-08-02T01:14:04+08:00`；profile 修复时间见 ledger 的 `profile_refreshed_at`；发现后消失的路径保留为 `not_readable`。

## 请求数据集覆盖

| 请求项 | session/package 数 | 有媒体 | manifest-only | 典型状态 |
|---|---:|---:|---:|---|
| `SANPO` | 1631 | 917 | 714 | FOUND |
| `EgoWalk` | 10 | 3 | 7 | FOUND |
| `Bonn` | 10 | 2 | 8 | FOUND |
| `REveL` | 32 | 5 | 27 | FOUND |
| `JRDB` | 45 | 22 | 23 | FOUND |
| `Shiraz` | 17 | 6 | 11 | FOUND |
| `Shanghai` | 10 | 8 | 2 | FOUND |
| `self_collected` | 227 | 151 | 76 | FOUND |
| `replay` | 983 | 281 | 702 | FOUND |
| `canonical_input` | 2 | 1 | 1 | FOUND |
| `segmentation_520` | 17 | 15 | 2 | FOUND |
| `event_eval_1920` | 3 | 3 | 0 | FOUND |
| `consumed` | 130 | 84 | 46 | FOUND |
| `burned` | 8 | 1 | 7 | FOUND |
| `fresh` | 90 | 51 | 39 | FOUND |
| `reserved` | 1 | 0 | 1 | MANIFEST_ONLY_NO_PHYSICAL_MEDIA |
- SANPO split inference：`dev=121`, `heldout=12`, `mixed=10`, `replay=63`, `test=98`, `train=272`, `unspecified=1055`；`unspecified` 表示路径/manifest 未给出可审计 split，不等于 train/test 缺失。

## 质量缺口与风险

- 损坏/不可解码 session：`62`；示例 source_id：SRC-01cad4d4988ba8fe8a0c, SRC-043bbb5254e25fa212e7, SRC-051117ebde39116f9419, SRC-13b7f4ffc356566c0088, SRC-16a698898d3af36134ad, SRC-1869569340f59374657d, SRC-1c1b8c83ed11a7a50103, SRC-1eb1d318d8c4ff0121ee, SRC-1f9c5bb40a52dc3b9cb9, SRC-2b9f288ba3845b24b240, SRC-2da6b52614f3a218571c, SRC-2f04f617e46d95606cbf。
- 文件级 profile 状态：`readable=135352`，`readable_probe=101`，`corrupt_or_unreadable=283`，`not_readable=229`，`not_evaluable=269`，`not_checked=39`。`not_readable` 表示发现后路径消失/权限失败，不等于内容损坏。
- 文件名/frame-key 级缺口：识别到 missing frame keys `32849`，duplicate frame keys `4058`；无法建立 frame key 的记录保持 `not_evaluable`。
- 哈希快照与重做 profile 的文件大小不一致：`0`；非零时应重新哈希后再把 profile 与 hash 作为同一快照使用。
- RGB-mask-depth-pose frame-key 对齐为 partial/misaligned：`20` 条。未建立 frame key 的 pose/metadata 不会被判定为对齐。
- 没有可解析 timestamp：`4286` 条；没有可解析 fps：`4482` 条。很多 image sequence 只有 frame index，不能从文件名安全推导真实时间。
- 发现内容重复组：`19070`；其中涉及排他角色冲突：`29`。重复内容不自动合并，因为不同 evidence role 仍需保留。
- `.bag`/`.db3`/部分点云仅做非空/结构级检查；如果没有可用 rosbags/codec，报告会保留 `not_evaluable_dependency_or_codec`，不写成可解码。
- archive (`zip/tar/gz/7z`) 只记录容器大小、hash 和非空状态，未擅自解压；archive 内部 session 需在后续单独 materialize 后再补扫。
- 角色主要来自路径 token 与同目录 JSON/JSONL metadata；这不是对历史研究文档的语义重判。相同内容在 fresh/reserved 与 consumed/burned 目录出现时必须先 HOLD。

## 研究问题边界

- RGB-only 资产最多支持连续性、检测/跟踪 Development 或 replay regression；不能单凭 RGB 证明 obstacle truth、pose、TTC 或安全。
- mask/depth/pose 的可支持问题取决于 frame-key、timestamp 和解码状态；存在对齐 gap 时，ledger 只支持 gap localization。
- `consumed/burned` 资产仍可用于 Development、回归、诊断或机制解释，但不能被重新称为 fresh/unseen/independent；历史协议终态保持不可变。
- 本文档是资产完整性/gap 报告，不授予新实验、Confirmation、Android、默认 App 或生产权限。
