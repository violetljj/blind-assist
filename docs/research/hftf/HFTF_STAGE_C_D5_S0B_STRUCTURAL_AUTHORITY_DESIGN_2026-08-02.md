# HFTF Stage C D5-S0B：structural authority 设计

## 结论

S0A.1 已证明目录容量充足，但没有证明这些 ZIP 能为 HFTF 提供可解释、共同对齐的
结构权威。S0B 要回答的是：对冻结 catalog 中除一个 schema sentinel 外的 197 个
`Data_diff` 父体，受限官方元数据是否
能同时建立 robot height、精确 robot→`lcam_front` 外参、动态 front pose，以及
image/depth/seg/pose 共同连续 25 个 10 Hz raw frames。

本设计只授权提交以及后续冻结 provider-resolution/S0B execution contract；现在不授权
读取新 toolkit blob、请求数据托管端或打开 ZIP。

## 必须先冻结的执行权威

为避免“未读 provider source 就先猜 URL”的循环，执行分三段：

1. P0 先冻结并执行 source-only provider resolver：只读 exact toolkit commit 中由
   download-ground 路由机械引用的 tree/blob，禁止 dataset host 请求，输出 hash-bound
   URL template 与 198-parent archive mapping。
2. P1 在 P0 成功后，机械取 catalog `parent_id` 字典序最小者作为唯一 schema sentinel。
   它永久退出 structural pool、payload、ecology 和 effect，仅用于冻结 member path、
   basename/index、camera/modality、metadata field、extrinsic、pose 与 ZIP method schema。
3. R0 再绑定 P0/P1 结果、exact catalog hash/198-row order 和排除 sentinel 后的 exact
   197-row order，完整 census，不得用 census parent 改 schema。

R0 还要冻结 HTTP/redirect/range/retry、每请求/每 archive/每 parent/全局 byte budget、
ZIP/ZIP64 EOCD、central directory，以及仅目标 metadata/pose member 的 bounded local
header 与 compressed range。只允许 stored/deflate；encryption、data descriptor、
ZIP64 extra、CRC32、compressed/uncompressed size 与 decompression-ratio 必须有明确
fail-closed 规则和对抗性 fixtures。

允许读取的上限只有：

- ZIP EOCD/ZIP64 locator 与 central directory；
- exact trajectory metadata JSON；
- exact `lcam_front` dynamic-pose member，但只流式计算 SHA、bytes 和 nonempty line
  count，不解析或保留 pose 值。

禁止 image/depth/seg member payload、其他 scene member 和任何 outcome。

## 结构门

每个合格 parent 必须有：

- 带精确 field path 与单位权威的有限正 metric robot height；
- 带方向、frame names、convention 与单位的有限 rigid robot→camera transform；
- exact front dynamic-pose member 至少 25 行；
- 官方 schema 对 pose row ordinal ↔ source frame index 的明确绑定；
- central-directory 中 image/depth/seg indices 与上述 pose indices 的共同连续 25 帧。

若有多个窗口，机械选择共同 index set 中数值最早的连续 25 帧；随后相对窗口起点固定
取 offsets `0,2,…,24`，得到 13×5 Hz、跨度 2.4 秒。若 pose-row index
映射、height、extrinsic、member identity 或单位无法权威观察，整个 S0B 必须
`SOURCE_AUTHORITY_NOT_EVALUABLE`，不能把 UNKNOWN 当作 parent 不合格或 pool 不足。

三类 image/depth/seg 的 member path、basename、camera/modality/index regex 与
duplicate/ambiguity 规则必须由 P1 冻结，不能从名字临场猜。

分类优先级也预先互斥：

1. provider/schema/field/unit/frame/index/member/解释权的全局缺失或歧义：
   whole `SOURCE_AUTHORITY_NOT_EVALUABLE`；
2. transport/range/budget/hash/implementation/protocol-parser/partial failure：
   whole `INVALID`；
3. authority 与 parser 全局有效且读取成功，但单 parent 缺项、malformed 或确定违反门
   （height 非正/非有限、extrinsic 非 rigid、pose <25 行、无共同 25 帧）：
   `parent_ineligible`。

S0B R0 必须按冻结 catalog 顺序完整 census 197 个非 sentinel parents，不早停、不人工挑选。至少
64 个结构合格 parents、覆盖至少 8 个环境才过 capacity/coverage 门；相同环境轨迹仍
是 cluster。成功也只授权另冻 environment-cluster-aware D5-M0 设计，不自动授权
payload、ecology、effect、student、主线/App/Android、生产或 safety。
