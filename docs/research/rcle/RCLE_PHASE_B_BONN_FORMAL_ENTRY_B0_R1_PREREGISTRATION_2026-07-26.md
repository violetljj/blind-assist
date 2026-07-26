# RCLE Phase B Bonn Formal Entry B0 R1 预注册

状态：`DESIGN_FROZEN / REVIEW_PENDING / IMPLEMENTATION_AUTHORIZED_IF_REVIEW_PASS / CANONICAL_EXECUTION_REQUIRES_SEPARATE_IMPLEMENTATION_REVIEW / NOT_STARTED`

日期：2026-07-26

## 目标与上游

B0 R1 是 R0 preclaim HEAD 合同失效后的唯一版本化 recovery。它不改变 cohort、
分母、指标或门，只做固定六条 Bonn sequence 的 archive acquisition、member/CRC
inventory 与 timestamp-only 10 秒 window denominator。

- 上游 R3 authority receipt：
  `05a283b84f62bee000447bb567eadd63b424afaa9d81f5f0d83d36a9ed02489b`
- cohort identity：
  `513b770d18489fd0caf84874e9fb89456eb3a992fc262b037220b66b5caae86e`
- R0 execution-contract result：
  `PRECLAIM_NETWORK_OBSERVATION_CONTRACT_VIOLATION / R0_CLOSED`

R0 六次无 body HEAD 只作 non-authoritative transport discovery，禁止进入 R1
选择、expected-size gate、window、PASS/HOLD 或 archive identity 证据。

## 固定 cohort 与 URL

严格按 rank 使用：

1. `rgbd_bonn_crowd2`
2. `rgbd_bonn_balloon_tracking`
3. `rgbd_bonn_balloon_tracking2`
4. `rgbd_bonn_moving_obstructing_box2`
5. `rgbd_bonn_balloon2`
6. `rgbd_bonn_moving_nonobstructing_box2`

唯一 URL 为
`https://www.ipb.uni-bonn.de/html/projects/rgbd_dynamic2019/{sequence_id}.zip`。
禁止 redirect 后改 identity、mirror、替换、增删或重排 sequence。

## Canonical 路径与 one-run

- archives：
  `artifacts.local/datasets/rcle_phase_b_bonn_b0_r1/archives/{sequence_id}.zip`
- evidence：
  `artifacts.local/evidence/rcle_phase_b_bonn_b0_r1/formal_entry_b0_r1/`
- output 与 archive directory 可在 claim 前按 design lock 预创建，但 claim
  `fsync` 前禁止对任何 destination 做任何网络操作，包括 DNS、search/API、
  mirror/CDN、任何 URL/host、任何 method 或任何 transport library 的
  HEAD、GET、redirect/range/metadata probe；claim 后只允许访问六个 frozen
  official URL；
- runner 必须在首次允许的网络调用前，以 `O_EXCL` 创建并 `fsync`
  `run_claim.json`；claim 必含 protocol/cohort/R3/prereg/design/implementation
  hashes、canonical absolute output/archive/claim identity、command、timestamp、
  具名 object `pre_r1_head_disclosure`、`maximum_run_claims=1` 与
  `network_operations_before_claim=0`；
- claim 一旦创建不得删除、替换或重写；canonical output/archive/claim 路径
  不接受 CLI/env override；
- R1 最多一个 claim；success、HOLD、异常或中断均禁止第二次 R1；
- 每 URL 最多三次同 run bounded GET attempt；每次以 `wb` 将同一 `.part`
  truncate 为零，并重置 SHA-256 与 byte counter；不发 Range、只接受 HTTP
  `200`、final URL 必须逐字等于 requested URL、`Content-Encoding` 必须缺失或
  `identity`、`Content-Length` 必须是正整数且与完整 body 相等；Content-Type
  去除 `;` 后参数并 ASCII lowercase 后必须精确为 `application/zip`，缺失、
  `application/octet-stream` 或其他 MIME 均失败；
- transport exception、非 200、redirect、header/type/magic/length mismatch
  均可消耗一次 bounded attempt 后重试；每次 ledger 保留失败类型与已写 bytes；
- canonical destination 在 run 前和发布瞬间都必须不存在；完整 body fsync
  后只允许 no-overwrite atomic rename，禁止 `replace`/覆盖；一旦 rename，ZIP/member/CRC/timestamp
  failure 属 archive authority failure，不再回到 transport retry；
- 每个 attempt 完成或失败后、下一 attempt/sequence 前，必须原子持久化并
  `fsync` ledger；中断时保留此前全部 attempt。不得换 URL 或 sequence。

## Payload firewall

允许：

1. GET status、headers、archive bytes 与流式 SHA-256；
2. ZIP central directory、member name/size/compressed-size/CRC；
3. materialization invocation 中每个 file member 只完整流式解压一次；三个允许文本在该同一流中同时做
   CRC/byte counter、raw-member SHA-256 与 timestamp first-token parse，其他
   member 只做 CRC/counter；不得第二次打开 member；
4. 唯一 `rgb.txt`、`depth.txt`、`groundtruth.txt` 的非注释行第一 token；
5. 第一 token 的 finite、strictly-increasing timestamp 与 hash；
6. timestamp-only window denominator。

独立 validator 是单独的 read-only recomputation invocation：每次调用也必须让
每个 file member 恰好流式解压一次，并复算同一 CRC/counter/text/window receipt；
它不得复用 materializer 内存结果、联网、decode 或读取额外 payload 类型。

禁止：

- RGB/depth image decode、persist、cache、sample 或 visual/model inspection；
- `groundtruth.txt` 后续 pose token 数值解析、保留或统计；
- static map、legacy trace/support/residual/score；
- RCLE、raw/compensated expansion、scale proxy 或任何 Phase B metric。

## Archive/member contract

每条 sequence 必须：

- requested URL 与 final URL 均为冻结 official URL；
- response 是非 HTML 的 ZIP，完整落盘并固定 local SHA-256；
- central directory 无 absolute/backslash/drive/`..` traversal，规范化 member
  name 按 Unicode codepoint 与 `casefold()` 均不重复；
- relative root 规则唯一：若全部 file member 共享同一个第一 path component
  且每条均至少两段，则仅移除这一公共 top-level component；否则不移除；
- 应用上述 root 规则后，唯一精确匹配 `rgb.txt`、`depth.txt`、
  `groundtruth.txt`，并至少有一个 `rgb/<file>` 与一个 `depth/<file>`；
- 每个 file member 流式读到 EOF，uncompressed byte counter 等于 declared
  size 且 CRC 全通过；
- 文本以 raw bytes 分行；blank 或首个非 ASCII-whitespace byte 为 `#` 的行
  跳过；第一 token 必须为 ASCII 且匹配
  `[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?`；
- token 用 Python `Decimal` 精确解析，必须 finite、strictly increasing；
  canonical token 规则：`-0`/`+0` 为 `0`；科学计数法转非指数 fixed decimal；
  小数部分删除尾随 `0`，随后删除孤立末尾 `.`，整数不带小数点；
- canonical token ledger bytes 为每个 normalized token 的 ASCII bytes 加单个
  `LF`，包括最后一 token 后的末尾 `LF`；不得用 JSON、平台换行或 BOM；
- 每个文本同时记录 raw complete-member SHA-256 与 canonical normalized
  token-ledger SHA-256；pose 后续 bytes 不解析、不保留、不统计。

任一失败使该 sequence 为 `NOT_EVALUABLE_ARCHIVE_AUTHORITY`，但仍保留在固定
六序列分母，不得替换。其他 sequence 继续处理。

## Window denominator

每条可评价 sequence：

1. 用同一 exact `Decimal` 计算
   `t0=max(first_rgb, first_depth, first_pose)`；
2. `t1=min(last_rgb, last_depth, last_pose)`；
3. `N=max(0,floor((t1-t0)/Decimal("10")))`；`t1-t0==10` 必须产生 1 个窗口；
4. 对整数 `k=0..N-1` 生成连续、不重叠半开窗口
   `[t0+10k,t0+10(k+1))`；
5. 尾部不足 10 秒丢弃；
6. 不按画面、pose、support、metric、header 或名称挑窗口；
7. 零窗口仍保留在六序列分母。

R1 receipt 物化的全部 window 是未来 B1 可冻结协议的唯一完整 denominator。

## Gate 与终态

PASS 必须同时满足：

- 6/6 official URL identity 与 local archive SHA-256 固定；
- 6/6 archive/member/CRC inventory 可独立复算；
- 6/6 timestamp-only firewall 有效；
- 至少 2/6 sequence 各有至少一个完整 10 秒 window；
- 所有失败/零窗口单位完整保留。

PASS：

```text
PHASE_B_B0_R1_INVENTORY_PASS_B1_METRIC_PROTOCOL_MAY_BE_FROZEN
```

否则：

```text
HOLD_PHASE_B_B0_R1_NOT_EVALUABLE_NO_REPLACEMENT_NO_RERUN
```

设计审查 PASS 只授权完成 hash-bound implementation lock、independent validator
与 fixture-only offline contract tests；只有实现审查逐项 PASS 才授权唯一
canonical execution。B0 PASS 也只允许另立结果前冻结的 B1 metric protocol；不自动授权 RGB/depth
metric decode、pose 数值使用、Kill Gate B、Replay、Android、人体、安全或生产。
