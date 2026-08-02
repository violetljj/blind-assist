# HFTF Stage C D5-S0B-P0B.1：sharded semantic evidence repair 设计

## 结论

P0B 的 INVALID 只证明原单体 evidence JSON 超过冻结的 8 MiB cap；没有 durable
candidate count、URL、provider 或 source outcome。允许继续，但只能另立 P0B.1：
新的版本化表示协议、新 root、新 attempt/preflight、重新独立审计。它是 consumed
source population 的证据恢复，不是 fresh validation，也不是旧 root 的 resume、
rerun 或只把旧 cap 调大。

## 唯一变化：内容无关的去重与分片

科学问题、18 个 exact blobs、runtime、AST/role/classification、完整覆盖和终端含义
全部不变。表示层机械改为：

- 每个 exact source path 一个 durable shard，文件名只用 manifest index；
- canonical AST object 每个 node 只存一次，不再把 JSON 作为转义字符串重复；
- expression record 只引用 node receipt，不重复 canonical object；
- generic expression 不保存重复的 source-segment 文本，但保留 SHA-256、UTF-8
  byte length、encoding 与 span；
- call 的 callee/full-call、assignment 的 target/value/full-assignment 以及全部 string
  literal value 仍保留文本与哈希；
- index 绑定每个 shard 的 bytes/SHA-256/counts 和全局 totals。

不得删 candidate、采样、截断、early success，所有 node/expression/string/call/
assignment/function/import exact coverage invariants 均不变。

P0B.1 锁定的是完整 AST-semantic structure，不宣称与旧 P0B generic exact lexeme
完全等价。string/call/assignment lexeme 仍 durable；P0C 不得把省略的 generic
lexeme 当作已观察文本。若 P0C 确实需要它，必须另冻 source-reread 权限，并按
OID/SHA、encoding、span 与已存 segment hash/length 复核。

## 预先冻结的容量

每个 shard 的 cap 只由 P0A 在 P0B 前已经冻结的 blob bytes 决定：
`max(1 MiB, 512 × blob_bytes)`。18 个 cap 的预计算总和是 129,690,624 bytes。
这不是按本次未发布结果拟合的 cap。任何单 shard 或 aggregate overflow 仍为
INVALID；attempt/preflight/not-evaluable/index/result/failure 各有独立 1 MiB cap。

execution contract 必须逐项冻结 18 个 cap 及其 canonical hash
`a7e3203057f17467dfe50e5671ab51fa578b832d439305764895a7c845f0a9f8`。

## 终端闭集与 durability

attempt/preflight durable 后，先在内存完成 18/18 object receipts、全部 parse/
extract、全部 shard serialization 与 cap 校验，此前不写任何 shard：

- syntax/encoding NOT_EVALUABLE：0 shard、0 index，只写 exact
  `not-evaluable.json` 后写 result；
- LOCKED：按 000..017 exclusive-create + fsync 18 shards，18/18 验证后才写
  index，index 后才写 result；
- 任一 shard、not-evaluable、index 或 result 写入/fsync 的异常/中断：只允许
  INVALID failure，绑定除 failure 自身外所有已存在顶层文件的 exact
  name/bytes/SHA，即使该文件是 partial 或 invalid JSON；不得恢复、补齐或重读
  source。failure closed set 允许 exact shard prefix，以及 optional index、
  not-evaluable、result。

terminal validator 必须先尝试完整正常终端；只有没有完整正常终端可验证时才接受
INVALID。后继测试必须注入 NE payload、NE result、LOCKED index 与 LOCKED result
四个写入/fsync 中断。

每 shard 必须绑定 manifest index/path、commit、OID、source bytes/SHA、encoding、
parse node/depth、runtime/algorithm，并保存 all-node canonical objects 与全部派生
records。validator 必须重算 same-shard refs、node IDs、canonical DFS、exact coverage。
index 必须按 P0A 顺序绑定 18 shard 的 filename/bytes/SHA/counts 与全局 totals。

## 权限边界

本设计只授权提交 invalid result 与本设计，并在推送后冻结 hash-bound P0B.1
execution contract。当前不授权 source blob reread、P0B.1、fetch/network、
dataset host/ZIP、P0C、P1/S0B、payload、主线/App、生产或 safety claim。
