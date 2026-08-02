# HFTF Stage C D5-S0B-P0B.1：sharded semantic evidence execution contract

## 状态与结论

本文件与同名 JSON 已冻结为
`FROZEN_EXECUTABLE_AFTER_P0B1_IMPLEMENTATION_TEST_DOUBLE_AUDIT_BEFORE_FIRST_SOURCE_REREAD`。
planner、durability helper、复用的纯 AST semantic helper、terminal validator、
focused tests、runtime、父项与本文均由 path/SHA-256 绑定；focused tests
`18/18`、绑定运行时的 HFTF full suite
`520/520`，四个规定的 write/fsync failure injections 全绿，独立科学与工程审计
均须为 `CLEAR`。

本合同只授权在提交推送后、tracked clean、`HEAD=origin/master`、canonical root
不存在且 core 二次复核完整 formal context 时，执行一次 exact P0B.1 source
recovery。它不授权打开旧 P0B root、fetch/network、dataset host、ZIP/payload、
P0C/P1/S0B effect execution、主线/App、生产或 safety 行为。

## 固定父项与 source authority

- P0B.1 design：
  `6b2523091a967b2a64e2062c9314d1cc4d6eaf37b99de204f4fd9ccf953f5d9d`；
- P0B INVALID：
  `357ea359b7346253c8916d79809dd636e098c047063321fba2d02518fba00164`；
- P0A locked result：
  `15f0bc4c96a1adea45aaa1ee1d1dddba4341f3390500147c165a4c343b523137`；
- exact toolkit commit：`158a6844d782942110967325ca3082f50ab2bfc7`；
- exact P0A closure：18 blobs、250,569 source bytes、ordered manifest SHA-256
  `e1ceefc4b25126a296753e8b9824f941801036f9d6a1f991cd44d9e4281768b2`。

未来执行只能按 P0A manifest 顺序，从 local Git object store 对每行进行
commit:path OID、type、size、blob bytes 的一次读取与复核。18/18 object receipts
及其 set hash 必须在首次 AST parse 前完成。旧 P0B root、旧进程内存、failure
side effect、外部配置、network 与 dataset host 都不是 source authority。

## Runtime、算法与证据等价边界

Runtime 沿用冻结的 CPython 3.11.9 launcher/base executable、`python311.dll`、
stable ABI、`ast.py` 与 `tokenize.py` 的 exact path/SHA lock。canonical JSON 为
UTF-8、`ensure_ascii=false`、sorted keys、compact separators、单个 LF。

node ID、canonical shallow AST、runtime `_fields` 与 list item 顺序的 Module-root
DFS、lexical scope、literal role、URL scheme 与 archive suffix 分类规则均保持
P0B 不变。唯一表示变化是：

- canonical AST object 每个 node receipt 只保存一次，不保存为 escaped JSON；
- expression 通过 same-shard `node_id` 引用 node receipt；
- generic expression 不保存 source text，但必须保留并复算 segment SHA-256、
  UTF-8 byte length、encoding 与 span；
- string、call callee/full call、assignment target/value/full assignment 的文本、
  SHA-256 与 UTF-8 length 保持 durable。

P0B.1 锁定完整 AST structure，不宣称 generic exact lexeme parity。P0C 不得把
省略文本当作 observed lexeme；若后继确需文本，必须另立 hash-bound source-reread
合同。

## Exact artifact schemas

同名 JSON 逐项冻结每类 artifact 的 exact key set：

- `attempt.json` 绑定 contract/design/INVALID/P0A hashes、commit 与禁止的外部权限；
- `preflight.json` 绑定 clean Git gate、父终端、manifest、runtime 与 algorithm；
- 每个 shard 顶层绑定 manifest index/path、commit、OID、source bytes/SHA、
  encoding、parse status、node/depth、runtime/algorithm、object receipt、七类 records
  与 counts；
- node receipt 保存 path、preorder、type、span、depth、parent field/id 与唯一
  canonical AST object；
- expression 保存 node/parent/type/span/role/scope 及 segment hash/length/encoding，
  不含 generic segment text；
- string/call/assignment/function/import 保持原 P0B 字段；call 保留
  callee/positional/keyword argument node IDs，assignment 保留 target/value
  node IDs，keyword item exact schema 为 `name + node_id`；并新增完整 span 与
  retained-text length/hash binding；
- `index.json` 绑定 attempt/preflight、commit、P0A manifest、18-receipt set、
  runtime/algorithm/cap manifest、18 rows、aggregate bytes、global counts 与 depth；
- `not-evaluable.json` 只允许冻结的 encoding/AST grammar/P0A dynamic reason，不含
  source text；
- `result.json` 通过互斥的 `index_sha256` 或 `not_evaluable_sha256` 绑定正常终端；
- `failure.json` 保存 reason 与所有已存在顶层 artifact（failure 自身除外）的
  exact name/bytes/SHA-256，即使 artifact 是 partial 或 invalid JSON。

任何 missing、duplicate 或 unknown key 均为 INVALID；JSON duplicate key 在 schema
验证前拒绝。所有 counts、indexes、bytes 与 spans 必须是整数。

## Exact coverage 与 validator

每个引用必须在同 shard 内唯一解析。validator 必须重算 node IDs、parent/depth、
canonical child edges、runtime `_fields`/list-order DFS，以及 node/expression/string/
call/assignment/function/import 的 exact one-to-one coverage。不得有 dangling ref、
extra record、抽样、截断、candidate filter 或 early success。

index validator 必须重新读取 durable shards（不得重读 source），逐项重算 filename、
bytes、SHA-256、counts、depth、顺序、单 shard cap、18-row canonical hash、aggregate
bytes 与 global totals。result 必须重算完整 hash chain，并保持 provider resolution、
dataset-host 与全部越权 claim 为 false。

terminal validator 的固定顺序是：

1. 先拒绝 non-regular、unknown、duplicate 或 oversize top-level entries；
2. 尝试完整 LOCKED terminal；
3. 尝试完整 NOT_EVALUABLE terminal；
4. 只有两个正常终端均不成立时才尝试 INVALID failure；
5. 否则拒绝 root。

## Capacity manifest

每 shard cap 为 `max(1,048,576, 512 × P0A blob bytes)`：

| Index | Source path | Blob bytes | Maximum shard bytes |
|---:|---|---:|---:|
| 000 | `tartanair/__init__.py` | 49 | 1,048,576 |
| 001 | `tartanair/customizer.py` | 32,995 | 16,893,440 |
| 002 | `tartanair/dataloader.py` | 12,963 | 6,637,056 |
| 003 | `tartanair/dataset.py` | 40,478 | 20,724,736 |
| 004 | `tartanair/downloader.py` | 32,876 | 16,832,512 |
| 005 | `tartanair/eval_utils/trajectory_evaluator_ate.py` | 4,551 | 2,330,112 |
| 006 | `tartanair/eval_utils/trajectory_evaluator_base.py` | 9,774 | 5,004,288 |
| 007 | `tartanair/eval_utils/trajectory_evaluator_rpe.py` | 4,972 | 2,545,664 |
| 008 | `tartanair/evaluator.py` | 4,644 | 2,377,728 |
| 009 | `tartanair/flow_calculation.py` | 18,471 | 9,457,152 |
| 010 | `tartanair/flow_utils.py` | 3,252 | 1,665,024 |
| 011 | `tartanair/iterator.py` | 16,432 | 8,413,184 |
| 012 | `tartanair/lister.py` | 1,314 | 1,048,576 |
| 013 | `tartanair/reader.py` | 8,297 | 4,248,064 |
| 014 | `tartanair/tartanair.py` | 32,905 | 16,847,360 |
| 015 | `tartanair/tartanair_module.py` | 15,206 | 7,785,472 |
| 016 | `tartanair/unzipper.py` | 3,004 | 1,538,048 |
| 017 | `tartanair/visualizer.py` | 8,386 | 4,293,632 |

18 caps 合计 129,690,624 bytes；canonical manifest SHA-256 为
`a7e3203057f17467dfe50e5671ab51fa578b832d439305764895a7c845f0a9f8`。
`attempt/preflight/not-evaluable/index/result/failure` 各有独立 1 MiB cap，不计入
shard aggregate。任何 record/global/shard/aggregate/control cap overflow 都是
INVALID。

## Closed sets 与 write order

attempt 与 preflight 必须 exclusive-create、flush、fsync，并在首次 source blob
读取前 durable。随后必须先在内存完成 18/18 receipts、全部 encoding/parse/extract、
18 shard serialization、schema/reference/coverage/count 与全部 cap 校验；此前不能
写 shard 或 not-evaluable。

LOCKED exact set 为 `attempt + preflight + shard_000..017 + index + result`。写入顺序
严格为 000..017 各自 exclusive-create/flush/fsync，随后从 durable shard 重验，
再写/fsync index，再重验 index/hash chain，最后写/fsync result。

NOT_EVALUABLE exact set 为
`attempt + preflight + not-evaluable + result`，且必须是 0 shard、0 index。
dynamic-evidence reason 严格为 0 source reads/receipts/parse-prefix；syntax/encoding
reason 必须 durable 保存 18/18 object receipts、set hash、source total bytes、成功
parse-prefix receipts、AST node/depth 与六类 record prefix counters，并逐项验证
global caps。两类 reason 共用 exact schema，未适用 identity/error 字段严格为 null。
其中 dynamic reason 的 `manifest_index/source_path/detected_source_encoding/error_type`
均为 null；encoding detection 失败绑定 exact manifest index/path 且 encoding 为
null；decode 或 AST grammar 失败还绑定已检测 encoding。

INVALID failure set 为
`attempt + preflight + exact shard prefix(0..18) + optional index +
optional not-evaluable + optional result + failure`。failure observed rows 按名称排序，
绑定除 failure 自身外每个 present regular file 的 raw bytes 长度与 SHA-256；读取
partial artifact 只允许按 raw bytes 哈希，不得解析、修复、覆盖、删除或补齐。

任一 shard、not-evaluable、index 或 result 的 write/fsync 中断均只能进入 INVALID；
不得 resume、retry 或重读 source。后继 tests 必须至少注入：

- not-evaluable payload write/fsync interruption；
- not-evaluable result write/fsync interruption；
- LOCKED index write/fsync interruption；
- LOCKED result write/fsync interruption。

## Firewalls

只允许 AST-only 静态编码；禁止 import、module initialization、code object、
bytecode、compile/eval/exec、template evaluation、control-flow/data-flow/reachability
解释或 provider resolution。禁止 fetch/checkout/network、dataset-host、ZIP/payload
读取和旧 P0B root/process-memory 恢复。

无论 URL candidate 是 0、1 或多个，P0B.1 正常锁定都不产生 provider conclusion；
它不建立 official URL template、198 parent mapping、HFTF opportunity/effect、fresh
validation、algorithm-selection increment、mainline/App、production 或 safety 权限。
两个正常 result 都必须把 `source_evidence_role` 锁为
`CONSUMED_SOURCE_RECOVERY_NOT_FRESH_VALIDATION`，并写入 8-key 全 false
`claim_ceiling`。只有 LOCKED result 的 `next_authority` 可为
`FREEZE_SEPARATE_HASH_BOUND_P0C_PROVIDER_RESOLUTION_CONTRACT`；NOT_EVALUABLE 必须为
`STOP_PROVIDER_RESOLUTION_SOURCE_EVIDENCE_NOT_EVALUABLE`。两者的
`p0c_execution_authorized_automatically` 均严格为 false，P0C 执行仍需独立合同与授权。

## Executable gate 与单次执行

JSON 使用自引用安全的 semantic hash：将
`document_pair.json_semantic_sha256` 临时替换为 64 个 `0` 后，对完整 canonical
contract JSON 计算 SHA-256；本文使用普通 raw-file SHA-256。正式入口与私有 core
都会验证 executable status、全部 parent/implementation/test/document/runtime
bindings、P0A locked terminal、exact closure/cap manifest、positive/negative
authorization、tracked clean、`HEAD=origin/master`、canonical root 与 execution
commit。任何一步失败都必须在创建 root 和读取 source 前停止。

单次调用开始后，attempt/preflight 先 durable；随后才允许 exact 18 local Git blob
reads。canonical root 一旦存在，任何 terminal 均禁止 rerun。若得到 LOCKED，只能
另冻独立 hash-bound P0C contract；若得到 NOT_EVALUABLE 或 INVALID，则保留对应负
终端并停止本路线的本次恢复，不得调参救援、resume 或 source reread。
