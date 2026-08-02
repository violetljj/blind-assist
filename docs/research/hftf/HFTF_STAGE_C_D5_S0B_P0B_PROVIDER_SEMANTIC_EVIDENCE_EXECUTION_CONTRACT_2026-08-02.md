# HFTF D5-S0B-P0B provider semantic evidence execution contract

P0B 复用 P0A 已存在的 local Git object store，不进行任何 fetch、checkout 或网络
请求。attempt/preflight durable 后，严格按 P0A closure 的 18-row 顺序，对每行依次
核对 exact commit:path OID、object type=`blob`、object size，再读取一次 raw blob
并核对 bytes/SHA-256。18/18 immutable object receipts 与 canonical set hash 全部
形成后，才允许 encoding detection 与 AST extraction。

运行时冻结为 CPython 3.11.9，并绑定 `ast.py` 与 `tokenize.py` 哈希。每个 AST
occurrence（包括 CPython 复用的 operator/context singleton）都按父边独立编号。
先按 `ast.iter_fields` 字段顺序及 list 顺序深度优先冻结 preorder，再从叶到根生成
shallow canonical AST dump；node ID 是 UTF-8 编码的 NUL-joined
`(path, preorder_index, node_type, start/end span, canonical_shallow_dump)` 的
SHA-256，因此直接 child occurrence ID 形成 Merkle 式证据链。JSON 标量原样，
bytes 用 hex，complex 用 real/imag repr，Ellipsis 用 type tag。canonical JSON 固定
`ensure_ascii=false, sort_keys=true, separators=(',', ':')`；artifact 以 LF 结尾，
嵌入 record 的 shallow dump 去掉末尾 LF。source segment 只用
`ast.get_source_segment`，不求值。

URL class 只按大小写不敏感的 `http://` / `https://` 前缀；ZIP class 只按大小写
不敏感的 `.zip` 结尾。literal role 按 docstring、call arg、assignment、default/
annotation、ordinary Expr、other 的优先级机械判定。所有 cap 均为 global，只有
single-string 与 single-segment cap 是 per record；AST depth 是 per blob，记录
全局最大值。任何截断、cap、receipt、runtime、hash、OID、partial 或实现失败都为
INVALID；verified source 与冻结 encoding/grammar 不兼容才是 NOT_EVALUABLE。
LOCKED validator 重算全部 node ID、父子边、preorder/depth、每 path node/depth
收据；它从每个 Module 根严格按 runtime `_fields` 顺序与 list index 顺序重走
完整 DFS，并逐项匹配冻结 preorder。随后从 canonical dump 重建 literal role、
call/assignment、function argument
及 import alias；strings/calls/assignments/expressions 必须对 all-node AST 一次且
完整覆盖，禁止删除或复制。每个成功 blob 后立即执行全部 global record caps，
因此后续 syntax failure 不能掩盖既有超限。syntax NOT_EVALUABLE 采用 exact
observation/parse-receipt schema，携带并核对 prefix AST/record cap usage，且与
LOCKED 共用完整 evidence artifact byte cap。

P0B 只锁定语法证据。URL-like literal、docstring、assignment、call spelling、
单候选或多候选都不是 provider authority。唯一 parser 路径是
`ast.parse`/`PyCF_ONLY_AST`；它内部会调用 compile 的 AST-only 模式，但不产生
code object/bytecode，也不执行源码/import/eval/exec。不解释控制流，不生成 URL
template 或 198-parent mapping。LOCKED 只允许另冻
hash-bound P0C contract，不自动授权 P0C、host、P1、S0B census、payload、主线、
App、生产或 safety claim。
