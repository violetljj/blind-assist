# HFTF D5-S0B-P0B provider semantic evidence design

P0B 不直接声称 provider resolution。它只读取 P0A 已锁定的 18 个 exact Git
blobs：先逐项复核 commit:path OID、object type/size、bytes、SHA-256，再用冻结的
encoding detector 与 Python AST 机械记录全部字符串 literal、import alias、function、
call site、assignment 和 bounded expression provenance。不得执行源码、`compile`、
import、dynamic import、`eval` 或 `exec`，也不得新增 Git fetch 或 dataset-host
请求。

读取集合不根据文件名或先验 provider 猜测筛选；18 个 closure blobs 全部且仅各读
一次。attempt 与 preflight 必须在第一个 source blob 前 durable。所有 records 有
冻结的数量、单项字节和总 evidence JSON 上限，并按 source/位置/type/canonical JSON
确定性排序。18 个 immutable object receipts 必须先全部复核并冻结 set hash，才开始
AST extraction；它们包含 expected/actual commit:path OID、object type、size、
content bytes 与 SHA。后续 parse receipt 独立记录 encoding/AST status，不回写
object receipt。cap 或 receipt/hash/OID/FETCH_HEAD 失败是 `INVALID`；exact source
无法 AST parse 或 P0A dynamic evidence 非零才是 `NOT_EVALUABLE`。

P0B 可以保留 URL literals 和语法 provenance，但不解释 provider 控制流，不建立
URL template 或 198-parent mapping。成功只允许另冻 hash-bound P0C resolver
contract；不自动授权 P0C、dataset host、P1 sentinel、S0B census、payload 或任何
主线/App/生产/safety 变更。

URL-like literal 不是 provider，单一候选不建立唯一 provider，多候选也不自动构成
歧义；docstring、logging/error/help/example、dead lexical branch、assignment 和
call spelling 都不能被升级成 runtime authority。JoinedStr、BinOp、`%`、`.format`
和 `urljoin` 只记录结构与 unresolved operands，绝不在 P0B 求值。P0A 的 8 个
unresolved imports、18/19 Python closure 以及禁止读取外部 txt/config 的限制必须
原样带入。

每条 string literal 必须直接带 stable node ID、完整 start/end span、qualified
lexical scope、role 与 docstring flag，以冻结 precedence 区分 docstring、call arg、
assignment、default/annotation、ordinary expression 和其他上下文。URL/suffix
predicate、node ID、AST dump、source segment、scope、role precedence、canonical
JSON 参数、各 cap 的 per-blob/global 作用域及 fixture hashes，必须在独立 execution
contract 中精确定义并 hash-bind，禁止结果后调整。

本设计提交本身不授权 P0B blob read 或 execution。它只授权提交/推送 P0A result 与
本设计，以及继续冻结 P0B implementation/tests/execution contract；P0B/P0C、host、
P1/S0B、payload、主线/App/生产/safety 权限均为 false。
