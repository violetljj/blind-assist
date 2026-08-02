# HFTF D5-S0B-P0A toolkit source-closure execution contract

P0A 只锁定 provider resolver 的 exact-commit Python import closure。它从
`tartanair/__init__.py` 出发，通过 AST 机械追踪相对 import 与 `tartanair.*` import；
不根据文件名搜索 `download`/`ground`，也不读取不可达源码。

正式执行只允许一次 exact toolkit fetch、一次 `tartanair/` tree-name listing，以及
closure 中 Python blobs 的读取。每个可达 blob 先读取 Git object-size metadata，
并在内容读取前检查剩余 blob/byte budget；最多 128 blobs / 4 MiB source。输出只保留
path、Git blob OID、SHA-256、bytes、import edges、unresolved imports 与
dynamic-import 计数；不提取 URL literal，不解释 provider 语义，不生成 archive
mapping。

直接的 `__import__` / `importlib.import_module` 调用及其简单 import、模块对象
赋值和 callable 赋值别名会被计数；无法可靠还原的 `getattr`、subscription、
container escape、`exec`、`eval` 等间接动态构造另记风险计数。任一计数非零时，
未来 P0B 必须终止为 `NOT_EVALUABLE`。零计数只表示冻结的语法检测器未发现证据，
不是运行时完整性证明，也不得把静态 import closure 解释为完整 provider authority。

P0A 不请求 dataset host、archive 或 ZIP。成功只允许另冻 P0B provider-resolution
contract；不自动授权 P0B、P1、S0B census 或 payload。
