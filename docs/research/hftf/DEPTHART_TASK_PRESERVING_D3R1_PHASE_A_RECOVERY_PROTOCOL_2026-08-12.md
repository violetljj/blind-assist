# DepthART task-preserving D3R1 Phase-A recovery protocol

状态：`PRE_OUTCOME / METADATA_ONLY / FROZEN`

D3 Phase-A 已完整处理原 48 身份，但固定 continuity 门只有 21 个通过，少于 32；原版本已
终止且没有 partial selection。D3R1 是独立 evidence version：不扩大旧池、不复用旧 21 个通过
身份，也不降低 `300 frames / adjacent gap <= 0.5 s / pose bracket <= 0.25 s`。

新池规模冻结为 127。这个数字只用于资源规划：把旧版 21/48 代入单侧 95% Clopper-Pearson
下界 `p=0.3150044506435995`，精确二项式计算得到 127 次中至少 32 次成功的概率
`0.9502686917714296`；126 次仅为 `0.944208372283701`。旧池不是随机 IID 样本，因此这不是
科学预测、质量证据或通过保证，也不改变任何门。

planner 只读取 pinned Apple Training split metadata，并扫描 published `origin/master` commit
`8d17a053dc6d345a688035cd298c49c70d36288f` 的不可变 `docs/research` tree；它不会重扫会把
roster 自身纳入排除项的 live docs。另 SHA 锚定 TARO R10 脚本中的 32 个并发 fresh identity，
对全部排除项做 union，要求 unique visit/session，按 `sha256(visit_id:video_id)` 固定排序取前 127。

本协议本身不授权 source-use scope、HEAD、GET、archive member、truth、模型、训练、Development
或 R2。metadata roster 锁定后必须停在
`EXPLICIT_D3R1_SOURCE_SCOPE_REGISTRATION_FOR_EXACT_127_METADATA_ROSTER`。
