# RCLE Phase B Bonn Formal Entry B0 R0 执行合同结果

状态：`PRECLAIM_NETWORK_OBSERVATION_CONTRACT_VIOLATION / FORMAL_RUN_NOT_STARTED / R0_CLOSED`

日期：2026-07-26

现场 provenance：`HEAD=e012655a292bb8f55430a4c23501b91e1276208c`；
HEAD 审计发生前工作树干净，B0 R0 canonical dataset/evidence 目录不存在。

## 结论

B0 R0 没有创建 canonical `run_claim.json`、没有 GET archive body、没有创建
canonical archive/evidence 目录，也没有运行 inventory、timestamp window 或任何
Phase B metric。但在正式实现与 claim 尚不存在时，只读数据可行性审计对固定六个
official archive URL 各发出一次 `HEAD` 请求。该行为读取了 status/headers，违反
R0 “network 前必须 exclusive-create claim”的执行顺序，因此 R0 不再具备正式
执行资格，不能把后续 GET 记为同一 R0 的合规运行。

## 已观察但降级的 transport discovery

六次请求均无 response body；审计报告记录为 `200`、`application/zip`、
`Accept-Ranges: bytes`，精确 `Content-Length` 合计 `2,262,988,443` bytes。
这些 header observation：

- 不进入 cohort 选择、window 分母或 B0 PASS/HOLD 判定；
- 不替代正式 archive bytes、local SHA-256、ZIP/CRC/member/timestamp 证据；
- 不授权 mirror、sequence replacement、payload decode 或 Phase B metric；
- 仅作为 R0 合同失效的完整披露。

## 后继边界

若继续 Phase B，只能另立版本化 B0 R1 recovery：保持 R0 固定六序列、official
GET URL、六序列全分母、10 秒 window、PASS/HOLD 门和禁止事项不变；在对六个
frozen URL/host 做任何新网络操作前创建 R1 独立 canonical claim，并把上述 HEAD 明确绑定为
non-authoritative discovery。R1 不是 R0 成功或重跑，也不得回写 R0。
