# HFTF Stage C D5-S0B-P0B：provider semantic evidence INVALID

## 结论

P0B 的唯一一次正式执行已关闭为
`D5_S0B_P0B_PROVIDER_SEMANTIC_EVIDENCE_INVALID_STOP`。失败原因为完整 AST
证据 JSON 超过冻结的 8 MiB cap；canonical failure validator 接受
attempt/preflight/failure 哈希链。`evidence.json` 与 `result.json` 均不存在。

这不是 provider authority 的正结果或负结果，也不是 TartanGround source
不可用。18 个 P0A closure blobs 的语义提取路径已经被本协议消费，但没有形成
可引用的 durable semantic evidence；不能从进程内中间态、failure reason 或
副作用恢复 URL/provider 结论。

## 不可重试边界

旧 root、旧合同与简单“把 8 MiB 调大再跑”全部关闭。后继若继续，只能是新的
版本化表示协议、新 canonical root、新 attempt/preflight 与新的独立执行前双审。
修复必须是内容无关的机械表示变化，不能根据本次未发布的候选数量、URL 内容、
source path 结果或 provider outcome 调规则。

## 仍然关闭的权限

本终态没有发生新 fetch/checkout/network、dataset-host 请求、ZIP/payload 读取、
源码执行/import/eval/exec 或 P0C。它不授权 provider resolution、P1/S0B census、
主线/App/生产或 safety claim。
