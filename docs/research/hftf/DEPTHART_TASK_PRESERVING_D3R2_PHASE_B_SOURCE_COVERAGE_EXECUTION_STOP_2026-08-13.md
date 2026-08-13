# DepthART-S D3R2 Phase-B source-coverage execution stop

状态：`INVALID_INCOMPLETE / NO_SCIENTIFIC_TERMINAL / NO_SUCCESSOR`

D3R2 的 exact-64 coverage-only census 已在受保护 `master` 上合并 activation 后正式启动。前 `44/64`
个资产形成连续、sealed checkpoint 前缀；第 `45` 个请求（selection `23`、pool `58`、visit
`466238`、video `44796744` 的 `lowres_depth.zip`）首试得到 HTTP `200`，但流式下载正文长度与
冻结 `Content-Length` 不一致。producer 按冻结 transport policy 将其记录为 terminal
`DownloadFailure <- ValueError: download length mismatch` 并停止。

这不是 missing-source 或科学 FAIL。本次没有完成全部 64 个资产，没有生成 manifest，也没有运行独立
coverage validator；因此不能发布 partial coverage aggregate，更不能登记 missing-source policy、读取
像素/truth 或进行 Phase-B selection。当前：

- 计划 `32` identities、`64` assets、`9,600` stems；完成 `44` 个 asset checkpoints / `22` 个成对身份；
- 保留 `44` 个 source bodies，共 `4,223,537,610 bytes`；第 23 个 identity 目录为空；
- sealed failure receipt、sidecar 与 `_temporary_downloads` marker 均保留；
- `scientific_terminal=null / selection_evaluated=false / selected_phase_b=null`；
- `next_gate=null / successor=null`。

停止后的独立 metadata-only auditor 验证了 attempt、001..044 连续 checkpoint seals、failure receipt、
冻结 HEAD URL/header/length 绑定，以及 source 路径/文件名/长度。它没有读取或哈希 source body，没有打开
ZIP container/directory/member，也没有复现或发布任何 partial missing-frame 观察。

当前 r0 root 永久不可修改或恢复：`--resume` 会被 temporary marker、failure receipt 和 orphan identity
inventory 依次拒绝。不得删除这些证据后重试，不得复用 44 个 partial bodies，不得同版本重跑，也不得运行
要求完整 64+manifest 的 validator。任何未来恢复都必须再次由用户授权，另立版本、协议和 fresh root；它
不是本版本的 successor。

RGB、模型、角色、训练、Development outcome、R2、性能、默认 App、production 和 safety 均未打开。

机器回执：[JSON](DEPTHART_TASK_PRESERVING_D3R2_PHASE_B_SOURCE_COVERAGE_EXECUTION_STOP_2026-08-13.json)。
