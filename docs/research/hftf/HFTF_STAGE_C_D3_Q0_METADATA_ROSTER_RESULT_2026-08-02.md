# HFTF Stage C D3-Q0 metadata-only roster 结果

## 结论

唯一一次 metadata-only 扫描成功锁定 40 个 D3-Q0 truth-screening slots：

`D3_Q0_METADATA_ROSTER_40_SLOTS_LOCKED`

这是 roster 可执行性成功，不是 HFTF 效果成功。它只证明 official-train 中存在 40 个
满足冻结 source-shape 合同、且未进入历史 84-parent exclusion union 的候选源。没有
打开 RGB、mask、depth 或 pose 内容，没有计算 support、future truth、clearance 或
effect。

本节点只授权冻结下一份独立的 qualifier/sealed-truth/effect-skeleton execution
contract；逐 slot 媒体获取、truth screening 和效果评估仍未获准。

## 唯一执行与 durable evidence

- execution contract 由 commit
  `f4b5b2581f6b56d6847148bc1ce5e829a3a0ef1f` 推送；执行前正式校验
  `HEAD == origin/master`。
- `attempt.json` 在首个网络请求前以 exclusive write、flush 与 `fsync` 落盘，
  SHA-256 为
  `a2f1764b7af0f5a9f50d28e7e489be38f98a88947e606bda0712ef7dce409800`。
- 只启动一次 CLI，原进程链被监控、从未重启；约 928 秒后自然退出。
- `roster.json` 共 1442033 bytes，SHA-256 为
  `8720a68855e0ddcbee9ae174de69383dd6d596329d76f83d0798197e333ba7db`。
- stdout 最终终端精确匹配，stderr 为 0 bytes，`failure.json` 不存在。

## 冻结 roster

official-train split 共 1560 个 session。按 session ID 字典序扫描到第 236 个账本条目
时凑齐前 40 个 metadata-eligible slots：

| 项目 | 数量 |
|---|---:|
| 完整 frozen exclusion union | 84 |
| 扫描区间内遇到的 frozen exclusions | 77 |
| 3 次重试后 camera-pose metadata 为 404 | 115 |
| source fps 不是 5 或 20 | 4 |
| 锁定 slots | 40 |
| 5 Hz slots | 16 |
| 20 Hz slots | 24 |

40 个 session ID 严格升序、互不重复，并与完整 84-parent exclusion union 完全不相交。
首个是
`1839645812f8fd05942a2b1f3d612674ef23155a3b090ea19f5042ab771547c6`，
第 40 个是
`26aec7ce13338007233419ea2d9389ffdc7bbc2e101f40cf00c6e5bcc83fd8ae`。
完整 slot、ID 和 fps 清单保存在同名 machine-readable result JSON；ordered-ID
SHA-256 为
`9e65ab09f19728fed5cabb3f3fc56f88cc75766a7d8c2d9d0f0ed34cb1398b75`。

5 Hz slots 固定选择 source frames `0..12`，20 Hz slots 固定选择
`0,4,...,48`。非选择帧不构成资格要求，也没有因为 D2 的旧 50-frame helper 被错误
要求存在。

## 离线核验

主审离线重算并全部通过：

- terminal、40/40 counts、236-entry ledger、ID 升序/唯一与完整 84-exclusion
  不相交；
- 40×3 共 120 个 RGB/mask/depth modality receipt-list hashes；
- 120×13 共 1560 个 selected-frame receipts、对象 session binding 与 5/20 Hz
  timeline；
- official split generation/text hash、contract hash 与 attempt 状态；
- 所有 slot-local 和 global content/support/truth/effect firewalls；
- roster 无 failure artifact，且唯一允许的后继权限只是冻结下一份执行合同。

独立只读审计另行重算 78+6 的互斥 84-parent exclusion union、40-slot 选择序列、
120 组 modality hashes、1560 个对象 receipts、完整 binding chain 与全部 firewall，
结论为 `CLEAR`，0 mismatch、0 blocker。审计未打开媒体内容，也未运行下游流程。

## 证据与权限边界

本结果没有打开媒体或 pose 内容，因此既不是 reference/support qualification
evidence，也不是 future truth、D3 effect、RGB student、模型、生产或 safety
evidence。40 个 slots 现在是不可替换、不可追加、不可重排的固定 screening budget；
后续即使某 slot 失败，也必须消耗该位，不能选择第 41 个。

当前仍禁止：

- 重跑 metadata roster，或依据后见结果替换、追加、跳过、重排 slots；
- 打开任何 slot 的 RGB、mask、depth、pose 内容或 sealed truth；
- 计算 qualifier、effect、student 或主线对比；
- 改变研究主线、默认 App、Android、生产或 safety 权限。
