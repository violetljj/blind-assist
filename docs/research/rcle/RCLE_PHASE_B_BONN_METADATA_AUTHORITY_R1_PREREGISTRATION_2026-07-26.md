# RCLE Phase B Bonn Metadata Authority R1 预注册

状态：`DESIGN_FROZEN / REVIEW_PENDING / EXECUTION_AUTHORIZED_IF_AND_ONLY_IF_REVIEW_PASS`

日期：2026-07-26

## 目标

R1 只修复 R0 的执行控制，不重新选择 cohort，也不读取 payload。唯一候选：

```text
BONN_CANONICAL_METADATA_AUTHORITY_ADOPTION_R1
```

R0 receipt `4386bbe3…f1764` 的 26-row denominator、历史排除和 6 条 cohort
内容已独立复算一致，但因为 runner override 无法证明 one-run/canonical-output，
永久保留为 diagnostic。R1 将该结果视为已见、不可更换的输入 identity，验证后
在新的 canonical-only execution domain 中一次性 adoption。

## 固定输入

- R0 receipt SHA-256：
  `4386bbe3b617abca3b73fc3070a65cef403fe270c12fd25f5034a579882f1764`
- R0 implementation lock：
  `a47cd39ea82c10828290def8bae54f61b28676190c8ab06acc93217b1590a617`
- official metadata page：
  `2bd8df16acad79c70e1021f1da039c78510034fd9091fd706f8a3f480ea5c186`
- historical manifest：
  `f02bd9f1313def45cc107d72ace5f7c7803f4ab816bf6e98c5f9173fa3bb1cc6`
- cohort：
  `513b770d18489fd0caf84874e9fb89456eb3a992fc262b037220b66b5caae86e`

R1 不允许改变顺序、替换 sequence、重新加盐、修改 size 门或重解释 metadata。

## Canonical-only 执行合同

1. runner 不接受 input、lock、output 或 receipt path override；
2. 所有路径由 repo root 与 implementation lock 固定；
3. implementation lock 必须逐项校验 canonical absolute path；
4. 正式启动首先使用 exclusive-create 写入 canonical `run_claim.json`；
5. `run_claim.json` 一旦存在，成功、失败、异常或中断均禁止第二次 materialize；
6. `--validate-existing` 只能读取 canonical receipt，不创建新 claim；
7. receipt 必须绑定 run claim、R0 receipt、R1 lock、全部 controls 和环境；
8. 任何 hash、path、26-row、9-exclusion、6-cohort、firewall 或 schema 不一致
   立即关闭 R1。

## 权限

允许：

- 读取 official metadata HTML；
- 读取 R0 metadata receipt；
- 复算 26-row denominator、9 条排除、固定 6 条及 cohort hash；
- 产生 canonical R1 authority receipt。

禁止：

- 读取/下载 ZIP、RGB、depth、pose 数值、static map；
- 读取旧 trace/support/residual/score；
- 重新选择或替换 cohort；
- 运行 payload inventory 或任何 Phase B metric；
- Replay、Android、人体、安全或生产。

## 终态

- PASS：
  `CANONICAL_METADATA_AUTHORITY_PASS_FORMAL_PHASE_B_PROTOCOL_MAY_BE_FROZEN`
- 任一失败：
  `CLOSE_BONN_CANONICAL_METADATA_AUTHORITY_ADOPTION_R1_NO_RERUN`

R1 PASS 只允许冻结并审查 formal Phase B 协议；payload inventory 继续关闭，
直到该独立协议完成设计审查并获得用户明确授权。R1 不等于 Phase B 指标 PASS。
