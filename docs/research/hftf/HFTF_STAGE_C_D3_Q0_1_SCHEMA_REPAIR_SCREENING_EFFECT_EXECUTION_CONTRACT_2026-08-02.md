# HFTF Stage C D3-Q0.1 schema-repair execution contract

状态：

`FROZEN_AFTER_Q0_INVALID_BEFORE_ANY_Q0_1_SLOT_2_MEDIA_SUPPORT_OR_TRUTH`

## 结论

Q0.1 是 Q0 `D3_QUALIFICATION_INVALID_STOP` 之后的独立执行后继，不是新的科学
协议、cohort 或门槛搜索。它只修复 runner 在 selector 顶层重复写入
`slot_attempt_sha256` 的闭合模式缺陷；合法的
`source_authority_and_content_hashes.slot_attempt_sha256` 保留。

原 roster slot 1 已打开 media/support/truth，永久 burned。Q0.1 在新 canonical
root 写入一份结果无关的 carry-forward receipt，计入原 40-slot 总预算，但不计
qualified、not-qualified 或 execution failure；不得读取旧 sealed payload、旧
selector 或旧日志，也不得导入其中任何 outcome 字段。首个可新开输入仍是原
slot 2，保持原 index 和字典序，最多再开 39 slots。

## 不变的科学合同

- required qualified sources 仍为 6，首 6 个合法 qualified source 立即停止；
- 每个 parent 的 body/head × `.4/.8 s` 四 strata，denominator 固定 252；
- common-known coverage/risk/safe 门仍为 `.10/5/20`，UNKNOWN→SAFE 必须为 0；
- slot failure 消耗原槽位，不 replacement、不 reorder、不 manual skip、不扩预算；
- future-blind 输入、42 条 prediction durable 前置、sealed payload open-once、
  D2 estimand 和全部 effect gates 与 Q0 完全一致；
- parent session 仍是独立统计单位。

## 执行顺序

1. 合同、实现和 40 个专属 tests 提交推送，确认 `HEAD == origin/master`；
2. 运行 `verify_git=True` 与独立科学/工程审计；
3. 首次 runner 调用只 durable 写 screening attempt 和 slot-1 carry-forward
   receipt，媒体、pose、support、truth 均保持零访问；
4. 再次 runner 调用才可执行原 slot 2，此后严格按原 2→40 顺序；
5. six-source success 才授权 future-blind；42 predictions durable 后才授权
   sealed effect。budget exhausted 或 invalid 立即停止。

冻结前独立科学与工程终审均为 `CLEAR`，0 blocker。专属 state/pipeline tests 为
23/23 与 17/17，HFTF 全集为 392/392；8319 个 canonical final/tmp/orphan paths
中最长 142 字符，小于 240 上限。冻结 JSON SHA-256 为
`268f1491835fb8b4d365a24064eac94edc5046633fa7861b7fbd1588ded7225a`，审计时新
canonical root 不存在。

## 权限边界

即使 effect 最终为
`CAUSAL_SIGNED_CLEARANCE_TRANSPORT_SUPPORTED_FOR_RGB_STUDENT_PROTOCOL`，也只授权
冻结独立 RGB-student contract。当前不授权 student 训练/执行、reserved official
test、研究主线或默认 App/Android 变更、生产或 safety 声明。
