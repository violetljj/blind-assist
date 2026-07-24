# USTRF truth-blind causal per-track attribution-token producer 可行性与 extra-token 审计 R0（2026-07-24）

## 结论

本轮按独立边界冻结并完成 `TRUTH-BLIND-CAUSAL-PER-TRACK-TOKEN-PRODUCER-AUDIT-R0`，终态为 **`HOLD_FOR_POLICY_GATE / VALID`**。

producer 可行性门通过，但 extra-token 风险门没有可信上界：

- C1/C2/C3 的 runtime 输入投影在 `123` 条 trace 上逐帧一致，已折叠为候选无关的 `41` 条完整序列 ledger、`62,229` 帧；
- ledger 在独立 producer 进程中先冻结并 hash inventory；此时 truth、event window 与 oracle token 解码数均为 `0`。第二个进程复验全部 ledger 后才做 post-hoc oracle/负暴露联结；
- 当前 `33` 个 oracle-supported candidate-event 单元达到 `33/33` causal coverage；它们对应 `11` 个唯一 oracle event、`13` 枚唯一 producer token，不把 C1–C3 三次联结冒充三个运行时 token；
- CrowdBot 1203 同一无 active relation event 的 `3` 个 candidate-event 单元继续 fail closed；unknown-route token、跨 reset token、重复 token ID 均为 `0`；
- 完整序列共生成 `5,126` 枚 producer token，其中只有 `13` 枚与 oracle-supported 单元联结，`5,113` 枚为 extra token；
- 冻结的 `836` 个负暴露 interval 合计仅 `4.95626851575` 分钟，其中出现 `153` 枚 token，点估计为 `30.869998167734 token/min`；
- 同一 track/reset 在首枚 token 后再次达到连续两帧资格门的重复激活共 `6,328` 次，全部记录并抑制，没有转成重复 token。

因此本轮不能进入 `causal token → isolated opener` 集成。`33/33` 只证明 reset-scoped 持续 token 在当前 oracle 支持单元上具备覆盖可行性；它没有证明 extra-token 可接受，更没有提供提醒级误报、clearance、人体或生产证据。

## 冻结 producer 合同

producer 只接收：

- 当前/历史 detector 形成的 T0 track identity；
- 当前/历史 per-track route relation；
- route validity；
- reset。

producer 明确拒绝 `event_id`、truth box/identity、alertable/event window、未来帧、clearance、oracle token 与 candidate identity。每枚 token 只由 `source + sequence + reset segment + track ID` 定义：

1. 同一 track 在 route known 下连续两帧 active relation 时，因果地产生一枚 token；
2. token 仅为本轮 post-hoc 可行性联结而在该 reset scope 内持续有效，不代表已经获得 opener 消费或 App runtime 权限；
3. unknown route 清空连续性且不得发 token；
4. reset 清空连续性和旧 token scope；
5. relation gap 清空连续性；
6. 同一 track/reset 再次资格化只写入 repeat-activation ledger，不生成第二枚 token。

没有修改或重跑 detector、T0、route、C1–C3、route-invalid guard、truth、event window 或 clearance，也没有改任何阈值和分母。

## 两阶段执行与 post-hoc 联结

第一阶段只读取现有 candidate-blind full-sequence formation facts。它先硬验证 C1/C2/C3 三份投影的以下字段逐帧完全一致：frame identity/timestamp、reset、route known、observed track IDs 与 active relation track IDs；随后折叠成 41 条候选无关 ledger。每帧只保留上述允许字段、token activation 和被抑制的 repeat activation。

第一阶段完成后，inventory 明确记录：

- `candidate_projection_count_verified=123`
- `candidate_independent_sequence_count=41`
- `frame_count=62,229`
- `truth_payloads_decoded=0`
- `event_windows_decoded=0`
- `oracle_tokens_decoded=0`

第二阶段在新的进程中先按 inventory SHA 复验 41 条 ledger，之后才读取上一轮 36 个 truth-assisted oracle token ledger和冻结负暴露 mask。coverage 以 `posthoc_oracle_cell_id` 报告，但 producer token identity 保持候选无关；同一 token 与多个 candidate cell 的联结不会增加 runtime token 数。

## 完整审计结果

| 审计项 | 结果 |
| --- | ---: |
| 完整序列 | 41 |
| 完整序列帧 | 62,229 |
| producer token | 5,126 |
| oracle-supported candidate-event coverage | 33/33 |
| 唯一 supported oracle event | 11 |
| 与 supported oracle 联结的唯一 producer token | 13 |
| 无 active relation、继续 fail closed | 3/3 candidate-event cell；1 unique event |
| full-sequence extra token | 5,113 |
| negative exposure interval | 836 |
| negative exposure | 4.95626851575 min |
| negative-exposure token | 153 |
| negative-exposure token/min | 30.869998167734 |
| repeat activation（已抑制） | 6,328 |
| duplicate token ID | 0 |
| unknown-route token | 0 |
| cross-reset token | 0 |

extra-token ledger 保留全部 `5,113` 行 extra token 和全部 `153` 个负暴露联结；repeat ledger 保留全部 `6,328` 次重复激活，不把不可接受或尚无门限的风险隐藏为零。

## 为什么是 HOLD 而不是通过或 REJECT

冻结判定规则为：

- supported coverage 少于 `33/33`，或 3 个 no-active-relation 单元未继续关闭，或 unknown/reset/duplicate 完整性失败：`REJECT`；
- coverage 和完整性通过，但 extra-token 风险没有冻结的接受阈值与可信上界：`HOLD_FOR_POLICY_GATE`；
- 只有二者都通过，才可能产生下一轮集成资格。

本轮 coverage/完整性通过，因此不是 coverage `REJECT`。但负暴露只有 `4.956` 分钟，低于既有 5 分钟最低支持线；协议也没有冻结可接受的 extra-token/min 门限。观察到的 `30.87 token/min` 与 `5,113` 枚完整序列 extra token 进一步说明不能把风险当作可忽略。因此 `credible_extra_token_risk_bound=false`，终态只能 `HOLD_FOR_POLICY_GATE`。该枚举仅表示“机制覆盖可行，但 policy/risk gate 未闭合”，不表示 L2/L3 promotion hold。

## 收据与验证

- config SHA-256：`74b299f14a6c82fa747ce16c0961b20fcf2259ff685ad8b08093514d1cd81d8d`
- truth-blind token inventory SHA-256：`6d3effc77b693f76e44bb7f1676169b496d8dd0d90f1b034a5ea3814b7714368`
- extra-token ledger SHA-256：`f64b37d1fe6e818483c75b8277d457e21a4e6fff624734237d39d43139a6f3c1`
- repeat-activation ledger SHA-256：`5f0eb26fc50663fa91159b566d3bd9a346261b66a8f4ab0d51a962ed9b6a3c5c`
- terminal SHA-256：`1ab95c5d9bf01d9c23dcd8198fc614ceef35231e85e633b86c9c62bbb9bca197`
- validation SHA-256：`e222d2e0db8b36a623c0991c0595a23b86af75ed4793defd33493746ba3efb17`
- focused tests：9 tests OK；覆盖连续两帧资格、one-token-per-track/reset、重复激活抑制、reset 新 scope、unknown route 清 streak、unobserved track、truth/event/window/future/clearance/oracle/candidate 字段拒绝及三候选投影一致性
- validator：`VALID`；精确重建 123 份输入投影、41 条 producer ledger、62,229 帧、33/36 post-hoc cell、全部 extra/negative/repeat ledger 与 terminal
- 本地 canonical evidence：`artifacts.local/evidence/ustrf-truth-blind-causal-per-track-token-producer-audit-r0/`

## 权限与下一独立边界

本轮最大权限为 `TRUTH_BLIND_CAUSAL_TOKEN_FEASIBILITY_AND_EXTRA_TOKEN_AUDIT_ONLY`。C1–C3、opener、clearance、候选比较、selection、L2/L3、Android shadow、H2、人体、独立行走和生产权限均未修改或开放。

下一独立边界不能冻结 `causal token → isolated opener` 集成。若继续，应先独立冻结 **candidate-independent causal token policy gate**：在不读取 truth/event window、不改 C1–C3 或 clearance 的前提下，为 token 的有效期、重资格/抑制和完整负暴露风险预注册接受阈值与可信上界；只有把 extra-token 风险压到冻结门内并由全序列审计复验后，才重新获得实际集成资格。
