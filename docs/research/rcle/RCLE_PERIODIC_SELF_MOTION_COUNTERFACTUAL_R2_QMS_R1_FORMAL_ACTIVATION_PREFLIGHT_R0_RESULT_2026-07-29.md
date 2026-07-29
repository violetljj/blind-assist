# RCLE periodic self-motion R2 QMS-R1 formal activation preflight R0

日期：2026-07-29（Asia/Hong_Kong）

## 结论

```text
PREFLIGHT:
  ACTIVATION_PREFLIGHT_PASS / VALID / FORMAL_NOT_RUN

W8:
  W8_GUARDED_HOST_QUALIFIED / PREFLIGHT_ONLY

ACTIVATION DECISION:
  QMS_R1_SUCCESSOR_FORMAL_EXECUTION_AUTHORIZED / ONE_SHOT
```

QMS-R1 successor 的一次性正式执行已经获得授权，但本任务没有运行新
`480 MAIN + 16 GUARD`。successor formal output path 仍不存在，正式 R3
pair-core call 为零。

## 冻结与不相交

- QMS-R1 operator source SHA-256：
  `5e66d270c1267d36e927cf47808337e6c1c0da68566e039c9a6ad35eb7c7e8c6`；
- 新 formal lock：496 sequences、88 latent clusters，
  identity-set SHA-256
  `dd1fe17a58e458db7bf2fd719e5a3dbcf968b8129b9441ed9d4792dd735ea759`；
- 新 PREFLIGHT lock：固定 8 sequences，identity-set SHA-256
  `dad43f637eaa5aa48276d7410139ec3739309ffdcbe0e540627d86e762ec692d`；
- 对旧 formal、QMS-R1 DEV、QMS-R1 CAL、旧 PREFLIGHT，以及新 formal 与
  新 PREFLIGHT 之间，`numeric seed / token / token hash / cluster id /
  sequence id / scene geometry hash` 六类交集全部为零。

仅改变 `sequence_id` 不算新身份；独立 validator 重新派生每个 uint64 seed，
并复核 scene、arm 和 trajectory identity。

## 未漂移复验

- all-seed geometry：G01–G14 为 `14/14 PASS`，manifest SHA-256
  `3dcf37496997a1edb2e47871c0dfc5185fd207016a26a86e29514412484e7ac6`；
- R3 transport：pair rows 与 PairState 均 exact equal，lock SHA-256
  `2037644bcc91503dd34d20e5e88904bdc44ffbcb3cbfb1454abf5e820036ebfb`；
- analysis implementation：9-member family、shared draws、max-t 等锁逐字节
  重建一致，lock SHA-256
  `09ed9a0716ed3af9979b2f5030168fb34650649dfa6eb16968557dfc5e811aaf`。

## W8 guarded-host

固定 8 条完整 identities 共运行 4,816 frames / 4,808 ordered pairs：

- measured wall：`1099.9671 s`；
- launch available RAM：`9,402,650,624 bytes`；
- minimum available RAM：`8,535,347,200 bytes`；
- swap in/out delta：`0 / 0`；
- maximum heartbeat interval：`20.0721 s`；
- residual worker PID：`0`；
- response / trigger value emitted：`false / false`。

QMS-R1 的冻结语义是同一 scene×motion 用一次 `render_pair` 同时产生 clean/low，
blur 再由同一 clean 派生。投影因此用每个 motion 三个 quality arm 中最大的独立
render 时间作为共享 render 的保守上界；R3 与 validation/hash 仍逐 arm全计。
W8 grouped projection：

```text
core:          37,104.6523 s
retry reserve:  3,710.4652 s  (10%)
total:         40,815.1175 s  = 11.3375 h
ceiling:       43,200.0000 s  = 12 h
```

一次 RAM 不足启动和两次 control-plane 实现错误都在首条 identity 前
fail-closed；专属 partial 目录经审计为空后删除。通过结果只来自最终完整 8/8
运行。

## 防火墙与证据

- predecessor formal tree 前后 SHA-256：
  `9575651890f917e24321f890c8cb69ba20d29f286a08f9bfbbec04dafc083691`；
- W8 success receipt SHA-256：
  `e5001cf67519a2aac4ed3c7a00e78b4e385d43490283f0dc6a5195b8fe8b7ea1`；
- independent receipt SHA-256：
  `c9c82e9c920b9aaedbb652b1b23c8f8c1f53efff3e8b365d72937ed097439d7e`；
- activation decision SHA-256：
  `49584c4f1d508927d196b51658ba1616bd1fed646b17ad4ac284b350cf48e0ed`。

decision 只授权 exact lock、exact operator、W8、cluster-grouped shared-render
scheduler 下的一次 successor formal execution。不得换 identity、改 operator、
改 R3/threshold/three-pair/analysis、访问 sequence16、进入 Android/realtime，
也不产生产品或安全 authority。
