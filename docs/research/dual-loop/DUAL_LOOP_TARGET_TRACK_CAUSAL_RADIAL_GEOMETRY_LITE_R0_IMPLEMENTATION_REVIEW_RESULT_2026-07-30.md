# Target/track-conditioned causal radial geometry LITE R0 implementation review

日期：2026-07-30（Asia/Hong_Kong）

```text
PROTOCOL: DUAL_LOOP_TARGET_TRACK_CAUSAL_RADIAL_GEOMETRY_LITE_R0
TERMINAL: IMPLEMENTATION_REVIEW_PASS
IMPLEMENTATION_LOCK_SHA256: 9e2173d0b0cea2959fa35d2e52c7f13378aae0e5c62642a13c1524cbd5971807
PARAMETER_SHA256: a7529f94b5a40e240d07f3cc4202d8471d0e387dabdeef866cffbfb3786c0ab4
SYNTHETIC_FIXTURES: 24/24 PASS
LOCK_VALIDATOR: VALID / errors=[]
FULL_REPLAY: NOT_RUN
TRUTH_JOIN: NOT_RUN
OLD_F1B_DECISION_ACCESS: SEALED / DECLARED_ZERO
```

## 结论

独立复核对当前精确 implementation lock、7 个实现/测试文件 binding、输入冻结
manifest 和参数 identity 重新实算后给出 `IMPLEMENTATION_REVIEW_PASS`。旧的中间
身份和中间评审结论作废，不构成本轮执行依据。

通过项包括：

- producer 每目标只保留一个 previous observation，lookback=1、lookahead=0；
- 两臂接收相同 frame、ROI、target、epoch 与 timestamp；
- box log-area 和 ROI sparse radial-flow 的公式、符号、support、quality 和
  abstention 与设计锁一致；
- current ROI 扩张为宽高总尺寸 `1.10×`；
- TTL 固定为 capture-anchored 100 ms，consumer 超期转为 `STALE_RESULT` 且不续期；
- producer 在 decode 前核 frozen replay SHA，并 fail-closed 拒绝 truth/event/Vicon/
  decision 路径、输入覆盖及 input-freeze/image-root 内写入；
- evaluator 在首次接触 truth/event ledger 前完成 receipt/output/replay identity、
  implementation/parameter identity、metadata、TTL、quality schema 及精确
  `13,014 × 2 = 26,028` keyset 核验；
- truth/event 加入后仍保持 469 个 primary parent events 的固定分母、wrong-sign、
  target、anchor-region、truth-state 与停止门。

复核只运行 synthetic fixtures 和 implementation-lock validator；未运行全量
REveL producer，未读取 truth/event 内容，未访问旧 F-1B decision 输出。

## 权限效果

本 PASS 只允许另立并独立评审一次性 activation decision。它本身不授权全量
producer、truth join、Development 科学终点消费、Confirmation、Android、融合、
提醒、产品或安全工作。implementation-only 修复预算已用尽；后续任何实现、参数、
fixture 或 lock 漂移都必须停止并重新评审。
