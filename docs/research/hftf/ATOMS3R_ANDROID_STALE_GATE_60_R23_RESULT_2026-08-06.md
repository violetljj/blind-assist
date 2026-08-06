# AtomS3R Android stale gate 60 ms R23 结果（2026-08-06）

## 结论

60 ms 阈值在 VGA/Q14 + decodeSampleSize=2 下仍然过严，10 秒丢弃 5/275 个 packet（约 1.8%），因此拒绝。当前只保留 65 ms 作为低丢弃率候选阈值。

## 10 秒结果

- processed frames：270
- source packets：275
- stale packets dropped：5（约 1.8%）
- sequence gaps：4
- 0 error，0 reconnect，0 overwrite

与 55 ms 的约 16.5% 丢弃相比，60 ms 虽明显改善但仍超过日常实时策略可接受的 1% 上限。不能用大量样本丢弃换取尾延迟下降。
