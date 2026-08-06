# AtomS3R Android stale gate 55 ms R22 结果（2026-08-06）

## 结论

VGA/Q14 + `decodeSampleSize=2` 下将过期帧准入门从 65 ms 收紧至 55 ms，10 秒即丢弃 45/272 个输入 packet（约 16.5%）。该路线明确拒绝，不进入 1 分钟。

## 10 秒结果

- processed frames：227
- source packets：272
- stale packets dropped：45
- sequence gaps：44
- 0 error，0 reconnect，0 overwrite

55 ms 阈值接近当前正常 frame age 分布上沿，会把正常网络/采集波动当成过期帧。任何统计尾延迟下降都主要来自删掉大量样本，不是真实链路优化。

继续保留 65 ms 作为候选阈值：此前 1 分钟只丢弃约 0.24%，且不会明显影响正常结果吞吐。
