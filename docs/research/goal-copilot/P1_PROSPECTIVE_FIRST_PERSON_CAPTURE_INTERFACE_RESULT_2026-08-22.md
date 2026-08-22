# P1 prospective first-person capture interface result

状态：`IMPLEMENTED / REAL_DEVICE_COHORT_NOT_CAPTURED / PROVIDER_CALLS=0 / PA3_INFERENCE_NOT_AUTHORIZED`

## 结论

S0v3 的 `6 visible episodes / 7 visible frames` 失败后，唯一 successor 所需的 physical-capture 接口已经实现。
它没有用预存公开视频、Mapillary、replay 或 synthetic video 冒充 prospective capture，也没有运行 proposal model。

新接口把证据顺序固定为：

```text
C0 public Goal Contract
→ immutable full-roster capture plan
→ device-owned continuous first-person recording
→ hash-bound device receipt
→ fixed outcome-blind frame extraction
→ private truth
→ visibility denominator gate
→ optional PA3 authorization
```

`arm_capture.py` 在任何录像前冻结至少 5 个 episode、媒体文件名、forward-camera/continuous-capture 要求、全局
capture instruction，以及结束前 `2.5 / 1.5 / 0.5 s` 三个抽帧点。`materialize_capture.py` 随后逐 episode 验证
`goal < arm < physical capture < receipt`、C0/plan/receipt body hash、media SHA-256、目录边界、单视频流、分辨率、
时长与 recorder timeline，并拒绝 partial roster、重复视频、truth/outcome 字段或 per-episode frame selection。

PyAV 的真实 MP4 probe、decode 与 JPEG extraction 路径已通过 focused integration test。异常路径会清理由当前调用创建的
temporary output；已存在的 durable output 不会覆盖。

## 当前证据边界

- 13 项新 capture-contract tests 通过，包括真实 MP4 probe/extraction。
- provider/model calls：`0`。
- real device cohort：未采集；本机 ADB 当前无连接设备。
- PA3 inference：未授权。
- 默认 App：未修改。

该接口只证明本地可审计的 provenance mechanics，不能外部证明设备时钟或 recorder 没有被伪造。真实 cohort 仍须由
device-owned sidecar 与 source custody 支撑。下一动作是用该接口采集完整、入口可见的 first-person cohort；只有私有 truth
达到预冻结的 `>=5 visible episodes / >=8 visible frames` 后，才允许一次 frozen PA3 semantic proposal run。
