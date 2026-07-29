# F-1A 负类补充 R1 结果

状态：`COMPLETE / READY / DATA_PROTOCOL_VALID`

执行者：`viojjet`

## 结论

R0 的固定标签修复保持 `HOLD_DATA` 不变；独立后继 R1 在一条 development-only、
SHA 绑定的既有 Ulm 连续 RGB 中，只补缺失的 `STATIC_SCENE` /
`LOW_TEXTURE_BLUR_OR_OCCLUSION` 负类。两次新鲜隔离复核对 `580.0–588.5 s` 的
`STATIC_SCENE` 达成一致。两条单方低纹理候选经第三模型裁决全部隔离。

合并账本因此达到：

```text
independent capture sessions = 4
positive events = 17
positive sessions = 3
negative windows = 20
negative categories with >=2 windows = 4
development sessions = 2
decision sessions = 2
decision sessions with positive and negative = 2/2
```

负类分布为：

```text
TURN_OR_NEAR_IN_PLACE_ROTATION = 11
NORMAL_WALKING_SHAKE = 5
LATERAL_PASS_OR_RECEDING = 2
STATIC_SCENE = 2
LOW_TEXTURE_BLUR_OR_OCCLUSION = 0
```

合同只要求五类候选中的至少四类各有两个负窗；因此全部 F-1A 最小条件通过，
`DATA_STATUS=READY / DATA_PROTOCOL_STATUS=VALID`。低纹理机制未覆盖，不能进入后续主张。

## 不可变性与污染边界

- R0 账本 SHA `ab9f7771...cf7cf70` 未改变；
- R0 validation SHA `bd9dd028...a8f12e` 由 R1 spec 预先绑定；
- Ulm 只作为 `DEVELOPMENT_SUPPLEMENT`；其历史 USTRF 输出访问状态被披露，不能成为
  decision 或确认数据；
- R1 复核未见 YOLO、Sparse LK、RCLE、历史 USTRF 输出、风险、提醒或双环候选输出；
- 不回收 R0 隔离项，不把两条单方低纹理候选计入分母，不降低类别或窗口门。

## 可复算凭据

```text
review_bundle_subject_sha256:
5fddcff183c68c8ff6b6150d7b02beaa555135bdc222263cc82f6773c8a542a1

review_a_raw_sha256:
321415ddf3b79157d17eb5b49c5c7fa8e06d6ff28e5b9c0b6489a3dc3ec132c3

review_a_normalized_sha256:
9112231d3ef7363d46a61a915caf90209d40e8d6cb85b4b50aa4a11a1b81068e

review_b_sha256:
b87e58355753879180a6ae2cd2012ad676d0ec8947da427ce309214d0c502661

comparison_sha256:
b5fdca80a3ad1373133eb6e4179ce350bbbefd658989a7897b3662d8306bfd67

adjudication_sha256:
9a46c1d0fa4c233ba96e73a2dbe30e19c0c03159b898a496b490ad8b7dd221e4

combined_ledger_sha256:
4f514b1277449d754d2ca45469d655610c55d4b56d2e2c1031efbf7c8b9d23c5

validation_sha256:
256121ee38c0390c1817993c81f52ceae11aea9c8fddf848ad1b5a3e18e822fe
```

正式本地凭据位于：

`artifacts.local/evidence/dual-loop/f1a-negative-category-supplement-r1/`

## 下一阶段

F-1A READY 只产生时间凭据审计资格。按用户连续推进授权，下一步为 F-1B0：先审计现有
生产 QNN 与隔离 Sparse LK 凭据是否包含合同要求的真实发布、可用与消费时间；字段不足
时只运行 baseline timing，不访问事件级增量、不选择质量门、TTL、区域或融合规则。
