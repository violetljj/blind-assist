# LITE R1 source RGB shape audit

日期：2026-07-30（Asia/Hong_Kong）

```text
STATUS: R1_SOURCE_AUDIT_PASS
CANDIDATE_OUTPUT_ACCESS: NONE
TRUTH_EVENT_ACCESS: NONE
OLD_F1B_DECISION_ACCESS: NONE
REPLAY_ROWS: 13014
UNIQUE_REPLAY_IMAGES: 8363
SAME_TARGET_SAME_EPOCH_PAIRS: 12876
SHAPE_MISMATCH_PAIRS: 32
EXPECTED_COMMON_ABSTENTION_ARM_ROWS: 64
```

可复算 source-only audit 的 SHA-256 为
`38802bacd9ec08de95556986445fe59c2c8ea815745f0f8a89147ffd2ec704bd`，
输入 replay SHA-256 仍为
`14f1f7f0f330d8b01146e37c31505240f3f0e8d301846ebcad44a628948e6440`。

## 观察

8,363 张 replay-unique RGB 全部可解码。主尺寸 `260×346` 有 8,329 张、对应
12,965 replay rows；其余为：

| decoded H×W | unique images | replay rows |
| --- | ---: | ---: |
| `258×346` | 13 | 16 |
| `260×259` | 12 | 15 |
| `260×344` | 6 | 12 |
| `250×346` | 3 | 6 |

13,012 个 same-target immediate pairs 中，12,876 个还满足 same epoch。
其中 32 个 decoded `(H,W)` 不同，占 `0.248524%`；track-000 为 12，track-001
为 20，current region 的 LEFT/CENTER/RIGHT 分别为 14/14/4。32 个 pair 的
`delta_t` 为 `37.704–52.751 ms`，原本都在 100 ms history 上限内。

34 张非主尺寸 RGB 集中于 11 个 source-consecutive 段：10 段长 3 帧，1 段长
4 帧。进入和退出边界形成 22 个 source transitions；展开到 target-conditioned
replay opportunities 后为 32 个。27/32 保持同 region，5/32 同时跨 region。

## 支持的推断

R0 失败由未定义的 cross-shape pair 接口触发，不证明 bbox 或 radial flow 的科学
成败。最小且不制造伪尺度的 R1 语义是：native decoded shape 不同即两臂共同
`FRAME_SHAPE_CHANGE` abstention；禁止 resize、pad 或 crop；当前帧仍成为该 target
下一机会的唯一 previous frame。

audit 不能知道这 32 个机会落入多少 truth-only natural events，因此不授权调整
event、coverage、deadband、target/region 参数或 readiness gate。完整明细与
mismatch image hashes 位于 ignored
`artifacts.local/evidence/dual-loop/target-track-causal-radial-geometry-lite-r1/source-shape-audit/shape_audit.json`。
