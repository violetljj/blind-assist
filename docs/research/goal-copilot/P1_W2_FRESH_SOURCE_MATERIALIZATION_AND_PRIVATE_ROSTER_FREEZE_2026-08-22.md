# P1-W2 fresh source materialization and private roster freeze

状态：`FRESH_SOURCE_PAYLOAD_ACQUIRED / PRIVATE_ROSTER_FROZEN / EXECUTION_NOT_AUTHORIZED / DEFAULT_APP_UNCHANGED`

终态：`P1_W2_FRESH_PRIVATE_ROSTER_FROZEN`

Claim ceiling：`FRESH_ADT_INDOOR_OBJECT_PROXY_ROSTER_ONLY / NO_ANCHOR_INTERFACE_OUTCOME / NO_BUILDING_ENTRANCE_OR_PRODUCT_AUTHORITY`

## 1. 授权与停止边界

本 successor 只执行已冻结的
[`P1-W2 implementation/data selection`](P1_W2_OUTCOME_BLIND_IMPLEMENTATION_AND_DATA_SELECTION_2026-08-22.md)：

1. 下载并校验 8 个既定 ADT parents 的 preview RGB 与 main groundtruth；
2. 只用 instance、bbox、visibility、camera pose 与时间戳机械生成 source/probe/confuser；
3. 分离 provider input 与 evaluator-private truth，并冻结 roster hashes。

没有下载 EfficientLoFTR checkpoint，没有调用 matcher、DINO 或其他模型，没有计算 geometry、identity 或 joint
eligibility outcome，也没有执行 consumed 17-case Development cohort。P1-W2 single execution 仍需再次授权。

## 2. Payload acquisition

8/8 frozen parents 的 16/16 members 已从 live ADT manifest 获取，并逐文件核对冻结 filename、byte size 与 SHA-1：

```text
RGB members                     8
groundtruth members             8
RGB bytes              835,513,449
groundtruth bytes       405,678,105
total bytes           1,241,191,554
manifest drift                    0
hash/size failure                 0
```

本地 acquisition receipt：

```text
artifacts.local/evidence/p1_w2_anchor_interface_v1/acquisition_receipt.json
sha256 fef62cf9b9c9c617572bd1149a160cc7a81bcbd57f7fb4998286ba424ab6cbe8
terminal P1_W2_FRESH_SOURCE_PAYLOAD_ACQUIRED
```

RGB 被标记为 `RGB_PROVIDER_INPUT`；groundtruth 永远是 `GT_EVALUATOR_ONLY`，不得进入 matcher/identity provider。

## 3. Mechanical private roster

Selector 位于：

- `scripts/research/goal_copilot_bridge/acquire_p1_w2_fresh_sources.py`
- `scripts/research/goal_copilot_bridge/select_p1_w2_private_roster.py`

它严格使用预冻结的 `static + rigid + object`、source/probe visibility、bbox min-side、same-prototype/same-category
confuser、pose strata、0.50 s reappearance gap、candidate cap 和 hash ordering。它不解码 RGB 内容作选择，也不读取任何
模型输出。

固定 8-parent 母分母的机械 accounting 为：

```text
PARENT_ROSTER_ELIGIBLE           7
PAYLOAD_OR_SCHEMA_FAILURE        1
fixed parent denominator         8
selected source/probe pairs     27
```

唯一失败 parent 是 opaque `p1w2-parent-02`：payload 本身 hash 正确，但只有 `3068/3073` GT timestamps 能在冻结
`20 ms` 窗口内映射到 RGB preview。没有放宽 alignment、换 parent、换对象或补样；它继续留在 8-parent accounting 中。

两条 Lite preview 各含 4 个重复但不倒退的 source timestamps。首次 selector 把“重复”错误当作 schema failure；在任何
模型 outcome 不存在时，只将接口修正为允许相等、仍拒绝时间倒退，并从同一 frozen parents 重建。没有修改任何
target/confuser/visibility/pose gate。最终 parent-02 的独立 alignment failure 仍 fail closed。

## 4. Frozen support

```text
ROTATION_DOMINANT          2
SMALL_TRANSLATION          7
LARGE_TRANSLATION          5
REAPPEARANCE               6
SAME_SCENE_CONFUSER        7
```

因此满足预注册的 data-support gate：eligible parents `>=6/8`、每个 frozen stratum 非零、same-scene-confuser probes
`>=6`。这只说明 roster 可评价，不是 anchor interface signal。

Provider file 只含 opaque parent/case/candidate IDs、frame indices 与 core boxes；递归 firewall 检查确认不含
`sequence_id`、`object_uid`、`source_object_uid`、`true_candidate_id`、truth role 或 confuser priority。Private
`support_buckets` 也只留在 private evaluator，不能向 provider 暗示 motion、reappearance 或 confuser truth。Private
evaluator 在 provider 输出封存后才可读取 true/confuser mapping。

## 5. Frozen receipts

权威目录：

```text
artifacts.local/evidence/p1_w2_anchor_interface_v1/private_roster_v1/
```

冻结 hashes：

```text
public_roster.json
  efa99053d7aee023f39ff283ec7a2cf15c8ab3fc43e62f4d06bf639485d5fd2e
provider_input.json
  46ef4fe18590b97cc03f4239ca0a42fce1f5006d793ebafa79b240ee97bcaa56
evaluator_private_truth_map.json
  1b17da21fad886649c1969d3e19a0d3755efb90da98ed1f9b25e384607b9afb0
parent_accounting.json
  10921df743038ba1445a99f0dfe77fd56191244dabaf3e7aad009adaec005541
roster_freeze_receipt.json
  b9c901147147262f65b87e74119ad558007b63d66ee2aadc7538a3695b05f96a
```

同一 selector 对相同 payload 做独立 replay，四个 authoritative output hashes 完全一致；disposable replay output 已
清空。最终 receipt 记录 `model=0 / matcher=0 / identity=0`，`execution_authorized=false`。

## 6. 当前终点

```text
implementation/checkpoints/gates: frozen, unchanged
fresh source payload: acquired and verified
private roster: frozen and evaluable
P1-W2 model outcome: absent
P1-W2 single execution: not authorized
```

唯一合法 successor：`P1_W2_SINGLE_EXECUTION`。若另行授权，只能读取上述 frozen hashes 运行一次；不得重选 parent、
target、source、probe、confuser，不能改变 crop、threshold、geometry model、identity representation 或判决门。
