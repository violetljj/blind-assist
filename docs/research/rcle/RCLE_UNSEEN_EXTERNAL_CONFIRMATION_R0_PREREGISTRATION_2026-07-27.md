# RCLE 未见数据外部确认 R0 预注册

状态：`BASE_PROTOCOL_FROZEN / SOURCE_CANDIDATE_LOCK_PENDING / RGB_OUTCOME_FORBIDDEN`

## 研究问题

本阶段只回答一个问题：在不改变 RCLE 底层连续 expansion、不改变
`0.01/s` 阈值，也不改变三 pair 因果确认状态机的条件下，
`CAUSAL_THREE_PAIR_CONFIRMATION_R1` 能否在新的 all-real、cross-source 数据上，
逐来源、逐窗口降低低参考触发，同时保留真实接近响应、限制首触发延迟，并保持
正/低参考角色方向。

当前文件先冻结确认设计和找数规则。具体来源、序列和窗口尚未选择，因此当前不是
F2 confirmation lock，也不授权读取候选 RGB algorithm outcome。合法顺序为：

```text
base protocol
-> official metadata candidate lock
-> payload identity and geometry-only selection
-> exact 2-source / 4-window identity lock
-> implementation and performance qualification
-> one exclusive old+R1 confirmation run
```

任何后一步都不得回写前一步已观察的结果。找数阶段由
[`RCLE_UNSEEN_EXTERNAL_CONFIRMATION_SOURCE_DISCOVERY_R0_CONTRACT_2026-07-27.json`](RCLE_UNSEEN_EXTERNAL_CONFIRMATION_SOURCE_DISCOVERY_R0_CONTRACT_2026-07-27.json)
约束。

本协议中的“未见”严格指
`RGB_ALGORITHM_OUTCOME_UNSEEN / GEOMETRY_SELECTED_EXTERNAL_CONFIRMATION`。
Geometry 会在 RGB outcome 前承担角色选择，因此所选窗不是 geometry-pristine；
不得把结果表述为完全未读取任何 claim-relevant 信息的
`PRISTINE_CONFIRMATION`。这一限定不削弱 RGB 算法盲法，但限制结论只能回答冻结
geometry 角色上的外部算法确认。

## 冻结实现

“旧版”定义为当前 pair 可评价且
`compensated_expansion_median_per_s > 0.01` 时立即触发。R1 只在该连续信号之后
增加因果状态机：

```text
above_t = evaluable_t AND expansion_t > 0.01/s
streak_t = streak_(t-1) + 1  if above_t else 0
r1_trigger_t = streak_t >= 3
```

来源、序列或窗口改变时重置 streak。任一弃权、缺失 pair 或
`expansion_t <= 0.01/s` 也重置 streak。状态机没有未来信息，lookahead 为 0。

冻结内容包括 rotation compensation、Sparse LK、local affine expansion、
observable support manager、pair evaluability、`0.01/s` strict-`>` threshold、
三 pair 长度、reset 规则及所有底层数值配置。外部确认不运行 2/4 pair、majority、
median filter、其他阈值、其他 estimator 或任何单项算法替代。

正式实现锁必须绑定实际执行链中的 `support_manager.py`，不能只绑定其
init re-export。旧版和 R1 不分别解码或运行 RGB：底层 pair evaluator
只执行一次，同一有序 pair ledger 同时写入连续 expansion、`old_trigger` 和
`r1_trigger`；独立 validator 再从连续信号重算两条触发路径。

## 未见与 all-real

候选必须来自真实传感器采集的连续 RGB-D 序列。渲染、仿真、生成深度或由单目模型
估计的深度不能承担本轮 all-real 角色。发布方提供的 metric depth、时间戳和
timestamped 6-DoF pose 必须能够与 RGB 对齐；pose 若来自重建而非外部测量，需在
来源清单中披露，但不能在看到 RCLE RGB outcome 后改变资格。

进入候选前要逐 sequence/capture 核对既往 access vector。当前 development 四窗、
TUM `fr2/rpy`、ETH3D `desk_changing_1`、TartanAir
`japanesealley/Hard/P002`、已观察 CID-SIMS、Bonn、EVIMO2 及
OpenLORIS office/cafe 污染范围不能重新包装为 confirmation。仅有 metadata identity
访问不自动烧毁新的 sequence；任何 claim-relevant geometry、RGB visual review、
RCLE/同机制 outcome 或 selection influence 则按最小可证明范围排除。

所选 cohort 必须包含两个 ancestry 独立的数据来源。每个来源恰好贡献一个
positive 窗和一个 below-reference 窗，共四个 10 秒窗。这样每个来源内部都有
正/低参考对照，角色方向无需依赖 pooled aggregate。

## 先协议后数据

来源搜索只允许读取 official page、论文/数据说明、许可、文件树、大小、hash、
传感器与 ground-truth 格式。候选 lock 必须在任何候选 payload member、数值 pose、
depth 或 RGB 字节前写入并 fsync。候选排序固定使用以下信息：

1. all-real 连续 RGB-D、timestamped 6-DoF pose 和公开可复算身份是否齐全；
2. exact payload 是否可通过普通公开渠道取得；
3. 标称 cadence 是否至少 10 Hz，单 capture 是否至少 30 秒；
4. 下载和解包成本是否落在本轮资源预算；
5. source family ID 与 sequence ID 的字典序。

不得使用序列名暗示的“接近”“静止”等语义、预览图、RGB visual review、已有算法
结果或非冻结几何结果改变排序。metadata 排名结束后冻结不超过四个 source family，
每 family 不超过两个 exact sequence/capture，累计下载预算不超过 40 GiB。

Payload 和 geometry 按 candidate lock 顺序处理。一个来源在冻结 sequence 集合中
找不到正/低参考双角色时，记录 geometry-only `SOURCE_ROLE_INCOMPLETE`，再进入下一个
预冻结来源；这不是 RGB 失败后的换窗。获得前两个双角色来源后立即停止找数并密封
剩余候选。预算耗尽仍不足两个来源时返回
`EXTERNAL_COHORT_NOT_EVALUABLE / VALID`，不读任何候选 RGB algorithm outcome。

## Geometry-only 选窗

每个 exact sequence 以最早同步 timestamp 为锚，枚举不重叠的半开 10 秒窗，不滑窗，
不按画面或算法结果移动起点。窗口至少包含 90 个 source-consecutive pair；pair
timestamp 必须严格递增，单 pair `dt` 不得超过 `0.100 s`。缺失或不同步 pair 保留为
abstention，不插值、不补帧。

每个可评价 pair 使用冻结的 pose、metric depth 和相机内参计算
`signed_radial_expansion_per_s`。角色规则沿用既有 geometry 定义：

- positive：geometry coverage 至少 `0.80`，`signed radial >= 0.05/s` 的固定分母
  比例至少 `0.80`，最长连续 positive 段至少 `5.0 s`；
- below-reference：geometry coverage 至少 `0.80`，`signed radial < 0.01/s`
  的固定分母比例至少 `0.80`，最长连续 below 段至少 `5.0 s`；
- 同一窗同时满足两种角色时判 `AMBIGUOUS`，不得选择。

每个来源在全部冻结 sequence 和固定窗中，选择字典序最早的
`(positive_sequence_id, positive_window_index, below_sequence_id,
below_window_index)` 可行 tuple。同 sequence 两窗的起点至少相隔 `20.0 s`；
不同 sequence 不施加伪时间距离。四窗身份、ordered RGB/depth/pose member identity、
CRC/bytes/hash 和 role ledger 必须在首次 selected RGB member read 前独占发布并
fsync。此后禁止替换来源、序列、窗口或成员。

## 一次性执行

正式运行前需完成：

- 完整算法与适配器 implementation lock；
- 独立 validator 和恶意反例测试；
- 非 confirmation fixture 上的 runtime/I/O preflight；
- host receipt、输入/代码 hash、空 canonical output、exclusive claim；
- 网络关闭或网络访问硬失败；
- machine-readable progress，且不得泄露 trigger/coverage/outcome。

正式 claim 只消费一次。所有四窗必须在同一 claim 和同一 implementation identity
下完成。若 runner、序列化、身份或资源异常造成 `INVALID`，保留原 evidence version；
不得原地覆盖。是否另立纯执行修复版本需单独判断，但不能改变数据、窗、算法、阈值或
门。科学 FAIL 或 RGB 后的 NOT_EVALUABLE 不允许换窗补救。

## 四门与判定单位

连续 pair 不是独立试验。窗口是最小科学判定单元，来源是跨窗复制单元。

每个 below-reference 窗分别计算：

```text
below_relative_reduction =
  1 - r1_trigger_coverage / old_trigger_coverage
```

旧 coverage 为 0 时该窗为 `NOT_EVALUABLE_OLD_BELOW_ZERO`，不得把 `0/0` 记为通过。
其余每个 below 窗必须满足相对下降 `>= 0.30`。

每个 positive 窗分别计算：

```text
positive_retention =
  r1_trigger_coverage / old_trigger_coverage
positive_first_trigger_delay =
  first_r1_trigger_timestamp - first_old_trigger_timestamp
```

每个 positive 窗必须满足 retention `>= 0.90` 且 delay `<= 0.25 s`。旧版没有
positive trigger，或旧版有而 R1 没有，均判该窗 gate FAIL，不以 undefined ratio
回避算法失效。

每个来源的 role direction 必须满足：

```text
r1_positive_window_coverage > r1_below_window_coverage
```

最终 `CONFIRMATION_PASS` 需要两个 below 窗、两个 positive 窗和两个来源方向全部
通过，并且执行证据为 VALID。任一适用窗或来源失败即
`CONFIRMATION_FAIL_STOP_AT_R1 / VALID`。Pooled 四门、pair 总数和总体 coverage
必须报告，但只作 diagnostic，不能覆盖局部失败。

## 报告合同

结果至少包含：

| 层级 | 必报字段 |
| --- | --- |
| pair | source/sequence/window/pair identity、timestamps、evaluable、continuous expansion、old/R1 trigger、reset reason |
| window | role、固定分母、old/R1 count 与 coverage、适用 gate 值、PASS/FAIL/NOT_EVALUABLE |
| source | positive/below window identity、role direction、全部局部门的 AND |
| cohort | 两来源终态、pooled diagnostics、execution validity、scientific outcome |

报告不得只给 pooled aggregate，不得删除弃权或失败窗，不得对失败来源做
leave-one-source-out 主结果。任何 FAIL、INVALID、NOT_EVALUABLE 或重大 abstention
都要记录 observation、supported inference、alternative explanations、跨来源失效
类别、可复用资产与后继假设。

## 停止与后继

`CONFIRMATION_PASS / VALID` 只说明 R1 在本次两来源四窗机制确认中通过，允许另立
Android/实时链路集成设计；它不自动授权集成、主动提醒、真人、产品或安全声明。

`CONFIRMATION_FAIL_STOP_AT_R1 / VALID` 表示实现停在 R1，开展跨来源失效归因，不
立即调三 pair、阈值、底层算法或窗。`EXTERNAL_COHORT_NOT_EVALUABLE / VALID` 表示
geometry-only 找数未形成可确认 cohort；不得用 RGB 结果、滑窗、降角色门或追加来源
回救本版本。
