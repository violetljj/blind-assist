# P1 facade-door semantic availability and leftmost-goal relation result

状态：`NO_USER_CAPTURE_OR_LABELING / S0V6_SOURCE_TRUTH_INDEPENDENCE_NOT_ESTABLISHED / S0V7_PUBLIC_SCHEMA_CONTRACT_MISMATCH / S0V8_FACADE_DOOR_AVAILABILITY_ESTABLISHED / S0V10_LEFTMOST_RELATION_TOP1_IMPROVED`

## Answer first

公开数据发现、审计、下载、固定 roster、RGB/label 隔离、YOLOE inference 与 private evaluation 均已自动完成；
用户没有选图、采集、标框、运行命令或判断中间结果。

FacadeElements 的 3,494 个 image/label pair 来自同一个已校验 Zenodo archive。source lock 在 selected pixel 与
private label access 前冻结；五个 door 类只从全局 `data.yaml` 得到，provider 只看到 RGB、pre-truth Goal
Contract 与全局 prompt `building entrance`。

## 自动 source audit

- S0v6 LSAA：其 door detection truth 的独立人工标注 provenance 无法建立，在下载 selected data 或调用模型前以
  `SOURCE_TRUTH_INDEPENDENCE_NOT_ESTABLISHED` 终止。
- S0v7 FacadeElements：网页摘要的 door taxonomy 与 archive `data.yaml` 不一致；旧合同在 selected pixel/label
  access 前 fail closed，没有事后改类。
- S0v9：新 relation roster 的 case/episode prefix 读取实现错误，首次 pixel access 前即因 C0 ID 不匹配终止，
  零模型调用。修复后使用新合同 S0v10，没有改写失败 run。

## S0v8: set-valued facade-door proposal availability

24 个固定样本中 private evaluator 得到 20 个可见 facade-door case。冻结的
`YOLOE-26n-seg / Ultralytics 8.4.52 / imgsz=640 / conf=0.001 / max_det=100 / K=10` 一次运行结果：

| IoU >= 0.30 | Result |
|---|---:|
| evaluable | 20 |
| Recall@1 | 17/20 |
| Recall@3 | 19/20 |
| Recall@5 | 19/20 |
| Recall@10 | 19/20 |

因此 goal-semantic YOLOE 在这个公开真实立面 cohort 上建立了 bounded facade-door target availability；它没有证明
具体物理实例 identity，也没有证明每个门都是真正可通行的公共建筑入口。

## S0v10: UNIQUE leftmost-entrance relation verifier

为避免把 SET_VALUED 的“找入口”伪装成 identity，S0v10 在未见的下一组 24 个 path-hash roster 上预先冻结产品目标：

`帮我找最左边的入口 -> LEFTMOST_BUILDING_ENTRANCE / UNIQUE -> building entrance`

private truth 后开，只把所有合法 door bbox 中 x-center 最小者定义为唯一目标，其余 door 是同类 distractor。
16 个可见 case 的 semantic proposal Recall@10 为 16/16。随后 GT-blind verifier 仅按候选 bbox x-center 从左到右
重排，原 provider rank 只作为 tie-break；没有模型、阈值或配置 sweep。

| IoU >= 0.30 | YOLOE rank | Frozen leftmost relation rank |
|---|---:|---:|
| all evaluable Top-1 | 12/16 | 13/16 |
| strict A/B/C contrastive cases | 0/2 | 2/2 |

`A/B/C` 要求 candidate pool 同时含目标 A、另一扇 door B，以及与两者 IoU 均小于 0.30 的背景 proposal C；
因此严格 contrastive 分母只有 2，结果只能视为当前帧 goal-relation feasibility evidence，不能推广成稳定收益。

## Claim ceiling and next algorithm gate

已建立的是两件窄事实：semantic proposal 能覆盖多数 facade-door target；公开 Goal Contract 中明确给出的空间关系可用于
候选重排。没有建立跨帧 instance re-identification、memory verifier、接近/到达控制、第一视角分布、产品或安全结论。

下一算法门应使用更大且独立的 UNIQUE relation cohort 验证关系重排，并另行寻找带独立 truth 的第一视角 approach
数据；不能把 facade-door bbox 自动升级为 traversable entrance truth，也不能把本轮 2-case contrastive 结果直接接入 AMRM。

## Evidence identity

- S0v8 roster SHA-256: `331364394df605828df6a1f69771e60f2ae85a6da6fd436c77aa64c5259ed3a4`
- S0v8 prediction SHA-256: `4f11f1bbe58fc952d0b8636e1092964824063a1096d50a15a98857cbf8aa9680`
- S0v8 evaluation SHA-256: `9e4224ea1aeb9f2280cc639e3414141ec41ad271d94b62a0b2d2f8b2ae0c8d64`
- S0v10 source lock SHA-256: `6c7c98a9d231664b68b7a1e90790f1b591ef9ef20a3c171bf39f553397c97335`
- S0v10 verifier manifest SHA-256: `41567abd78232a45e4a268a328f50cd4d9cee2d066cb7f914817f9d24f463ccc`
- S0v10 prediction SHA-256: `3306e5e3fbcebf1168fd9a77e7d56d553371d7de79209d2b44d99dda84c819aa`
- S0v10 PA3 evaluation SHA-256: `2c0fd967a4dd2f6489ccf26304515531228fad955459b11dd6854b476faf0497`
- S0v10 relation prediction SHA-256: `0a2ff090ba0a50522058e8a3dd549049ea69107ed5d23c83bd14d1ba30a29364`
- S0v10 relation evaluation SHA-256: `7eb604ecce67782c80778234fceb49a5017503ad83c37c01458b7083b0884142`

Ignored evidence roots:

- `artifacts.local/evidence/p1_pa3_s0v6_automated_lsaa_facade_v1/`
- `artifacts.local/evidence/p1_pa3_s0v7_automated_facadeelements_v1/`
- `artifacts.local/evidence/p1_pa3_s0v8_automated_facadeelements_v1/`
- `artifacts.local/evidence/p1_s0v9_leftmost_entrance_verifier_v1/`
- `artifacts.local/evidence/p1_s0v10_leftmost_entrance_verifier_v1/`
