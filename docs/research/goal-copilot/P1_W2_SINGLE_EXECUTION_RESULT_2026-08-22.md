# P1-W2 single execution result

状态：`SINGLE_EXECUTION_COMPLETE / PROVIDER_SEALED_BEFORE_PRIVATE_TRUTH / NO_RERUN / NO_STAGE_A_V2 / DEFAULT_APP_UNCHANGED`

终态：`P1_W2_RGB_REFERENT_INTERFACE_NOT_SUPPORTED`

Claim ceiling：`FRESH_ADT_INDOOR_OBJECT_PROXY_SINGLE_EXECUTION_ONLY / NO_BUILDING_ENTRANCE_PRODUCT_OR_SAFETY_AUTHORITY`

## 1. 冻结输入与执行纪律

本次是 [`P1-W2 private roster`](P1_W2_FRESH_SOURCE_MATERIALIZATION_AND_PRIVATE_ROSTER_FREEZE_2026-08-22.md) 的唯一
授权执行：固定 8-parent 母分母、7 个可运行 parents、27 cases、66 candidate pairs，没有重选 target/source/probe/
confuser，也没有改 EfficientLoFTR、DINOv2-S、crop、threshold、geometry model 或 terminal gate。

Provider runner 不接受 private-truth 参数；先封存 27-case provider output，随后独立 adjudicator 才读取
`evaluator_private_truth_map.json`。正式调用量：

```text
EfficientLoFTR candidate pairs       66
DINOv2-S encoded crops               73
DINOv2-S forward batches              5
provider cases                        27
```

第一次启动在第一个 candidate 的 EfficientLoFTR preprocessing 前失败：HF processor 不接受 2D grayscale PIL，
`completed={}`，没有 correspondence、candidate result 或 private adjudication。唯一修正是把同一 grayscale crop 复制为
3-channel processor input，checkpoint 自身仍按配置转灰；随后从同一 frozen input 完成正式 provider seal。该未封存
attempt 不进入结果或调用分母。

## 2. 冻结判决

| endpoint | frozen requirement | observed | sufficient |
|---|---:|---:|---|
| Geometry parent-macro true support | overall `>=0.70`; each viewpoint `>=0.50` | overall `0.00`; all strata `0.00` | no |
| Identity unique true on confuser probes | parent-macro `>=0.70`; false bind `0` | `0.00`; false bind `1` | no |
| Joint correct eligibility | parent-macro `>=0.60`; false bind `0` | `0.00`; false bind `0` | no |

固定 counts：

```text
true-candidate geometry supported      0 / 27
confuser unique-true identity           0 / 7
confuser identity NOT_OBSERVABLE        6 / 7
confuser unique-wrong identity bind     1 / 7
joint correct ELIGIBLE                  0 / 27
joint NOT_ELIGIBLE                     27 / 27
joint false bind                        0 / 27
```

`joint false bind=0` 不能作为正向安全信号，因为 frozen geometry veto 使全部 cases 都是 `NOT_ELIGIBLE`。

## 3. Failure-layer accounting

### 3.1 Geometry

EfficientLoFTR 并非完全没有产出：true-candidate raw correspondence count 为 `min 5 / median 84 / max 777`。但经过
source+probe core filtering 后降为 `min 0 / median 2 / max 58`：

```text
core correspondences >= 8          8 / 27
finite homography                  8 / 27
inliers >= 8                       0 / 27
inlier ratio >= 0.50               0 / 27
both endpoint quadrants >= 3       3 / 27
maximum inlier count                    7
maximum inlier ratio            0.4545
```

因此失败发生在 **referent-core geometric consistency**，不是因为把 matcher 的 raw match count 直接当作 geometry
authority。Context/raw matches 没有越权救回 core。

### 3.2 Identity

27 个 true candidates 对 consumed P1-A2 四门的单门通过数为：

```text
anchor_match_fraction              2 / 27
median match_confidence            8 / 27
spatial_consistency                7 / 27
anchor_coverage                    6 / 27
all four AND                       0 / 27
```

True-candidate median 分别为 `0.0703125 / 0.5948849 / 0.36 / 0.6875`，均低于 frozen threshold。same-scene confuser
denominator 上没有 unique true separation，并有一次 unique wrong selection；因此 identity failure 不能解释为“只是过度
abstain 而没有错绑”。

## 4. 解释边界

这次结果同时否定了当前冻结 geometry support 与 identity separability 在该 fresh ADT indoor-object proxy 上的
feasibility signal，所以按预注册决策树进入：

> `P1_W2_RGB_REFERENT_INTERFACE_NOT_SUPPORTED`

它不证明所有 RGB referent representation 都不可行，也不直接评价真实建筑入口；但已经足以拒绝把当前
`EfficientLoFTR core geometry + historical DINOv2-S four-gate identity` 自动推进到 Stage A v2、tracker、memory、SAM2、
SLAM 或 App。没有自动 successor。

不得在这 27 cases 上降低 8-inlier、0.50 ratio、DINO AND gate，换 DINOv3/RoMa，改 crop/context 或挑选更漂亮的
case 来救结果。任何新 representation 都必须是另行授权、带新问题边界的 successor，而不是 P1-W2 rerun。

## 5. Evidence identity

```text
runner source
  22ef3c41164c399f44397fedd0c589697fb262c3d8208c5eec5479f53bbf3b0d
adjudicator source
  d617b890e462a5a1ce8e69b7275567299bc76a18047a1cee390facc55d56f2f0
provider_output.json
  fd1395799dfa914ce9d05ed759b8a912e1fb71a768a5b6bdf38d7fd7606c426a
provider_execution_receipt.json
  4813622a43c12492a1634576b65bcfea2549c53e47da57f17fdd8cb08aebfb4f
result.json
  8c63ce2749d32054dae7158c458a17bb2380c65c91e9bbca993e09abce76e104
result_receipt.json
  a5ca94f3bb2ed5f532e099c4dddc6d860d55cc1b2a6087847b677939b57e5347
```

权威本地目录：

```text
artifacts.local/evidence/p1_w2_anchor_interface_v1/single_execution_v1/
```

Provider/output hashes、truth firewall、66/66 progress、result bindings 与所有 finite 数值均已校验；独立 adjudication
replay 产生完全相同的 `result.json` hash。没有运行中的 task-owned process 或 disposable replay 文件。
