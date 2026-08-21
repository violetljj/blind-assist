# P1-A2 fixed-reference dense-identity validity protocol V1

状态：`FROZEN_BEFORE_PRIVATE_TRUTH / EXECUTION_NOT_STARTED`。

## 唯一研究问题

在 P1-R0 的 frozen sparse-LK candidate box 完全不变时，固定初始 physical-target crop 的 dense visual
correspondence，能否在 egocentric background drift 形成长期 wrong lock 前让系统主动失信？这是 materially different
validity representation，不是 A1 optical-flow health threshold 的续扫。

```text
frozen P1-R0 candidate bbox
  -> NEW fixed-reference dense identity validator
       VALID     -> 原 candidate
       UNCERTAIN -> null
       INVALID   -> null
  -> frozen P1 state machine
  -> frozen evaluator + private truth
```

Validator 不能生成、移动或扩大 bbox，不能搜索 bbox 外画面，不能 reacquire，不能读 post-init GT，不能更新
target memory，也不能改变 candidate generator、P1 state machine 或 evaluator。数据仍为已消费 P1-D0 15 episodes /
1,724 frames；本轮没有 Sky、fresh data、CoTracker/TAPIR、Android、训练或产品改动。

## 冻结 representation

唯一 encoder 为 `facebook/dinov2-small` revision
`ed25f3a31f01632728cabb09d1542f84ab7b0056`（Apache-2.0）：

```text
model.safetensors       AE1E99FCEFD534ED978CDEB8326F08030C96E28B7A81FFCBC98A857C84D14BE1
config.json             1809F83E3BDB1609A501A610AD4A742F4FD8AE44D72CA4AA0DF52D1F2AC8628D
preprocessor_config.json 14E780D86FA1861F8751F868D7F45425B5FEB55C38CA26F152CA5097AB30F828
```

初始 oracle bbox crop 只编码一次：精确 crop、直接 resize 到 `224×224`、ImageNet mean/std，取冻结 last hidden
layer 的 `16×16×384` patch tokens，逐 patch L2 normalize 后成为永久 `TargetIdentityMemory`。每个当前
`sparse_lk_flow` bbox 用完全相同的 crop/encoder 合同编码。memory update 次数必须为 0。

初始 256 patches 与当前 256 patches 建立全 cosine matrix，再取 mutual nearest correspondences。每帧固定报告：

- `anchor_match_fraction`；
- mutual match 的 median `match_confidence`；
- normalized patch coordinates 上 partial-affine RANSAC 的 `spatial_consistency`；
- 初始 4×4 coarse cells 的 `anchor_coverage`；
- affine residual 的 `correspondence_dispersion`；
- 仅诊断的 global mean-token cosine。

Policy 必须同时使用前四个量，不能退化为单 cosine gate。

## 一次性极小 development sweep

在 private truth 打开前，从全部 consumed RGB flow candidates 的每个 policy feature 分布固定产生
`q20/q35/q50/q65/q80`。唯一 policy family 是四个 `>=` predicate 的 AND，共 `5^4 = 625` 个；无单特征、双特征、
OR、classifier、layer/encoder/patch/geometry sweep、二轮 threshold rescue 或 learned head。

四项全过为 `VALID`，恰有一项不过为 `UNCERTAIN`，其余为 `INVALID`；后两者在 evaluator 输入都删除当前
candidate。Frame-0 oracle initialization 保留且只负责创建 fixed memory。

## 冻结 admission gate 与诊断

Frozen reference 仍为 P1-R0：`correct=87`、`wrong=1,221`、max wrong-lock `8,498 ms`。一个 policy 只有同时满足：

```text
correct assertion retention >= 90%       (integer result >= 79/87)
wrong assertion reduction   >= 60%       (integer result <= 488/1,221)
max wrong-lock reduction    >= 60%       (duration <= 3,399 ms)
post-init GT reads = 0
candidate generator unchanged
online target-memory updates = 0
global search / added reacquisition = 0
```

才建立 representation signal。仍报告 background/other-instance wrong、episode-macro reduction、coverage、false loss、
switches 与 frozen evaluator 全量结果，不生成加权总分。

新增 `pre_drift_warning_lead_ms = first GT-wrong assertion timestamp - first validator UNCERTAIN/INVALID timestamp`。
每 episode 从首次 wrong 前最后一个 correct assertion 的下一帧开始寻找 warning；正值表示提前失信，负值表示晚报，
无 warning 为 null。这只是 private-truth 后诊断，不参与 policy 排名或准入。

Policy 排名依次最大化 aggregate wrong reduction、wrong-lock reduction、episode-macro wrong reduction、correct
retention，最后按 canonical predicate 稳定排序。

## 三种穷尽终态

1. 至少一个 policy 通过全部 gate：`DENSE_IDENTITY_VALIDITY_SIGNAL_ESTABLISHED / NO_POLICY_ADMISSION /
   NO_SCIENTIFIC_VERDICT`；下一步才可设计 loss/reacquisition。
2. 没有 policy 通过 retention，但至少一个同时通过两项 60% safety reduction：
   `DENSE_IDENTITY_GAIN_ONLY_BY_ABSTENTION / NO_POLICY_ADMISSION / NO_SCIENTIFIC_VERDICT`。
3. 其余：`DENSE_IDENTITY_NOT_SUFFICIENT / NO_POLICY_ADMISSION / NO_SCIENTIFIC_VERDICT`；这才授权另立 materially
   stronger temporal-correspondence representation 设计（CoTracker/TAPIR class），但不在本轮执行。

任何终态都不授权保留 discovered threshold、修改 App、声称 safety/产品改善或运行 Sky。
