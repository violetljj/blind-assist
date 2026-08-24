# Spatial-Layout Identity Verification V0 Protocol

状态：`FROZEN_BEFORE_DATA_DOWNLOAD_OR_MODEL_EXECUTION / DEVELOPMENT / EXECUTION_AUTHORIZED / NO_OUTCOME`

## 唯一问题与 claim ceiling

`SPATIAL_LAYOUT_IDENTITY_VERIFICATION_V0` 只回答：在全新公开 source、全新 physical instances、固定单参考、固定候选
与相同冻结 DINOv2-S patch tokens 上，一个不训练的显式 spatial-layout scorer，能否比现有双向 mean-nearest
appearance scorer 更可靠地区分同一实体与同类不同实体。

本实验不回答绝对分数能否跨域标定，不包含 target-absent、阈值、`NONE`、risk--coverage、NearID、PDM、fusion、
Deep Sets、multiple references、主动观测、tracker、P1、控制、安全或 App。即使通过，最高结论仍是受控公开数据上的
Development identity-ranking signal，不是开放世界 physical-instance authority。

## 冻结数据与 roster

唯一数据源为 University of Washington 官方 RGB-D Object Dataset evaluation set：

```text
https://rgbd-dataset.cs.washington.edu/dataset/rgbd-dataset_eval/rgbd-dataset_eval.zip
Content-Length: 673456874
```

该数据含 300 个具名 physical instances、51 个 category；每个 instance 含 3 条不同相机高度的 turntable video。
只读取 `*_crop.png` RGB，不读取 depth、mask、pose 或官方 category-recognition split。archive 下载完成后先记录
SHA-256，再只读取 ZIP central directory，以如下规则冻结 roster，此前不得解码像素：

1. 按 category 名与 instance 数值排序；只接纳同时具有 video `1` 和 `3`、且 category 至少有 2 个合法 instance 的实体。
2. 每个 target 的 hard negative 是同 category 排序中的下一个 physical instance，末项循环至首项。
3. reference 固定为 target video `1` 的 `q=.50` 帧；candidate 固定为 video `3` 的 `q={.25,.50,.75}` 帧。
4. 帧按文件名中的数值 frame 排序，以 `floor(q*(n-1))` 取样；target 与 hard candidate 共用 video 与 quantile。
5. candidate slot 由 `SHA256(pair_id)` 最低位固定；任何缺帧、重复 member、instance/category 解析冲突均 fail closed，
   不得换图补齐。

数据源、physical instances 和 capture videos 均未进入此前 SUN3D、T-LESS、CORe50 identity experiments。这里的
`source-disjoint` 指本实验相对既有 identity cohorts 的新 dataset，以及每一 pair 内 video 1 reference 对 video 3
candidate 的 capture-source separation；它不代表自然场景、跨设备或跨域 Confirmation。

## 冻结两臂

两臂共享 `facebook/dinov2-small@ed25f3a31f01632728cabb09d1542f84ab7b0056`、224×224 ImageNet
normalization、最后层 16×16×384 L2-normalized patch tokens、原始 cropped RGB 和独立 unary candidate scoring。

Baseline 是现有双向 mean-nearest patch cosine：reference→candidate 与 candidate→reference 的平均值再取均值。

Layout arm 不训练参数，也不读取 category、instance、slot、另一个 candidate 或 outcome。对每个 unary pair：

1. 计算 256×256 patch cosine；仅以此建立双向 nearest 与 reciprocal mutual matches。
2. reciprocal matches 按 cosine、reference index、candidate index 的固定顺序，保留前 `K=64`。
3. 在 4×4 coarse cells 上计算双侧 coverage；计算双侧 convex-hull dispersion。
4. 计算双向 4-nearest→8-nearest local-neighborhood preservation。
5. 分别归一化两侧坐标后计算 orthogonal Procrustes median residual，并以 `exp(-residual/.5)` 转为一致性。
6. 从全部双向 nearest 的唯一目标数量计算 collision/conflict consistency；mutual support 截断于 `64/256`。
7. 六个 `[0,1]` 结构项取等权 geometric mean。最终 `score(R,C)` 是显式 forward/reverse 两方向结构分数的保守最小值。

appearance cosine 只决定 mutual-match 图和固定 top-K 次序，不作为 layout composite 的独立加权项。不得在 outcome 后修改
K、cell grid、邻域大小、residual scale、结构项、权重或聚合。

## Paired 判决与晋级门

每个 pair 都严格比较 target unary score 与 hard-negative unary score；相等记为未正确 outrank。不存在接受阈值或
`NONE`。逐 pair 保存两臂 raw scores、margin 与 layout diagnostics，truth 只在所有 raw unary scores 写入后用于 adjudication。

Challenger 晋级必须同时满足：

- paired `rescue > collateral`；
- baseline-correct control retention `>=80%`；
- `score(R,C)=score(C,R)` 在绝对误差 `<=1e-9` 上为 `100%`；
- candidate permutation invariance `=100%`；
- stable same-class distractor stratum 至少包含 5 个 target instances；该 stratum 定义为同一 reference 对 3 个冻结
  candidate quantile 的 baseline 全部错误，layout 在这些 pairs 上 target outrank `>=50%`。

报告还必须给出两臂 overall target-outrank、rescue/collateral、control retention、stable stratum、按 category 与 quantile
的最坏组、direction residual，以及完整 raw diagnostics。任何一门失败均不晋级；不得以 subgroup、threshold、fusion、
新 backbone 或另一 layout 公式 rescue。

## 单次、恢复与终态

正式 run 必须绑定协议文件、实现、roster、archive、model files 的 SHA。roster 冻结后可在相同 hashes 下恢复确定性
feature extraction；`raw-scores.json` 与 `final-report.json` 原子写入。final report 存在后入口必须拒绝覆盖、重跑或
另开同 roster 的第二次判决。无外部模型调用。

合法科学终态只有：

```text
SPATIAL_LAYOUT_IDENTITY_SIGNAL_SUPPORTED_DEVELOPMENT
SPATIAL_LAYOUT_IDENTITY_MIXED_WITH_COLLATERAL_DEVELOPMENT
SPATIAL_LAYOUT_IDENTITY_NOT_SUPPORTED_DEVELOPMENT
SPATIAL_LAYOUT_IDENTITY_NOT_EVALUABLE
```

无论终态如何，都保持 `OPEN_SET_CALIBRATION_NOT_RUN / RELIABLE_VERIFIER_NOT_ESTABLISHED / NO_P1 /
DEFAULT_APP_UNCHANGED`。若 layout 在 fresh roster 上仍产生大量 collateral，则该结果是停止被动单参考 RGB verifier
mechanism zoo、转向主动 distinctive evidence 或改变输入合同的依据，而不是继续换 backbone 的授权。
