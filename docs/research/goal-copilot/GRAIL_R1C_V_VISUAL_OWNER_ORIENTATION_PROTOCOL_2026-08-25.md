# GRAIL-R1C-V Visual Owner Orientation Protocol

日期：2026-08-25（Asia/Hong_Kong）

状态：`FROZEN_BEFORE_IMPLEMENTATION_AND_OUTCOME / PROJECT_CONSUMED_DEVELOPMENT / DETERMINISTIC_RGB_PROPOSAL_PROBE / AXIS_AND_SIGN_FACTORIZED / R1C_V_FINAL_SINGLE_OBTAINABLE_ARM / FORMAL_TEST_UNOPENED / STOP_BEFORE_M2 / DEFAULT_APP_UNCHANGED`

## 研究问题与固定机制终态

> Ownership is largely observable, but image-frame relations are view-dependent. A source-native owner-centric frame restores the relational ceiling; the remaining question is whether its directed orientation is visually observable.

R1B 已建立 reference owner-group exact=`74/78`，但 image-frame privileged cross-view slot agreement=`54/78`。R1C-O 以 native owner-local coordinate 得到 agreement=`78/78`、referent=`75/78`、complete=`58/78`。R1C-V 只检验 directed owner orientation 的 RGB/proposal obtainability；不得重开 ownership、relation fields、slot quantization或下游。

## 输入防火墙

预测侧唯一允许输入：

```text
full-scene RGB
+ independently predicted owner/part proposal group
+ proposal bbox/mask when already present
+ semantic type
```

预测侧禁止：

```text
native yaw / native position / native owner coordinate
camera pose / world pose / depth truth
object ID or its lexical/component suffix
sample ordering
reference-query joint alignment or correspondence
R1C-O label, evaluator truth, outcome or metric
```

Native coordinate、object ID 与 projected axis/sign 只允许 evaluator 在预测结果持久化后使用。query/reference 必须各自独立预测；不得先看另一视角再消除 sign ambiguity。

## 冻结数据与下游

- 同一 consumed 78-case Development、43 个 same-type wrong-target cases、78 个 absence pairs；不打开 formal test。
- query 使用 frozen M1 V2b full-scene RGB/proposals/features；reference 使用 R1B full-scene RGB/proposals/masks/features。
- 两侧复用 R1A/R1B 的 frozen `predict_groups()`；不得调 affinity、group threshold、context crop 或 mask encoding。
- 3×3 slot 使用既有 `rank_bin` 三等分及 `LEFT/CENTER/RIGHT × TOP/MIDDLE/BOTTOM` 标签；不得调边界或加入 front axis。
- selector 字段固定为 `semantic type + sibling ordinal + nearest stable object type`；nearest stable type、appearance tiebreak、pose head、threshold=`0.9353410602`、negative pairing 与 evaluator均保持 R1C-O 不变。

## 冻结 deterministic estimator

每个视角、每个 predicted group 独立执行：

1. 以 proposal bbox center 作为 part center；group support region 为成员 bbox 的并集包围框，向四边各扩张一个 group width/height 的 `10%`，裁到图像边界。
2. **Estimated undirected axis**：对 group centers 做 2D PCA；有至少两个不同 center 时，选择两个特征向量中绝对 image-x 分量较大的向量作为 owner left-right axis。若 center covariance 不足，则使用 support region 的 image-horizontal axis。向量先确定为无向 line，不携带 canonical sign。
3. **Estimated sign**：将 support RGB 转为固定 luminance `0.299R+0.587G+0.114B`，用中央差分梯度幅值作为权重。沿无向 axis 计算相对 support center 的 normalized signed first moment。为消除 eigensolver 任意符号，axis 先规范为 image-x 为正（若 x 为零则 image-y 为正）。若 signed moment 的绝对值 `<0.05`，输出 `UNKNOWN/NOT_EVALUABLE`；否则 canonical right 指向较高 gradient-weighted moment 的一侧。不得为 UNKNOWN 补 camera/image-order sign。
4. vertical direction 固定为 image up；将成员 center 投影到 estimated directed right 与 image-down，继续使用冻结 3×3 rank，生成该视角 sibling slot。sign UNKNOWN 时，多 sibling group 的 horizontal slot 为 `NOT_EVALUABLE`；single-horizontal group 保留 `SINGLE`。

该 estimator 是一次性、无训练、无 outcome tuning 的最小视觉 falsifier。结果暴露后不得改变 `10%` support padding、PCA axis选择、`0.05` sign threshold、gradient定义或 UNKNOWN 规则。

## 三个 arm

| Arm | Axis | Sign | 作用与权限 |
|---|---|---|---|
| `AXIS_ONLY_DIAGNOSTIC` | estimated | oracle | 只归因 axis；oracle sign 不得形成 obtainability claim |
| `SIGN_ONLY_DIAGNOSTIC` | oracle | estimated | 只归因 sign；oracle axis 不得形成 obtainability claim |
| **`R1C_V_FINAL`** | **estimated** | **estimated** | 唯一 obtainable arm与唯一可进入成功门的结果 |

Evaluator oracle directed axis 由同一视角内 native local-right 与 proposal center 的固定最小二乘方向取得；native local-right 无方差或投影不可辨时标为 oracle-axis `NOT_EVALUABLE`。两个 diagnostic 不得选择参数、arm 或 successor。

## 指标顺序与分母

必须按以下顺序报告：

1. **Cross-view canonical slot agreement**：positive 78-case；任一侧 target slot UNKNOWN 计为不一致，不从分母删除。
2. referent / complete / wrong-target / absence。
3. 相对 R1C-O 的 uplift recovery。
4. axis accuracy、sign accuracy；同时报告 `axis-evaluable/78`、`sign-evaluable/78`、`sign accuracy | evaluable` 与 UNKNOWN count。
5. yaw/axis angular error，仅诊断；不得覆盖 slot 或端到端裁决。

还必须分 arm 报告 permutation consistency、R1B 23 个 view-disagreement failure 的 rescue，以及失败互斥归因：`OWNER_GROUP_OR_AXIS_UNKNOWN / AXIS_ERROR / SIGN_UNKNOWN / SIGN_FLIP / SLOT_COLLISION_OR_TIEBREAK`。

## 预注册成功门与停止条件

只有 `R1C_V_FINAL` 同时满足以下条件，才建立 deterministic visual orientation obtainability：

- cross-view canonical slot agreement `>=70/78`；
- referent `>=70/78`；complete `>=50/78`；
- wrong-target `<=1/43`；absence false commit `<=1/78`；
- candidate permutation=`156/156`；
- selector collateral=`0`，complete collateral=`0`。

axis/sign accuracy 不单独替代上述门。若 final 未通过，终态按 dominant attribution 选一项：

```text
GRAIL_R1C_V_AXIS_NOT_VISUALLY_OBTAINABLE_BY_DETERMINISTIC_PROBE_STOP
GRAIL_R1C_V_SIGN_NOT_VISUALLY_OBTAINABLE_BY_DETERMINISTIC_PROBE_STOP
GRAIL_R1C_V_SLOT_STABLE_BUT_DOWNSTREAM_CEILING_NOT_RECOVERED_STOP
```

若通过，终态为：

```text
GRAIL_R1C_V_DETERMINISTIC_VISUAL_OWNER_ORIENTATION_ESTABLISHED
```

无论通过或失败，本轮都不授权调 estimator、bin、matcher、threshold、fusion、pose head，不授权训练 relation/orientation network、formal test、M2、Android/default-App。任何 successor 必须另立结果后协议。

## Claim ceiling

本协议最高只允许 `PROJECT_CONSUMED_DEVELOPMENT_DETERMINISTIC_RGB_PROPOSAL_ORIENTATION_PROBE` synthetic ProcTHOR/AI2-THOR mechanism claim；不建立自然 RGB、跨数据集、学习、formal generalization、设备、产品或安全 authority。

