# GRAIL-R1C-O Owner-Local Canonical Coordinate Protocol

日期：2026-08-25（Asia/Hong_Kong）

状态：`FROZEN_BEFORE_R1C_O_OUTCOME / PROJECT_CONSUMED_DEVELOPMENT / PRIVILEGED_NATIVE_OWNER_FRAME_ONLY / R1C_V_UNOPENED / FORMAL_TEST_UNOPENED / STOP_BEFORE_M2 / DEFAULT_APP_UNCHANGED`

## 唯一问题

R1B 已把 reference target owner-group exact 提至 `74/78`，但 privileged query/reference image-space ordinal 仅 `54/78` 一致。R1C-O 只检验：将 sibling position 从 camera/image frame 换成 ProcTHOR/AI2-THOR source-native owner frame 后，R0 relational uplift 能否恢复。

## 冻结合同

- cohort、78-case denominator、candidate set、M1 V2b checkpoint、local-match appearance score、collision tiebreak、pose head、threshold=`0.9353410602`、negative pairing 和 evaluator 全部不变。
- owner 依次取 AI2-THOR native component prefix（`owner___part`）、native `parentReceptacles`、standalone self；不得读取 camera pose、RGB、bbox、mask 或 outcome 来定轴。
- frame 使用 owner native yaw 的逆变换，得到 `(right, up, front)`；label 使用同一 native owner、同一 semantic type 的全部 runtime siblings，而非当前视角可见集合。
- horizontal 为 local-right 的 `LEFT/CENTER/RIGHT` 三等分 rank；vertical 为 negative-local-up 的 `TOP/MIDDLE/BOTTOM` 三等分 rank。`front` 只审计，不进入本轮 selector。
- owner/position/yaw 缺失时标为 `NOT_EVALUABLE`；不得用 camera/world axis 或样本顺序补值。
- selector 只使用 R1 已确定的最小字段：`semantic type + canonical sibling ordinal + nearest stable object type`。不改变 nearby-type 信息源。

## 预注册门

R1C-O 只有同时满足以下条件才允许另立 R1C-V：

- canonical target evaluable=`78/78`，query/reference canonical label agreement=`78/78`；
- referent `>=70/78`；complete `>=50/78`；
- wrong-target `<=1/43`；absence false commit `<=1/78`；
- candidate permutation=`156/156`。

任一失败即终止于 `GRAIL_R1C_O_CANONICAL_COORDINATE_CEILING_NOT_ESTABLISHED_STOP`，不得训练 orientation student、调 matcher/threshold、融合 R1B 或启动 M2/formal test。通过只能建立 synthetic privileged-coordinate mechanism ceiling；R1C-V 仍须另立协议。

