# HFTF Stage C SANPO cross-split body/head temporal-student canary F0.1

日期：2026-08-01

状态：`FROZEN_BEFORE_F0_1_SOURCE_OUTCOME`

## 1. 为什么在执行前替代 F0

F0 已冻结了 body/head field、`.4 s` causal teacher、UNKNOWN、三臂模型、训练参数与
effect margins，但原 source design 把 train/dev/heldout 都切自 official train split。

在没有下载 F0 media、没有计算 geometry label、没有生成 corpus、更没有训练或打开
student outcome 时，独立只读审计确认 official SANPO-Synthetic test split 也提供相同
RGB/mask/depth/pose source schema。为降低 same-split 乐观偏差，F0.1 将 heldout
提升为 official test split。这个变化基于 source metadata 与 split identity，不基于
几何或模型结果。

F0 原合同与 planner 实现保留为审计记录；F0 same-split heldout 不执行。F0.1 继承
F0 除 source split/role 外的全部数值、teacher、模型和 stop rules。

## 2. 固定 source roles

train/dev pool：

- official train generation：`1692794964120907`；
- split SHA-256：
  `f9c5dc4c289fa87342abc0d2cc49f112fcc78c7e02e0b6b081e296a99344173c`；
- 排除 60 个 effective burned sessions；
- 按完整 session ID 字典序选前 9 个 metadata-eligible sessions；
- rank 1–6 为 train，7–9 为 dev。

heldout pool：

- official test generation：`1692794964058506`；
- split SHA-256：
  `0f701db54d2cc26b32bf2c636537a1353beb5d7e09f8914279cde2e7c06400df`；
- exact 401 个 session IDs；
- 按 official split 文件顺序选前 3 个 metadata-eligible sessions；
- 排除 burn union 与已选 train/dev；
- 三个全部固定为 heldout。

所有 role 以 parent session 为单位互斥。任何 geometry/student outcome 打开后都不得
换样。F0 same-split planner 的前九个 train/dev metadata candidates 可被 F0.1 精确
复用；其原 heldout ranks 不成为评价 source。

## 3. Test split importer 边界

acquisition 必须显式接收并记录实际 `train/test` split，绑定相应 split object 的
generation/hash。test media 只能用于 ordered heldout evaluation，不能进入
pretraining、dev checkpoint/threshold selection 或 augmentation statistics。

禁止通过删除原 importer 的 train check 来绕过来源身份；必须扩展为受枚举约束、
hash-bound 的 split-aware importer，并增加回归测试。

## 4. 其余合同完全不变

以下全部精确继承 F0：

- body/head × current/`.4 s` × 6 directions × 6 distances；
- physical-time history `[-.8,-.6,-.4,-.2,0] s`；
- history-only causal origin、anchor-yaw orientation；
- future pose 只重投影 observation，future modalities 不进入 student；
- stride-8 candidate teacher 与 disjoint stride-4 reference；
- UNKNOWN 防火墙及 12/12 pre-training opportunity gates；
- `SF_CURRENT / SF_FUTURE / HIST_FUTURE` 同结构三臂；
- 三 seeds、训练参数、`.5` threshold；
- heldout F1/recall/FPR/per-height/worst-source success margins。

成功仍只支持 SANPO-Synthetic body/head temporal geometry-proxy signal。foot-ground、
完整 HFTF、自然或人类事件、主线、Android/App、生产与安全 claim 均保持未授权。
