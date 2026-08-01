# HFTF Stage C foot-ground student canary result E0.1

日期：2026-08-01

终态：`E0_1_FOOT_GROUND_STUDENT_CANARY_NOT_EVALUABLE`

## 1. 结论

E0.1 在 student training 前按 role-opportunity 门停止。新 dev/heldout 的 exact
transport、ground plane、history-speed、`.4 s` known coverage、known loss 与
UNKNOWN 防火墙全部通过；dev 也有 4 个 risk cells/4 anchors。heldout 只有 1 个
risk cell/1 anchor，低于冻结的 2/2，不能把单一正格当作可评价学生分母。

这不是 student 负结果，因为没有生成完整 corpus、训练模型或读取 student output。
不得降低 heldout 2/2 门、换一条已知有更多 risk 的 source，或在该 heldout 上调参。

## 2. 报告绑定

- source lock SHA-256：
  `03035bbe59aa7e8a48114c9df1925675fa484acce5c24ac3e51ddeb14f0a5ff7`
- acquisition SHA-256：
  `5ba6c3e98d88741b380644a43b97cb81ae4b30fc79c3ad45faedafc956b4696b`
- transport SHA-256：
  `a75a5b18d4d325b80617bdca83179c870200f5b22156fa3fc6b345a9f42a5173`
- teacher-opportunity report：
  `artifacts.local/evidence/hftf/stage-c-e0-1-teacher-opportunity-20260801/teacher_opportunity.json`
- teacher-opportunity SHA-256：
  `44240751e577dff8ae1ad55cc4263e143cf6d2762a68f61430c5226837d22e99`

所有正式 runner 均第二遍 byte-exact；`.8 s` output 明确未计算。

## 3. Fresh source metrics

| role | source | anchors | known `.4 s` | risk cells | risk anchors | directions | known no-risk |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| dev | `2024_12_01...` | 167 | `.9329` | 4 | 4 | 3 | 775 |
| heldout | `2024_07_10...` | 173 | `.8312` | 1 | 1 | 1 | 718 |

两条 plane known 与 speed eligible 均为 1.0，candidate known loss 与
UNKNOWN→SAFE 均为 0。heldout blocker 精确是 risk-positive support，不是 geometry
transport 或 known coverage。

## 4. Successor 边界

唯一允许的 successor 是一次性、有限预算的 multi-source E0.2 evaluation
qualification：

- 不再逐条抽样直到某条通过；
- 在任何新媒体前一次锁定 3 dev + 3 heldout，且 recording date 与全部 consumed
  dates 互斥；
- 角色按 metadata 排序位置交替分配；
- model/training/threshold/success margins 保持 E0.1 不变；
- 若固定六条仍不能形成 role-level opportunity，则关闭该 student canary source
  route，不继续扩大。

当前不授权 corpus、training、完整 HFTF、主线、App 或安全 claim。
