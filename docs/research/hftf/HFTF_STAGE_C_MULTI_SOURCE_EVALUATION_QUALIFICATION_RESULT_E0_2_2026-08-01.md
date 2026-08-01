# HFTF Stage C multi-source evaluation qualification result E0.2

日期：2026-08-01

终态：`E0_2_FIXED_BATCH_TEACHER_MECHANICS_NOT_EVALUABLE`

## 1. 结论

固定 3 dev + 3 heldout batch 的 exact acquisition、transport、role risk/no-risk
opportunity 与 determinism 均通过，但 3/6 source 未达到每 source `.4 s`
candidate known fraction `.70`；其中一条同时未达到 plane-known `.95`。

按预冻结 stop rule，EgoWalk foot-ground student source route 在 student training 前
关闭。不降低门、不删除失败 source、不继续扩大 inventory，也不把 role-level
opportunity 通过冒充 teacher mechanics 通过。

## 2. 报告绑定

- report：
  `artifacts.local/evidence/hftf/stage-c-e0-2-fixed-batch-qualification-20260801/qualification.json`
- SHA-256：
  `a58aff72e0207871ef80d9aa6f94bc9ef7db21ba08d15e7405436b0a60558eee`
- protocol commit：`6b1d76b`
- runner commit：`09d036f`

固定媒体 1,232,000,737 bytes 全部 size/hash 匹配，六条 source 永久 burned。完整
transport 通过，第二遍 teacher payload byte-exact；`.8 s` output 未计算。

## 3. Source mechanics

| role | source | anchors | plane known | known `.4 s` | risk cells | result |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| dev | `2024_12_15...` | 148 | `.9088` | `.3257` | 16 | plane + known fail |
| heldout | `2024_09_28...` | 219 | 1.000 | `.9854` | 0 | pass |
| dev | `2024_11_07...` | 171 | `.9708` | `.6515` | 9 | known fail |
| heldout | `2024_11_13...` | 153 | `.9935` | `.7346` | 15 | pass |
| dev | `2024_08_16...` | 165 | 1.000 | `.9006` | 10 | pass |
| heldout | `2024_12_28...` | 190 | 1.000 | `.5000` | 22 | known fail |

所有 source history-speed eligible fraction 为 1.0。失败不是 motion eligibility 或
media transport，而是 D0/D1 near-ground observation support 跨自然 source 不稳定。

## 4. Role opportunity

| role | risk cells | physical anchors | risk sources | directions | no-risk cells |
| --- | ---: | ---: | ---: | ---: | ---: |
| dev | 35 | 32 | 3/3 | 5 | 1,506 |
| heldout | 37 | 32 | 2/3 | 5 | 2,079 |

role opportunity 全部大幅过门，但不能覆盖 source mechanics failure。risk cells 仍是
geometry proxy，不是真实危险或 prevalence。

## 5. 下一路线

该结果只关闭 `EgoWalk + current D0/D1 foot-ground reader + RGB student` source
version，不关闭 HFTF。下一候选不再筛 EgoWalk，而应回到 R4 已有强 reference 支持且
同时具备 RGB/depth/pose 的 SANPO obstacle/body/head source：

- 先冻结未消费 SANPO sessions 与 body/head temporal teacher/student canary；
- 保留 `.4 s` causal-origin、history-RGB-only 与 UNKNOWN 防火墙；
- foot-ground 在找到具备自然 elevation truth + RGB 的新来源前保持未评价；
- SANPO student 成功也只支持 body/head obstacle branch，不能冒充完整 HFTF。

当前没有 student corpus/training/output、研究主线、Android/App 或安全 claim。
