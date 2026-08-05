# DA V2 模型变体门 R0 canonical baseline 结果

日期：2026-08-05

终态：`P1_ENGINEERING_GATE_READY / CANONICAL_STANDALONE_TASK_NOT_SUPPORTED`

## 结论

冻结 roster、TUM 注册深度和 canonical `518x686 FP16` aligned-depth cache 后，独立重算
成功。canonical 对自身的 11 项工程非劣化门全部通过，几何状态和 transition agreement
均为 100%；因此 P1 执行器具备接收等价候选的基本自检。

同一 baseline 的绝对 task 门只通过 paired-valid 与 temporal 两项，clearance、collision
agreement、false-clear 三项失败。它可以作为 P2 的高频相对结构 canonical，却不能独立承担
米制 clearance。后续即使真机 P95 大幅下降，也不得用性能覆盖这三个失败。

## Baseline

| 指标 | 结果 | 解释 |
|---|---:|---|
| 每帧 raw metric AbsRel 中位数 | 29.43% | 直接对注册 RGB-D，不做真值尺度拟合 |
| 每帧 scale-aligned AbsRel 中位数 | 8.33% | 只衡量相对结构；一次乘性 oracle scale 不可部署 |
| raw metric AbsRel frame P95 | 42.65% | 固定 120 帧 |
| scale-aligned AbsRel frame P95 | 13.10% | 固定 120 帧 |
| sensor-valid 上地面恢复成功率 | 100% | 不等于 status truth 全一致 |
| sensor/canonical status exact agreement | 99.17% | 1 帧 sensor `UNKNOWN_GROUND`、canonical `VALID` |
| paired-valid | 119/120 = 99.17% | 绝对门通过 |
| clearance MAE | 0.38042 m | `>0.25 m`，失败 |
| collision agreement | 75.27% | `<90%`，失败 |
| false-clear / 全部 known decisions | 203/837 = 24.25% | `>5%`，失败 |
| false-clear / truth occupied | 203/325 = 62.46% | 诊断分母，不替代冻结主门 |
| temporal clearance-delta MAE | 0.11313 m | `<=0.15 m`，通过 |

这里的 24.25% 来自 FP16 dense cache 对恢复后的官方 TUM archive 重新计算；与早期报告
24.29% 的微小差异不改变终态。当前结果没有打开任何新候选，也没有据此修改阈值。

## 典型失败集

固定 12 帧索引为 `13, 30, 68, 70, 76, 85, 89, 93, 100, 107, 116, 119`，覆盖：

- walking_static 与 walking_xyz 的 false-clear clusters；
- 最大 clearance 与 raw metric-depth 误差；
- sensor `UNKNOWN_GROUND` / canonical `VALID` 状态不一致；
- walking_xyz 尾部时序。

它们是强制输出的 failure atlas，不允许候选运行后替换为更好看的帧。

## 可复现证据

- protocol SHA-256：`7AD758A829DB3AA07EC295F31090DDF8A8B1E7E6D6943B86C2A25FE063EBC664`；
- roster SHA-256：`62D41DBFCA64BBD7964B146427911A384A8E0CE57FF606F9BA74682E380D43F2`；
- canonical aligned-depth SHA-256：
  `9A7FC55DB6B3E7C467B5BAFE68D3603F4B463C498558B7236D603569595D3A34`；
- machine result SHA-256：`9FBF26296AB0FA9EBCEEC9725C209B3B66D3B59DB60BB3F7F195420E7BEBCAEB`；
- TUM walking_static / walking_xyz archive SHA-256：
  `8ACA832BF01746AA95FCC0B15BF42D78AD3BB8EDEF672BA4CC94A404B194F6C1` /
  `1459E9488AC0E61A2EC80DFBC35CFB77942F6D8EABDED1C8D26A70BE650D0E1D`。

机器可读结果为
`DAV2_MODEL_VARIANT_VALIDATION_R0_BASELINE_RESULT_2026-08-05.json`。

## P2 放行规则

候选先完整物化并锁定深度 cache，再一次性运行质量门。工程非劣化未过则不进入真机深
profile；工程门通过但绝对 task 门未过，只能作为 student/相对结构/disagreement 支线。
同设备固定 APK 端到端 P95 至少 `1.15x` speedup 才值得保留。最终独立米制 authority 仍需
新 final-camera、session/parent-disjoint 真值集。
