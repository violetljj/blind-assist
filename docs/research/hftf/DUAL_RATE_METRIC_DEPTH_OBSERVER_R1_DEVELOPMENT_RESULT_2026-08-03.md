# Dual-rate metric depth observer R1 Development result

日期：2026-08-03

终态：

- `R1_DEVELOPMENT_TASK_GATES_NOT_SUPPORTED`
- `R1_PHONE_SHARED_HTP_FEASIBILITY_NOT_SUPPORTED`
- `R1_FULL_PROMOTION_NOT_EVALUABLE`

决策：`D_NOT_SUPPORTED_ON_CONSUMED_DEVELOPMENT_HOLD_TOF_DECISION`

## 结论

冻结的 D 臂没有通过。异步稳健仿射校正把已知帧上的 clearance MAE 降到
`0.15241 m`，但包络一致率只有 `86.21%`、false-clear 为 `8.43%`，严格
fail-closed 后只有 `58/120 = 48.33%` 帧保持已知。它通过 2/5 个既有任务门，少于 C
臂的 3/5；不能作为 C 的改进版，更不能替代 ToF。

手机资源审计同样为负：使用已经实测的 Metric3D HTP `1500.794 ms` 和共享加速器
边界，120 个 DA 帧中 112 个落入 Metric3D busy 区间，中断比例 `93.33%`；锚点首次
完成时 source-age P95 已为 `1.500794 s`，超过冻结的 `1.0 s` 有效期。当前手机实现
边界不支持这个在线双模型方案。

这不是“必须购买 ToF”的证明。fresh session-disjoint 数据、最终外接摄像头、共驻峰值
内存和持续温度都没有评价，所以采购决策保持 `HOLD`。准确停止范围是：不在这批已消费
结果上继续调整 period、TTL、拟合器、斜率范围或 UNKNOWN 规则来救当前 D。

## 四臂同口径结果

| 臂 | 已知/paired | Clearance MAE | 包络一致率 | False-clear | Temporal delta MAE | 稳态均值 / P95 | 通过门 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| A DA V2 | 119/120 | 0.38014 m | 75.24% | 24.29% | 0.11261 m | 54.82 / 59.18 ms | 2/5 |
| B Metric3D | 119/120 | 0.11832 m | 93.45% | 4.03% | 0.09204 m | 142.65 / 145.97 ms | 5/5 |
| C period-5 offset | 118/120 | 0.16506 m | 88.62% | 6.97% | 0.12216 m | 79.41 / 198.59 ms | 3/5 |
| D async affine | 58/120 | 0.15241 m | 86.21% | 8.43% | 0.08543 m | 54.82 / 59.18 ms | 2/5 |

D 的延迟列是独立 CUDA worker 假设下的 DA 输出延迟，Metric3D 完成不阻塞 DA；它不是
capture-to-clearance 端到端延迟。B/C 的旧报告计时也主要覆盖推理，不足以替代真实采集
链路测量。

## D 臂失败分解

因果约束本身通过：24 个 anchor receipt 全部只在 completion timestamp 后使用，
causality violation 为 0；已知输出的 anchor source-age P95 为 `0.60415 s`，也通过
1 秒门。失败来自校正可识别性和包络决策，而不是 PC 锚点单纯太慢：

- `UNKNOWN_ANCHOR_STARTUP`: 8 帧；
- `UNKNOWN_INSUFFICIENT_FIT_PAIRS`: 10 帧；
- `UNKNOWN_SLOPE_OUT_OF_BOUNDS`: 18 帧；
- `UNKNOWN_FAST_BAND`: 26 帧。

已知帧 MAE 的改善不能抵消 51.67% UNKNOWN，也不能抵消中心带 false-clear
`15.52%`。低 coverage 下只看已知对会产生选择性乐观，因此 R1 以 paired-valid 门明确
判负。

现有 A0 `false-clear` 的分母是所有 known `band × horizon` decision，不是仅对 truth
occupied 条件化的漏检率；本结果保持旧口径以保证四臂可比。后续 fresh 合同应同时冻结
并报告 conditional false-clear、session/scene paired delta、age bins 和 worst-session，
但不得把这些事后诊断用于重判本次终局。

## 证据边界

四臂只使用四个已消费 TUM RGB-D 窗口。sensor comparator 是 registered RGB-D
派生的 clearance proxy，不是人工通行真值。它们足以否定当前窄实现的 Development
门，不足以证明 RGB 共模误差已解决或 ToF 已不必要。

R1 只实现了固定仿射参数向后传播，没有把光流、遮挡、新区域融合混入同一候选。稠密
Metric3D depth-map flow propagation 若继续，必须作为新候选，先冻结 flow/checkpoint、
occlusion/reacquisition、new-region、TTL 和独立数据角色；不能在本 cohort 上追加后
重新挑最好结果。

手机侧只做资源时序审计。已测手机 DA 资产是 relative-only Qualcomm checkpoint，
不是 PC 质量臂的 Hypersim metric checkpoint，因此不允许把 PC 质量和手机时序拼成一个
已验证端侧系统。

## 可复现性

协议 SHA-256：
`BABF828028D8C7C4200724C94184D7975E67CA85F0DD08E17984096B94BBDC9B`。
执行器 SHA-256：
`B80D33510C4CEF2E7E59EE05592D69F99425D66EEDA64E0CFDEE0C0B3E27CD57`。
机器汇总见
[DUAL_RATE_METRIC_DEPTH_OBSERVER_R1_DEVELOPMENT_RESULT_2026-08-03.json](DUAL_RATE_METRIC_DEPTH_OBSERVER_R1_DEVELOPMENT_RESULT_2026-08-03.json)。

执行命令：

```powershell
E:\codex-tools\bin\blindassist-python.cmd -m unittest discover `
  -s scripts/research/dual_rate_metric_depth_observer_r1 `
  -p "test_*.py" -v

E:\codex-tools\bin\blindassist-python.cmd `
  scripts/research/dual_rate_metric_depth_observer_r1/evaluate_r1.py `
  --output artifacts.local/evidence/hftf/dual-rate-metric-depth-observer-r1/result-hash-bound-final.json `
  --trace-output artifacts.local/evidence/hftf/dual-rate-metric-depth-observer-r1/trace-hash-bound-final.json
```

第一次不可覆盖输出 `result.json` 保留，SHA-256 为
`78ED0E09F3B3ABC94B90A58D136FFBC384F857F64B13544CC1A2C039C0A54576`。最终复核只修复
A/B 稳态延迟的汇总显示、把未测 D 内存从 `0` 改为 `null`，并清理静态检查；D 的算法
与指标逐字段一致。最终实现另加一项回归测试，保证 invalid fast field 不会被 valid fit
误标为 `VALID`；当前 cohort 的结果仍逐字段一致。最终 ignored summary SHA-256 为
`E182258083BFEFB947CD96461E3757018752B32A0C09EE51B97588DDEC131557`，trace SHA-256
为 `256DDEB26E94DBCD3AC5E2BD6B2AE0AC7571D05FE8B45F1F47876FDF3D186C41`。

## 下一动作

当前最合理的次序不是在旧 outcome 上救 D，也不是直接宣布购买 ToF：

1. 保留 B 作为 PC/离线教师，停止当前 D 的 consumed 调参；
2. 若要继续纯 RGB，另立“离线教师蒸馏小型 calibration head”或完整稠密 flow 传播协议，
   使用新的 session/parent-disjoint 数据；
3. 只有新候选先过 fresh 任务门，才值得做共驻内存、长时间温度和最终摄像头实测；
4. 若独立新数据仍显示视觉共模或时效失败，再把 VL53L5CX 作为预冻结 E 臂与 D 同场比较。
