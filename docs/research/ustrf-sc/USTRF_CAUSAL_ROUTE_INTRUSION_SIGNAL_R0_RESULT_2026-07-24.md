# USTRF causal route-relative intrusion signal R0

日期：2026-07-24

终态：`SIGNAL_REJECT / VALID`

最大权限：`NEW_SIGNAL_SEPARABILITY_PROBE_ONLY`

## 结论

本轮第一次不再调整资格时长、TTL 或 renewal，而是加入新的候选无关测量变量：目标 bbox 相对 causal route UV 的径向接近、横向收敛与 bbox 尺度扩张趋势。结果没有形成可接受的 coverage/risk 组合，信号应直接淘汰，不能进入 successor policy 或 opener。

冻结信号为 `causal_route_relative_intrusion_trend_2_of_3`：同一 track/reset 必须连续 5 帧 route known 且 active relation 成立；对 route-relative radial distance、route-relative lateral distance、normalized log bbox height 分别取 5 帧全部 10 个 past-only pairwise slope 的中位数。径向斜率 `<0`、横向斜率 `<0`、尺度斜率 `>0` 各计一项，至少 2/3 才激活。窗口、组合与零阈值均在任何 signal output 前冻结，本轮没有事后扫描或调参。

完整 producer 先复核 C1/C2/C3 的 `123` 条 preoutput trace 在 bbox、active relation、route/reset 与时间上逐帧一致，再折叠为 `41` 条候选无关序列 / `62,229` 帧。signal inventory 冻结时 truth、event window、oracle 与负暴露解码数均为 0；第二进程复验全部 ledger SHA 后才做 post-hoc audit。

## 定量结果

| 指标 | 新信号 R0 | 冻结要求 | 结论 |
| --- | ---: | ---: | --- |
| supported unique event coverage | `7/11` | `11/11` | fail |
| supported candidate cell coverage | `21/33` | `33/33` | fail |
| full-sequence activation | `1,903` | 仅作完整披露 | 偏密 |
| 负暴露 activation | `43` | `<=2` | fail |
| 负暴露点率 | `8.6759/min` | `<=0.50/min` | fail |
| 单侧 95% Poisson UCB | `11.1877/min` | `<=0.50/min` | fail |

负暴露 activation 分布为 CrowdBot 0410 MDS `19`、CrowdBot 1203 shared-control `17`、LILocBench long-term changes `7`。

相对已拒绝的 current-input timing family，这不是增益：旧 family 的乐观 coverage 上界为 `8/11 = 24/33`，新信号只有 `7/11 = 21/33`，少 1 个 unique event / 3 个 cell；同时风险远超门。终态因此为 `SIGNAL_REJECT`，不是“能力增强但证据不足”。

## 能力解释与限制

这仍是排除错误方向的研究结果，不是 BlindAssist 提醒能力提升。它证明“简单 route-relative image-plane convergence + bbox growth 的固定 2-of-3 组合”也不足以区分危险与普通路侧目标。

route-relative position 使用同帧 bbox footpoint 减 causal route UV，可消除两者共有的图像平移；但本轮没有统一可靠的相机旋转、尺度、深度或完整 ego-motion 补偿输入，因此不把它称为完整相机运动补偿。bbox expansion 也不是物理 TTC、距离或碰撞概率。

当前负暴露仍只有 `4.9563min`，低于零事件也需要的 `5.9915min` 可信 floor；不过本轮无需依赖该限制来拒绝，因为 coverage 与经验点风险都已经明确失败。

## 权限与下一独立边界

本轮没有训练模型，没有修改 detector、T0、route、reset、C1–C3、truth、oracle、clearance、two-frame guard、TTL 或 renewal；没有运行候选、调提醒规则、生成 policy、连接 opener，selection、L2/L3、Android shadow、H2、人体、独立行走与生产权限均关闭。

下一独立边界不得包装或微调本信号。若继续，应换成真正不同的信息源，优先是具有统一因果输入与明确可用性门的 **ego-motion-compensated radial expansion / optical-flow residual**，或具有可比跨来源语义的 metric relative motion；仍先做候选无关 separability probe，失败即淘汰。

## 收据与验证

- config SHA-256：`c76664ba61c356c40b8e01f194bc5bc56862a668947aac0f01323770ee342d31`
- signal inventory SHA-256：`892abf429cfa4947b7f019f73258e4eeac380ab97bc8c8ebd2bd945091cd361d`
- terminal SHA-256：`0bf3a272bbf447a4e8ca868ffd3bfe1b33b60b18646260cdb113d1ceff5fff45`
- validator：`VALID`；重算 `41` 序列、`62,229` 帧、`1,903` activation，终态仍为 `SIGNAL_REJECT`
- canonical local evidence：`artifacts.local/evidence/ustrf-causal-route-intrusion-signal-r0/`

```powershell
.\.venv-export312\Scripts\python.exe scripts\run_research_tool.py ustrf-route-target-evidence-closure run_causal_route_intrusion_signal_r0.py --repo . --config configs/ustrf_causal_route_intrusion_signal_r0.json --phase producer
.\.venv-export312\Scripts\python.exe scripts\run_research_tool.py ustrf-route-target-evidence-closure run_causal_route_intrusion_signal_r0.py --repo . --config configs/ustrf_causal_route_intrusion_signal_r0.json --phase audit
.\.venv-export312\Scripts\python.exe scripts\run_research_tool.py ustrf-route-target-evidence-closure validate_causal_route_intrusion_signal_r0.py --repo . --config configs/ustrf_causal_route_intrusion_signal_r0.json
.\.venv-export312\Scripts\python.exe scripts\run_research_tool.py ustrf-route-target-evidence-closure test_causal_route_intrusion_signal_r0.py -v
```
