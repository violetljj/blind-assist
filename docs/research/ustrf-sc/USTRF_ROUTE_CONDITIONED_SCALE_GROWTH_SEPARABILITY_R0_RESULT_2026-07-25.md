# USTRF route-conditioned scale growth separability R0 结果（2026-07-25）

状态：`FAIL_CLOSED_INPUT_OR_CONTRACT_BLOCKED / VALID`

权限：`RESEARCH_ONLY / SIGNAL_AVAILABILITY_AND_DISCOVERY_SEPARABILITY_ONLY`

## 结论

本轮没有计算任何 `loomingScore`，也没有生成 threshold frontier 或 discovery candidate。原因不是尺度增长已经被证伪，而是冻结的 candidate-blind producer 输入不足以证明 bbox 位于已绑定的 canonical source-frame 坐标：

- `62,229 / 62,229` 帧没有逐帧绑定 canonical `source_size`；
- `62,229 / 62,229` 帧没有 source-to-canonical rotation/orientation receipt；
- `263,680 / 263,680` 个 observed-track 记录没有 authoritative severe-truncation 状态。

旧 causal route-intrusion R0 在实现中硬编码 `640×480`；上游部分 detector ledger 虽出现 `source_size=[640,480]`，但该字段和 rotation authority 没有进入当前 123 条候选投影及其 41 条 candidate-independent trace。将其解释为“rotation=0”或仅以 bbox 触边代替 severe-truncation authority，都会违反本轮冻结合同。因此合法终态为：

```text
FAIL_CLOSED_INPUT_OR_CONTRACT_BLOCKED
```

这只关闭当前输入合同下的执行，不支持“纯尺度增长不可分”或“尺度信息永远无用”的结论。

## 冻结边界

配置 `configs/ustrf_route_conditioned_scale_growth_separability_r0.json` 在任何 signal outcome 前冻结：

- 主信号 `S_t=0.5*log(w_norm*h_norm)`；
- past-only `600ms`、至少 `5` 个有效观测、最大相邻 gap `150ms`；
- 真实 timestamp 与固定 Theil–Sen median pairwise slope；
- bbox 触边或 severe truncation 无效，不插值；
- 唯一可扫描变量为 `loomingScore >= threshold`，断点为全部唯一有限实际 slope；
- discovery 门为 `11/11` unique supported events、`33/33` mechanically mapped cells、负暴露 first opportunity `<=2`；
- 冻结负暴露为 `836` 个 interval、`297376110945ns = 4.95626851575min`。

父 evaluator 没有首次合格延迟数值门，并明确留下 `alertable_deadline_not_frozen`。本轮因此在 signal outcome 前独立冻结 `5000ms` 门，参考既有 USTRF continuous-event first-correct-alert ceiling；配置明确标注该值不是父 evaluator 的继承结论。若未来更改延迟定义或数值，必须新建版本，不能在本 R0 上回填。

## 父证据复验

执行前重新绑定并验证：

| 父证据 | 终态 / 关键结果 | 当前验证 |
| --- | --- | --- |
| current-input policy feasibility bound R0 | `CURRENT_INPUT_POLICY_FAMILY_NOT_FEASIBLE`；最大 `8/11=24/33`，风险 `<=2` 时 `2/11=6/33` | config / certificate / terminal / validation SHA 全部匹配 |
| causal route-relative intrusion signal R0 | `SIGNAL_REJECT`；`7/11=21/33`，负激活 `43/4.956min` | config / 41-ledger inventory / terminal / validation SHA 全部匹配 |
| route-target metric eligibility R2-L1 | `836` interval、`4.95626851575min` | config / mask / denominator / validation SHA 全部匹配 |
| truth-blind causal per-track producer R0 | `123` projection、`41` sequence、`62,229` frame | config / inventory / terminal / validation SHA 全部匹配 |

父结果只提供冻结输入与已关闭方向；本轮没有重新调整 qualification、TTL、renewal、association、opening timing 或旧 `2-of-3` 组合。

## 两阶段与独立验证

producer-preflight 只读取 candidate-blind bbox、track/reset、route relation/validity 和 timestamp，先证明三个候选的投影逐帧相同并折叠 `123 → 41`。在发现 geometry contract 缺口后立即终止：

| 项目 | 结果 |
| --- | ---: |
| signal scores | `0` |
| truth payload decode | `0` |
| event-window decode | `0` |
| oracle-token decode | `0` |
| negative-exposure decode | `0` |
| candidate-cell decode | `0` |
| producer inventory | 未生成 |
| complete frontier | 未生成 |
| frozen candidate | 未生成 |

独立 audit 在新进程重算 `123 → 41`、复核 blocker 与 terminal SHA，并确认 inventory/frontier 不存在。第三个 validator 进程再次重算同一合同缺口，终态为 `VALID`。交付前最终复跑的进程 ID 分别为 `33360 / 56232 / 60172`，只用于证明本次阶段隔离，不是可复用身份。

## 机器收据

证据位于忽略目录：

`artifacts.local/evidence/ustrf-route-conditioned-scale-growth-separability-r0/`

| 文件 | SHA-256 |
| --- | --- |
| `contract-violation-receipt-r0.json` | `49bacb09642427538e0c75a50dd001282d3e38885bf139342c56dacb5cefaa8f` |
| `terminal-receipt-r0.json` | `dcd72213aec7ff9247af3a825851e59943c5e097a013d3ecbc41787a46a33a31` |
| `audit-receipt-r0.json` | `86eb94244b4f127532616794d78bae17c62ed9eaf4565c0e760dfbff6c086d3d` |
| `validation-receipt-r0.json` | `99c2e8f7e1088b674222c01a6028126bf20ffb4a2298a47e8843951311577649` |

配置 SHA-256 为 `48d2d3f79d41db1fc5e345891dda1c705fe0d074ca01d7d93f047228c9930d9c`。

## 验证

```powershell
.\.venv-export312\Scripts\python.exe scripts\run_research_tool.py ustrf-route-target-evidence-closure test_route_conditioned_scale_growth_separability_r0.py -v
.\.venv-export312\Scripts\python.exe scripts\run_research_tool.py ustrf-route-target-evidence-closure run_route_conditioned_scale_growth_separability_r0.py --repo . --config configs\ustrf_route_conditioned_scale_growth_separability_r0.json --phase producer
.\.venv-export312\Scripts\python.exe scripts\run_research_tool.py ustrf-route-target-evidence-closure run_route_conditioned_scale_growth_separability_r0.py --repo . --config configs\ustrf_route_conditioned_scale_growth_separability_r0.json --phase audit
.\.venv-export312\Scripts\python.exe scripts\run_research_tool.py ustrf-route-target-evidence-closure validate_route_conditioned_scale_growth_separability_r0.py --repo . --config configs\ustrf_route_conditioned_scale_growth_separability_r0.json
```

focused tests 为 `10/10 OK`；validator 为 `VALID`。

## 下一独立边界

不得在本轮自动继续 Gate 2，也不得以硬编码尺寸、假定 rotation=0、显示层坐标或事后 heuristic truncation 回救。唯一合法下一动作是另立 input-contract repair goal：

1. 从 canonical raw / source transport authority 重建逐帧 width、height 与 source-to-canonical rotation receipt；
2. 将 severe-truncation authority绑定到每个 observed track，或以预注册、可验证的上游规则重新物化；
3. 重新生成 candidate-blind frame membership receipt，并证明 123 个候选投影逐帧一致；
4. 使用新版本配置重新启动本 R0；旧 blocked terminal 保持不可变。

Android、Kotlin runtime、opener、shadow、H2、人体和生产权限全部保持关闭。
