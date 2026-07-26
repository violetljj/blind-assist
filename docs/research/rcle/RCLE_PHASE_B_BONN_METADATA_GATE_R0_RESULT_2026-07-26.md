# RCLE Phase B Bonn Metadata Gate R0 结果

状态：`CONTENT_VALID / EXECUTION_CONTRACT_FAIL / FORMAL_PHASE_B_CLOSED`

日期：2026-07-26

## 结论

按设计锁 `e49d1f88…31c9e9` 实现了唯一候选
`BONN_METADATA_BLIND_AUTHORITY_AND_COHORT_FREEZE_R0`。Metadata 内容与
cohort 可复算，但独立审查发现 one-run / canonical-output 合同未被实现强制，
因此 R0 不能形成权威 metadata-gate PASS：

```text
official universe          26/26
historical exclusions       9/9
display-size exclusions     4
eligible not selected       7
deterministically selected  6
receipt content             VALID
execution contract          FAIL
```

终态：

```text
CLOSE_BONN_METADATA_BLIND_AUTHORITY_AND_COHORT_FREEZE_R0_EXECUTION_CONTRACT_FAIL
```

下面的 cohort 只能作为可复算 diagnostic identity，不能升级为 formal
admission authority。没有下载或解码任何 ZIP、RGB、depth、pose 数值或 static
map，没有读取旧 trace/support/residual/score，没有计算 RCLE 或 Phase B 指标。

## 锁与 receipt

| 项目 | SHA-256 |
| --- | --- |
| Phase B 入口设计锁 | `e49d1f88f13e2a190714211cfe46bb7d9f8518eaca93b5981736dcdd7231c9e9` |
| 历史排除 manifest | `f02bd9f1313def45cc107d72ace5f7c7803f4ab816bf6e98c5f9173fa3bb1cc6` |
| implementation lock | `a47cd39ea82c10828290def8bae54f61b28676190c8ab06acc93217b1590a617` |
| official metadata page | `2bd8df16acad79c70e1021f1da039c78510034fd9091fd706f8a3f480ea5c186` |
| receipt | `4386bbe3b617abca3b73fc3070a65cef403fe270c12fd25f5034a579882f1764` |
| receipt validation | `544a470471a9bd28ab1610efec62a53276754ecd9c2b00e28a27e8b0b364b718` |

输出位置：

```text
artifacts.local/evidence/rcle_phase_b_bonn_entry_r0/metadata_gate_r0/
```

只包含 `receipt.json` 与 `receipt_validation.json`。

## 确定性 cohort

按冻结规则排除历史 9 条、排除官方 display size `>550 MB` 的条目，再按
`SHA256("rcle-phase-b-bonn-entry-r1\t" + sequence_id)` 升序选择前 6：

1. `rgbd_bonn_crowd2`
2. `rgbd_bonn_balloon_tracking`
3. `rgbd_bonn_balloon_tracking2`
4. `rgbd_bonn_moving_obstructing_box2`
5. `rgbd_bonn_balloon2`
6. `rgbd_bonn_moving_nonobstructing_box2`

Cohort identity SHA-256：

```text
513b770d18489fd0caf84874e9fb89456eb3a992fc262b037220b66b5caae86e
```

这 6 条只具有 `NONAUTHORITATIVE_DIAGNOSTIC_COHORT_IDENTITY` 角色。metadata
名称不得被解释为 pure rotation、approach、static surface 或 mixed-motion
truth，也不得触发 payload acquisition。

## 防火墙和分母

Receipt 保留官方全部 `26` 条 metadata denominator，每条都有 universe rank、
selection hash、eligible rank、disposition 和 reason：

- `EXCLUDED_HISTORICAL = 9`
- `EXCLUDED_DISPLAY_SIZE = 4`
- `SELECTED = 6`
- `ELIGIBLE_NOT_SELECTED = 7`

以下 receipt 计数全部为零：

- RGB/depth payload members；
- pose numeric values；
- static-map points；
- video visual inspection；
- legacy trace/support/residual/score；
- candidate signal 与 Phase B metrics。

失败单位和零窗口 sequence 将来必须保留，不允许换 sequence 回救。

## 验证

- 8 项 focused unit tests：PASS；
- Python compile：PASS；
- implementation lock validator：PASS；
- receipt `--validate-existing`：`VALID`；
- denominator：`26`；
- selected：`6`；
- payload authority：`false`；
- formal Phase B authority：`false`。

上述 `VALID` 只说明 receipt 内容可以从同一 HTML、历史排除和选择算法复算，
不证明执行合同有效。

## 独立审查阻断

独立只读审查确认：

- runner 暴露 `--output-root` 与 `--lock` override；
- one-run 防护只检查调用者指定的 output root；
- implementation lock 虽声明 `maximum_gate_runs = 1` 和唯一
  `metadata_output_location`，validator 却没有把实际参数强制绑定到该路径；
- 因而换一个 output root 可以再次 materialize，无法证明全局唯一 run。

这是执行控制失败，不是 cohort 数学或 metadata 内容失败。现有 receipt 降为
diagnostic，R0 候选按 fail-closed 关闭；不得补丁后重跑、不得以当前 selected
identity 继续 Phase B。

第一次 launcher 启动在读取 metadata 前因 Windows Python 缺少 `tzdata` 终止；
当时没有输出目录、receipt、cohort 或结果可见。只将时间戳实现改为标准库固定
`UTC+08:00`，新增无 `tzdata` 依赖回归测试并重新锁定 controls 后，才进行上述
唯一一次正式物化。选择算法、universe、历史排除和 gates 均未改变。

## 权限和下一边界

本次结果不授权：

- 下载、解压或读取所选 6 条 payload；
- 物化 RGB/depth/pose 公共时间窗；
- 实现或运行 raw、rotation-compensated 或 scale proxy；
- 计算 rotation leakage、closing error、RSR/CRR、support/runtime；
- Kill Gate B、Replay、Android、人体、安全或生产后继。

若要恢复，只能另立明确授权的版本化 R1 设计：移除 override 或强制实际
lock/output 等于 canonical path，增加第二输出与 override 负回归，并预先决定
是否允许在 R0 结果已见后重放同一确定性 metadata。当前授权不允许自动开始该
revision。
