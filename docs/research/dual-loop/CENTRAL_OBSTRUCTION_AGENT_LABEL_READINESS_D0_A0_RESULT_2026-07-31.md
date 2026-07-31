# Central obstruction Agent label readiness D0-A0 result

状态：`COMPLETE / VALID / D0_A1_NEXT`

时间：2026-07-31（Asia/Hong_Kong）

## 结论

`CENTRAL_OBSTRUCTION_D0_A0_INPUT_UNIVERSE_R3` 已完成 reuse-first 输入宇宙冻结，
独立 validator 从源账本和 payload 重新计算后为 `VALID`。最小 production-labeling
bundle 固定为 6 个完整连续 RGB session、34,279 帧、5 个 source ancestry group，
不再为 D0-A0 扩数据。唯一允许的下一动作是
`D0-A1_EXCLUDED_CALIBRATION_AND_LOCK_ONLY`。

本阶段没有生成 observation label，没有读取 truth/review label 或 candidate output，
没有启动 D0-A2 正式标注，也没有授权 D0-B、融合、Android、真人、可通行性、产品或
安全结论。

## 冻结输入

| 数据来源 | session | 帧数 | 时间语义 |
| --- | --- | ---: | --- |
| CrowdBot | `defaced_2021-03-27-11-51-18_filtered_lidar_odom` | 2,239 | source-native ROS bag ns |
| CrowdBot | `defaced_2021-03-27-11-55-00_filtered_lidar_odom` | 2,183 | source-native ROS bag ns |
| Walking in Matoaka | `commons_file_walking_in_matoaka_west_virginia_h1yohswxnqo` | 10,724 | video start 起固定 10 Hz 派生 |
| Shanghai night walk | `commons_page_153983964_capture_2024-10-10` | 5,662 | video start 起固定 10 Hz 派生 |
| Shiraz city tour | `commons_page_143041813_capture_2021-11-20` | 4,891 | video start 起固定 10 Hz 派生 |
| REveL Dynamic | `revel_dynamic_full_capture_2024-02-21` | 8,580 | source filename timestamp ns |

全部 payload 为 3,726,047,583 bytes；ordered frame identity SHA-256 为
`c5e4604f90d2982073752617dcdad4099e00cd6d1ab5c03b5f7d3e373d0a2968`。

REveL 原生尺寸审计保留了 7 种尺寸：`346×260=8,534`、`346×258=13`、
`259×260=15`、`344×260=6`、`346×250=3`、`12×260=6`、`346×13=3`。
46 帧不是主尺寸，其中 9 帧存在极窄边。D0-A1 必须逐项 fail closed 或标为
`NOT_EVALUABLE`，不得通过补边、删除或事后缩短 session 隐去该 burden。

## Reuse-first 角色审计

角色账本共 107 行：

| disposition | 行数 |
| --- | ---: |
| `ADMIT_D0_A_PRODUCTION_LABELING` | 6 |
| `ADMIT_D0_A_CALIBRATION_ONLY` | 61 |
| `NOT_EVALUABLE_FOR_CURRENT_QUESTION` | 40 |

每行披露 content identity、ancestry、当前问题适配度、缺失要求、既有内容/算法输出
访问、claim overlap、selection/tuning influence、当前角色与局部排除原因。AI
source-fitness review 只看每个 admitted session 的 8 个等距帧且 candidate output、
truth 和 review label 均不可见；它只支持 bounded visual canary，不是标签正确性或
模型效果证据。

## 失败前身与信息增益

- R0 虽冻结了 25,699 帧，但缺少当前协议强制的 `reuse-role-ledger.jsonl` 和逐
  session reuse disclosure，封存为 `INVALID_IMPLEMENTATION_INCOMPLETE`。它促成
  mandatory-artifact mutation test，不构成科学失败。
- R1 完成同一 34,279 帧来源冻结后，独立验证发现 producer 与 validator 之间当前
  协议哈希由 `a49169ff…f619c` 漂移为 `6b1ae640…9443`。R1 不覆盖，封存为
  `INVALID_CURRENT_PROTOCOL_BINDING_DRIFT`。
- R2 在稳定协议上通过 payload 复算，但最终披露审查发现 source-fitness review
  实际位于当前 primary task 的 source-only 视图，而 spec 错写为独立上下文。R2
  封存为 `INVALID_REVIEW_CONTEXT_DISCLOSURE`；R3 只把该字段改为
  `isolated_context=false / source_only_view=true`，保持 candidate/truth/review
  label 不可见，并再次全量复算。

这两个失败只说明 implementation/artifact coverage 与协议绑定约束，不否定现有 RGB
是否适合后续中央阻塞标注。

## 证据身份

本地 evidence root：
`artifacts.local/evidence/central-obstruction-agent-label-readiness-d0-a-r3/`

| artifact | SHA-256 |
| --- | --- |
| `source_universe_r3.json` | `c1da49cb7d17077e7816b4a9c922096ea452550150034195169fb1b981d622c8` |
| `input-universe-manifest.json` | `3d860a827f988e29c4a2ed08524b1fb40c3706301bde75361c6bf5fbe7a43fbe` |
| `input-universe-receipt.json` | `6d5d25f88c0f82da4d077204554a1e77a1c319aaa4436b9f442c277b1fa8c045` |
| `reuse-role-ledger.jsonl` | `2fcc96be1a4c74fb6ff19c727c8323d8997e37718a9e6de59d8e537bf6001d2e` |
| `reuse-fitness-review.json` | `207ffcb2a0a35af13dbf314d08a8e40e94218281e025f7347f00fa779bbef59b` |
| `input-universe-validation.json` | `4440a9f58beadcb6a49ff75755bf4720d187cd4ebd4cf612a4e9e535dc7abb6a` |

## Claim ceiling 与下一门

当前只可声称：D0-A0 的输入身份、顺序、payload、来源 ancestry、reuse role 和
source-fitness canary 已冻结且可独立复算。不能声称 Agent 标签可靠、中央阻塞事件
存在或分布充分、模型 B 有增量、提醒改善、可通行或安全。

D0-A1 必须使用排除于 production-labeling bundle 的 calibration 单元，冻结 ROI、
prompt、parent-event 边界/匹配、歧义分层、抽样审计和 readiness 门。该 lock 完成前
不得启动 D0-A2。
