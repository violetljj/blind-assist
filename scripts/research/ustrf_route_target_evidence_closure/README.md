# USTRF route-target evidence closure

状态：R1 `DATA_BLOCKED / STOP_SOURCE_SEARCH` / evidence maturity V2 governance active at L0 / R2-L1X-L2P `FAIL_CLOSED_EXECUTION_ABORTED` / L2+L3 prereg frozen / candidates unrun

## R2-L1X-L2P recovery and preregistration

`configs/ustrf_route_target_r2_l1x_l2p_prereg_r1.json` 在任何新 C1–C3 输出前绑定旧 R1 failure receipts，并冻结 L2 fresh-selection 与 non-executable L3 lockbox。`run_r2_l1x_l2p.py` 只在独立 namespace 恢复逐 ledger canonical raw；`validate_r2_l1x_l2p.py` 复建 41 ledger / 62,229 frame / 15 reset、权限和唯一终态。`validate_l2_l3_prereg_r1.py` 独立校验 L2 required metrics/门/primary/tie-break/source/veto/role/selection 语义，以及 L3 的 `executable=false`、`candidate_id=null` 和 lockbox/statistics floors。

原 R2 在三次远端清理白名单失败后保留 `FAIL_CLOSED_EXECUTION_ABORTED`。outcome-unseen A1 只修远端路径白名单，但两次 instrumentation 仍无法从 app external-files 识别 shell materialized manifest，第三次又触发不可降低的 6 GiB 内存门；A1 尝试耗尽并成为最终合法终态。当前仍只有 2/41 ledger、4,594/62,229 frame canonical input，C1–C3、trace/profile、机制成绩审计和 selection 均为 0。详见 [R2-L1X-L2P 日期化结果](../../../docs/research/ustrf-sc/USTRF_ROUTE_TARGET_R2_L1X_L2P_RESULT_2026-07-24.md)。

验证入口：

```powershell
python scripts/run_research_tool.py ustrf-route-target-evidence-closure validate_l2_l3_prereg_r1.py --repo .
python scripts/run_research_tool.py ustrf-route-target-evidence-closure test_l2_l3_prereg_r1.py
python scripts/run_research_tool.py ustrf-route-target-evidence-closure test_r2_l1x_l2p.py
python scripts/run_research_tool.py ustrf-route-target-evidence-closure test_r2_l1x_l2p_transport_amendment_a1.py
python scripts/run_research_tool.py ustrf-route-target-evidence-closure validate_r2_l1x_l2p.py --config artifacts.local/evidence/ustrf-route-target-r2-l1x-l2p-a1/frozen-merged-prereg-a1.json --repo .
```

## R2-L1E receipt-aware exploratory profiles

`configs/ustrf_route_target_l1_exploratory_profile_r1.json` 精确绑定父 R2-L1 protocol/mask/denominator/validation、冻结 C1–C3 实现、41 条 masked sequence ledger、62,229 帧和 15 个 discontinuity reset。独立 runner 只允许逐 ledger Android Canvas canonical raw 分片、host compact successor、冻结 T0 replay-local association 和因果 route input；truth 只能在候选输出后 join。终态 schema 只允许 `EXPLORATORY_PROFILES_COMPLETE`、`FAIL_CLOSED_INPUT_BLOCKED` 或 `FAIL_CLOSED_EXECUTION_ABORTED`，且所有 selection、Android shadow、H2、人体和生产权限固定关闭。

运行：

```powershell
python scripts/run_research_tool.py ustrf-route-target-evidence-closure run_metric_eligibility_exploratory_profiles_r2_l1.py --config configs/ustrf_route_target_l1_exploratory_profile_r1.json --repo .
python scripts/run_research_tool.py ustrf-route-target-evidence-closure validate_exploratory_profiles_r2_l1.py --config configs/ustrf_route_target_l1_exploratory_profile_r1.json --repo .
python scripts/run_research_tool.py ustrf-route-target-evidence-closure test_exploratory_profiles_r2_l1.py
```

当前机器收据为 `FAIL_CLOSED_EXECUTION_ABORTED`：冻结的 6 GiB 系统可用内存门在初始尝试和两次有界重试中均触发，首个 CrowdBot device attempt 未创建。validator 重建为 2/41 ledger、4,594/62,229 帧 canonical raw 已验证，39 ledger、57,635 帧缺失；候选、trace 和 profile 均为 0。结果见 [R2-L1E 日期化结果](../../../docs/research/ustrf-sc/USTRF_ROUTE_TARGET_L1_EXPLORATORY_PROFILE_R1_RESULT_2026-07-24.md)。

## R2-L1 metric eligibility materialization

`configs/ustrf_route_target_metric_eligibility_r2_l1.json` 将当前 6,369 个 LILocBench/CrowdBot 事件或提案逐项物化为 8 指标 eligibility mask，并把非事件粒度的负暴露和 preoutput frame support 放入独立 ledger。输入只允许读取 11 个哈希冻结、candidate-blind 的 truth/route/review 文件；禁止目录扫描、候选模块执行和候选输出读取。

运行：

```powershell
python scripts/run_research_tool.py ustrf-route-target-evidence-closure materialize_metric_eligibility_r2_l1.py --config configs/ustrf_route_target_metric_eligibility_r2_l1.json --repo .
python scripts/run_research_tool.py ustrf-route-target-evidence-closure validate_metric_eligibility_r2_l1.py --config configs/ustrf_route_target_metric_eligibility_r2_l1.json --repo .
python scripts/research/ustrf_route_target_evidence_closure/test_metric_eligibility_r2_l1.py -v
```

输出位于忽略的 `artifacts.local/evidence/ustrf-route-target-metric-eligibility-r2-l1/`：`eligibility-mask-r2-l1.json`、`denominator-receipt-r2-l1.json` 和 `validation-receipt-r2-l1.json`。mask 同时包含 62,188 个相邻 frame-pair 的 eligibility/exclusion audit 与 62,229 行显式 preoutput frame ledger。validator 会完整重建前两者并检查规范 JSON 精确一致，同时硬拒绝 0/0 pass、pre-clear 进入 clearance、truth pool 冒充 repeat 分母、pair universe 缺口、负暴露重叠和任何候选输出访问。

当前物化结论是：`critical_miss`、`clearance`、`unknown_or_stale_alert` 为 L1 探索资格；`repeat`、`evidence_age` 为候选观测完整后才成立的条件资格；`event_recall`、`regeneration`、`false_alerts_per_minute` 仍为 L0。该结论只授权另开独立任务生成单次探索 profile，不授权选择候选或进入 Android/H2/生产。

下一独立任务使用 [R2-L1E 单次探索 profile 通宵目标](../../../docs/research/ustrf-sc/USTRF_ROUTE_TARGET_L1_EXPLORATORY_PROFILE_OVERNIGHT_GOAL_2026-07-24.md)：先检查全量 canonical input，再让 C1–C3 各对 41 条 masked sequence ledger 单次 replay，并在冻结观测断点重置状态；canonical raw 逐 ledger 分片验证和清理，输出只有分指标探索 profile 与机器收据，不产生 winner、排名或晋级。

## 稳定 Interface

运行 `python scripts/research/ustrf_route_target_evidence_closure/validate_prereg.py --config configs/ustrf_route_target_evidence_closure_r1.json --repo .`。validator 重算父 evidence 哈希，并冻结五态逐人路线角色、三条 oracle 臂、最多三个累积结构候选和逐来源 holdout 门；任一漂移均 fail closed。

运行 `python scripts/run_research_tool.py ustrf-route-target-evidence-closure validate_evidence_maturity_v2.py --config configs/ustrf_route_target_evidence_maturity_v2.json --repo .` 校验当前证据成熟度协议。V2 不改写 R1：它只允许按 event recall、critical、repeat、clearance、false-alert exposure、evidence age 和 unknown-route veto 分别冻结 eligibility/分母，并按 L0–L4 提升权限。空分母必须是 `not_evaluable`，低样本只能是 `evaluable_underpowered`；L1 不选胜者，L2 不直接进 Android，L3 才能申请 production-isolated shadow admission。

`prepare_route_role_review_bundle.py` 在 detector/candidate 输出隐藏状态下，逐帧重算 4,594 张 RGB 哈希并联结 source timestamp、因果 route receipt、既有 target/negative person seed truth。seed box 只提供既有审查事实；每帧仍要求 full-frame all-person discovery，不能把未发现的共现者当作 absent。

`annotate_seen_persons_closed_vocab.py` 是第二条独立 person annotation proposal pass：固定闭词表模型 SHA、960 输入、`.01` proposal floor，只生成 truth-review proposal，不是 detector 候选，也不获得 benchmark 或晋级 credit。它与既有 prompted YOLOE-11s-seg pass 融合；非 seed 的单模型 proposal 必须隔离，不能直接进入真值。另保留既有负窗 YOLO11x-960 proposal 作为负窗附加审计证据。

`fuse_seen_person_proposals.py` 先以既有冻结 seed truth 为优先，以固定 Ultralytics ByteTrack 默认参数分别形成 annotation-only 身份提议，再按预注册 IoU 联结两条 proposal pass，并只生成 `proposal_track_id`。ByteTrack 在未观察到的帧号间隙强制重置，且片段编号写入 ID，不能跨不连续窗口串联。含单模型节点、多人关联歧义或 identity 冲突的 tracklet 必须进入第三模型 adjudication；在此之前禁止命名为稳定 `person_id`。

`prepare_third_model_adjudication_bundle.py` 只抽取争议 tracklet 涉及的去重帧；`annotate_third_model_disagreements.py` 使用冻结 SHA 的 YOLO11x-960 作为第三条闭词表 person proposal pass，仍看不到评分标签、App detector 或候选告警。`resolve_third_model_adjudication.py` 对单模型节点和一对多身份歧义执行 fail-closed 裁决；未被第三模型重合确认或仍跨 tracklet 冲突的整段必须 quarantine。

`build_route_role_model_proxy_truth.py` 只把 registered RGB-D 用作离线注释支持，route 只读 causal prediction；route/depth/脚点无效均输出 unknown。`run_seen_oracle_attribution.py` 运行 T0 与三条单接缝 oracle。`candidates.py` 是冻结的 C1–C3 因果状态机实现；它不读取 proxy truth 或 RGB-D，也不修改 detector/tracker。

`inspect_remote_zip_inventory.py`、`extract_remote_zip_entry.py` 与 `stream_remote_zip_entry.py` 只通过官方 Range 请求核对 ZIP 目录/条目并做 CRC+SHA 收口；`stream_remote_zip_entry.py` 可显式启用有限并发 Range，把压缩分片暂存到独立缓存盘，按原始字节顺序解压并复用同一 ZIP CRC、输出 SHA 与帧哈希合同，成功后删除分片。长时间运行的旧物化器也可由 evidence root 中不可变、版本化的 `transport-acceleration-r*.json` 在下一条 stream 子进程边界启用同一模式，优先使用最高受支持版本；旁路配置必须声明 candidate-blind，并将自身路径和 SHA 写入下载收据。`qualify_crowdbot_route_capacity.py` 只用发布 tracks 与同步 pose 做来源容量代理，不运行候选。`materialize_crowdbot_rgbd_sequence.py` 无损导出 forward RGB 与精确同时间戳 aligned depth；`rgb8` 原样保存，来源原生 `bgr8` 只允许通过通道反转规范化成 RGB PNG，不允许颜色增强或坏行修复。`materialize_crowdbot_holdout_sources.py` 支持同一来源的多 part raw inventory，每次只保留一个临时 bag，bundle 验证通过后删除 raw 并留下可重取收据；Windows 若在子进程退出后短暂保留 raw bag 文件句柄，只允许有界退避重试同一已验证文件的删除，不跳过 bundle/hash 校验。全量完成后，`audit_materialized_holdout.py` 才会逐文件复算 RGB/depth 哈希并执行跨来源 exact SHA 与 dHash 近重复审计；该审计本身不授予来源准入或 H2 权限。

Holdout candidate route 不得复用容量筛查的未来真实轨迹。`materialize_crowdbot_rgbd_sequence.py` 同时保存 candidate-blind TF frame inventory，`backfill_crowdbot_tf_inventory.py` 只用于给已验证的早期 bundle 补齐同一 raw bag 的 TF 绑定。16/16 后，`build_crowdbot_causal_route_ledger.py` 继承 R3 的 past-pose-prefix-only 合同：仅用当前/过去 Qolo pose，经静态 `tf_qolo→camera` 外参与相机内参生成 causal route UV；future pose 只允许生成 annotation route truth，并以所有可投影 pose 样本形成完整 UV polyline，不能只保留终点。

`build_crowdbot_projected_track_role_proposal.py` 把发布 LiDAR track 投影进 RGB，仅生成 candidate-blind identity/metric-role proposal；它不能据 track 缺失宣称画面无人。`approaching_route` 必须由 actual-future track 在 1.6s 内实际进入 route 支持，`receding` 必须有 prior intersection；missing/间断不会形成 clear。该 proposal 还必须与两条独立 visual person pass 做全帧共识与歧义隔离，才能冻结为最终 holdout truth。

`annotate_crowdbot_holdout_person_proposals.py` 只按预注册的 YOLOv8n/Yolo11x 模型哈希和 `.01` proposal floor 生成两条全帧 person proposals；它们看不到 App detector/event 或 C1–C3。首组 truth/window 因“无关未知人使整窗失效”和“继承相机不可见 LiDAR onset”以 `0/2` 失败，相关来源只保留协议诊断权限。替换协议由 `configs/ustrf_route_target_evidence_closure_r1_replacement_holdout.json` 哈希冻结：`fuse_crowdbot_holdout_person_role_truth.py` 只从视觉确认的 metric-person 连续角色生成正事件；raw LiDAR event 只作容量代理。负帧要求 causal route known 且所有路线相关人物已解决，路线外未知人不抹掉整帧，路线内或可能路线相关的未知人仍使帧不可评。`freeze_crowdbot_holdout_truth_windows.py` 只有在隔离后仍逐来源满足正/critical 事件、同序列等长负窗和 10 分钟负 exposure 时才签发 2/2 selection authority。

`prepare_crowdbot_holdout_detector_device_bundle.py` 只有在上述 2/2 truth/window 收据已冻结后才可整理完整 RGB manifest；随后复用 hash-bound Android `ImagePreprocessor` Canvas + 正式 App TFLite CPU-4-thread exporter 产生 canonical raw tensors。`run_crowdbot_holdout_app_detector.py` 只解码该设备 raw stream，并绑定权重、labels、`.35/.45`、manifest/receipt 与完整 RGB 哈希；禁止用 host PIL reconstruction 冒充 App detector。`run_crowdbot_holdout_candidates.py` 固定使用 T0 association，让 C1–C3 各自对每条完整 sequence 一次运行；truth 只在状态更新后用于归因。候选告警若重合 unresolved person，则该来源以 `unknown_person_active_alert_count > 0` 硬失败。`configs/ustrf_route_target_evidence_closure_r1_scoring_amendment.json` 在候选运行前修正 false-alert numerator：完整序列内所有 route-known、无真事件归因且非 unresolved-person 的 delivery 都计入，不能只计匹配窗口却除以全量负暴露。报告保留逐序列 trace hash、delivery/closure 收据、逐来源全部硬门与 worst-source tie-break，任何来源失败都不会打开 Android shadow 或 H2。

Replacement 的 23 条完整序列最终仍只形成 2 个可接受事件，说明纯 LiDAR/pose 容量代理不能预测 camera-visible metric identity continuity 与 terminal clear。后续来源必须先过 `configs/ustrf_route_target_evidence_closure_r1_camera_source_prescreen.json`：每来源只解码两条 candidate-blind canary，事件 canary 按 positive/critical/active 容量选取，负窗 canary按 `negative_route_seconds / compressed_GiB` 选取；两条均永久排除未来 lockbox。canary 门按正式门乘 `2 / metadata_sequence_count` 冻结，只具有 reject-only 权限；失败立即停止其余下载，通过也只允许物化非 canary，不能直接准入来源或运行候选。`inspect_remote_zip_inventory.py` 在读取 body 之前硬检查 HTTP 206、Content-Range 与 Content-Length，服务器忽略 Range 时不得误拉整包。

## 输出

首组诊断 evidence 保留在 `artifacts.local/evidence/ustrf-route-target-evidence-closure-r1/`；替换来源写入 `artifacts.local/evidence/ustrf-route-target-evidence-closure-r1-replacement/` 与 `artifacts.local/datasets/ustrf-route-target-evidence-closure-r1-replacement/`。`0410 mds + 1203 shared-control` 已完成 23/23，但 truth/window admission 为 0/2，只保留诊断和回归权限。0327 reject-only canary 位于 `artifacts.local/camera-source-prescreen-r1/`，物理存储在获用户授权的 D 盘；两条共 4,422 RGB，最终为 0 positive、0 critical、0 matched negative、0.0764min negative exposure，已拒绝并停止剩余 11 条。NavWareSet 只下载 181.6MB Stage A 即因 robot/GRS 原始时间区间不重叠而拒绝，没有启动约 5.97GB Stage B。Bi3 完整 41.7GB 包也未下载。候选输出始终关闭。

最终有界来源审计还拒绝了 REveL：已查看的 `dynamic` 只能保留 development/diagnostic 权限，排除后全部未查看 footage 的发布总时长上界约 7.903 分钟，低于每来源 10 分钟负暴露门；匿名小包又不同时包含相机、稳定身份、metric sensor/person pose 与 terminal clear。JRDB、Oxford-IHM、KTP/IAS-Lab、FLOBOT、THÖR-MAGNI、SCAND 也均未同时通过五项来源门。最终收据为 `artifacts.local/evidence/ustrf-route-target-evidence-closure-r1/source-search-final-bounded-decision-r1.json`，决定为 `DATA_BLOCKED_STOP_SOURCE_SEARCH`、可用来源 `0/2`、本轮新增下载 `0 bytes`。不继续扩大来源搜索，不运行候选。

## 安全边界

当前 15+15 seen 窗口只做故障归因，不能选择候选或调标量。detector、`.35`、NMS、tracker 均冻结；深度、TTC、route-risk flip、Android shadow、训练和生产权限关闭。模型生成的 route-role truth 是 hash-bound benchmark evidence，不是真实用户安全事实。

## 停止条件

R1 来源搜索已经停止。V2 每轮最多审计两个新来源 family、每来源两个 canary，默认自动下载上限 2 GiB；连续两个 family 不合格即 `STOP_DATA_COLLECTION_AT_CURRENT_LEVEL`，保留已有局部指标证据。真实性/unknown 硬门失败才 `STOP_MECHANISM`。轮内不得改语义、分母、阈值或 tie-break；轮间改变必须升协议版本并让已查看数据失去 selection/confirmation/shadow lockbox 权限。
