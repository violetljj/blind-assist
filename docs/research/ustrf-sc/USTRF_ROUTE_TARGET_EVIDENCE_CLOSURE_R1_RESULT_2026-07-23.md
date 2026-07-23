# USTRF route-target evidence closure R1 阶段结果（2026-07-23）

状态：`SEEN_TRUTH_FROZEN / O1_NOT_EVALUABLE / O2_PROXY_NO_GAIN / O3_LIFECYCLE_GAP_CONFIRMED / C1_C3_HASH_FROZEN / FIRST_HOLDOUT_ADMISSION_0_OF_2 / REPLACEMENT_TRUTH_ADMISSION_0_OF_2 / CAMERA_NATIVE_SOURCE_PRESCREEN_FROZEN / 0327_CANARY_REJECTED / NAVWARESET_STAGE_A_REJECTED / REVEL_REJECTED / DATA_BLOCKED_STOP_SOURCE_SEARCH / CANDIDATES_UNRUN / ANDROID_SHADOW_CLOSED / H2_CLOSED`

## 结论

本轮已经把 15+15 seen 诊断集从“只有目标框”推进到窗口内逐人身份提议、五态路线角色代理与 person-bound lifecycle，并运行三条 oracle 归因；但没有获得候选选择权。最强结论是：oracle lifecycle 可把当前口径的误提醒与 repeat 同时降到 0，却仍保留原来的 14/15，说明 lifecycle 是主要可修复缺口，但不能补回那一个当前 evidence 缺失事件。

O1 因 34 个共现身份片段无法通过三模型 fail-closed 裁决而正式 `not_evaluable`；O2 的模型代理路线关系反而增加 repeat/误提醒，不能据此宣称路线关系已闭合。C1–C3 的无深度、无 TTC 因果实现已经在解码任何新 holdout 前哈希冻结。首组 CrowdBot 两来源以 `0/2` 失败后，候选盲替换集 `0410 mds + 1203 shared-control` 已完成 23/23 RGB-D/TF 物化、完整性审计、双视觉 pass、路线/轨迹融合和真值冻结，但仍以 `0/2` 失败：发布 LiDAR 容量代理严重高估了“相机可见、可唯一绑定 metric identity、连续到 terminal clear”的事件容量；每来源最终只接受 1 个正事件，负暴露不足 3 分钟且没有匹配负窗。两次失败均未运行 C1–C3，因此没有候选胜负；Android shadow 与 H2 继续关闭。

## 逐人身份与路线角色

- reviewer-facing bundle：4,594 张 RGB，隐藏正负标签、future truth、App detector 与候选告警；SHA-256 `fcca02de...e386`。
- proposal A：YOLOE-11s-seg + MobileCLIP；proposal B：YOLOv8n closed vocabulary；争议帧第三模型：YOLO11x-960。三者均只作 annotation proposal，不获得 detector/candidate credit。
- 修正了两处会污染真值的实现错误：ByteTrack ID 加入不连续片段号；身份键加入 `blind_window_id`。最终 823 个窗口内 tracklet 中 789 接受、34 隔离，跨盲窗 tracklet 为 0。
- 既有冻结 seed 身份具有优先权；模型补出的非 seed 延伸若冲突，只隔离延伸，不删除已冻结 seed。最终 15/15 既有目标身份均覆盖。
- 路线角色代理使用 causal route prediction + registered RGB-D 离线注释支持；候选 H2 深度仍关闭。3,804 个可见 person-frame 中 geometry known `2,567`、unknown `1,237`；unknown 不会被写成 safe 或 cleared。
- lifecycle clear 对既有 15 个目标使用隔离 scorer binding 的冻结 clear anchor；它只服务 oracle truth，不进入候选输入。最终 15/15 目标各保持一个 person-bound event、15/15 alertable、15/15 有 clear anchor。

核心 truth：`artifacts.local/evidence/ustrf-route-target-evidence-closure-r1/route-role-model-proxy-truth-r6.json`，SHA-256 `a0d51ca5...7d29`。它是 model-proxy benchmark evidence，不是人工真值、设备几何或生产授权。

## Seen oracle 归因

新口径把负窗提醒和正窗中归因到非目标人的提醒都计入 false alert numerator，因此 T0 的 false alerts/min 高于父结果只报负窗的 `8.620/min`。

| 臂 | dynamics_0 | lt_changes_dynamics_0 | 解释 |
|---|---:|---:|---|
| T0 current | recall `3/3`，FA `16.13/min`，repeat `0` | recall `11/12`，FA `11.60/min`，repeat `12` | 精确保留父结果 14/15 首因 |
| O1 oracle person + current route/event | 仅 `1/6` 窗口全人可评 | `14/24` 窗口全人可评 | 因 34 个隔离共现身份，整体 `not_evaluable` |
| O2 current detector + oracle relation proxy | recall `3/3`，FA `32.26/min`，repeat `2` | recall `12/12`，FA `21.41/min`，repeat `48` | proxy relation 抖动与事件核组合更差；不得作为路线闭合证据 |
| O3 current evidence + oracle lifecycle | recall `3/3`，FA `0`，repeat `0` | recall `11/12`，FA `0`，repeat `0` | lifecycle 能闭合误提醒/repeat/clear，但不能创造缺失 evidence |

oracle receipt：`artifacts.local/evidence/ustrf-route-target-evidence-closure-r1/seen-oracle-attribution-r1.json`，SHA-256 `33dbadc7...c56b`。

## 三个结构候选

冻结实现通过稳定入口 `python scripts/run_research_tool.py ustrf-route-target-evidence-closure candidates.py` 调用，implementation SHA-256 `82fb1a63...16c4`：

1. C1：T0 person lineage 上的 causal image-space route relation FSM；只用连续三帧严格单调关系，不增加标量。
2. C2：继承 C1，以连续 route occupancy episode 为键，只交付一次；track fragmentation 不能在同一 episode 内重发。
3. C3：继承 C2，以 person lineage + route episode 双键；只有全部 episode lineage 有连续 receding/outside 且 route release 才 clear，missing/unknown 不能 clear。

29 个 focused contract tests 已通过；detector、`.35`、NMS、T0 association、深度、TTC 与 route-risk flip 均未改变。

## Sealed holdout 准入与物化

- Bonn `person_tracking` 已被双 reviewer 以旋转主导、0 event 拒绝；`crowd` route unknown `.976268`，不能复用。
- JRDB 元数据高度适配，但官方下载要求登录；当前没有用户凭据或代建账号权限。
- CrowdBot_v2 主记录的官方 `hasPart` 关系把 processed archive、raw rosbag 与模型权重分开发布；Range-only ZIP inventory 证明 raw bag 内有 forward defaced RGB-D，不需要先盲下 43.8GB processed 包。
- SCAND 有 138 轨迹/8.7 小时与两机器人多模态数据，但官方元数据未建立本门所需全体人 time-consistent route-role truth。

候选实现 SHA `82fb1a63...16c4` 冻结后，先以发布的 2D LiDAR tracks + 同步 robot pose 做纯来源容量筛选。`0325/shared_control`、`0424/rds`、`0424/mds`、`0325/rds` 分别仅有 `8/6/7/8` 个保守正事件容量，均按 `<10` 淘汰；没有放宽 `.45m` corridor、`2` 帧 positive 或 `3` 帧 critical 定义回救。

最终保留：

| 来源 | scorable route | negative exposure proxy | positive / critical capacity | cooccurrence |
|---|---:|---:|---:|---:|
| `crowdbot_0410_shared_control` | `18.003min` | `17.671min` | `20 / 19` | `100%` proxy frames |
| `crowdbot_0424_shared_control` | `14.251min` | `14.054min` | `17 / 16` | `100%` proxy frames |

两组同属 CrowdBot/Qolo/shared-control，故仍有共同硬件与采集协议局限；但捕获日期相隔 14 天、parent sequence/person-route trace 断开，且与 seen LILocBench 完全新鲜。发布 tracks 只证明来源容量，不是真值，也不会进入候选得分。

`0424` 最小 bag 的 modality probe 已通过：`640×480 rgb8` 849 帧/64.9s、aligned `16UC1` depth 361 帧、精确 RGB-depth 同时戳 339 帧；3 个首/中/末样本均见明显人脸模糊，但车牌仍可能可辨，因此仅限内部研究、无外部再发布权。非精确 depth match 一律 unknown，H2 不开放。首组无损物化最终为 `16/16` sequence、两来源各 `8/8`；22,856 个唯一 RGB 时间戳全部复算一致，所有 bundle 均保持 `candidate_outputs_executed=false`。物化后发现 ROS bag 同时间戳重复 RGB message 会覆盖同一路径而保留旧 manifest 行；修复为 `same_bag_timestamp_last_message_wins_v1`，将 24,733 条 message 规范化为 22,856 个唯一时间戳，并留下逐序列与汇总收据。早期 probe 序列 `13-25-24` 的 TF 缺口也按原 bag SHA receipt 回填为 17 对。最终完整性审计为 2 来源、16 序列、22,856 RGB，精确重复对 0、近重复对 0，审计 SHA-256 `9d9e0928...27b8`；所有临时 raw bag 在 SHA 核对后删除，可按 URL+receipt 重取。

容量筛查使用的 actual-future robot polyline 只能进入 annotation route truth，禁止进入候选。Holdout 的候选 route 输入已另行冻结为既有 R3 `past_pose_prefix_only_no_future_ground_truth_v1`：RGB 只联结不晚于当前帧且 age≤200ms 的 published Qolo pose，以 12 帧历史外推 24 帧，并通过 raw-bag TF 相机外参投影为当前 route UV；future pose、stale join、缺失或不可投影均输出 unknown，不能告警或 clear。相应 TF inventory、首条旧 bundle 的可验证 backfill 和最终 causal route ledger 工具已预置，仍未运行任何候选。

Holdout 真值冻结协议也已在候选运行前锁定：YOLOv8n 与 YOLO11x 只作 candidate-blind 全帧 person proposal，发布 LiDAR track 只能提供稳定 identity/metric-role prior，不能据其缺失宣称画面无人；投影 track center 必须唯一落入视觉共识框。`approaching_route` 不再等同于任意距离下降，而要求当前在 corridor 外、连续三次靠近，且该人的 actual-future track 在 1.6s 注释窗口内确实进入 actual-future route；`receding` 必须有 prior intersection。纯视觉角色回退也必须使用该窗口内所有可投影 pose 样本形成的 actual-future UV polyline 与 person box 相交，禁止退化为只检查 1.6 秒终点。单模型、歧义或未建立 metric role 的 person episode 隔离，相关窗口不能用于路线指标。事件以连续两帧 active role 进入 alertable，以三帧 route-intersecting 定义 critical，只有 terminal clear 后才能生成新事件；正窗继承前后各 30 帧上下文，负窗必须同序列、等帧长、非重叠且 full-role complete。候选与 App detector/event 输出在 truth/window 哈希冻结前完全隐藏。

候选执行接缝也已在解封前固定：App `yolo11n_fp16_320.tflite`、COCO labels、Android Canvas `ImagePreprocessor` exporter 与 T0 配置分别做 SHA-256 绑定；只有 truth/window 两来源 admission 通过并冻结哈希后才允许生成设备 manifest。Detector ledger 必须解码 Android CPU-4-thread 导出的 canonical raw tensor，禁止以既有已知不逐像素等价的 host PIL 输入冒充 App evidence。C1–C3 各自对每条完整序列只运行一次，不按评分窗口重置；truth 只在状态更新后用于告警归因。评分逐来源检查 recall、critical miss、false alerts/min、clearance、repeat/regeneration、evidence age 与 unknown-route active alert，再按预注册 worst-source tie-break 选胜者；任一来源失败即 `STOP_NO_ANDROID_SHADOW_KEEP_H2_CLOSED`。

首组双视觉 pass 已完整覆盖 22,856/22,856 帧：YOLOv8n proposal SHA-256 `67b09f7c...a00cd`，YOLO11x proposal SHA-256 `308de313...00c01`。causal route 在 `0410/0424` 分别 known `10,902/7,889` 帧，投影轨迹角色 proposal 覆盖 `25,525/44,246` person-frame。融合产物 SHA-256 `d15f88a1...c336f`，但仅 9,210/22,856 帧达到旧的 all-person role complete；6,340 个事件 proposal 全部因整窗至少一个未知/隔离人而 quarantined，负暴露只剩 `0.336/1.126min`。冻结窗口收据 SHA-256 `df8fbc25...ea6e`，两来源四项门均失败，`admitted_source_count=0/2`。

该失败还暴露了两个协议结构错误：26 个发布轨迹事件直接继承 LiDAR onset，其中没有一个在 onset→clear 全程保持视觉身份连续；同时，路线外的无关未知人也会抹掉整个正/负窗口。由于任何候选输出均未运行，这只授权修正下一组新鲜来源的真值协议，不授权回救首组结果。替换预注册冻结为：

- `crowdbot_0410_mds`：scorable route `23.824min`、negative proxy `23.603min`、positive/critical capacity `21/17`；
- `crowdbot_1203_shared_control`：scorable route `25.598min`、negative proxy `25.417min`、positive/critical capacity `35/24`；
- 两来源 cooccurrence proxy 均为 `100%`，完整 10+13 条序列与 raw bag 一一匹配；`1203 manual` 因只有 `4.526min` 路线暴露被拒，未降低 10 分钟门回救；
- 新正事件只从视觉确认的 metric-person 连续角色生成；旧 LiDAR event 只保留容量权限；
- 正窗要求事件人身份/角色连续，不再要求无关所有人全完整；负窗只接受 causal route known 且所有路线相关人已解决的帧；
- 路线外未知人不再抹掉整帧；路线内或可能路线相关的未知人仍让该帧不可评，候选若对未知人告警则逐来源硬失败；
- `.35`、视觉 IoU `.30`、route margin `.08`、前后 30 帧、事件/负暴露门、C1–C3、detector/NMS/tracker 均未改变。

两来源 modality canary 均已通过。`0410 mds` 为 `rgb8`，1,526 个唯一 RGB、701 depth、642 exact、17 对 TF；`1203 shared-control` 为 `bgr8`，通过纯通道反转无损规范化为 RGB，947 个唯一 RGB、446 depth、398 exact、17 对 TF。两者都有静态 `camera_left_color_optical_frame → tf_qolo` 直连，首/中/末预览均见真实高密度共现行人及官方人脸模糊；`1203` 存在来源原生水平坏行/彩条，作为 worst-source 图像质量限制保留，不做清洗。两者都只获内部研究权限，不获外部再发布、H2 或生产权限。

替换预注册最终 SHA-256 `f68a59cf...7f72a1`。候选执行静态审计随后发现旧 runner 只在匹配窗口内累计误提醒，却用全来源负暴露分钟作分母；在任何候选输出运行前，以 scoring amendment `9af8c307...34d148` 修正为：完整序列内每个 route-known、未归因到真事件且不属于 unresolved-person 的 delivery 都进入 false-alert numerator，正窗错人和匹配窗外误提醒均不再漏计；unknown route 与 unknown person 仍分别硬失败。候选、门槛和标量均未改变，36 个 focused contract tests 已通过。

正式 10+13 条序列已完成 `23/23`：`0410 mds=10/10`、`1203 shared-control=13/13`，每次只保留一条 raw bag，按 ZIP CRC 与 SHA-256 校验，生成 lossless RGB-D/TF bundle 后删除 raw。最终共 34,779 RGB、16,931 depth、15,416 exact RGB-depth；跨来源精确重复 0、dHash≤4 近重复 0，完整性审计通过，SHA-256 `ff26e9e9...97d99`。候选输出从物化到真值冻结始终为 false。

传输最终保持 `8` 个并发 Range worker。R5 用 32 个小分片动态调度，R6 在不增加连接数的前提下改为 64 分片；R7 再把 socket timeout 从 120 秒缩短到 45 秒以恢复零字节连接卡死，config SHA-256 `01bce46f...f36b`。重试还暴露了预存零字节 cache 用 `xb` 重开会触发 `FileExistsError`，已改为“文件存在即 append”，加入零字节续传回归测试。最后两条正式收据均为 8 worker、64 连续分片、严格 206 Content-Range、分片字节/SHA、ZIP CRC、输出大小/SHA 全部一致。Windows raw bag 句柄延迟则使用有界 unlink retry；所有 attempt1/2/3 失败收据均保留，未跳过完整性门。

由于候选盲的传输恢复改变了冻结物化器哈希，首次 post-materialization validator 按设计失败关闭。随后追加 materialization amendment，绑定原/新物化器、streamer 和 R1→R7 传输配置哈希链，明确候选、真值算法、门槛、标量、Android shadow 与 H2 均未改变；修正 SHA-256 `7f878cde...d614`。缺少修正、哈希不符、候选输出暴露或打开 H2 均会失败；当前 45 个 focused tests 通过。

候选盲真值链最终完成：

- causal route：`0410/1203` known `9,472/12,500` 帧，evidence age P95 `47.78/47.92ms`，SHA-256 `7c83a2c0...60eed`；
- projected metric role：`59,236/98,261` person-frame，SHA-256 `9a08f3b4...e6b7`；
- YOLOv8n/YOLO11x 全帧 pass：各 `34,779/34,779`，SHA-256 `4dcb846d...18a1b` / `3e80b30a...8539`；
- 融合：route-relevant complete `17,267` 帧、accepted metric person `237,990` frame、presence-only unknown `65,405`、quarantined visual candidate `13,892`，SHA-256 `154dbd8a...b00da`。

冻结窗口最终只接受两个事件：`0410` 为 1 positive/1 critical，`1203` 为 1 positive/0 critical；另有 12 个事件因缺 terminal clear、身份/角色不连续、可见间断或固定上下文不足而隔离。`0410/1203` 可评分负暴露仅 `2.948/2.585min`，均无同序列等长 matched negative window，四项准入门全部失败，truth SHA-256 `ebba22d5...d9036`，`admitted_source_count=0/2`、`selection_authority=false`。

因此该替换集只保留故障归因权限：发布 LiDAR track 的正/负容量代理不足以预判 camera-visible metric identity continuity 与 terminal clear。归因收据为 `artifacts.local/evidence/ustrf-route-target-evidence-closure-r1-replacement/holdout-truth-admission-failure-attribution-r1.json`。不得回看本批结果放宽真值协议、事件/负暴露门或候选标量；下一轮若继续，必须在下载前用相机原生的小样本 identity-continuity + terminal-clear prescreen，再冻结新的两来源 lockbox。C1–C3、Android shadow 和 H2 均继续关闭。

## Camera-native 来源预筛与下一轮

两次 `0/2` 的教训已经固化为新的 reject-only 合同，而不是继续盲目扩大下载量。通用 policy SHA-256 为 `06743dfb...f3f2d`：每来源先冻结两条 canary，事件 canary 最大化 `(positive, critical, active frames)`，第二条在排除事件 canary 后最大化 `negative_route_seconds / compressed_GiB`；两条完整复合键及其 parent/近重复永久排除未来 lockbox。门槛按正式 `10 positive / 2 critical / 10 matched negative / 10min negative exposure` 乘 `2 / metadata_sequence_count` 得出。canary 失败立即拒绝来源，不下载剩余序列；通过也只允许继续物化非 canary，仍需完整正式门，绝不直接运行 C1–C3。

`crowdbot_0327_shared_control` 的 metadata proxy 为 14 条、49,008 帧、42.122 分钟负暴露、60/54 positive/critical，但 `forward_camera_visibility_verified=false`。三个官方 raw inventory 只证明 13 条，`12-34-13` 缺 raw 清单，已从可物化容量排除。冻结 canary 为：

- event：`11-55-00`，11/10 proxy event，6,546,266,842 compressed bytes；
- negative information/byte：`11-51-18`，171.817 秒 negative proxy，6,828,360,294 compressed bytes。

0327 roster SHA-256 `d87d2fb5...b70c`，pre-run validator receipt SHA-256 `7bb6a673...43b3e`。两条合计 13,374,627,136 compressed bytes，按 2/14 比例的 reject-only 门为 2 positive、1 critical、2 matched negative、1.428571 分钟负暴露；排除两条 canary 与缺 raw 序列后仍有 11 条、41/36 proxy event、33.525 分钟 proxy negative。候选、App detector/event、Android shadow 与 H2 均关闭。物理存储使用用户已授权的 D 盘，仓库内只暴露 `artifacts.local/` junction；传输固定 8 worker、64 分片、45 秒 socket timeout，逐条完成 CRC/SHA、lossless RGB-D bundle 后删除可重取 raw bag。

两条 0327 canary 最终均完成物化：共 4,422 RGB、2,130 depth、297 exact RGB-depth。因果路线 known 3,939 帧，unknown rate `10.923%`，evidence-age P95 `49.64ms`；两条视觉 person pass 各覆盖 4,422/4,422 帧。融合后只有 1,650 帧达到 route-relevant person role complete，冻结窗口得到 `0 positive / 0 critical / 0 matched negative / 0.0764min negative exposure`，四项 scaled gate 全失败。prescreen decision 为 `REJECT_CROWDBOT_0327_STOP_REMAINING_11_SEQUENCE_DOWNLOADS`，SHA-256 绑定在 `artifacts.local/camera-source-prescreen-r1/evidence/prescreen-decision-r1.json`。因此没有下载剩余 11 条，也没有运行 App detector 或 C1–C3。

该结果进一步证明 metadata 的 LiDAR/pose event capacity 不能作为 camera truth 的下载授权。后续来源预筛必须在扩下载前直接验证：相机内全人物发现、metric identity 唯一绑定、连续 route role、terminal clear，以及足够的可评分负暴露；若微型样例无法覆盖这些字段，就应淘汰来源而不是继续用更大的 metadata proxy 猜测。

CrowdBot 外还审计了 Bi3 与 NavWareSet。Bi3 官方材料证明 37 个实验、74 人、UM/LAAS 两站点、Stretch/PR2 两平台、10.5 小时、30Hz 顶视 RGB 与 120Hz 全场 MoCap；但 camera↔MoCap 唯一身份、逐帧同步/标定、camera-visible terminal clear、负窗和媒体许可尚未闭合。Dropbox 完整包约 41.7GB 且忽略 HTTP Range，因此没有下载完整包。

NavWareSet 原本更适合作为独立第一视角候选：48 scenes、HSR/Jackal RGB-D、TF/pose、framewise person UUID 与 world position。为避免先拉约 5.97GB scene 26，先冻结 scene 13 的 181,596,612-byte Stage A microcanary。第一次执行暴露 GitHub weak ETag 不能冒充 content SHA-256；修正只重新绑定下载内容哈希，且在 bag/annotation 解码前留下 amendment。正式解码后，robot bag 为 `[1730821395049328348, 1730821396048308123]ns`，GRS bag 为 `[1730821379971961548, 1730821380969381053]ns`，原始区间相隔 `14.079947295s`，没有已注册 temporal offset，违反预注册的跨模态重叠门。Stage A receipt 因而为 `REJECT_NAVWARESET_STAGE_A_DO_NOT_DOWNLOAD_STAGE_B_OR_ADDITIONAL_SCENES`；Stage B 下载量为 0。该失败门未在看过内容后放宽。

REveL 的本地完整链也完成最终有界审计并拒绝。现有 `dynamic` 的 8,580 RGB、两类 helmet identity、sensor/person Vicon 与 camera calibration 对开发期身份和跨模态回归有价值，但该 sequence 已被 detector、resolution、Vicon/radial 和 crop/tiling 实验查看，不能重新获得 sealed freshness。官方总量约 14.1 分钟；永久排除 371.805 秒的 `dynamic` 后，全部未查看 sequence 的总时长上界约 `7.903min`，即使没有事件和 unknown 帧也达不到冻结的每来源 10 分钟负暴露门。session 2 的小型匿名 images/labels 不含 metric person/sensor Vicon，最小完整 bag 为 1,581,570,924 bytes，因此也不存在同时覆盖相机身份、route 与 terminal clear 的微型 canary。REveL final audit SHA-256 绑定于 `artifacts.local/evidence/ustrf-route-target-evidence-closure-r1/revel-source-final-bounded-audit-r1.json`。

外部清单的最终有界审计同时结束：JRDB 的数据能力最接近，但完整传感器下载需要外部访问状态，匿名 labels-only 文件不能组成 camera-native canary；Oxford-IHM 需要先提交研究目的和 GDPR 保护方案；KTP/IAS-Lab 总量不足正式负暴露门；FLOBOT、THÖR-MAGNI 与 SCAND 分别缺 camera-bound persistent identity、公开相机媒体或 human tracking truth。当前没有任何来源同时通过“匿名 content-hash-bound 微型 canary、相机可见稳定全人身份、source-native causal route、同一人连续到 terminal clear、可评分共现负暴露”五项。

因此本轮来源边界正式结束为 `DATA_BLOCKED_STOP_SOURCE_SEARCH`，可用来源 `0/2`，本轮最终审计新增下载 `0 bytes`。决定收据为 `artifacts.local/evidence/ustrf-route-target-evidence-closure-r1/source-search-final-bounded-decision-r1.json`。不再以“继续找数据”为下一步；只有外部状态先提供至少两个独立且五项全部通过的来源，才允许重新建立新 canary 合同。在此之前不运行 detector、C1–C3、Android shadow 或 H2。当前 focused tests 为 `55/55`。
