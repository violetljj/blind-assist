# HFTF H0.1/H0.2 SANPO source-specific proxy authority result

日期：2026-08-01

workflow：`DEVELOPMENT_STANDARD`

终态：`HFTF_H0_2_INDEPENDENT_SESSION_REPLICATION_ADMITTED`

下一步：`H1_GEOMETRY_TEACHER_CANARY`

主线/App：`UNCHANGED / UNCHANGED`

## 结论

HFTF 已从“通用 H0 只能做静态投影”推进到可执行 H1 geometry teacher canary。准入只
覆盖 SANPO-Synthetic 的 source-derived geometry proxy mechanics，不覆盖物理人体标定、
精确 capture time、真实风险、student 效果、Android、提醒、安全或生产。

## H0.1 发现

source-specific verifier 固定：

- official repository commit
  `11faca999b5c223b804cd3196541a1427834918b`；
- `sanpo_dataset/lib/common.py` SHA-256
  `25f93fbe61a61fff61cccf29c4bb0047cbbc120eea3f51b67c64dd123412043e`；
- replay `description.json` 与 `camera_poses.csv` 的 GCS object name、generation、
  size、MD5、CRC32C 和本地 MD5；
- official loader 的 `camera_poses[frame_num]` 与相同 `frame_num` RGB/depth/mask
  filename 规则。

因此 pose row ↔ source frame index 获得 source-specific 权威。`frame_num / session
fps` 只叫 nominal relative time，不叫实测 capture timestamp。

发现会话 `e1ae36e0…de856` 的 48 个 signed-permutation/direction 假设比较结果：

| 指标 | 结果 |
| --- | ---: |
| 正确公式 | `p_world = R_xyzw @ p_opencv_camera + translation_m` |
| canonical rank | 1 |
| median relative depth error | `0.000564621` |
| p75 relative depth error | `0.001288149` |
| valid coverage | `0.700302` |
| local-ground plane frames | `25/25` |
| source-derived vertical | `+Z` |
| median camera-to-plane proxy | `1.283588 m` |
| clearance IQR | `0.071925 m` |
| median plane residual | `0.007686 m` |

H0.1 报告 SHA-256：
`5141cbd59250ec8aee095365518185b482dfb56e31be32934429654ab68abb60`。

## H0.2 独立 session 复现

选择规则在读取几何 outcome 前固定为：official SANPO-Synthetic train split 中，排除
H0.1 session 后，按 session ID 字典序选择具有 `camera_chest/left`、intrinsics、pose、
RGB、mask 与 metric depth 的最小三个 eligible sessions。frames 只是重复观测，独立
单元是 source session。

H0.2 不要求每个 session 重新发现 48 假设中的唯一公式；它冻结 H0.1 canonical
transform，要求该公式在每个新 session 中 rank 1、误差/coverage 过门，并独立复算局部
地面代理。

| session | frames | canonical median/p75 | coverage | vertical | camera-plane median/IQR | plane residual |
| --- | ---: | ---: | ---: | --- | ---: | ---: |
| `001217c6…910a` | 25 | `.000763/.001629` | `.962409` | `+Z` | `1.306577/.012023 m` | `.003688 m` |
| `0099b54c…864c` | 25 | `.000583/.001590` | `.596975` | `+Z` | `1.251454/.041488 m` | `.018647 m` |
| `00bdf8ce…5896` | 25 | `.000369/.000985` | `.772760` | `+Z` | `1.229026/.161340 m` | `.012879 m` |

三份 session 报告 SHA-256：

- `1ed2b83d7c82e427ecb4ce00c8e2b3ae16a76a65dae0a3af9deff81ce7d0c429`
- `c41da78654a21e9285c6d09d6ea1c2a69de0260d263dfd40692effa34c0b697b`
- `ec45665a8f760b7b8bfc3b75812e50b012c0b77f31ea74211482442146e59a04`

cohort 报告 SHA-256：
`79ae922cb38e65f2a89723238359c86cad959ecae293ca10b4e8c0c92df72059`。

## 协议修正与证据边界

原 H0 条款要求 H1 前获得物理 camera-to-body/ground calibration。独立资料审计确认：
官方格式没有这项 receipt；若 H1 要声称真实身体碰撞，该缺口确实必须停止。

本次按风险分层把 H1 收窄成可逆的 synthetic geometry-proxy mechanics：

1. 不使用默认人体身高或伪造外参；
2. 身体代理中心定义为 camera world position 到每帧 source-derived local ground plane
   的正交投影；
3. 相机到平面的距离只作 proxy 尺度诊断，不称为 wearer 身高或官方标定；
4. H1 成功上限仍只是 `GEOMETRY_PROXY_MECHANISM_SUPPORTED`；
5. H2/H3、participant/event、产品与安全层仍要求独立权威。

这是对过度严格前置条件的显式、可审计替换，不是把缺失 receipt 静默当成已存在。

## 坐标标签冲突

official loader feature 名称为 `camera_quaternions_right_handed_y_up`，但四个当前回放的
metric-depth reprojection 与 semantic-ground plane 都导出 `+Z` vertical。处理规则：

- 当前 evidence version 使用 source-derived `+Z`；
- 不外推到 SANPO-Real 或其他 SANPO versions；
- 不声称 official feature label 全局错误；
- 任何跨版本复用都必须重新运行 source-specific verifier。

## 明确未获得

- physical camera-to-person/body calibration；
- participant-specific body dimensions；
- exact capture timestamp、drop/jitter 或 clock synchronization；
- human event/collision/safety truth；
- multi-height 或 future representation 增量；
- causal RGB student、事件级 utility、主线晋级；
- Android、提醒、TTS、振动、默认 App 或生产权限。
