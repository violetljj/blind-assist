# RCLE real positive-approach source-authority Discovery R0 result

日期：2026-07-27

## 终态

`SELECT_CID_SIMS_FLOOR3_AS_SOLE_NEXT_R2_CANDIDATE_FAMILY / VALID`

本次只完成 metadata/source-authority Discovery。唯一下一数据族候选冻结为：

`CID-SIMS V6 / office_building / floor3 / fixed three-run family`

它不是 role admission，也没有得到
`REAL_POSITIVE_APPROACH_ROLE_ADMITTED / VALID`。因此：

- ETH3D `sofa_3` R1 保持
  `HOLD_ALGORITHM_CANARY_APPROACH_ROLE_INCOMPLETE / VALID`，不修补、不重跑；
- 不创建或运行 geometry、RGB algorithm、performance qualification；
- 不下载 RGB、depth、pose、pointcloud 或 archive payload；
- 不判断 CID-SIMS、CoRBS 或 KITTI-360 谁能通过 approach 门；
- 不把 CoRBS 或 KITTI-360 设为同一正式 claim 的 fallback。

结果前冻结的
[Discovery R0 预注册](RCLE_REAL_POSITIVE_APPROACH_SOURCE_AUTHORITY_DISCOVERY_R0_PREREGISTRATION_2026-07-27.md)
SHA-256 为
`13ccc944366345287248f5980a55ca4eda395666f96fe4433f892bf54ca7720c`。
机器可读的完整 access vector、来源身份和后继限制见
[candidate lock](RCLE_REAL_POSITIVE_APPROACH_SOURCE_AUTHORITY_DISCOVERY_R0_CANDIDATE_LOCK_2026-07-27.json)，
其 SHA-256 为
`683ee8486d19a2ef3818a9c3b397166ef14ee73125dd791801c1053f02002728`。

## 冻结选择的解释

CID-SIMS 官网把 `Floor3` 描述为三个走廊 sequence，而 ScienceDB V6 的
file-tree metadata 将它闭合为三条精确 archive。因预注册没有在三条 run
之间冻结 tie-break，本版只允许选择唯一**数据族及固定成员全集**，不得事后
根据长度、行走描述或未来 geometry 结果挑其中一条。

下一份独立 R2 只能审计 CID-SIMS，并且必须在任何 payload GET 前用
outcome-blind deterministic rule 冻结 `floor3_1`、`floor3_2`、
`floor3_3` 中恰好一个。R2 claim 失败后不得换同族 run、CoRBS、KITTI-360、
mirror 或 repack。

## 三来源权威比较

| 数据族 | 精确身份与 archive | 许可、大小、checksum、入口 | 坐标链 | 本版处置 |
| --- | --- | --- | --- | --- |
| CID-SIMS Floor3 | ScienceDB DOI `10.57760/sciencedb.ai.00003`，dataset ID `358179a76abd47bea52f8f5aeecd301b`，V6；固定 `floor3_1/2/3.zip` | `CC BY-NC-ND 4.0`；三 archive 共 `9,809,610,684` bytes；每条有官方 MD5、file ID、精确 bytes，canonical HEAD 均 `200` | D455 对齐 RGB-D；`+Z` 前向；depth `/1000`；`T_DtoC=I`；camera-to-world pose；GeoSLAM trajectory/pointcloud 同一 world | 唯一下一数据族候选 |
| CoRBS E3 | `Electrical Cabinet / E3`；历史官方 `E3_raw.zip`、`E3_pre_registereddata.zip`、`E3_Trajectory.zip`、`Geometry_E.zip` | `CC BY 3.0`；当前四个官方 endpoint HEAD 均 `403`；官方 bytes/checksum 未发布或未核实 | Kinect v2；mocap + hand-eye 到 color camera；depth-to-color registration；scanner mesh 对齐同一 global frame | `HOLD_CURRENT_DOWNLOAD_AUTHORITY_INCOMPLETE`，只作未来新协议的 contingency map |
| KITTI-360 | 压力地图固定 `2013_05_28_drive_0000_sync`；官方组件为 perspective helper、`calibration.zip`、`data_poses.zip`、`data_3d_semantics.zip` | `CC BY-NC-SA 3.0`；下载/使用需注册并说明用途；只发布组件级 `128G / 3K / 8.9M / 12G`；官方 checksum 未核实 | perspective `+Z` 前向；IMU-to-world 与 cam0-to-world 均明确；无 source-native dense RGB-D depth，使用 Velodyne/SICK 和 world-frame static PLY | 仅 `VEHICLE_DOMAIN_STRESS_CANDIDATE`，不能替代室内候选 |

## CID-SIMS 固定来源身份

ScienceDB V6 metadata 给出：

| Run | V6 path | File ID | Bytes | Official MD5 |
| --- | --- | --- | ---: | --- |
| `floor3_1` | `/V6/CID-SIMS/office_building/floor3/floor3_1.zip` | `c595882daafe788a29d687872cc1fc2a` | 2,211,008,069 | `585d38855ad7d04817991cdbbb72016b` |
| `floor3_2` | `/V6/CID-SIMS/office_building/floor3/floor3_2.zip` | `4909999130e6752b5e2147a0684b59ac` | 3,274,014,381 | `e1a369f7c13cbb777a90d7e792085afa` |
| `floor3_3` | `/V6/CID-SIMS/office_building/floor3/floor3_3.zip` | `28066e8768d0dd9c7854cb25adfb6770` | 4,324,588,234 | `c5a4f709f5f47b698c002500a2d1856b` |

共享 static geometry 是
`/V6/CID-SIMS/office_building/floor3/pointcloud.las`，
file ID `1534d1103e7a30876a3998a02746f3be`，
`1,052,342,483` bytes，MD5
`d79247bcdca29fa5e1caf9ada30490fc`。本次只读取 file-tree metadata，
没有读取该对象。

CID-SIMS 官方页面说明 Floor3 是走廊中的三条真实 robot sequence，并提供
对齐 RGB/depth、内外参、camera-to-world ground truth 和同一 world frame
中的 3D pointcloud。官方 calibration 固定 `depth_factor=1000`、
`T_DtoC=identity`；D455 optical frame 的 `+Z` 为前向。因此未来 source-native
链为：

`aligned depth pixel -> D455 optical camera -> camera-to-world pose -> unified world/static pointcloud`

这只证明数据结构可审计，不证明任何 sequence 具有 positive approach outcome。

V6 数据许可为 `CC BY-NC-ND 4.0`：隔离的非商业内部研究可兼容，但必须署名，
不得商用，也不得公开分享改编后的 payload/派生材料。若项目用途转为商业，
该来源不能沿用当前许可结论。

## CoRBS E3 限制

DFKI 论文和历史官方 HTML 闭合：

- canonical sequence ID 是 `E3`；`Schaltkasten_Take10` 只是内部媒体 stem；
- duration `165.3 s`，trajectory `47.0 m`；
- `CC BY 3.0`；
- color/depth/IR、mocap trajectory 与 scanner geometry 共享 global frame。

但当前 historical official host 上四个精确对象都返回 `403`，没有
Content-Length；Wayback 只保留 HTML，未发现 exact ZIP body。官方 archive
大小与 checksum 均为 `NOT_PUBLISHED_OR_NOT_VERIFIED`。因此 E3 不能在当前
metadata authority 下成为可执行唯一候选。

## KITTI-360 限制

KITTI-360 官方文档完整定义 perspective camera、GPS/IMU、Velodyne/SICK、
world frame、`poses.txt` 与 `cam0_to_world.txt`；许可为
`CC BY-NC-SA 3.0`。但它要求注册并声明用途，且是 station-wagon urban driving
域，不提供 source-native dense RGB-D depth。即使来源权威比 CoRBS 当前入口
更稳定，也只能保留 vehicle-domain stress 角色，不能替代室内低位 robot
candidate，更不能形成 head-worn 或 human claim。

## 跨项目访问与独立性

在本 Discovery 新文件之外：

- tracked repository text hits：`0`；
- non-artifact worktree text hits：`0`；
- Git history candidate-string hits：`0`；
- `artifacts.local` exact-candidate filename hits：`0`。

因此没有发现三个 exact candidate 身份曾读取 geometry、RGB 或 claim-relevant
outcome，或曾影响当前选择。该结论限于本 checkout、Git history 和本机
artifact filename audit；不夸大为对所有外部机器或人员的全局证明。

本次访问向量为：

`metadata_identity=YES / payload_presence=NOT_PROBED / geometry_access=NO / rgb_visual_access=NO / other_algorithm_outcome_access=NO / claim_relevant_outcome_access=NO / selection_or_tuning_influence=METADATA_RULE_ONLY`

## 唯一合法后继

可以另立：

`RCLE_REAL_POSITIVE_APPROACH_ROLE_ADMISSION_R2_CID_SIMS`

但它在任何 GET 前必须：

1. 只从固定 `floor3_1/2/3` universe 中用结果盲规则选一个 exact archive；
2. 绑定该 archive 的 ScienceDB file ID、MD5、bytes 和 canonical URL；
3. 冻结 source descriptor SHA-256、完整 burned/exclusion 与 access vector；
4. 闭合 identity、ancestry、independence、reuse 与 confirmation isolation；
5. 冻结 admission contract 和 exclusive claim。

只有该独立 R2 得到
`REAL_POSITIVE_APPROACH_ROLE_ADMITTED / VALID`，才可以另立 algorithm
implementation 与 performance qualification。Discovery R0 本身没有这些权限。

## 官方来源

- CID-SIMS：
  [ScienceDB DOI / V6 authority](https://doi.org/10.57760/sciencedb.ai.00003)、
  [official overview](https://cid-sims.github.io/overview/index.html)、
  [calibration](https://cid-sims.github.io/calibration/index.html)、
  [ground truth](https://cid-sims.github.io/groundtruth/index.html)、
  [published calibration.yaml](https://raw.githubusercontent.com/CID-SIMS/CID-SIMS.github.io/main/calibration/calibration.yaml)。
- CoRBS：
  [DFKI publication authority](https://www.dfki.de/web/forschung/projekte-publikationen/publikation/8230)、
  [historical official E3 page](https://web.archive.org/web/20160604072058id_/http://corbs.dfki.uni-kl.de/electrical-cabinet/)、
  [WACV paper](https://www.cs.princeton.edu/courses/archive/fall16/cos526/papers/wasenmuller16.pdf)。
- KITTI-360：
  [official dataset and license](https://www.cvlibs.net/datasets/kitti-360/index.php)、
  [official frames and coordinate chain](https://www.cvlibs.net/datasets/kitti-360/documentation.php)、
  [official component sizes and entries](https://www.cvlibs.net/datasets/kitti-360/download.php)、
  [official utility repository](https://github.com/autonomousvision/kitti360Scripts)。
