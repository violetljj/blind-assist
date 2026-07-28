# RCLE source authority repair R1 checkpoint

日期：2026-07-27

冻结合同：
[RCLE_SOURCE_AUTHORITY_REPAIR_R1_CONTRACT_2026-07-27.json](RCLE_SOURCE_AUTHORITY_REPAIR_R1_CONTRACT_2026-07-27.json)

当前状态：

`OPENLORIS_CORRIDOR_AUTHORITY_PASS / MULTISCAN_AUTHORITY_PASS`

当前外部 cohort 仍为：

`EXTERNAL_COHORT_NOT_EVALUABLE`

## OpenLORIS corridor

本轮闭合了 `corridor1-1` 与 `corridor1-2` 的最小范围历史访问向量和完整
container member directory，没有提取任何 member、查看 RGB、读取 pose/depth 数值或
运行 RCLE 算法。

历史与本轮访问向量：

| 字段 | 状态 | 范围与说明 |
| --- | --- | --- |
| `metadata_identity` | `YES` | 官方页面、GitHub/Hugging Face tree/API、exact path/bytes/LFS identity |
| `payload_presence` | `NO_FULL_PAYLOAD / DIRECTORY_RANGES_ONLY` | 本机未发现 corridor payload；本轮仅取容器签名/目录 range |
| `geometry_access` | `NO` | 未提取或读取 pose、depth、calibration、timestamp member 内容 |
| `rgb_visual_access` | `NO` | 未提取 RGB member，未看 preview/contact sheet/video |
| `other_algorithm_outcome_access` | `YES` | 旧官方/论文页面曾自动返回 SLAM benchmark 描述；未参与本轮 member audit |
| `claim_relevant_outcome_access` | `NO` | 未读取 RCLE 或同机制 outcome |
| `selection_or_tuning_influence` | `YES_METADATA_ONLY` | OpenLORIS 由公开 metadata 进入 authority repair；未影响窗、门、算法或参数 |

完整 member identity：

| capture | cryptographic binding | directory | required modalities |
| --- | --- | ---: | --- |
| `corridor1-1` | HF LFS SHA-256 `c7ff1a472ca54da82198521eda8c18f2065691075a05e706880f7fb58fda8415`，`13,853,763,765` bytes | `42,601` members；`42,595` files 全部有 7z CRC32 | `8514` color、depth、aligned-depth frames；`groundtruth.txt`、`sensors.yaml`、`trans_matrix.yaml` 与时间索引均存在 |
| `corridor1-2` | 外层 HF LFS SHA-256 `a2de290b85bdefc5388aad27858125206a19b007d9b735de5faff954cc473413`；nested 7z 为外层 immutable TAR 的 exact slice `offset=512, length=6,075,166,411` | `17,408` members；`17,402` files 全部有 7z CRC32 | `3477` color、depth、aligned-depth frames；同一组 groundtruth/calibration/time-index member 均存在 |

这里的 member hash 不偷换概念：

- 发布方 cryptographic hash 绑定 direct LFS object，或绑定 outer hashed object 中的 exact
  byte slice；
- 7z 目录为每个文件提供 path、uncompressed bytes 与 CRC32；
- 没有发布方逐 member SHA-256，因此记录为 `NOT_PUBLISHED`，不把 archive SHA-256
  冒充逐 member SHA-256；
- 两份完整目录 receipt 自身 SHA-256 分别为
  `7137b377d81c719b2f7644318bc0bd7785b46b16c97094d47936a0d0711d063c` 与
  `2e6c142008b46ffc692e16819b4206e289bfb3e249ff25adfa8ef476c3226840`。

range-only 目录审计实际读取 `351,285 + 144,534` bytes，共 18 个成功 HTTP range；
member extraction 为 `false`。完整 receipt 位于
`artifacts.local/evidence/rcle_source_authority_repair_r1/`。

独立离线 validator 不发起网络请求、不复用目录 producer，重新检查冻结 exact
capture 集合、`60,009` members、`59,997` files、逐文件 CRC32、required modality
计数、range guard、container identity 与所有 firewall，结果为 `PASS / errors=[]`。
验证 receipt SHA-256 为
`c18428c3fd47ae3b9fb8ed260b253de6560e4f2d6f7368eb26e12535e5a23856`。

官方来源：

- <https://lifelong-robotic-vision.github.io/dataset/scene.html>
- <https://huggingface.co/datasets/shixuesong/openloris-scene>
- <https://github.com/lifelong-robotic-vision/OpenLORIS-Scene/blob/master/download.md>

结论：两个 exact capture 的 source/container/member identity 与访问防火墙足以通过
本轮 source-authority gate；这不授予 geometry role，也不签发 candidate lock。

## MultiScan

用户报告已接受许可后，受控浏览器实时复核显示 acknowledgment form 已消失，
`Files` tree 可访问，故
`MULTISCAN_LICENSE_ACCEPTANCE=USER_REPORTED_AND_CONTROLLED_ACCESS_VERIFIED`。本轮没有
代用户提交许可表单，也没有读取 cookie、token 或其他登录凭据。

冻结的两个 exact capture 均已闭合 publisher-controlled Git LFS/Xet identity：

| capture | exact path | bytes | LFS SHA-256 | Xet hash |
| --- | --- | ---: | --- | --- |
| `scene_00000_00` | `scans/scene_00000_00.zip` | `594,277,493` | `4f7278e8…31d0` | `5b6a6004…b27c` |
| `scene_00000_01` | `scans/scene_00000_01.zip` | `783,429,134` | `c085b30e…450c` | `89a66e40…468f` |

两者绑定同一发布 commit `93674c024ea7371a8ac1ac308e71b6c96093f0f5`。选择是
冻结首 capture 加同 scene 的字典序配对 scan，没有使用语义描述、RGB preview 或
motion-role inference。

官方 acquired-data 文档逐字段给出 `scene_00000_00.json`：RGB、depth、confidence 与
camera-info 均为 `5763` frames、`60 Hz`，nominal duration `96.05 s`；camera-info
JSONL 每帧含 timestamp、intrinsics、pose transform、quaternion、euler angles 与
exposure duration。因此该 capture 的 `>=30 s` 与 source-native frame-count/frequency
同步关系为 `PASS`。

登录 official CLI 后，按
[metadata access clarification](RCLE_SOURCE_AUTHORITY_REPAIR_R1_METADATA_ACCESS_CLARIFICATION_2026-07-27.json)
只读取两个 ZIP 的 EOCD、central directory、local header 与 exact `.json/.jsonl`
metadata member。合计 range bytes 为 `2,930,400`，远低于 `32 MiB` guard；完整 ZIP、
RGB、depth、confidence、mesh 与 texture bytes 均为 `0`。

| capture | source-native streams | nominal duration | camera-info | timestamp |
| --- | --- | ---: | ---: | --- |
| `scene_00000_00` | RGB/depth/confidence/camera-info 均 `5763 @ 60 Hz` | `96.05 s` | `5763` JSONL lines | 严格递增；raw span `96,073,214,084` |
| `scene_00000_01` | RGB/depth/confidence/camera-info 均 `7789 @ 60 Hz` | `129.8167 s` | `7789` JSONL lines | 严格递增；raw span `129,857,997,333` |

两份 JSONL 的首末行均包含 timestamp、intrinsics、transform、quaternion、
euler_angles 与 exposure_duration；逐行 timestamp 全量检查严格递增。metadata
member bytes 与 ZIP central-directory CRC32 均一致。

独立 validator 不 import producer，重新检查 exact capture 集合、远程 LFS pointer、
archive bytes、metadata member set/offset/CRC、range byte accounting、四流帧数/频率、
JSONL 行数、duration、timestamp 与 firewall，结果
`MULTISCAN_SOURCE_AUTHORITY_PASS / errors=[]`；validation receipt SHA-256 为
`9595aea00fe51b6211e8061c46dce381eba4aad85d2978a5bd2a0ffca37f67fe`。
token-only HTML 对 blob page 返回 `401`，故 Xet hash 的独立重放明确记为
browser-session observation only；可独立重放的官方 LFS SHA-256 与 bytes 是硬身份门。

机器可读审计：
[RCLE_MULTISCAN_SOURCE_AUTHORITY_R1_AUDIT_2026-07-27.json](RCLE_MULTISCAN_SOURCE_AUTHORITY_R1_AUDIT_2026-07-27.json)

## 停止边界

OpenLORIS corridor 与 MultiScan 现均通过 source-authority gate，满足 `2/2`。
这只允许另立并评审 Source Discovery R1 candidate lock；在该 lock 的独立 review
PASS 前仍不得下载 candidate geometry payload 或运行 geometry-only 选窗。候选列表、门、
算法和 Android 权限均未改变。

## 后继 Source Discovery R1

独立 candidate lock SHA-256
`c1a0ea53dc698b1f12107db9951f7ecf88def7aa4aed2d2edd0e186093cb5a3c`
已通过 separate review；review errors `[]`，且 review 前 payload root 为 `0 bytes`。

review PASS 后的 OpenLORIS geometry-only transport preflight 发现两个 archive 均为
solid 7z，depth/aligned-depth 与 color member 在同一 solid stream 中交错。冻结
10 秒网格共有 `28 + 11 = 39` 个 cadence-eligible windows，但每窗在到达最后一个
所需 aligned-depth member 前至少经过 `299` 个 color members。因此 reviewed lock
下 RGB-free geometry acquisition window 为 `0`。

独立 validator 结果：
`OPENLORIS_GEOMETRY_ONLY_TRANSPORT_NOT_EVALUABLE_VALID / errors=[]`，
receipt SHA-256
`b03d5b354c2df6fc610d3419f113be800bc14059886d319126b9eb92efcddf98`。

按冻结 candidate order 与 `transport failure` stop rule，没有下载任何 candidate
geometry payload，没有继续 MultiScan depth acquisition，也没有运行 geometry formula。
后继正式终态见
[Source Discovery R1 result](RCLE_UNSEEN_EXTERNAL_CONFIRMATION_SOURCE_DISCOVERY_R1_RESULT_2026-07-27.md)：
`EXTERNAL_COHORT_NOT_EVALUABLE / VALID`。
