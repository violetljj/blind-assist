# RCLE Source Authority Repair R1 result

日期：2026-07-27

## 结论

`SOURCE_AUTHORITY_REPAIR_R1_PASS / TWO_SOURCE_FAMILIES_AUTHORITY_READY`

本结果只闭合 OpenLORIS corridor 与 MultiScan 的 exact-source authority。它不授予
motion role、candidate geometry、RGB algorithm、Android、产品或安全 authority。

## OpenLORIS corridor

`OPENLORIS_CORRIDOR_AUTHORITY_PASS`

- exact captures：`corridor1-1`、`corridor1-2`；
- `corridor1-1`：HF LFS SHA-256
  `c7ff1a472ca54da82198521eda8c18f2065691075a05e706880f7fb58fda8415`，
  `13,853,763,765` bytes；
- `corridor1-2`：outer HF LFS SHA-256
  `a2de290b85bdefc5388aad27858125206a19b007d9b735de5faff954cc473413`，
  nested 7z exact slice `offset=512 / length=6,075,166,411`；
- complete directories：`42,601 / 17,408` members，
  `42,595 / 17,402` files；
- 每个 file 均记录 path、uncompressed bytes 与 7z CRC32；发布方没有逐 member
  SHA-256，明确记为 `NOT_PUBLISHED`，不把 archive SHA 冒充 member SHA；
- range-only header reads：`351,285 + 144,534 = 495,819` bytes；
- member extraction、geometry access、RGB visual access、RCLE algorithm execution
  均为 `0`；
- 独立 validator：`PASS / errors=[]`，receipt SHA-256
  `c18428c3fd47ae3b9fb8ed260b253de6560e4f2d6f7368eb26e12535e5a23856`。

历史访问向量已按 metadata identity、payload presence、geometry access、RGB visual
access、other/claim-relevant outcome access 与 selection influence 分项闭合，详见
[checkpoint](RCLE_SOURCE_AUTHORITY_REPAIR_R1_CHECKPOINT_2026-07-27.md)。

## MultiScan

`MULTISCAN_SOURCE_AUTHORITY_PASS`

用户完成 `CC BY-NC 4.0` 接受后，受控浏览器验证 acknowledgment form 消失且 Files
tree 解锁。本轮没有代用户接受许可，也没有读取、输出或记录 token。

| capture | bytes | LFS SHA-256 | frames/frequency | nominal duration |
| --- | ---: | --- | ---: | ---: |
| `scene_00000_00` | `594,277,493` | `4f7278e8…31d0` | `5763 @ 60 Hz` | `96.05 s` |
| `scene_00000_01` | `783,429,134` | `c085b30e…450c` | `7789 @ 60 Hz` | `129.8167 s` |

两个 capture 的 RGB、depth、confidence 与 camera-info 均有相同 source-native
frame count/frequency；camera-info JSONL 行数分别为 `5763 / 7789`，全量 timestamp
严格递增，首末行均含 timestamp、intrinsics、transform、quaternion、euler_angles
和 exposure_duration。metadata member CRC32 与 ZIP central directory 一致。

只读取 EOCD、central directory、local header 与 exact `.json/.jsonl` metadata，
合计 `2,930,400` range bytes。完整 ZIP、RGB、depth、confidence、mesh 与 texture
bytes 均为 `0`。

独立 validator 重放官方 LFS pointer 并复核所有 duration/sync/firewall：
`MULTISCAN_SOURCE_AUTHORITY_PASS / errors=[]`，receipt SHA-256
`9595aea00fe51b6211e8061c46dce381eba4aad85d2978a5bd2a0ffca37f67fe`。
token-only HTML 无法重放 Xet UI hash，故 Xet hash 明确作为 browser-session
附加证据；可独立重放的 LFS SHA-256 与 exact bytes 是硬身份门。

机器审计：
[RCLE_MULTISCAN_SOURCE_AUTHORITY_R1_AUDIT_2026-07-27.json](RCLE_MULTISCAN_SOURCE_AUTHORITY_R1_AUDIT_2026-07-27.json)

## Gate 结论

source-authority pass count 为 `2/2`。因此后继可以、且只能另立并评审
Source Discovery R1 candidate lock；本结果本身不允许 payload acquisition。

后继 candidate lock 已作为独立 artifact 建立并评审，不回写本 source-authority
结果。
