# USTRF JRDB 单帧 RGB/time/transform canary R1 结果（2026-07-25）

状态：`RGB_TIME_TRANSFORM_CANARY_PRESENT / VALID`

权限：`SOURCE_AUTHORITY_CANARY_ONLY / G1_CLOSED / SIGNAL_CLOSED / ROUTE_TRUTH_CLOSED / ANDROID_CLOSED / HUMAN_CLOSED / PRODUCTION_CLOSED`

## 结论

用户自行建立 JRDB 登录态后，登录页首次公开了官方 image/timestamp/calibration 精确 URL 与体积。R1 没有下载 22.5 GB image archive，而是通过官方 `Accept-Ranges: bytes` 对 ZIP64 central directory 和一个目标成员做有界读取，首次把同一 test sequence/frame 的四类权威证据闭合：

1. test labels 中存在 `cubberly-auditorium-2019-04-22_1 / 000000.jpg`，含 9 个对象；
2. `frames_img.json` 将 `stitched_image0/000000.jpg` 绑定到 capture timestamp `1555960991.4668088`；
3. remote ZIP 唯一成员为 `images/image_stitched/cubberly-auditorium-2019-04-22_1/000000.jpg`；
4. 2019 calibration archive 与官方 toolkit 的 `defaults.yaml`、`cameras.yaml` 均 hash-bound。

独立 validator 重新读取 remote central directory 与同一 JPEG，完整复算为：

`RGB_TIME_TRANSFORM_CANARY_PRESENT / VALID`

这解除的是“JRDB 新来源 RGB/time/frame identity 能否合法物化”的 access/transport blocker，不改写当前 41-sequence G0 的 `SOURCE_AUTHORITY_ABSENT`，也不自动授权 G1、route truth 或任何 signal。

## 登录后官方清单

登录后的 JRDB 页面列出：

- JRDB 2022 test images：22.6 GB；
- JRDB 2022 test calibration：5.5 MB；
- JRDB 2019 test images：21 GB；
- JRDB 2019 test timestamps：1.9 MB；
- JRDB 2019 test calibration：4 KB。

本阶段采用旧版 test images/timestamps/calibration，因为三者对同一 27 条 test sequence 提供明确绑定；27/27 timestamp sequence 与已下载 test labels sequence 精确重合。登录、下载链接发现和账户操作均由用户完成或授权；脚本不读取 cookie、password 或 session store。

## ZIP64 与资源门

remote archive：

| 项目 | 结果 |
| --- | ---: |
| archive bytes | 22,527,101,047 |
| ETag | `"53eb84877-5e0c07f2eed9c"` |
| ZIP64 entries | 166,135 |
| central directory offset | 22,505,185,483 |
| central directory bytes | 21,915,466 |
| producer network bytes | 22,257,329 |
| validator network bytes | 22,257,329 |
| per-process budget | 67,108,864 |
| full archive downloaded | false |

原 16 MiB access-audit 门小于 21.9 MB central directory，因此在读取目录前另立 R1 并显式冻结 64 MiB；没有在结果后放宽。每个进程只做四个 range：

- 完整 central directory；
- 30-byte local header；
- 93-byte filename/extra；
- 341,740-byte compressed JPEG。

任何 206/Content-Range、预算、唯一成员、CRC、解压尺寸或 JPEG SOF 不一致都会 fail closed。

## 单帧结果

| 项目 | 结果 |
| --- | --- |
| sequence | `cubberly-auditorium-2019-04-22_1` |
| frame | `000000.jpg` |
| source timestamp row id | `2` |
| capture timestamp | `1555960991.4668088` |
| label objects | 9 |
| JPEG | 3760×480、8-bit、3 components |
| uncompressed bytes | 354,706 |
| SHA-256 | `d958f5a8e9416409580726c44c25175cbfa08ddc84e88be55ff6876d8156e676` |
| CRC-32 | `2380758809` |

可视检查确认它是有效 stitched indoor panorama；可视内容不承担 route-role、事件或性能 truth。

## 复算凭据

- config SHA-256：`1b40abd7b709f9c26b04e0c05348a8b3f289704860545af7d11fb9c79566a08c`
- producer PID：`57864`
- receipt SHA-256：`c190a220ef4d170c2f969c89e72ee9ff078a7952c791d8324aa721f340535d97`
- validator PID：`60072`
- validation SHA-256：`3b5cf4261a3bb4db693ac106d1bbad3eb5ec2d04f1e93896b4378f7eabd11c20`
- validator checks：schema、stage、PID isolation、deterministic recomputation、image hash、same-frame、bounded network、no-full-archive 与全部高权限关闭均为 true

## 下一合法边界

本结果只允许另立最小 `JRDB_RGB_CONTINUITY_AND_EGOMOTION_AVAILABILITY_R0`：

1. 冻结同一 sequence 的短连续窗口和总字节门；
2. 只评估 timestamp monotonicity、RGB continuity 与 background sparse-LK/RANSAC 的 availability/abstention；
3. 不读取 route role、event outcome 或 person truth 来调 ego-motion；
4. 不扩展到整条 sequence，除非短窗口先通过 transport/quality 门；
5. 若背景特征或 affine 质量不足，必须 abstain，不能默认 ego-motion=0。

route-role truth 仍不存在，因此不能把短窗口变成 G2/G4 signal、正式 validation 或 U0。

