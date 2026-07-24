# USTRF JRDB 单帧 RGB/time/transform canary R1（2026-07-25）

状态：`FROZEN_BEFORE_EXECUTION`

## 唯一问题

在不下载 22.5 GB image archive 的前提下，能否通过官方 byte-range、timestamps、calibration 与 labels，把同一 JRDB test sequence/frame 的真实 stitched JPEG、独立 capture timestamp 和静态 projection/calibration 合法绑定？

## 冻结 canary

- sequence：`cubberly-auditorium-2019-04-22_1`
- label/image frame：`000000.jpg`
- timestamp camera：`stitched_image0`
- image archive：登录后官方清单所列 `https://jrdb.erc.monash.edu/static/downloads/test_images.zip`
- archive identity：22,527,101,047 bytes；ETag `"53eb84877-5e0c07f2eed9c"`
- ZIP64：166,135 entries；central directory offset 22,505,185,483；size 21,915,466 bytes

选择首序列首帧只为最小运输/authority canary，不承担性能、路线或事件代表性。

## 资源门

- 单次 producer 或 validator 网络读取上限：64 MiB；
- 只允许读取 ZIP64 central directory、唯一目标成员 local header 与该成员压缩数据；
- 禁止下载 full archive、pointcloud、rosbag 或整条 sequence；
- 任一 HTTP range 退化为 full response、成员不唯一、预算超限、CRC/尺寸不符都 fail closed。

16 MiB 旧门无法容纳 21.9 MB central directory，因此在读取 central directory 前显式版本化为 64 MiB；不是运行后放宽结果门。

## 合法终态（按优先级）

1. `FAIL_CLOSED_AUDIT_INCOMPLETE`
2. `RANGE_EXTRACTION_RESOURCE_BLOCKED`
3. `SAME_FRAME_IDENTITY_OR_TIME_INSUFFICIENT`
4. `RGB_TIME_TRANSFORM_CANARY_PRESENT`

成功必须同时证明：

- labels 存在 `000000.jpg`；
- timestamps 的 `stitched_image0` URL 以同一 `sequence/000000.jpg` 结尾，并给出数值 capture timestamp；
- remote ZIP 中唯一成员以 `images/image_stitched/<sequence>/000000.jpg` 结尾；
- JPEG CRC、解压尺寸与 SOF geometry 有效；
- calibration archive 与官方 toolkit calibration hash-bound。

## 权限

只授权 source-authority canary。G1、signal、route truth、event、Android、human、independent walking、production、commit、push、PR 均关闭。

