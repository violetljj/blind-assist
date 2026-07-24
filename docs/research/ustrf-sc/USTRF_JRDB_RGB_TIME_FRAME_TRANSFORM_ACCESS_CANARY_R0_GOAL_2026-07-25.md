# USTRF JRDB RGB/time/frame-transform access canary R0（2026-07-25）

状态：`FROZEN_BEFORE_EXECUTION`

## 唯一问题

不绕过 JRDB 登录、许可或资源门时，官方公开材料能否把同一 test sequence/frame 的 stitched RGB、capture timestamp 与 label/calibration transform 一起物化为最小 canary？

## 冻结输入

- JRDB 未登录公开下载页的日期化 HTML；
- 官方 `JRDB_sample_structure.zip`；
- 官方 `jrdb_toolkit` 固定 commit `4fbf7d6eba3255746000eb8c15f707af69561c5d` 中的 README、visualiser 与 calibration；
- 前一阶段已审计的 test labels 只作为 frame-id schema 背景，不在本阶段读取 signal、route truth 或 outcome。

## 合法终态（按优先级）

1. `FAIL_CLOSED_AUDIT_INCOMPLETE`
2. `ACCESS_BLOCKED_LOGIN_REQUIRED`
3. `FRAME_IDENTITY_OR_TIME_AUTHORITY_INSUFFICIENT`
4. `RGB_TIME_TRANSFORM_CANARY_PRESENT`

`RGB_TIME_TRANSFORM_CANARY_PRESENT` 必须同时有真实 RGB 文件、独立 capture timestamp、同 sequence/frame identity，以及官方或可核验的 projection/calibration 绑定。目录名、标签键名或固定帧率不能替代 capture time。

## 资源与权限

- 不猜测受限 archive URL，不读取 cookie/password，不绕过登录；
- 不下载全量 RGB、point cloud 或 rosbag；
- 公开 metadata canary 总下载预算 16 MiB；
- 只允许 source-access/authority 结论；G1、signal、route truth、Android、human、production 均关闭。
