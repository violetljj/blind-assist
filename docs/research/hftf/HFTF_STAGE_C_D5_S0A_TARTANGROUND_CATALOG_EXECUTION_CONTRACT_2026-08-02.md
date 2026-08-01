# HFTF Stage C D5-S0A：TartanGround `Data_diff` 精确提交目录锁定合同

## 结论先行

本合同只授权一次目录级盘点：在正式提交并推送后，把官方
`castacks/tartanairpy` 仓库精确提交
`158a6844d782942110967325ca3082f50ab2bfc7` fetch 到唯一的
`artifacts.local` 根，读取该提交中的 `.gitmodules` 与
`tartanair/download_ground_files.txt` 两个 Git 对象，核对三个 gitlink，
并锁定清单实际枚举的 `Data_diff/P1xxx` 父体。

它不授权子模块 checkout，不请求数据托管端，不打开 ZIP，不读取图像、深度、
分割、pose 或 metadata 成员。目录容量达标也只能到达
`D5_S0A_TARTANGROUND_DIFF_CATALOG_LOCKED_REQUIRES_S0B_STRUCTURAL_AUTHORITY`；
不能据此宣称 D5-S0 可行，更不能进入机会生态、效应或学生模型执行。

## 冻结源身份

- 官方工具仓库：`https://github.com/castacks/tartanairpy.git`
- 根提交：`158a6844d782942110967325ca3082f50ab2bfc7`
- 清单对象：`tartanair/download_ground_files.txt`
- 子模块声明对象：`.gitmodules`
- 三个 gitlink 的 path、URL 与 commit 均由机器合同逐项冻结。
- 不执行 `git submodule init/update`，也不读取子模块对象。

## 一次性顺序

1. 本地验证设计、实现、测试的精确哈希以及 `HEAD == origin/master`。
2. 独占创建唯一证据根，写入、`fsync`、关闭并重新读取验证 `attempt.json`
   和 `preflight.json`。
3. 只执行一次带 `--recurse-submodules=no` 的精确提交 fetch；任何 transport、
   对象、格式、gitlink 或清单
   漂移都关闭为 `D5_S0A_TARTANGROUND_DIFF_CATALOG_INVALID_STOP`，不得续跑或重跑。
4. 读取两个允许的 Git 对象，按清单实际行枚举父体；正则只验身份，不生成父体。
5. 写入完整 `catalog.json`，然后写入绑定哈希链的 `result.json`。

## 目录门与含义

目录完整父体必须同时列出：

- `image_lcam_front.zip`
- `depth_lcam_front.zip`
- `seg_lcam_front.zip`
- `metadata.zip`

每个清单行还必须服从官方清单的 `<path> <positive-decimal> G` 结构；空白清单、
重复路径、不安全相对路径或异常 size 结构都使一次性盘点 INVALID。

目录容量门保持 D5 设计中预先冻结的至少 64 个不同轨迹父体、至少 8 个环境。
该门只检验目录容量和覆盖范围。它不检验 ZIP 内共同时间线、动态 pose、机器人
高度或精确 robot-camera 外参；这些只能由另一个先冻结、后执行的 D5-S0B 合同
决定。相同环境内轨迹仍视为聚类，不能冒充独立样本。

## 永久禁止

- 请求数据 ZIP 的 URL、HEAD、Range 或内容；
- 初始化、更新或 checkout 子模块；
- 解码或保留任何图像、深度、分割、pose、IMU、LiDAR 或 occupancy；
- 计算机会、support、truth、clearance、effect 或学生输出；
- 把合成差速机器人代理提升为盲人安全、生产、默认 App 或主线证据。
