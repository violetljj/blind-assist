# HFTF Stage C D5 TartanGround differential-drive source authority S0

## 结论

D4-R0 已因 transport failure 永久关闭。D5 不修 D4，也不复用 SANPO 的 1442 个已尝试
parents；它把 source population 改为 **TartanGround differential-drive trajectories**，
先问这个新总体是否具备足够、可验证、可前瞻抽样的 HFTF 输入权威。

父终态见 [D4-M0 invalid result](HFTF_STAGE_C_D4_M0_METADATA_CENSUS_INVALID_RESULT_2026-08-02.md)。
官方资料称 TartanGround 覆盖 63 个环境、878 条 ground-robot trajectories 和约
144 万 samples，提供同步多视角 RGB、metric depth、semantic segmentation、6-DoF
pose、IMU、LiDAR 与 semantic occupancy；目录把 differential-drive 单独放在
`Data_diff/P1xxx`，并提供 `lcam_front` 与 robot-height metadata。
来源：[dataset page](https://tartanair.org/tartanground/)、
[official documentation](https://tartanair.org/tartanground.html)、
[paper](https://arxiv.org/abs/2505.10696)、
[toolkit](https://github.com/castacks/tartanairpy)。

这些 publisher claims 目前只是候选理由，不是本地 inventory evidence。

## S0 只做什么

本文件只是 S0 **设计**。提交推送后也不直接授权 clone、list 或 remote request；
必须再冻结一个独立 execution contract，先绑定 exact toolkit/submodule commits、
implementation/tests、remote manifest identity、allowed calls、attempt-first、canonical
receipts、payload sentinel 和 failure closure。

该 execution contract 最多可授权：读 bound manifest；对 ZIP 做 bounded central-directory
range reads；只提取 exact trajectory metadata JSON 以验证 robot height/extrinsic；对
`lcam_front` pose member 只流式计算 SHA/行数，不解析、保留或按 pose 值排序。

除上述 exact metadata JSON 与 exact pose-member structural authority reads 外，禁止
下载或打开 RGB、depth、seg、LiDAR、IMU、occupancy payload 或 full pose archive；
禁止解析/保留 pose values，也禁止计算 support、truth、opportunity、clearance、
effect 或 student。

## 前置门

只有同时满足以下门，S0 才可裁决 source population feasible：

- 至少 64 个 distinct differential-drive trajectory parents；
- 至少跨 8 个 distinct environments；
- 每个 parent 同时有 robot height **和** exact robot-camera extrinsic authority；
- `lcam_front` image/depth/seg/dynamic-pose 具有共同至少 25 个连续 10 Hz raw frames；
- 只使用 raw indices `0,2,…,24` 形成 13 个 5 Hz frames，跨度 2.4 s；
- 不按 scene content、semantic class 或 future opportunity 排序。

UNKNOWN 不能默认存在；缺失或无法绑定的 modality 直接 metadata-ineligible。通过也只
授权另冻 D5-M0 parent allocation/acquisition contract，不自动授权任何 payload。
如果 manifest + bounded authority reads 根本看不到 height/extrinsic/common timeline，
整个 S0 必须是 `SOURCE_AUTHORITY_NOT_EVALUABLE`，不能把所有 parents 记成 ineligible
后误报 pool insufficient。

64 trajectories + 8 environments 只是一条容量/覆盖门，不证明 64 个统计独立单位。
同环境 trajectories 是 cluster；未来 ecology/effect 必须 environment-disjoint，并以
environment 为独立单位，或在任何 payload 前另冻 cluster estimand 与 cluster-aware inference。

## Claim ceiling

TartanGround 是 synthetic ground-robot proxy。Differential-drive motion 不是盲人步态，
semantic labels 不是人类 safety truth，robot height 也不自动等于人体 body envelope。
S0 成功只证明一个新 source population 的元数据可招募性，不证明 opportunity prevalence、
HFTF effect、主线替代、App/Android、生产或 safety。

机器设计 SHA-256：
`122eccb74d0eb83e231c4e1fa02a36284bab9e6b5df7d251845a7284eeff6b2d`。
[机器设计 JSON](HFTF_STAGE_C_D5_TARTANGROUND_DIFF_SOURCE_AUTHORITY_AND_RECRUITABILITY_S0_2026-08-02.json)。
