# USTRF sensor replay R3：OpenLORIS 来源替换结果（2026-07-22）

状态：`SOURCE_REPLACEMENT_EXHAUSTED / admitted=0 / three_source_count_credit=false / DO_NOT_SELECT_HARDWARE`

## 结论

本轮只替换来源，没有修改 R3 的 `horizon`、`unknown`、candidate、事件门或审核阈值。OpenLORIS dynamic office/cafe 的 9 条官方轨迹已全部完成预筛：7 条 office 在稀疏视觉预筛被拒绝；2 条 cafe 虽进入完整连续片段审核，但都被两位隔离 reviewer 一致拒绝。因此没有轨迹获得 route/event truth authority，三源计数仍为 `0/3`，冻结 evaluator 不运行。

## 冻结边界

- R3 prereg：`configs/ustrf_sensor_replay_r3_prereg_v1.json`，SHA-256 `3aa3fdb460c697d6d669e2174f7f7c9d17f1fc06b6f6392d35ba1ac5b2b73eaa`，字节未变。
- 来源替换配置：`configs/ustrf_sensor_replay_r3_openloris_prescreen_v1.json`，SHA-256 `e7c9268133560e23972ea93dd484e94b6bdbce7a8557c29193a49f6e926390ad`。
- 完整审核继续要求：两位 reviewer 都准入、互相输出不可见、candidate 告警不可见、每个连续帧恰好出现一次、事件锚点容差 `15 frames`。
- 只有至少 3 条完整片段都通过双模型审核后，才可获得三源计数并运行同一冻结 evaluator。

## 来源与坐标处理

OpenLORIS 官方说明 D435i 提供 30 Hz color/depth/aligned depth；office 使用 OptiTrack ground truth，cafe 使用离线 LiDAR SLAM ground truth。来源许可为 `CC-BY-ND-4.0`，本地证据不作为衍生数据再分发。

新增 `openloris_package` adapter 后，office camera pose 按 `world_T_marker × inverse(base_T_marker) × base_T_color` 转换，cafe 按 `world_T_base × base_T_color` 转换；没有把 marker/base 真值直接冒充相机位姿。单元测试覆盖两条坐标链和 fail-closed 三源计数。

## 预筛与完整审核

| 轨迹 | 归一化帧 | 处置 | 来源计数 |
| --- | ---: | --- | ---: |
| office1-1 | 809 | 稀疏视觉拒绝：静态扫描，无完整路线障碍生命周期 | 0 |
| office1-2 | 899 | 稀疏视觉拒绝：静态扫描 | 0 |
| office1-3 | 360 | 稀疏视觉拒绝：短时工位扫描 | 0 |
| office1-4 | 870 | 稀疏视觉拒绝：静态扫描 | 0 |
| office1-5 | 1589 | 稀疏视觉拒绝：长时静态环扫 | 0 |
| office1-6 | 1079 | 稀疏视觉拒绝；官方包缺 1 个 aligned-depth 帧 | 0 |
| office1-7 | 1140 | 稀疏视觉拒绝：相机扫视；可见人员不构成事件 | 0 |
| cafe1-1 | 1708 | 完整 18 页/1708 帧双模型一致拒绝 | 0 |
| cafe1-2 | 2698 | 完整 27 页/2698 帧双模型一致拒绝 | 0 |

cafe1-2 的 sanitized RGB-D VO 覆盖率为 `1.0`；冻结 candidate 产生 `route_known=2216`、`alerts=16`。这两项只描述候选器输出。两位 reviewer 在看不到 candidate 的条件下都判断序列以静止、原地/近原地旋转和横向扫视为主，路线投影有长段稀疏或缺失，无法建立可信的身体绑定前向路线，也没有完整的 `onset → alertable → passed_or_cleared → end` 障碍生命周期。

## 证据与停止条件

- metadata 预筛：`artifacts.local/evidence/ustrf-sensor-replay-r3/source-replacement-openloris-v1/prescreen-report.json`，SHA-256 `4940cd81c5d84a96ae1507b12e4c4fb6231961c97a8d1655e23c503186ff9377`。
- 9 轨迹视觉处置：`visual-prescreen-decisions-v1.json`，SHA-256 `14aee1db52ce4922d9e2f111af1306c30f73e5e139dd121b0eb2cbdc9b07482f`。
- cafe1-2 完整审核 manifest：`80fd7cc2a840f368985aa277eebfbdeff8b14c936d50d88c31e3be3756a79e4a`；共识：`review-consensus-cafe1-2.json`，SHA-256 `28c035927c2a3f5e8363ef15e9b4e129eeff990ed1653f90785819a4cc88094f`。
- cafe1-1/cafe1-2 独立 7z 成员都通过全包 CRC；一次并发下载污染的整包已隔离重命名为 `cafe1-1_2-package.tar.corrupt-do-not-use`，不能作为证据输入。

停止条件已触发：OpenLORIS office/cafe 候选穷尽且 `admitted=0`。下一步仍然只能替换到新的、具有持续身体前进和可观察 passed/cleared 生命周期的 RGB-D+独立轨迹来源；不得回调本轮门禁，也不得据此开放 120 episode、U0、Android、硬件选择或生产权限。
