# RCLE Phase B 真实 positive approach 数据角色准入 R1 预注册

日期：2026-07-27

状态：

`PREREGISTERED / SOURCE_AUTHORITY_LOCKED / CANDIDATE_PAYLOAD_NOT_ACCESSED`

机器合同：
[RCLE_PHASE_B_REAL_POSITIVE_APPROACH_ROLE_ADMISSION_R1_CONTRACT_2026-07-27.json](RCLE_PHASE_B_REAL_POSITIVE_APPROACH_ROLE_ADMISSION_R1_CONTRACT_2026-07-27.json)

合同 SHA-256：

`e2a3dfdecfbfb660a6c708e8f1146e7c3652c3192c34fdb19b9f13c47f92dc38`

## 结论与唯一范围

R1 只补真实 positive approach 数据角色，不实现、不导入、不运行 RCLE RGB
algorithm，也不建立 performance qualification。

唯一候选冻结为 ETH3D SLAM training `sofa_3` 的官方 RGB-D archive：

`https://www.eth3d.net/data/slam/datasets/sofa_3_rgbd.zip`

选择不依赖画面或 geometry outcome。冻结候选池中能满足“真实 RGB-D +
source-native pose/depth”的未准入对为 `sofa_3/sofa_4`；R1 按最低数字
source-native ID 选 `sofa_3`。`sofa_4` 不是失败后的替补，也不能保留为本命题的
confirmation。

合法终态只有：

- `REAL_POSITIVE_APPROACH_ROLE_ADMITTED / VALID`
- `HOLD_ALGORITHM_CANARY_APPROACH_ROLE_INCOMPLETE / VALID`
- `INVALID_R1_EVIDENCE / INVALID`

只有第一种终态允许另立一个新的 performance qualification 任务。R1 本身无论成功
与否都不实现或运行 performance qualification。

## Burned 与排除清单

统一机器清单：
[R1 burned/exclusion manifest](RCLE_PHASE_B_REAL_POSITIVE_APPROACH_ROLE_ADMISSION_R1_BURNED_AND_EXCLUSION_MANIFEST_2026-07-27.json)

SHA-256：

`0b54cecc1f3908264f3d4bd06a37b7c27b6f149c05e92e5b3949c0a6ef201593`

清单明确区分 outcome-burned、历史硬排除、not-real、其他程序已访问、R1 未选替代项
和本次 prelock 误触发的 process contamination。至少包括：

- TUM `fr2/rpy` 全 sequence、正式窗 `0/3/4/6`，以及已完整 prescreen 的
  `freiburg1_xyz`、`freiburg3_sitting_xyz`；
- Bonn 历史硬排除 9 条与已消费 cohort 6 条，共 15 条具名 sequence；其余 Bonn
  只作 family-level R1 候选排除，不虚称 outcome-burned；
- `RCLE_PHASE_A_SYNTHETIC_GENERATOR_FAMILY` 及 support fixtures；
- `EVIMO2_V2_FLEA3_SANITY_LL_CAPTURE_FAMILY` 的 13 条 sequence、所有窗和
  derivative；
- 其他程序已访问的 ETH3D `cables_1`、ICL-NUIM `traj0` 与所有 pre-existing
  `artifacts.local` payload。

在建立本清单前，一次过宽的本地文本检索误读并回显了既有
`artifacts.local` 数据 JSON 的部分内容。它不是 `sofa_3`，没有参与候选选择或
geometry 推断；但 R1 按 fail-closed 规则把全部 pre-existing local payload 排除，
不能用作候选、替代或 confirmation。

## License、identity、ancestry 与 reuse

机器锁：
[R1 source authority and candidate lock](RCLE_PHASE_B_REAL_POSITIVE_APPROACH_ROLE_ADMISSION_R1_SOURCE_AUTHORITY_AND_CANDIDATE_LOCK_2026-07-27.json)

SHA-256：

`7fc127f42ab50516d198b36938c396d9a1d3bcbbf219c02a72b991853ed7eccf`

官方 authority metadata 已在不请求 candidate archive 的前提下冻结：

| 对象 | bytes | SHA-256 |
| --- | ---: | --- |
| ETH3D license page | 7,587 | `1114c1af…2363` |
| ETH3D SLAM datasets page | 100,602 | `e66b2664…cea9` |
| ETH3D SLAM format documentation | 25,281 | `39c7be78…dfee` |

许可为 `CC-BY-NC-SA-4.0`，本协议只允许带 attribution 的非商业论文、学位研究和
研究原型评价；不产生商业、生产、安全或用户研究 authority。若分发 adaptation，
必须保持相同许可。

source descriptor SHA-256 冻结为：

`11ac41e221ec6bdc16f12e071a9befdb55a2466e00bc8a78ee7fe67185b04756`

ETH3D authority page没有发布 archive checksum，因此访问前不伪造 payload hash。
上面的 descriptor hash 同时绑定 exact candidate ID、official payload URL、license、
dataset-page hash、format documentation 和 no-replacement 规则。claim 后第一次
且唯一一次完整 GET 的 response-body SHA-256 自动成为不可替换 content identity。

ancestry 冻结为：

`ETH3D_OFFICIAL -> ETH3D_SLAM_TRAINING -> ETH3D_SLAM_SOFA_SCENE ->
ETH3D_SLAM_SOFA_3 -> ETH3D_SLAM_SOFA_3_RGBD -> R1`

independence group 为 `ETH3D_SLAM_SOFA_SCENE_CAPTURE_FAMILY`。一旦触及
`sofa_3` payload 或 geometry，整个 sofa scene family（`sofa_1-4`、
`sofa_dark_1-3`、`sofa_shake` 和所有 derivative）永久烧掉，只能作 canary、
source characterization、counterexample 或 regression，永不 confirmation。

## Claim 前后的硬顺序

1. 只验证 burned manifest、source-authority lock、机器合同及 SHA；
2. 冻结 bootstrap、acquisition、producer、独立 validator 与测试的 implementation
   lock；
3. 用 exclusive create 建立并 `fsync` claim，同时绑定 contract、source
   descriptor、burned manifest、source-authority lock 与 implementation lock；
4. claim 前禁止 request、resolve、stat、list、open 或 create 任意 `sofa_3`
   candidate payload path；
5. claim 后只允许 exact official URL 的一次 GET；禁止 HEAD、range prescreen、
   retry、mirror、repack、第二 archive、`sofa_4` 或任何替代；
6. 第一次完整 response 的 SHA 与 ZIP member name/size/CRC inventory 先冻结，再做
   geometry。

官方传输失败、冻结 URL policy 内的 HTTP 失败、ZIP 不可读，或单一 `sofa_3`
root 缺少 published required members 时直接 HOLD；不换源、不重试、不降低门。
unsafe/encrypted/drive-relative member、normalized duplicate、multi-sequence root、
下载后 control-text 解析或 cross-index identity 失败、禁止访问、冻结 binding 漂移、
实现失败或独立验证不一致属于已消费证据破坏，例外终态为
`INVALID_R1_EVIDENCE / INVALID`，不得包装成 source HOLD。

## 唯一窗口与冻结门

只评估 `sofa_3` 首个共同 RGB-D/pose timestamp 开始的第一个完整、半开
`10.000 s` 窗。只用 `associated.txt`、`calibration.txt`、`groundtruth.txt`、
`rgb.txt`、`depth.txt` 和该窗的 depth PNG；禁止解码 RGB。

门保持与 R0 一致：

- candidate-pair coverage `>= 0.80`；
- evaluable pair `>= 8`；
- window median signed radial expansion `>= 0.05 s^-1`；
- window median positive fraction `>= 0.75`。

任何门失败即 HOLD。禁止第二窗、滑窗、best-window rescue、重心化、翻转 pose 方向、
调整分母、降低阈值或查看 RGB。

producer 与 validator 必须分别从 source-native pose/depth 重算。validator 不得
import producer；identity、pair、window、gate、ancestry、reuse、forbidden access
与 replacement count 必须逐项核对。
