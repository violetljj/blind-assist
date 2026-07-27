# RCLE Phase B 真实 positive approach 数据角色准入 R0 结果

日期：2026-07-27

终态：

`HOLD_ALGORITHM_CANARY_APPROACH_ROLE_INCOMPLETE / VALID`

最大权限：

`DATA_ROLE_AUDIT_COMPLETE / RGB_ALGORITHM_IMPLEMENTATION_AND_EXECUTION_NOT_AUTHORIZED`

## 结论

唯一冻结来源 `EVIMO2 v2 / Flea3 / sanity_ll` 没有准入真实 positive
approach role。13 条 source-native sequence 各自只使用从首个共同 pose/depth
timestamp 开始的首个非重叠 `10.000 s` 窗，共 `3,895` 个 candidate pair；
`0/13` 个窗同时通过冻结的 coverage、signed radial expansion 和 positive
fraction 门。

因此本版本按预注册直接 HOLD。没有换到 TXT、其他 EVIMO2 camera/group、TUM、
ETH3D、ICL、镜像或第二来源；没有滑窗、改分母、降门或查看 RGB。后继
`RCLE_PHASE_B_RGB_ALGORITHM_CANARY_R0_IMPLEMENTATION_AND_PERFORMANCE_QUALIFICATION`
不得启动。

## Pre-access claim 与 source identity

在第一次 official index GET 前，exclusive-create 并 `fsync` 的 claim 已绑定：

| 对象 | SHA-256 |
| --- | --- |
| machine contract | `048e53baf51c26032ef86c3012bdecaa84b72f1e72ff5a6d04a1ba1090de0bed` |
| frozen source descriptor | `7f5170061170bb1fa4ac78fc1af8a172bb7a690720776c6295f4aaf683509a8e` |
| claim file | `d87b5d9131a4faf54ee51df9e40afd088faf5ffcde172215c1f6bf6c9474ec8e` |

official index 第一次 GET 为 HTTP `200`，原始 HTML `313,545` bytes、SHA
`fe3d7032…aef94`。冻结上游证据中的“约 1.706 GB 最低成本完整 Flea3 truth
package”唯一对应 official NPZ 链接；TXT 被显式记录为非替补。单次 payload
GET 为 HTTP `200`、`1,706,608,737` bytes，最终 archive SHA：

`b6a45becaf2d750df9f5f91bcb5c1d61d2f4942c419da8f22dd6ac40981945a8`

archive 有 13 条 sequence、108 个成员。正式审计只提取 13 个
`dataset_info.npz` 与 13 个 `dataset_depth.npz`，共 `605,414,286` bytes；
classical RGB、events 和 mask 提取数均为 `0`。

## 冻结几何结果

门为：

- candidate-pair coverage `>= 0.80`；
- evaluable pair `>= 8`；
- window median signed radial expansion `>= 0.05 s^-1`；
- window median positive fraction `>= 0.75`。

| Sequence | coverage | signed radial `s^-1` | positive fraction | admitted |
| --- | ---: | ---: | ---: | --- |
| `checkerboard_d_flat_fb_000000` | 0.9465 | 0.212799 | 0.6700 | no |
| `checkerboard_d_flat_lr_000000` | 0.9730 | 1.102634 | 0.5212 | no |
| `checkerboard_d_flat_rot_lr_000000` | 0.9600 | 0.483585 | 0.7189 | no |
| `checkerboard_d_flat_rot_ud_000000` | 0.9633 | 0.566676 | 0.6649 | no |
| `checkerboard_d_flat_ud_000000` | 0.9000 | 0.410476 | 0.5279 | no |
| `checkerboard_d_tilt_fb_000000` | 0.9433 | 0.256523 | 0.5815 | no |
| `checkerboard_d_tilt_rot_z_000000` | 0.8933 | 0.110859 | 0.5365 | no |
| `depth_var_0_d_fb_000000` | 0.9700 | 0.021701 | 0.5072 | no |
| `depth_var_0_d_lr_000000` | 0.9500 | 0.546200 | 0.5134 | no |
| `depth_var_0_d_rot_z_000000` | 0.9600 | -0.032445 | 0.4957 | no |
| `tabletop_d_flat_lr_000000` | 0.9533 | 0.740646 | 0.5164 | no |
| `tabletop_d_flat_rot_ud_000000` | 0.9767 | 0.446558 | 0.6069 | no |
| `tabletop_d_flat_rot_z_000000` | 0.9767 | -0.007270 | 0.4989 | no |

多个窗有较大的 signed radial median，却没有达到 `0.75` 的径向正向一致性；
这正是 PB-H1 已指出的限制：absolute 或单一 signed aggregate 不能把横移/混合
运动包装成 coherent approach。`fb` 名称也没有被当作真值标签。

## 独立验证与 validity

独立 validator 没有 import producer；它复用了此前独立实现审查过的
`real_data_geometry_canary_r0.validator._independent_geometry`，并从全部
`dataset_info/depth` 重新计算：

- producer / validator pair record：`3895 / 3895`；
- pair replay mismatch：`0`；
- window replay mismatch：`0`；
- binding mismatch、forbidden extracted source、access violation：均 `0`；
- ancestry/confirmation overlap violation：`0`；
- algorithm outcome read：`false`；
- replacement source count：`0`。

validation 终态为 `VALID`。result、receipt、validation SHA 分别为
`ecc35e27…821a`、`dcf7cdf1…c3dd`、`6567f898…a2f4`。

启动时有三个实现层事件完整保留：一次零数据访问的 script-path import failure；
随后 source replay 暴露 4 个缺 depth/pose frame，初版错误地硬失败。修复只把合同
已规定的 missing depth/pose 写成 abstention，未改变 source、window、denominator、
formula 或 gate；正式 output 在修复前均未创建。独立全量重算关闭了这些实现风险，
但不把 HOLD 改成准入成功。

## Identity、ancestry、access 与 reuse

来源 ancestry 为 `EVIMO2_V2_OFFICIAL -> EVIMO2_V2_FLEA3 ->
EVIMO2_V2_FLEA3_SANITY_LL`，与 burned TUM、Bonn 和 Phase A synthetic
independence group 不重叠。尽管没有准入 role，本轮已发生 geometry access，
因此整个 `EVIMO2_V2_FLEA3_SANITY_LL_CAPTURE_FAMILY` 及其 derivative 永久排除
在未来 confirmation partition 之外，只能保留为 source characterization、
counterexample 或 regression。

科学缺口仍然存在。按 R0 的 no-replacement 规则，本任务到此合法终止；不得在
本版本另找来源、换窗或降低准入门。
