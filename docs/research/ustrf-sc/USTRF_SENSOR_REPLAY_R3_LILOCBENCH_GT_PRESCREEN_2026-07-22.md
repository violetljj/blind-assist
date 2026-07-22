# USTRF sensor replay R3：LILocBench 动态来源闭环（2026-07-22）

状态：`DYNAMICS_0_ADMITTED / LT_CHANGES_DYNAMICS_0_ADMITTED / admitted=2/3 / BONN_NEGATIVE_EVIDENCE / evaluator_not_run / DO_NOT_SELECT_HARDWARE`

## 结论

LILocBench `dynamics_0` 与 `lt_changes_dynamics_0` 已分别从 GT-only 候选闭合为 R3 准入轨迹。两条都先用官方 `base_link` GT 通过冻结的 24/12/0.03/0.50 路线拒绝型预检，再完整下载官方 RGB-D、复核 ZIP/标定/时间戳，将 `camera_front` 原始深度按 `T_color_depth` 注册到 color optical raster，并覆盖每个连续帧一次生成隐藏 candidate 的审核输入。每条轨迹都由两位隔离 AI reviewer 判定路线有效；第三模型只裁决不同的事件集合。最终分别冻结 3 与 12 个路线相交行人事件，因此累计计 `2/3`。

这不是 evaluator 结果。两条冻结 candidate 各产生 50 个 alerts，但三条准入轨迹尚未集齐，所以没有计算 event recall、critical miss、false alerts/min 或 clearance，也没有运行 `run_replay.py` evaluator。硬件选择、U0、Android、训练和生产权限保持关闭。

## 数据使用边界

- 官方页：[LILocBench](https://www.ipb.uni-bonn.de/html/projects/localization_benchmark/)；官方评价仓库：[PRBonn/LILocBench](https://github.com/PRBonn/LILocBench)。
- 项目规则已改为：普通公开渠道可下载的数据可直接缓存到 `artifacts.local/` 并用于隔离内部研究。数据页未明确给出 dataset license、参与者同意或隐私条款的事实仍如实记录，不再阻塞下载与研究使用。
- 许可/隐私澄清 issue 已提交：[PRBonn/LILocBench#1](https://github.com/PRBonn/LILocBench/issues/1)；截至本记录仍为 OPEN、0 回复。它只影响引用、再分发及更宽使用边界，不阻塞当前工作。
- 公开可下载不等于已获再分发、商业发布、参与者同意或生产授权；这些字段继续为 false/unknown。

## 冻结输入与 GT 预检

- R3 prereg SHA-256：`3aa3fdb460c697d6d669e2174f7f7c9d17f1fc06b6f6392d35ba1ac5b2b73eaa`；24/12/0.03/0.50、candidate、事件阈值和 15 帧审核容差均未改。
- `dynamics_0` manifest SHA-256：`4d4f0d8ee1ce8c14452f5288486cd4a062e0b70551f1f3fff4f17262983dd1d4`。
- GT：`518851 bytes`，SHA-256 `606f9392c1a992c9acdb5e680a9108ef7647ba8943d747435c90dfa56feaa38f`；3194 poses、159.951s、中位约 20Hz、轨迹长 54.052m。
- 因 RGB-D 为 15Hz、GT 为约 20Hz，预检以 nominal 15Hz 建时间线，24/12 帧仍是 1.6/0.8s。pose 对齐覆盖 `0.999583`；truth/causal route proxy unknown 为 `0.07875/0.088333`，均通过冻结 `<=0.50`。
- GT-only 最终报告：`dynamics_0-gt-prescreen-v6.json`，SHA-256 `88bd7f2cd08e1ff3403989191f5c6c234ef746daf7c20bf25010b606efd3dc5f`；它只有拒绝权，来源计数来自后续完整审核。

## 完整 RGB-D 与坐标适配

- 官方 ZIP：`3,753,768,874 bytes`，SHA-256 `0a1016e7e1759c1a67be8170f6d232ce0e302c22ec16c2bd9fc03dcca1a1b34c`；14411 members，完整 CRC 通过。archive receipt SHA-256：`5b483c6ea8c11f1ccefc3a43183f27006aee09b458f0e6b9581cf68124c4ec15`。
- 只解出 `camera_front` 与必要标定：2398 color + 2398 raw depth。四个关键成员哈希与 fail-closed manifest 预期精确一致。
- 位姿链为 `global_T_base_link × base_link_T_camera_front_link × camera_front_link_T_camera_front_color_frame × camera_front_color_frame_T_camera_front_color_optical_frame`；color optical `+Z` 在 base 中为 `(0.999154, 0.004129, -0.040917)`，确认为前向相机。
- `extrinsics_depth_to_color.yaml` 声明 `parent=color / child=depth`，按 `T_color_depth` 使用，不按文件名反转。注册使用 plumb-bob color distortion 与 nearest-z buffer，不填洞；raw `0/65535` 都映射为 unknown，发现并记录 22341 个饱和值。
- 完整连续包 2397 帧、159.892s；RGB-depth 对齐率 `0.999583`，delta p95/max `0.0501/0.0510ms`；注册后最小/中位有效深度率 `0.61833/0.78832`。preparation receipt SHA-256：`8f73075f86378b0d143b87384a4a4b33a4ff0e97bc92ad54284a93b682ef849c`。
- normalized bundle SHA-256：`ba735f30b372a4911945247c49639bf3f0973a673140a44f72ebe9b51b7f3a9e`；单源 normalization 的全局 `ok=false` 是因为尚未满 3 个来源，不代表本来源失败。

## Candidate 冻结与隔离审核

- `prepare_estimator_inputs.py` 物理删除 GT pose 字段；独立 ORB RGB-D pose estimator 的 GT access 为 false，估计覆盖率 `1.0`。pose receipt SHA-256：`8e959596cda14d2adee44193fa482695c2af1a0db167979fbd5ce50d9c56204e`。
- 冻结 candidate 覆盖 2397 帧，route truth known 2089、causal route prediction known 2000、alerts 50；SHA-256 `6481193cfa1af062efa7298da57fa1aa1b8da621143ad86cbf810fedf2e0d697`。这些 alerts 对 reviewer 不可见，也未被评测。
- review bundle 共 24 张 sheet，按原始顺序覆盖 0–2396 每帧一次；review inputs SHA-256 `9d367a63d69b8217105961a37fb8b1b0b2954be19748414585424498c747aae9`。
- Reviewer A/B 都 `route_valid=true`、`verdict=accept`，置信度 `0.86/0.90`；receipt SHA-256 分别为 `1768842761d219de5fba7f40ab85d6ec00948d2652b2c706cf3469f6a9066859` 与 `5460a39861fac814c0ca261c83f51dc28db1b460d3d4e370d893b78086c16ac1`。
- A 冻结 3 个事件，B 冻结其中第三个。按 `ustrf_event_review_v1` 启动新上下文第三模型；它仍看不到 candidate，并裁定三个事件都具有路线相交与完整生命周期。adjudication SHA-256：`16837bf22de94fa8447217dc97dea3dc7ef77f3c1a3367a1134e458043894fdb`。
- Canonical anchors：`139/142/161/161`、`1790/1796/1850/1850`、`2068/2070/2080/2082`，均为 `onset/alertable/passed-or-cleared/end`，且标为 critical。15 帧容差未改。
- consensus v2 SHA-256：`395a3b4cf1a14a49f0814dd27f7650de0e748e9b5c67a03706632dba7321bb73`；`route_event_admitted=true`、`source_count_credit=1`、`admitted_source_count=1`、`minimum_admitted_sources_met=false`、`event_truth_authority=false`。

## 第二条准入：`lt_changes_dynamics_0`

- GT：`1,814,325 bytes`，SHA-256 `277699961867b9d5d3bcaa14de1e577bf7ef9e6737b6f971460231461ff1d2d0`；11156 poses、558.948s、轨迹长 206.557m。nominal 15Hz pose 对齐 `0.999284`，truth/causal unknown `0.008706/0.009302`，均通过冻结门。GT report SHA-256：`bfc1a32b3e9de4c15280f3461a4be56f75092fdc580bd234d92bdad440f2f964`。
- 官方 ZIP：`12,940,694,854 bytes`，SHA-256 `98c42aa5f22e839749c6221e3c9ed63ffd460a66b28a66b7003f29c186640ee9`；50298 members、解压体积 14171916297 bytes，完整 CRC 通过。只解出 `camera_front` 与必要标定，共 16761 members、5404168455 bytes。
- `transformations.yaml` 与三项相机标定成员哈希逐项复核；前向轴仍为 `(0.999154, 0.004129, -0.040917)`。完整连续包 8377 帧、RGB-depth 对齐率 `0.999881`、delta p95/max `0.0501/0.0510ms`、最小/中位有效深度率 `0.63792/0.85221`；113332 个 raw 65535 饱和值映射为 unknown。preparation receipt SHA-256：`afe6b31d79db6d648491b458ac282213f56057ae48a0b21795b4373932917697`。
- normalized bundle SHA-256：`fb02e862f365cac0a68a3da4ccc2d2c0e9ad7483b3b79fd36477da0aeca5dd42`；独立 pose coverage `0.999164`，pose receipt SHA-256 `24f552069c0e1a9d80ce9723f089b0b532ed74137d8c25c202bb9ea797fe2204`。
- 冻结 candidate SHA-256：`591fd45075f28a5a2792af9c7f8e55c67e9b31adfd5369e6a42439ea972ed830`；route truth/prediction known 为 `7901/8071`，alerts 50。84 张 sheet 覆盖 0–8376 每帧一次；review inputs SHA-256 `75a119d59f40ab867b15c31b0c8d161d478547355663e9d933f2c5c9300aba19`。
- Reviewer A/B 都 `route_valid=true`、`verdict=admit`，分别冻结 7/8 个事件；receipt SHA-256 为 `9faf97b8049b68f60d31640f5b9859fae8d0091e35bad3003592fe2c85b83582` 与 `ee5e5ec8a41b6cc861999e042ebbf2c867b7c6b1fb2d51905e62ef96b1485078`。第三模型复核全部 84 张表后冻结 12 个 canonical events；adjudication SHA-256 `eb6ed21844317b35abc22ed91f70030e55cfa561500373e7ebdfc324813b73bf`。
- Canonical anchors：`532/548/582/589`、`1070/1090/1147/1152`、`1848/1860/1981/1990`、`2126/2134/2209/2212`、`2425/2432/2470/2481`、`3589/3592/3604/3610`、`3710/3738/3906/3915`、`4526/4551/4780/4799`、`4968/4974/5050/5055`、`5331/5338/5394/5400`、`5746/5755/5812/5820`、`6188/6200/6312/6322`。
- consensus SHA-256：`00b0b1627610a6e2023ea4ff1b154a423e3443cbb9c44aff1c1129595b53e91a`；本来源 `route_event_admitted=true`、`source_count_credit=1`。与 `dynamics_0` 合并后的项目累计是 `2/3`，不是该单来源 consensus 内的局部计数。

## Bonn 独立数据族负证据

官方 Bonn RGB-D Dynamic Dataset 的 registered RGB-D、OptiTrack sensor pose 与 TUM 格式均可直接适配，但手持运动风险在真实审核中兑现：

- `moving_obstructing_box`：589 个同步帧，GT 隔离 pose coverage `1.0`；两位隔离 reviewer 都认为横向旋转/扫视主导、无可信身体绑定前向路线，0 events，一致拒绝。consensus SHA-256 `42032693643d8b26a5cbe3e289c6645b21d67cc87f5b8b13ac31ec5f655597c2`。
- `person_tracking`：官方 329482910-byte ZIP，SHA-256 `a4810fd91ef2ea1d630b53fe0df5d76144c1b18d86ca91fb3a035debd0c9c5f5`，CRC 通过；580 个同步帧、4.341m camera path。两位隔离 reviewer 都认为路线稀疏且后半段旋转主导，0 events，一致拒绝。consensus SHA-256 `323627cee037393b9f8e7493cca8d497cd22035ca0a6338daee37292c6f49c40`。
- `crowd`：官方 515941922-byte ZIP，SHA-256 `76b30c86e4ccb78ab668dd427dab696faef4885633dba436bc391b3a8142df70`，CRC 通过；927 个同步帧中冻结路线只 known 22 帧，unknown `0.976268 > 0.50`。按“稀疏预筛只有拒绝权”直接拒绝，未启动 reviewer。

因此 Bonn 没有贡献第 3 条准入轨迹，也不应继续无界扩张同族下载。累计达到三条准入轨迹前，evaluator 继续不运行。
