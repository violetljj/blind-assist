# JRDB single-sequence native multisensor person geometry canary R1 结果（2026-07-25）

状态：`ANNOTATION_DERIVED_PERSON_GEOMETRY_AVAILABLE_WITH_ABSTENTION / VALID`

权限：`SEEN_DEVELOPMENT_AVAILABILITY_ONLY / DIAGNOSTIC_CEILING`

## 结论

R1 修正了 R0 的 claim dependency 误设，不改写 R0 历史事实。对同一 `meyer-green-2019-03-16_0 / 000000..000119` immutable packet：

- source-native robot-relative 3D geometry：`1,350/1,350` object-frame 可计算，`AVAILABLE_COMPLETE`；
- source-annotation-derived 3D motion：`1,336/1,336` 个相邻同轨 pair、14 条 track 可计算，`AVAILABLE_COMPLETE`；
- cross-modal 2D/3D identity：从 3D 分母看 `1,321/1,350`，29 个局部 abstention；从 2D 分母看 `1,321/1,345`，24 个局部 abstention，`AVAILABLE_WITH_DEGRADATION`。

29 个缺 2D 的 3D object-frame 没有被删除，也没有再关闭 3D-native claim；它们只对 cross-modal identity claim 弃权。R0 的 `FAIL_CLOSED_LABEL_JOIN / VALID` 仍作为旧合同的真实执行结果保留，R1 是版本化纠错。

## 分母与处置

| Claim | Expected | Eligible | Abstained | Invalid | Coverage | R1 status |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| robot-relative 3D geometry | 1,350 | 1,350 | 0 | 0 | 100% | `AVAILABLE_COMPLETE` |
| annotation-derived 3D motion pair | 1,336 | 1,336 | 0 | 0 | 100% | `AVAILABLE_COMPLETE` |
| cross-modal identity，3D-native denominator | 1,350 | 1,321 | 29 | 0 | 97.8519% | `AVAILABLE_WITH_DEGRADATION` |
| cross-modal identity，2D-native denominator | 1,345 | 1,321 | 24 | 0 | 98.2156% | `AVAILABLE_WITH_DEGRADATION` |

每行均满足 `expected = eligible + abstained + invalid`。coverage band 只作描述，不是新的 universal pass line，也没有把交集改成分母。

局部缺失保持原样：

- 3D-only `pedestrian:17`：frame `000079..000086`，8 个；
- 3D-only `pedestrian:12`：frame `000099..000119`，21 个；
- 2D-only `pedestrian:14`：frame `000033..000056`，24 个；
- duplicate/ambiguous ID：0。

## 真实性与权限上限

1,350/1,350 个 3D object-frame 和 1,336/1,336 个 motion pair 都包含 JRDB source 的 `attributes.interpolated=true`；direct 3D observation 与 direct motion pair 都是 0。

所以本结果只证明“官方 source annotation trajectory 经 source time、动态 pose 与静态 frame chain 后可计算”，不能称为直接传感器测得的人体速度，不能证明轨迹精度、泛化性能或人体安全有效性。authority ceiling 固定为 `DIAGNOSTIC`；`SELECTION`、route risk、event lifecycle、提醒逻辑、Android、人体/独立行走与生产全部关闭。

## 独立复算

- elastic standard document SHA-256：`08adac1c09ad7d369a58d20336e86f39237aa403b310d353b12df838259508a0`
- elastic standard config SHA-256：`772a39c9d1a3c53d202e7fdfdae6fc127617a11a329b034f5b5793e637bb5109`
- elastic standard validation SHA-256：`96bc1867b55f73be3da94afc2b436d0eeb7caeea2cde8bd01d17a637eff79d20`
- R1 config SHA-256：`18b72fb9b6e978bc48a05436147865bf9c34b13a16c3ffa5ea7d816fb3028d64`
- eligibility ledger SHA-256：`2f3fd20170373a22fec2be39dc974f8ae432da66ea100c588e0c2985db3d10b2`
- receipt SHA-256：`012a458845229dcf575a08b49993137515db608338b1978d21a37471a3d7281a`
- validation SHA-256：`63aae98ef0ce759eed88342b2fb03082411b847eb400b8e79abc4f26b05421dd`
- R1 validation：12/12 checks true；
- focused tests：5/5；Python compile 通过。

## 下一边界

本轮没有自动进入更高阶段。若继续研究，下一份独立边界应验证 annotation-derived trajectory 的误差/偏差或寻找 direct measurement authority；不得直接进入 route risk、event lifecycle、提醒、Android、人体或生产。
