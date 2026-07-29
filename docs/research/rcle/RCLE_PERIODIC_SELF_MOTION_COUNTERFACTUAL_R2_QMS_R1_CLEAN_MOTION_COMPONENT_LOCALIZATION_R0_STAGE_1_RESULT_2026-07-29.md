# RCLE clean motion-component localization R0：Stage 1 结果

日期：2026-07-29（Asia/Hong_Kong）

终态：

```text
STAGE 1 EXECUTION: COMPLETE / 16 OF 16
INDEPENDENT VALIDATION: VALID
ROUTING DECISION: OPEN_STAGE_2
LATER STATUS: STAGE_2 COMPLETE / VALID; SEE STAGE_A_RESULT
FORMAL 480+16: AUTHORIZED_ONE_SHOT / NOT_CONSUMED / NOT_RUN
```

## 一句话结论

在四个冻结 block 的首个全新 scene cluster 上，rotation-only 相对 static 的
`compensated_absolute_p90` 为 `4/4` 正，translation-only 相对
rotation-only 的 `compensated_signed_p90` 也为 `4/4` 正；full-6DoF 相对较大
single-component arm 只有 `2/4` 正。前两项均满足预先冻结的 `>=3/4` 方向规则，
因此独立 validator 打开 Stage 2。该结论只定位受控 generator 内部
“旋转补偿后的残余局部扩张”对运动分量的响应，不是报警、危险、总体效应或产品
结论。

## 冻结设计

- 四臂：`STATIC / ROTATION_ONLY / TRANSLATION_ONLY / FULL_6DOF`；
- 四个 block：`ADVIO_13 / 14 / 15 / 17`；
- Stage 1：每个 block 使用 ordinal `0` 的一个新 scene seed，共
  `4 clusters / 16 sequences / 9,616 pairs`；
- Stage 2：ordinal `1` 的另外 `16` 条已在 Stage 1 前冻结，初始状态为
  `SEALED_NOT_EXECUTABLE`；
- pair 只是 longitudinal repeat；方向判断单位是四个
  `block × scene-seed cluster`；
- signed response 与 absolute leakage 分离；absolute 指标是
  `median_g(abs(cell expansion))`，不是 `abs(median_g(cell expansion))`；
- R3、`>0.01/s`、三连续 pair、QMS-R1 operator、transport 和 analysis lock
  均未改动。

## 独立重算结果

### 冻结路由 contrast

| Contrast | ADVIO 13 | ADVIO 14 | ADVIO 15 | ADVIO 17 | 正方向 | 路由 |
|---|---:|---:|---:|---:|---:|---|
| `ROTATION_MINUS_STATIC`，absolute P90 | 0.091956 | 0.103836 | 0.169853 | 0.087018 | 4/4 | 打开 |
| `TRANSLATION_MINUS_ROTATION`，signed P90 | 0.003462 | 0.072374 | 0.145109 | 0.205583 | 4/4 | 打开 |
| `FULL_MINUS_MAX_SINGLE`，signed P90 | 0.038792 | 0.006952 | -0.055879 | -0.113457 | 2/4 | 不打开 |

`OPEN_STAGE_2` 由前两项共同满足冻结规则而产生；第三项没有被事后改写为正结果。

### 主要 arm 指标

| Block | Motion | signed P90 | absolute P90 | trigger density / 601 | evaluable fraction |
|---|---|---:|---:|---:|---:|
| ADVIO 13 | STATIC | 0.000000 | 0.000000 | 0.000000 | 1.000000 |
| ADVIO 13 | ROTATION_ONLY | 0.032879 | 0.091956 | 0.033278 | 0.988353 |
| ADVIO 13 | TRANSLATION_ONLY | 0.036341 | 0.220761 | 0.049917 | 0.985025 |
| ADVIO 13 | FULL_6DOF | 0.075133 | 0.178635 | 0.166389 | 0.983361 |
| ADVIO 14 | STATIC | 0.000000 | 0.000000 | 0.000000 | 1.000000 |
| ADVIO 14 | ROTATION_ONLY | 0.039936 | 0.103836 | 0.049917 | 0.998336 |
| ADVIO 14 | TRANSLATION_ONLY | 0.112310 | 0.365373 | 0.096506 | 0.996672 |
| ADVIO 14 | FULL_6DOF | 0.119262 | 0.237781 | 0.247920 | 1.000000 |
| ADVIO 15 | STATIC | 0.000000 | 0.000000 | 0.000000 | 1.000000 |
| ADVIO 15 | ROTATION_ONLY | 0.056145 | 0.169853 | 0.059900 | 1.000000 |
| ADVIO 15 | TRANSLATION_ONLY | 0.201255 | 0.559241 | 0.143095 | 0.998336 |
| ADVIO 15 | FULL_6DOF | 0.145375 | 0.406039 | 0.229617 | 0.998336 |
| ADVIO 17 | STATIC | 0.000000 | 0.000000 | 0.000000 | 1.000000 |
| ADVIO 17 | ROTATION_ONLY | 0.024853 | 0.087018 | 0.028286 | 1.000000 |
| ADVIO 17 | TRANSLATION_ONLY | 0.230437 | 0.384499 | 0.198003 | 0.995008 |
| ADVIO 17 | FULL_6DOF | 0.116980 | 0.256277 | 0.251248 | 0.995008 |

三点值得保留：

1. static 为严格零，说明新场景和观测 hook 没有自行制造基线响应；
2. rotation-only 的 signed 中位趋势接近零，但 absolute P90 明显非零，支持把
   “方向性 response”与“旋转残余泄漏”分开审计；
3. translation-only 的 signed P90 在四个 block 都高于 rotation-only，但
   full-6DoF 并不稳定超过较大单分量，当前不能声称简单叠加或协同增强。

## 执行与资源

- 成功 run wall time：`2315.0792 s`（约 `38.58 min`）；
- launch available RAM：`8.2192 GiB`；
- minimum available RAM：`7.1002 GiB`；
- heartbeat 最大间隔：`20.0328 s`；
- swap in/out delta：`0 / 0`；
- residual worker PID：`[]`；
- 正式 sequence：`0`；
- 正式 R3 pair-core call：`0`。

成功 run 前有两次宿主生命周期启动失败：均在完整 arm 落盘前退出，只留下
`progress/telemetry` 和空 staging，无 sequence receipt、pair ledger 或 reduced
metric。失败痕迹保存在：

```text
artifacts.local/evidence/rcle_periodic_self_motion_counterfactual_r2/
  qms_r1_clean_motion_component_localization_r0/
    stage1_failed_background_launch_20260729T133609HKT/
    stage1_failed_unobserved_exit_20260729T133900HKT/
```

随后使用持续监控的同一冻结 identity set 完成执行；没有换 seed、降门、修改
operator 或把失败输出并入分析。

## 证据链

```text
contract sha256:
  8863f2bb804dddfc1b366f9c08552ce529c1ea336600acd4cecb6f40c570e541
identity lock sha256:
  88d32767a7aec86a7eaa933fc336ae62bbff9797e6b935f4f2ecd01111a2c0ce
stage 1 activation sha256:
  79e125217f08a290df4b3194f152911344146c7c45e26427d946477f47fc4ca0
run receipt sha256:
  3c630df442040772b8f5a0a935456e307b425a074549e8763ecf400a4ef75328
independent analysis sha256:
  d6d92813c81f6a012d04189da2a248c3314a6d7f0e8e64199bdd39b68ce10c57
independent receipt sha256:
  90ddc4efc1e6ed2681f109ea3edb280342adcde990c2009a456ff12aa146321a
routing decision sha256:
  a35be9379f73cb040589278c0aada792597ea764017e1f8702dce95fb5a6efe1
```

## 当前边界

本文件记录 Stage 1 当时的独立结果。其后的资源门已在不降门、不换 identity、
不减少 cluster 的条件下满足；预冻结的 Stage 2 已完成并通过独立验证。两批合并
收口见
[Stage A result](RCLE_PERIODIC_SELF_MOTION_COUNTERFACTUAL_R2_QMS_R1_CLEAN_MOTION_COMPONENT_LOCALIZATION_R0_STAGE_A_RESULT_2026-07-29.md)。

successor formal 的一次性授权仍为 `AUTHORIZED_ONE_SHOT / NOT_CONSUMED`；
正式 `480+16` 仍为零。本 Stage A 不会自动运行 formal，也不形成 Android、
产品或安全权限。
