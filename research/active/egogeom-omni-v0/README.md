# EgoGeom-Omni（头戴有无 ToF 通吃基础几何路线）

所属：**避障线**（独立 `EXPLORE` 研究路线；不改动默认 `:app` 基础 ToF 辅助）。

当前核心结论、实机数据与下一步决定以 [CURRENT.md](CURRENT.md) 为准。

## 目录结构

| 路径 | 内容 |
| --- | --- |
| [CURRENT.md](CURRENT.md) | 紧凑主线状态：`v4.1` 硅片精确 + H3 有符号 CNH、`pc_fp16dec_a16 + G2` 实片验收与 Android QNN 闭环 |
| [OVERNIGHT_LOG_20261122.md](OVERNIGHT_LOG_20261122.md) | 6 底座选型、`v3.2 → v4 → v4.1` 逐轮训练、零样本与 S24 Ultra NPU 明细日志 |
| [RUNS.md](RUNS.md) | 关键算力消耗、运行记录与收据索引 |
| [math_theory/](math_theory/RESULTS.md) | 14 道数学实测与反例边界：[V4 五题 (M1–M5)](math_theory/RESULTS.md)、[V5 九题与 Aria 三轮 (M6–M14)](math_theory/RESULTS_V5.md)、[M13 混合精度帕累托](math_theory/m13_fisher_so_c_w8a8_silicon/SILICON_PERCHANNEL_PARETO.md) |
| [cane_blind_band_probe/](cane_blind_band_probe/CAMERA_IMU_OPT_TOF_ARCHITECTURE.md) | 盲杖盲区与可穿戴架构：[架构与预算](cane_blind_band_probe/CAMERA_IMU_OPT_TOF_ARCHITECTURE.md)、[中高部探测](cane_blind_band_probe/RESULTS.md)、[三区消融](cane_blind_band_probe/STEP1_ABLATION_RESULTS.md)、[悬空/玻璃反事实](cane_blind_band_probe/OVERHANG_GLASS_COUNTERFACTUAL_RESULTS.md)、[无尺度 TTC 与 G2 玻璃头](cane_blind_band_probe/SCALEFREE_TTC_AND_GLASS_HEAD_RESULTS.md) |
| 顶层 `.py` 脚本 | `v4.1`/`v4.2` 模型与训练（`egogeom_omni_v41_model.py` 等）、时序 BEV v3 与极线视差（`egogeom_temporal_bev_v3.py`）、S24U 温升与真实 Atom+H3 回放（`bench_s24u_thermal_soak.py`、`replay_atom_h3_e2e_s24u.py`） |

## 对应 Android 工程实现

- 纯 Kotlin 几何、时序 BEV v3 与穿戴三通道风险引擎：`core/vision/src/main/java/com/linnan/blindassist/vision/egogeom/`（单元测试在同模块 `src/test/`）
- 原生 QNN HTP JNI 与运行库：`app/src/dav2Snpe/cpp/egogeom_qnn_jni.cpp`、`app/src/dav2Snpe/java/com/linnan/blindassist/vision/EgoGeomQnnRuntime.kt`
- 隔离的 debug 实时演示页：`app/src/debug/java/com/linnan/blindassist/debug/EgoGeomWearableDemoActivity.kt`

## 目录维护约定

后续新增多实验子课题须在当前目录下建立独立主题子目录（参照 `math_theory/`、`cane_blind_band_probe/`），不再向路线顶层平铺大量带日期脚本或报告；所有权重、DLC/上下文二进制与大结果统一存入忽略的 `artifacts.local/work/egogeom-omni-v0/`。
