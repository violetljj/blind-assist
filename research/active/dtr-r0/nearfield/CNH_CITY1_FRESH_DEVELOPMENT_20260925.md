# City1 同场地新位姿 Development 采集

用户确认 City 可以继续采集，因此在旧 City1 物理场地上预选两个未用过的相机轨迹：从旧 City1 分别沿 x 平移 +2.0 m 与 +2.5 m，保留四段各两个位姿。两布局共用 `city-consumed-engineering-site-1`，**独立物理场地数为 1**。City0 停止追因，本批没有 City0 或正式 test 数据。

使用旧车辆近场临时清零方案，源地图、车辆远实例及元数据退出前恢复。固定曝光 EV100 12.2 在新图像前依据旧 City1 Development 图像预选。首次启动因所选 Python 缺少 OpenEXR 失败，留存 v1 回执；v2 使用任务本地固定版本 OpenEXR 后完成 16/16 帧七路采集，源身份、原生加载世界净空、场景/工程不变性和进程释放核对通过。

| 新布局 | 背景径向误差 P95 | 最大误差 | 30 mm 几何门槛 |
| --- | ---: | ---: | --- |
| `city1-fresh-development-00` | 6.285 mm | 29.942 mm | PASS，距上限约 0.058 mm |
| `city1-fresh-development-01` | 6.154 mm | 294.231 mm | FAIL |

两布局的覆盖及已抽查插入物近层门槛通过。32 张左右目 RGB 的中央亮度和对比度通过，但近场 p30 约 0.09、p60 约 0.13，低于沿用的 0.10/0.14 门槛；**32/32 张 RGB 不准入**。整批状态 `FAIL_FRESH_CITY1_DEVELOPMENT_NUMERIC`。布局 00 只保留为同场地 Development 的几何通过样本，不能作为可用 RGB 融合样本或新场地确认；布局 01 不按通过样本使用。

证据：`artifacts.local/evidence/cnh-city1-fresh-development-20260925-v2/capture/`、`artifacts.local/evidence/cnh-city1-fresh-development-geometry-20260925-v2/result.json`、`artifacts.local/evidence/cnh-city1-fresh-development-gate-20260925-v2/result.json`。现有 RGB-only replay 的源和回执契约只允许巷道；本批停止于一次预选采集，不用新的曝光重采结果覆盖失败。City1 后续可继续 Development 采集，但必须先有 City 适用的每布局测光固定曝光/同帧 RGB 回放路径和新的真实布局；不能把同一物理场地反复移动机位计作独立样本。
