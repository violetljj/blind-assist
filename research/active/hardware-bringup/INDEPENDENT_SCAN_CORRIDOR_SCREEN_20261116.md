# 独立扫描走廊样本筛查

2026-11-16，EXPLORE。搜索公开数据中的相机朝向与实际移向不一致片段。筛查前为小试设定的最低数量是两类各 10 段、各覆盖至少 3 个访次。初批计划上限为 450 MB 下载和 720 秒 CPU 命令时间，零自采、零 GPU。

## 数据和标注

- **SceneFun3D**。从 SceneFun3D 官方训练/验证访次和 Apple 官方 Training 的交集取 16 个公开录制，每访次一个录制。录制相机位姿和内参与独立 FARO 扫描坐标对齐。按时间戳和 CRC 核对选取的高分辨率扫描投影及内参。旧有 40 个 ARKitScenes 低分辨率深度访次中，官方元数据标记可得独立原扫的为 0 个。[文件格式](https://scenefun3d.github.io/documentation/dataset/file-structure/) · [官方位姿解析](https://github.com/SceneFun3D/scenefun3d/blob/main/utils/data_parser.py#L127-L208) · [ARKitScenes 官方源](https://github.com/apple/ARKitScenes)
- **走廊定义**。从位姿挑移动时刻，朝向取当前相机水平光轴，行进方向取随后约 0.8 秒的水平净位移。两条走廊都用现有 0.30–2.00 m 前向范围、左右各 0.30 m 及 HEAD/BODY 高度带。相邻同类时刻间隔不超过 1 秒时合为一段。
- **几何参考**。两个投影视角有足够的箱内扫描表面才标占据。所有参考视角均无箱内表面点，且有效射线至少确认每个高度带 75% 的箱内采样位置，才标该走廊为空。其余保留未知。扫描和后续位姿仅用于这一标注。手持 iPad 的位移是行进代理，标签适用范围是静态近场几何。

## 分布结果

原三视角筛查覆盖 16 个访次、175 个标注时刻。两条短时连接失败在降低并行度后恢复。检查扫描覆盖时复用同一录制内距离不超过 1 m、时间差不超过 2 秒的已缓存投影视角，原标签与预算记录保留。

| 片段类型 | 原三视角 | 扩展视角 | 扩展视角访次 |
| --- | ---: | ---: | ---: |
| 朝向有障碍，行进走廊为空 | 3 | 4 | 3 |
| 行进走廊占据 | 89 | 92 | 15 |
| 行进走廊为空的其他片段 | 2 | 4 | 3 |
| 行进走廊未知 | 23 | 14 | 9 |

4 个目标负例的朝向与移向差为 5.452°、13.086°、17.093°、18.272°。对应 3 条录制的 Apple 原生低分辨率深度和位姿下载地址已返回 HTTP 200，尚未下载或核对逐帧配对。小试数量门槛未达到，误报和漏报的算法对比为 `NOT_RUN`。

LaMAR 是后续优先核查的另一来源，提供日常步行录制，有对应的独立 NavVis 地图以及配准位姿。候选 CAB 公共录制仍需核对配准在短时段的精度和近距离扫描覆盖。本轮未获取 LaMAR 像素和扫描。[官方数据说明](https://github.com/microsoft/lamar-benchmark/blob/main/RAW-DATA.md)

## 复现

[筛查与离线复算脚本](host/scenefun_faro_corridor_screen_dev_20261116.py)。数据存于 `artifacts.local/work/independent-scans-heading-dev-20261116-codex/`，其中 `summary.json` 是首批，`supplement/summary.json` 是 16 访次合并结果，`reference_coverage_check.json` 是扩展视角的逐时刻记录，`shortlist.json` 是四个重点片段和原生下载入口可用性。位姿及来源清单存于 `artifacts.local/work/scenefun3d-heading-pose-scout-20261116-codex/`。扫描与内参 Range 请求预算记账 234,517,495 字节，含失败请求的预留字节，另取约 1.9 MB 的位姿及小型来源元数据。

已从缓存独立复算 175 个时刻的原标签和扩展标签，逐时刻与保留的结果一致。用已安装的 NumPy、OpenCV 和 requests 环境复算。

```powershell
python -B research/active/hardware-bringup/host/scenefun_faro_corridor_screen_dev_20261116.py `
  artifacts.local/work/scenefun3d-heading-pose-scout-20261116-codex `
  artifacts.local/work/independent-scans-heading-dev-20261116-codex --recheck
```
