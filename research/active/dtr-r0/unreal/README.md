# Unreal 行人避障实验场

可编辑的 Unreal Engine 5.8 场地，生成在
`artifacts.local/unreal/BlindAssistObstacleLab/`。源代码入库，Epic 模板、生成的
`.umap` / `.uasset`、缓存和截图仅保留在本地 artifact 树。

## 打开和操作

```powershell
python tools/unreal_obstacle_lab.py --engine <本机UE安装目录> --open
```

首次执行创建场地，后续执行保留已有地图及手工修改。也可以直接打开生成的
`BlindAssistObstacleLab.uproject`。地图为 `/Game/ObstacleLab/ObstacleLab`。
点击编辑器 **Play**，用 **WASD + 鼠标**行走和观察，**Esc** 退出。
第一人称角色复用 Epic 模板，步速设置为 1.2 m/s。

六条实验道：

| 区域 | 内容 |
| --- | --- |
| 静态障碍 | 路桩、低障碍、低悬横梁 |
| 窄通道 | 1.2 m 净宽通道、错位障碍 |
| 遮挡横穿 | 不透明挡板与横向行人 |
| 近失对照 | 路线侧方平行运动的行人 |
| 迎面接近 | 沿路线接近的行人 |
| 横向交通 | 遮挡与横穿箱形推车 |

青色线表示设计路线，短刻度间隔 2 m。各道设有 1.6 m 高、90 度水平视场的
`Sensor_<编号>_RGB_160cm` 相机。`Overview` 是鸟瞰相机。动态障碍由
`DynamicObstacles` Level Sequence 驱动，Play 时自动播放并每 20 秒循环；
可在 Sequencer 中调整关键帧。循环回跳和停留阶段用于场地演示，不能视为
连续自然运动实验。场景名称描述设计意图，不是已经测出的 CONTACT / SAFE 标签。

## 验证

```powershell
python tools/unreal_obstacle_lab.py --engine <本机UE安装目录> --verify
```

启动独立的离屏编辑器，渲染鸟瞰图和行人视角，运行 PIE，检查第一人称角色、
步速及动态人物实际位移，然后自动退出。结果在项目 `Saved/lab-smoke.json`，
截图为 `Saved/overview.png` 与 `Saved/pedestrian.png`。
构建详情保存在 `Saved/lab-build.json` 与 `Saved/Logs/build-lab.log`。

## 与避障线的衔接

这是 synthetic Development 场地，不是已完成的 DTR 评测。当前没有 RGB-D
数据导出、在线算法回接或告警覆盖层。已有相机和路线方便下一步接入。

最小接入对象是
[`SanitizedModelContract`](../carla/dtr_carla_rgbd_model_adapter.py)，新增 UE loader
即可复用模型侧 RGB-D 处理，不能给新数据套用 CARLA 专属 schema。
后续导出需要同一 tick 的 RGB、前向轴线性深度（米）、相机内参和外参、
wearer pose、时间戳及预先下发的计划。UE 位移单位为厘米，需要除以 100；
世界为 X 前、Y 右、Z 上，模型反投影相机坐标使用 Forward/Left/Up。
不要把径向深度或设备 Z 值直接作为前向轴深度。

Actor ID、真实速度、碰撞体、instance mask、未来接触和 TTC 必须单独进入
evaluator 数据，不能混入模型观测；不能用执行后的真实轨迹回填 issued plan。
先用已知距离平面核对深度和左右方向，再接入现有 DTR adapter。

## 官方复用来源

- 本机 Epic `TP_FirstPersonBP` 和其 `Characters` / `Input` /
  `LevelPrototyping` 共享内容；包含 Quinn 模型与官方步行动画。
- 本机 `/Engine/BasicShapes` 几何资产。
- [Epic 编辑器 Python 文档](https://dev.epicgames.com/documentation/unreal-engine/scripting-the-unreal-editor-using-python)
  与 [Sequencer Python 文档](https://dev.epicgames.com/documentation/unreal-engine/python-scripting-in-sequencer-in-unreal-engine)。

构建器复用本机已安装内容，不下载、不上传 Epic 二进制资产。
