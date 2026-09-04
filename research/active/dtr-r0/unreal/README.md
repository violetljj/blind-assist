# Willow Walk：行人避障街区

默认场景现在是完整步行街：连续店面、石材铺装、咖啡座、树池、长椅、街灯、车辆和侧巷。使用 PBR 扫描材质、Lumen 全局光照与反射、虚拟阴影和日光与自动曝光。原六道灰盒场地保留为对照。

## 打开

```powershell
python tools/unreal_obstacle_lab.py --engine <本机UE安装目录> --open
```

项目：`artifacts.local/unreal/BlindAssistStreetLab/BlindAssistStreetLab.uproject`。
地图：`/Game/StreetLab/StreetLab`。首次执行复制本机 Epic 模板并下载小型 CC0 材质/道具，随后生成地图；已有地图不覆盖。

点击 **Play**，用 **WASD + 鼠标**步行观察，**Esc** 退出。步速 1.2 m/s。
`StreetActivity` Sequencer 控制四名动态人物，30 秒循环。可在编辑器中调整店面、家具、车辆、人物轨迹、镜头和照明。

| 条件 | 场景实现 |
| --- | --- |
| 静态障碍 | 长椅、路桩、菜单牌、树池 |
| 窄通道 | 咖啡座、店面与街道家具之间的通行空间 |
| 遮挡横穿 | 侧巷、停靠车辆和横穿人物 |
| 近失对照 | 路线侧方平行运动人物 |
| 迎面接近 | 沿人行道接近的人物 |

物理导向铺装是场景内容，不代表算法输出。上述名称描述设计意图，不是已经测出的 CONTACT / SAFE 标签；循环跳变不能当作连续自然运动。

## 官方资产与补充内容

直接复用本机 UE 5.8 模板包中的：

- `ArchVis`：`HillTree_02` 树木及树皮、枝叶材质。
- `Building`：街灯及金属、发光材质。
- `Vehicles`：车辆车身和玻璃。
- `TP_FirstPersonBP` 与 `Characters`：第一人称操作、Quinn 模型及步行动画。

店面布局、檐口、窗框、阳台、咖啡座和导向铺装由脚本组装。人物仍为 Epic 模板角色，并非写实扫描行人。

补充使用 [Poly Haven CC0](https://polyhaven.com/license) 的[铺装](https://polyhaven.com/a/cobblestone_pavement)、[砖墙](https://polyhaven.com/a/brick_wall_001)、[灰泥](https://polyhaven.com/a/plastered_wall_02)、[木板](https://polyhaven.com/a/wood_planks)、[长椅](https://polyhaven.com/a/painted_wooden_bench)和[花箱](https://polyhaven.com/a/planter_box_02)。下载器使用官方 API，保留来源及文件校验信息。Epic 和 CC0 二进制资产、生成地图、缓存、截图均留在本地 `artifacts.local/unreal/`；Git 只保存脚本与说明。

## 检查画面与运行

```powershell
python tools/unreal_obstacle_lab.py --engine <本机UE安装目录> --verify
```

独立编辑器实际渲染 `Hero`、`Cafe`、`Crossing`、`Overview` 四个视角，随后在 PIE 中检查角色、步速和动态人物位移，并退出。结果在项目 `Saved/lab-smoke.json`；截图为 `Saved/hero.png`、`cafe.png`、`crossing.png`、`overview.png`。首次启用 SM6/Lumen 需要等待着色器编译。

原灰盒场地可通过同一命令附加 `--scene graybox` 打开或验证，项目仍在 `artifacts.local/unreal/BlindAssistObstacleLab/`。

## 避障线接口

当前是可编辑的 synthetic Development 场景，未完成 RGB-D 导出、在线 DTR 回接或告警覆盖层。预设相机包括 `Pedestrian` 和 `Crossing`，人行道相对世界零点高约 0.27 m，传感器视点相对人行道高 1.6 m。

下一步以新增 UE loader 连接现有 [`SanitizedModelContract`](../carla/dtr_carla_rgbd_model_adapter.py)。导出需要同一 tick 的 RGB、前向轴线性深度（米）、相机内外参、wearer pose、时间戳和预先下发计划。UE 位移厘米除以 100；世界坐标 X 前、Y 右、Z 上，而相机反投影使用 Forward/Left/Up。不能直接使用设备 Z 或径向深度。

Actor ID、真实速度、碰撞体、instance mask、未来接触和 TTC 单独放入 evaluator 数据。不能把执行后的真实轨迹回填 issued plan。
