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

已打通 **UE RGB-D 采集 → 现有 YOLO 分割 → 保留的 DTR X73 → 独立引擎碰撞对照 → 可拖动回放**。这是固定轨迹的离线 synthetic Development 接口演示，尚非在线闭环控制，也未评价自动绕行成功率。

使用装有 NumPy、Pillow、PyTorch、Ultralytics 的 Python 环境执行：

```powershell
python tools/run_street_experiment.py --engine <本机UE安装目录> --output artifacts.local/unreal/<新运行目录>
```

默认使用本地 `artifacts.local/models/yolo11n-seg.pt`，可通过 `--weights` 指定已有权重。渲染使用 UE GPU；检测器测量 CPU/GPU 后选择后端并保存 `predictions/backend.json`。`--capture-only` 仅采集；`--reuse-capture` 复用成功采集执行后续阶段（须尚无 `predictions` 目录）。不修改或保存原地图。

- 两条预先下发的直线路线，各 8 秒、5 Hz、41 帧，步速 1.2 m/s；Sequencer 按固定时间求值，RGB 与深度读取期间保持同一场景状态。固定相机与人物轨迹可复放，不保证 Lumen/植被等渲染位级一致。
- `model/`：640×360 RGB PNG、前向线性深度 float32 NPY（米）、位姿、内参和执行前写入的计划。模型 runner 只接受这棵目录，不读取 actor 真值。
- `evaluator/frames.json`：引擎胶囊碰撞查询、接触对象、人物实际位置。胶囊半径 0.30 m、半高 0.90 m，底部距地面 0.02 m，以 Visibility 通道查询阻挡。这是 5 Hz 采样接触，不是连续碰撞证明。
- `capture.json`：采集状态及 10 米正对平面的中央/离轴深度校验。
- `predictions/`：原有 X73 完整输出、检测候选、ONSET/HOLD/CLEAR 显示转换及支持状态。CLEAR 只表示正向模型风险结束，不代表安全；全局可观测性仍标记 UNKNOWN。
- `evaluation.json`、`replay.html`、`replay.gif`、`walk_a-preview.png`、`walk_b-preview.png`：真值核对及可见结果。只评价具备完整 3 秒未来窗口的帧，尾部不算负例。HTML 可直接打开并拖动时间轴。

预设相机包括 `Pedestrian` 和 `Crossing`，人行道相对世界零点高约 0.27 m，传感器视点相对人行道高 1.6 m。现有算法的检测覆盖不包括所有几何障碍；没有报险不等于无障碍。两条演示路线不能支持“优秀避障性能”或泛化结论。

新增 UE loader 连接现有 [`SanitizedModelContract`](../carla/dtr_carla_rgbd_model_adapter.py)，仅替换深度解码，沿用原有投影和 X73 算法链。UE 位移厘米除以 100；世界坐标 X 前、Y 右、Z 上，而相机反投影使用 Forward/Left/Up。不能直接使用设备 Z 或径向深度。

Actor ID、真实速度、碰撞体、instance mask、未来接触和 TTC 单独放入 evaluator 数据。不能把执行后的真实轨迹回填 issued plan。

### 2026-09-05 实际接通结果

本地结果：`artifacts.local/unreal/rgbd-chain-20260905-c/`。两条路线全部 82 帧完成采集、YOLO 和 X73 推理、真值核对与回放生成。10 m 平面中央及两个离轴采样均为 9.99922 m；两次成功采集的人物位置、穿戴者位置及采样碰撞序列一致，渲染像素和检测起报时刻不保证完全一致。

最终运行中，`walk_a` 在 2.6 s 发出 ONSET，共 18 帧正向风险。引擎首个胶囊接触发生在 4.6 s（路桩反光环），5.8 s 首次接触迎面人物。时间上相差 2.0 s 不证明算法识别了首个接触对象。`walk_b` 无采样接触，也无正向风险。完整 3 s 未来窗口内共有 52 帧，正向风险检测计数为 TP=13、FP=0、FN=5、TN=34；另外 30 帧尾部不评分。这些计数不将“未报险”提升为安全结论。

碰撞日志也记录了地面导向凸点，说明当前资产碰撞体仍需针对步行/跨越语义进一步校准；不能把任意胶囊阻挡等同于真实人体危险。接通时修复了 X30 的一个实际崩溃：XY 去重后只剩 17 个支持点，却直接调用要求至少 32 点的 OBB。现在按原有 32 点要求跳过不足支持组，没有调低阈值。
