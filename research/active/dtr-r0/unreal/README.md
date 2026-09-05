# Willow Walk：行人避障街区

展示后继版本为 `StreetLabV3`，保留 `StreetLabV2` 及其完整实验记录。它是可编辑的 synthetic Development 实验场，不宣称真实人体安全。

## V3 场景与实机画面

```powershell
python tools/unreal_obstacle_lab.py --engine <本机UE安装目录> --polish --open
python tools/run_street_closed_loop.py --engine <本机UE安装目录> --map StreetLabV3 --case low_obstacle_collision --output artifacts.local/unreal/<新运行目录>
```

`--polish` 使用本机 Epic `Building/Geometry/SM_Building` 曲面咖啡亭替换三个重复南侧楼组，完整 `ConceptCar/Car/SM_AutomotiveTP_Car` 替换旧运动汽车，并导入 [Poly Haven 的实拍阴天 HDR](https://polyhaven.com/a/overcast_industrial_courtyard)。该 HDR 为 CC0，下载器验证提供方 MD5 并记录 SHA-256；文件与许可信息保存在 `asset-downloads/environment/`。近景家具使用其 [Outdoor Table Chair Set 01](https://polyhaven.com/a/outdoor_table_chair_set_01) CC0 模型与 2K PBR 贴图，提供方校验及许可保存在同名下载目录。这些是实际三维资产和环境照明，不是背景效果图。City Sample Buildings 尚未下载或迁入，不能把当前场景称为 City Sample。

导向铺装使用原生厘米尺寸、圆顶凸条的 1 m 网格，避免把单个立方体拉伸到 71 m；3–4 mm 的凸起无阻挡碰撞。新地图独立绑定四名人物，保存后重载并在 PIE 检查实际位移及脚部骨骼变化。`Saved/lab-visual-v3.json` 保存地图哈希、源地图未改校验及运行结果；只有该回执通过，打开入口才默认选择 V3。

同镜头的 1920×1080 Hero / Cafe / Walking 对比由 `render_street_v3.py` 在实际 UE 中生成，输出到 `artifacts.local/unreal/street-v3-visual/`：`before/` 为 V2，`release/` 为最终 V3。V3 的新实验必须显式使用 `--map StreetLabV3`；下文 7/8 成功结果来自旧 V2 受控迭代，不能移用为 V3 成绩。

2026-09-05 的 V3 低障碍双分支验证位于 `closed-loop-v3-canary-20260905-a`，90 个实际新 RGB-D 帧、约 110 秒完成。直行分支在 3.54 s 接触低障碍代理；辅助分支在 1.0 s 由观测深度触发，1.2 s 实际偏离直线，9.6 s 无接触到达终点，命令与运动核对无不一致。此次仅验证一个条件的接通，评价器总体仍标 `INCOMPLETE`，不冒充 V3 全场景通过或 DTR 单独收益。保存地图哈希在实验前后均为 `17d7f5d35dff048e929be92895da6269a61be9cc51abbafcf6157729fc5e905b`；V2 源地图字节保持不变。

## 在线闭环 V2

```powershell
python tools/unreal_obstacle_lab.py --engine <本机UE安装目录> --upgrade --open
python tools/run_street_closed_loop.py --engine <本机UE安装目录> --output artifacts.local/unreal/<新运行目录>
```

第二条命令使用装有 NumPy、Pillow、PyTorch、Ultralytics 的 Python。`--upgrade` 仅在 V2 地图缺失时下载固定版本的人物并建图；已有地图不覆盖。没有已通过检查的 V3 时，打开项目默认进入 V2；原 `StreetLab` 和灰盒地图保留。

闭环是 **UE 实际 RGB-D → 独立传感器进程 → 原有 YOLO / DTR X73 与观测深度分支 → 下一步减速、停步或侧移 → UE 新观测**。每步模拟 0.2 秒，等待实际推理返回后再前进；这是在线锁步实验，不是已达到 5 Hz 墙钟实时性能。深度分支单独标为 `OBSERVED_DEPTH`，不能把它的绕行成绩归给 DTR。

| 场景族 | 接触对照 | 近失对照 |
| --- | --- | --- |
| 遮挡横穿 | 人物与直行者轨迹相交 | 相同速度，错开横穿相位 |
| 突然停步 | 前方人物停下 | 人物继续前进 |
| 窄道会车 | 迎面人物进入身体代理范围 | 侧向错开 |
| 低矮障碍 | 12 cm 障碍置于路线上 | 障碍侧移出路线 |

共 8 个场景，每个包含 `OPEN_LOOP` / `ASSISTED` 两个分支。模型只接收中性 episode 编号、已观察 RGB-D、位姿与执行前下发的导航计划。场景名、演员轨迹、预期接触和评价真值留在 `evaluator/`，控制器不读它们。

真值采用连续相对运动的圆盘/圆角矩形接触解析计算，区分 `BODY_COLLISION_PROXY` 与 `FOOT_TRIP_PROXY`。地面由引擎射线测量；4 mm 导向地面起伏作为可通行负对照，不再把任何胶囊阻挡都当成危险。该分层是明确的实验代理，不是步态、生物力学、伤害或普遍可通行性标准。

输出包括 `run.json`、`model/`、`evaluator/episodes/`、`sensor-worker/backend.json`、逐步检查点与 `evaluation.json`。评价器核对接触、告警先于轨迹改变、返回命令确实用于下一步，以及无接触到达 8 m 终点；只停在半路不算成功。HTML/GIF 成对显示真实传感器帧和实际轨迹，失败也显示。

中断后仅在源码、地图和场景选择不变时，使用同一命令附加 `--resume` 继续；已经完成的分支保留，未完成分支从逐步检查点恢复。渲染不保证位级一致，已写入的待处理帧会复用。若修改实现，应使用新的运行目录并保留旧失败。

仅修改 `street_live_policy.py` 时，可附加 `--reuse-open-loop <完整基线目录>` 复用已完成、已通过评价的 8 个直行对照，只重新执行 8 个辅助分支。入口要求地图、场景、采集器、模型代码与权重相同；`baseline-reuse.json` 记录来源，HTML、GIF、PNG 均标注 reused。它是复用 Development 对照，不是 16 个全新分支或新确认集。

2026-09-05 的低障碍实机 canary 中，无辅助分支首次代理接触为 3.54 s；辅助分支在 1.0 s 深度告警、1.2 s 实际改变轨迹，9.6 s 到达终点且无代理接触。完整组的首次尝试发现人物动画未刷新，并记录了辅助横穿仍接触的失败，已终止并保留。单个 canary 成功不能当作八场景总体成功。

上述结果分别在本地 `closed-loop-canary-20260905-c` 与 `closed-loop-suite-20260905-a`。后者是 `ABORTED_IMPLEMENTATION_DEFECT`，不是通过的实验。

后续组固定使用 100° 水平视场。先前 80° 视场下，横穿者中心保持约 45° 方位，长时间位于画外，混合了几何遮挡和画外来人两种条件。这个传感器条件变更单独记录，不解释为算法改进；先前失败不覆盖。车辆亦改用官方四轮组件与导入骨架位置，保留原资产。

人物同步验证使用 `verify_street_live_pose.py`：实际 UE 渲染两个步态时刻，再重复第一个；检查组件空间骨骼、动画时间和四秒墙钟等待期间不漂移。`human-pose-qa-20260905-c` 已通过，三个实际截图与骨骼记录保留。车辆前侧视图位于 `vehicle-repair-20260905`，官方轮胎位置来自导入骨架，未使用猜测的轮心位置。早先车身失真没有在充分等待后的对照中复现，不能认定 Nanite 是已证实的根因。

`closed-loop-suite-20260905-b` 完整执行 16 个分支：8/8 直行对照符合预期，辅助成功 5/8。失败为横穿接触、停步后回中线接触，以及窄道无接触但未到终点；四个近失与低障碍辅助成功。根据实际轨迹，只修正本地响应控制器的三个已观察缺陷：近距离正前方障碍先完全制动；记录回归通道中最后观测到的障碍前向位置，经过后再回中线；统一前向和侧向检测宽度，避免把窄道侧墙当作正前方障碍。DTR X73 保持不变。修正版 `closed-loop-suite-20260905-c` 复用 b 的直行对照，重新执行全部 8 个辅助分支，结果以其 `evaluation.json` 为准。

修正版 c 已完成，8 个新辅助分支中 **7/8 无接触到达终点**；四个碰撞条件全部通过告警、执行命令、实际轨迹改变与无接触到达核对。唯一失败是窄道近失：DTR 持续告警造成等待，14 秒时仅到 7.48 m，按原时限判失败；没有延长时限或删掉该分支。原基线该近失通过，不能声称全部条件都改善。

| 碰撞条件 | 修正版辅助到达时间 | 首次触发动作来源 |
| --- | --- | --- |
| 遮挡横穿 | 10.8 s | 观测深度 |
| 突然停步 | 9.6 s | 观测深度 |
| 窄道会车 | 11.0 s | DTR X73 |
| 低矮障碍 | 9.6 s | 观测深度 |

这是来源分离的动作时序记录，不是对 DTR 单独贡献的消融证明。c 的 407 个新传感器帧共耗时约 927 秒，各分支平均 worker 推理时间约 0.90–2.73 秒，未达到墙钟 5 Hz。窄道近失在相同前五个位姿下仍产生不同 RGB 像素，0.8 秒的 DTR 风险输出开始分叉；现有证据不能隔离具体渲染或检测敏感源。严格重复应复放保存的传感器输入，不能承诺重渲染位级一致。

这组结果属于同组受控 Development 迭代，不是新确认集、人体安全结论或 App 默认策略升级。实验索引登记命令被现有另一条 L10 记录的 `input_fingerprint` 不匹配阻断；未修改该记录，本地 `identity.json` 与完整结果保留。

## 打开

```powershell
python tools/unreal_obstacle_lab.py --engine <本机UE安装目录> --open
```

项目：`artifacts.local/unreal/BlindAssistStreetLab/BlindAssistStreetLab.uproject`。
地图：V3 构建验证通过后为 `/Game/StreetLab/StreetLabV3`，否则使用已升级的 `/Game/StreetLab/StreetLabV2`；基础地图为 `/Game/StreetLab/StreetLab`。首次执行复制本机 Epic 模板并下载小型 CC0 材质/道具，随后生成地图；已有基础地图不覆盖。

点击 **Play**，用 **WASD + 鼠标**步行观察，**Esc** 退出。步速 1.2 m/s。
V2 的 `StreetActivityV3` Sequencer 控制四名动态人物，30 秒循环。`--upgrade` 会修复旧序列失效的四个人物绑定并保存服装材质和步态。重新加载后，四人在 PIE 的 2.3 秒观察期间分别移动约 281、281、281、179 cm，骨骼姿态均发生变化，步速设置为 1.2 m/s。回执为 `Saved/lab-playback-repair.json`，实际画面为 `Saved/playback-repair/clothed-reloaded.png`。

这次演示修复将地图从 SHA-256 `832f2cc3d5c2bdb5fe3f4ef6fc86d9746fff94a52c59a9eea6da24f7bb52e3a4` 更新为 `ed2fe68272e931798019e4ab5047ec624fa0cd1d19b53749fc30de55b1068571`，属于实验后的演示后继版本；原实验记录、旧序列和失败截图均保留，不能把新地图哈希回填已有实验。可在编辑器中调整店面、家具、车辆、人物轨迹、镜头和照明。在线实验临时替换为明确的场景演员，按模拟时间驱动，不保存这些替换到地图。

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

店面布局、檐口、窗框、阳台、咖啡座和导向铺装由脚本组装。V2 人物与兼容步行动画来自 [Microsoft Rocketbox 官方仓库](https://github.com/microsoft/Microsoft-Rocketbox)，MIT 许可，固定提交 `0943055db6ec570bcef9f2c8b41c9e5467c808f9`；不是 Epic MetaHuman。来源、许可和下载哈希保存在本地 `asset-downloads/rocketbox/sources.json`，进口资产记录为项目 `Saved/lab-visual-upgrade.json`。

补充使用 [Poly Haven CC0](https://polyhaven.com/license) 的[铺装](https://polyhaven.com/a/cobblestone_pavement)、[砖墙](https://polyhaven.com/a/brick_wall_001)、[灰泥](https://polyhaven.com/a/plastered_wall_02)、[木板](https://polyhaven.com/a/wood_planks)、[长椅](https://polyhaven.com/a/painted_wooden_bench)和[花箱](https://polyhaven.com/a/planter_box_02)。下载器使用官方 API，保留来源及文件校验信息。Epic 和 CC0 二进制资产、生成地图、缓存、截图均留在本地 `artifacts.local/unreal/`；Git 只保存脚本与说明。

## 检查画面与运行

```powershell
python tools/unreal_obstacle_lab.py --engine <本机UE安装目录> --verify
```

独立编辑器实际渲染 `Hero`、`Cafe`、`Crossing`、`Overview` 四个视角，随后在 PIE 中检查角色、步速和动态人物位移，并退出。结果在项目 `Saved/lab-smoke.json`；截图为 `Saved/hero.png`、`cafe.png`、`crossing.png`、`overview.png`。首次启用 SM6/Lumen 需要等待着色器编译。

原灰盒场地可通过同一命令附加 `--scene graybox` 打开或验证，项目仍在 `artifacts.local/unreal/BlindAssistObstacleLab/`。

## 原始离线接口 V1（保留）

V1 打通 **UE RGB-D 采集 → 现有 YOLO 分割 → 保留的 DTR X73 → 独立引擎碰撞对照 → 可拖动回放**。下述命令保留固定轨迹的离线 synthetic Development 接口，不包含 V2 在线运动控制。

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
