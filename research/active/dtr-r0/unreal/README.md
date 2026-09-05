# Willow Walk：行人避障街区

展示版本为 `StreetLabV4`，使用 Epic City Sample Buildings；V2、V3 的实验记录保留。它是可编辑的 synthetic Development 实验场，不宣称真实人体安全。

## 算法研究入口

从 2026-09-06 起，UE V4 是避障线的默认 Development 实验场，CARLA 保留历史证据和必要的补充验证。统一入口为 `python tools/run_obstacle_research.py`，提供 `status`、`replay`、`closed-loop`、`compare`、`calibrate` 和 `geometry`。默认使用 V4、10 Hz Development 难例组、增量感知及 `DEPTH_ONLY` 运动基线。当前验收见 [实验场补齐记录](UE_LAB_ACCEPTANCE_20260906.md)，首次迁移历史见 [UE 主实验场记录](UE_PRIMARY_LAB_20260906.md)。

实测记录见 [2026-09-05 算法实验场报告](ALGORITHM_LAB_20260905.md)。

最新实现见 [增量状态与候选动作重构](INCREMENTAL_ACTION_REFACTOR_20260905.md)。固定回放及 live worker 默认使用增量 X73：同一检测账本的预测耗时 `370.28 -> 24.02 s`，733 帧历史输出一致；完整固定回放实测 `448.95 -> 51.34 s`，后者是历史同工作量比较。原批量实现保留作差分参照，固定回放可指定 `--engine batch-prefix`，闭环可指定 `--prediction-engine batch`。

感知修改先用固定 RGB-D 输入，运动决策修改再运行 UE 闭环。固定回放保留原始相机、位姿和已下发计划，因此不能为新的运动策略提供反事实轨迹或接触成绩。

```powershell
# 在具备 CUDA PyTorch、Ultralytics、NumPy、Pillow、psutil 的项目 Python 环境执行。
python research/active/dtr-r0/unreal/ue_fixed_replay.py export --source-run artifacts.local/unreal/closed-loop-v4-suite-20260905-a --output artifacts.local/unreal/<固定输入目录>
python research/active/dtr-r0/unreal/ue_fixed_replay.py replay --dataset artifacts.local/unreal/<固定输入目录> --output artifacts.local/unreal/<新感知结果目录>
python research/active/dtr-r0/unreal/scenario_bank.py freeze --manifest artifacts.local/unreal/<新难例库.json>
python tools/run_street_ablation.py --engine <本机UE安装目录> --scenario-manifest artifacts.local/unreal/<新难例库.json> --output artifacts.local/unreal/<新三组对照目录>
python tools/run_street_closed_loop.py --engine <本机UE安装目录> --map StreetLabV4 --scenario-manifest artifacts.local/unreal/<新难例库.json> --scenario-split development --controller-mode JOINT --output artifacts.local/unreal/<新难例运行目录>
```

原有三个控制模式为 `DTR_ONLY`、`DEPTH_ONLY`、`JOINT`。DTR 单独模式关闭整个深度控制通道，包括有效性停步、侧向路径选择和回归路线判定，保留原有 DTR 风险到等待/恢复的接口；没有另外创造 DTR 路径规划器。深度单独模式忽略 DTR 对动作的影响。各模式仍计算并记录两个原始感知分支，便于诊断，不能拿其 worker 耗时当作单分支计算成本。回放画面区分 raw X73 输出和实际启用的动作来源。

新增研究模式 `CANDIDATE_DEPTH` 和 `CANDIDATE_DTR` 使用同一组候选动作和当前足迹，后者允许预测交会改变动作选择。两者要求增量引擎，保留即时深度刹车和无效深度停步；未下发的候选不继承旧计划凭据。八场景固定比较入口为 `tools/run_street_candidate_comparison.py`，参数为 `--engine`、`--scenario-manifest`、`--output`，详细结果与使用范围见重构报告。

UE 启动器和 worker 的运动模式现在默认 `DEPTH_ONLY`：已有三组对照中深度单独为 8/8、联合为 7/8；修正采样窗口后的候选动作对照中两组均为 8/8，到达时间相同，新控制器改变三次动作但没有增量收益，因此保留为显式研究选项。X24/X25 共享拟合接口允许调用方指定采样窗口，候选模式按源时间戳冻结为 0.60 s，修复 5 Hz 数据在旧 0.50 s 窗口内无法满足四帧要求的问题；原默认值和 X73 输出保持兼容。`--action-footprint-state frozen` 可复现旧足迹配置。此处默认值只作用于 UE 实验入口。

三组入口冻结脚本、地图、渲染配置、模型及案例库，为每组重新运行八对直行/辅助分支，输出 `comparison.json`。`--resume` 只续接同一冻结运行的检查点，算法失败不会触发调参重跑。改动输入或源码须使用新的运行目录。

难例库包含八个原始回归条件、十六个 Development 变体及八个保留配方。速度、出现时机、遮挡位置/尺寸与低障碍位置会改变实际演员脚本；直行接触标签依据连续代理几何重新计算。`--case` 可做小范围接通检查，但评价器仍保留完整分组的分母，子集不会冒充整组完成。案例定义和真值只进入 evaluator，传感器 worker 不接收它们。

保留配方当前不参与调参或运行。将来明确启用时，先用 `scenario_bank.py release-held-out --manifest ... --reason ...` 记录用途，再以 `--scenario-split held_out --allow-held-out` 运行；首次访问即记录使用，保留原 manifest 及 admission/consumed 侧文件。中断后只用同一输出的 `--resume`，不将已访问条件恢复成未见条件。

## V4 官方建筑街区

```powershell
python tools/unreal_obstacle_lab.py --engine <本机UE安装目录> --city --open
python tools/run_street_closed_loop.py --engine <本机UE安装目录> --map StreetLabV4 --output artifacts.local/unreal/<新运行目录>
```

先通过 Fab/Epic 启动器将 [Epic City Sample Buildings](https://www.fab.com/listings/008fe959-5511-428e-93bd-f99b1179f6d5) 添加到 `BlindAssistStreetLab` 项目。`--city` 使用已下载资产，启用虚拟纹理，构建 V4、重载检查人物运动并生成三张实际截图；已有通过回执时直接使用现有地图。重建前应关闭该项目的编辑器。旧 V4 文件会按哈希归档到 `Saved/street-v4/previous-builds/`，源 V3 在编辑前另存字节备份。UE 下载内容、派生缓存和证据留在本地，Git 只包含脚本和说明。

新街区移除 1035 个程序立面部件，用四栋原尺度官方楼体组成沿街立面与远端广场背景：三栋 `BPP_Bldg_Hero_CHA_A01_N1` 遗产建筑和一栋 `BPP_Bldg_Hero_Low_SFD_Long_N1` 现代建筑。上部楼层与各自独立的 `Level01` 首层同时装配，共 1886 个网格实例；沿用原生门窗、檐口、材质和窗内景。原咖啡亭、家具、人物、5.5 m 横穿巷口与传感路线保留。包中的窗内景是原资产视觉效果，不表示已搭建可进入、可用于避障测试的室内空间。

最终地图 SHA-256 为 `3bb1ea3b8ed16d300e7fa3178542313f48548b5efdc276fc7ca188188c226e28`。重载后在仿真时间 7.126–9.454 秒检查四人分别移动 279.4、279.7、279.7、178.7 cm，足骨均更新。`Saved/lab-visual-v4.json` 包含构建、源地图检查、PIE 与逐图目检记录。实际 1920×1080 `Hero`、`Cafe`、`Walking` 截图在 `artifacts.local/unreal/street-v4-visual/release/`；首轮缺首层的失败画面保存在 `failed-first/`，未用于实验或当作交付。资产获取回执为 `Saved/city-sample-acquisition.json`。

V4 运行入口额外记录 `DefaultEngine.ini` 哈希，并将原始地图保存到运行目录的 `scene_snapshot/` 后核对 SHA-256，禁止将 V3 的成功率迁用到新地图。它延续相同四类、八条件设计，具体联合控制结果以 V4 的独立 `evaluation.json` 和配对回放为准。实验索引登记仍被 `experiments/index.jsonl:252` 既有记录的 `input_fingerprint` 不匹配阻断；未修改该记录，本地完整身份与结果保留。

### V4 全套实际运行结果

`closed-loop-v4-suite-20260905-a` 完成 **16 个全新分支、733 帧**，没有复用 V3 输入或重跑分支。8/8 直行对照成立，8/8 辅助分支无代理接触到达目标；四个碰撞条件均通过传感器、下一步命令和实际运动的时序核对。

| 条件 | 直行耗时 | 辅助耗时 | 辅助代理接触 |
| --- | --- | --- | --- |
| 遮挡横穿：碰撞 / 近失 | 8.0 / 8.0 s | 10.8 / 8.0 s | 无 / 无 |
| 突然停步：碰撞 / 近失 | 8.0 / 8.0 s | 9.6 / 8.8 s | 无 / 无 |
| 窄道会车：碰撞 / 近失 | 8.0 / 8.0 s | 11.6 / 13.0 s | 无 / 无 |
| 低矮障碍：碰撞 / 近失 | 8.0 / 8.0 s | 9.6 / 8.0 s | 无 / 无 |

窄道近失增加 **5.0 秒**，是明确的效率代价。三类碰撞条件首次避让来自 `OBSERVED_DEPTH`，窄道碰撞来自 `DTR_X73`；8/8 属于两分支联合闭环，不是 X73 单独成绩。worker 在 CUDA RTX 5060 Laptop 上的 733 次推理中位耗时 0.627 秒、P95 1.919 秒；锁步 UE 运行共 850.469 秒，不能称为墙钟 5 Hz 实时系统。

运行前后七个冻结源文件、V4 地图、原始地图快照与渲染配置哈希全部一致；配置 SHA-256 为 `ec779caaa624e0339a7a36e9727a9725f8a4598e2d08ade0b16167df55f429da`。运行所属 UE、worker 及其子进程、Zen、端口和锁均已释放。结果、同步双分支回放和预览保存在该目录的 `evaluation.json`、`replay.html`、`replay.gif`、`closed-loop-preview.png`。这是该组受控 Development 条件的实际成功，不扩展到人体安全或自然场景泛化。

## V3 场景与实机画面

```powershell
python tools/unreal_obstacle_lab.py --engine <本机UE安装目录> --polish --open
python tools/run_street_closed_loop.py --engine <本机UE安装目录> --map StreetLabV3 --case low_obstacle_collision --output artifacts.local/unreal/<新运行目录>
```

`--polish` 使用本机 Epic `Building/Geometry/SM_Building` 曲面咖啡亭替换三个重复南侧楼组，完整 `ConceptCar/Car/SM_AutomotiveTP_Car` 替换旧运动汽车，并导入 [Poly Haven 的实拍阴天 HDR](https://polyhaven.com/a/overcast_industrial_courtyard)。该 HDR 为 CC0，下载器验证提供方 MD5 并记录 SHA-256；文件与许可信息保存在 `asset-downloads/environment/`。近景家具使用其 [Outdoor Table Chair Set 01](https://polyhaven.com/a/outdoor_table_chair_set_01) CC0 模型与 2K PBR 贴图，提供方校验及许可保存在同名下载目录。这些是实际三维资产和环境照明，不是背景效果图。City Sample Buildings 尚未下载或迁入，不能把当前场景称为 City Sample。

导向铺装使用原生厘米尺寸、圆顶凸条的 1 m 网格，避免把单个立方体拉伸到 71 m；3–4 mm 的凸起无阻挡碰撞。新地图独立绑定四名人物，保存后重载并在 PIE 检查实际位移及脚部骨骼变化。`Saved/lab-visual-v3.json` 保存地图哈希、源地图未改校验及运行结果；只有该回执通过，打开入口才默认选择 V3。

同镜头的 1920×1080 Hero / Cafe / Walking 对比由 `render_street_v3.py` 在实际 UE 中生成，输出到 `artifacts.local/unreal/street-v3-visual/`：`before/` 为 V2，`release/` 为最终 V3。V3 的新实验必须显式使用 `--map StreetLabV3`；下文 7/8 成功结果来自旧 V2 受控迭代，不能移用为 V3 成绩。

2026-09-05 的 V3 低障碍双分支验证位于 `closed-loop-v3-canary-20260905-a`，90 个实际新 RGB-D 帧、约 110 秒完成。直行分支在 3.54 s 接触低障碍代理；辅助分支在 1.0 s 由观测深度触发，1.2 s 实际偏离直线，9.6 s 无接触到达终点，命令与运动核对无不一致。此次仅验证一个条件的接通，评价器总体仍标 `INCOMPLETE`，不冒充 V3 全场景通过或 DTR 单独收益。保存地图哈希在实验前后均为 `17d7f5d35dff048e929be92895da6269a61be9cc51abbafcf6157729fc5e905b`；V2 源地图字节保持不变。

### V3 完整场景验证

后续 V4 建图的首次尝试出现保存目标错误：UE `save_map` 写出副本后没有切换当前世界，随后保存触及 V3。构建哈希断言捕获后停止，未运行新实验。已从编辑前副本恢复 V3 的 1575 个演员，逐项核对标签、类型、位置相同；恢复后地图 SHA-256 为 `15a9908f68f6a2779dc6b33346921b1ee55c080f6776439eb1a2ba5f4729d041`。这不是旧 `17d7...` 文件的字节恢复，旧实验的原始地图字节副本已缺失；旧 RGB-D、轨迹、回执、截图仍保留，不回填哈希。恢复记录和原始副本保存在 `Saved/street-v4/recovery.json` 及 `recovery/`。V4 构建器已改为编辑前按哈希备份源文件，并明确载入、断言目标世界后才修改。

`closed-loop-v3-suite-20260905-a` 已完成四类、八条件的 **16 个全新分支**，738 个实际 RGB-D 帧，墙钟 1233.313 秒；没有复用 V2 对照或重跑分支。8/8 直行对照符合预设接触/近失条件，8/8 辅助分支无代理接触并到达目标；四个碰撞条件均通过传感器触发、下一步命令和实际轨迹改变的时序核对，接触回执及命令应用不一致计数均为零。七个运行源文件和上述 V3 地图哈希在运行前后保持一致。

这是保留 DTR X73 与观测深度近障碍分支的联合控制结果。横穿、停步和低障碍条件的首次实际避让来自 `OBSERVED_DEPTH`，窄道会车来自 `DTR_X73`，不能写成 X73 单独达到 8/8。近失条件仍暴露效率代价：停步近失增加 0.8 秒，窄道近失从 8.0 秒增加到 14.0 秒；全部八个辅助分支平均增加 2.05 秒。738 次 worker 推理的中位耗时 0.872 秒、P95 2.763 秒，仍是锁步运行，不是墙钟实时性能证明。

该目录的 `evaluation.json` 保存完整结果，`replay.html`、`replay.gif` 和 `closed-loop-preview.png` 可查看实际双分支画面与轨迹。运行完成后，所属 UE/worker/owner 进程、端口和锁均已释放。8/8 表示这组受控 Development 条件成功，不据此宣称人体安全、自然场景泛化或后续建筑版本自动通过。

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
