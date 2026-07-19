# 显式路线意图条件化风险模型合同（2026-07-19）

## 决策

不再训练模型从 RGB、过去光流或未对齐 IMU 中猜测使用者未来会走哪条路线。模型职责拆成：

1. 视觉网络输出场景障碍/风险轮廓，不直接决定左绕、右绕或直行。
2. 导航规划或用户明确选择提供 `route intent`，投影成当前相机归一化坐标中的 1/2/3 秒路线点或路线走廊。
3. 路线与风险轮廓的交叠使用确定性计算；连续两秒达到 `1/3` 交叠才进入 `intervention_needed`，连续两秒低于阈值才 `route_clear`。
4. 路线缺失、过期、低置信或缺投影回执时，只允许无方向的 `context_attention`，不允许模型补猜方向。

## 为什么这是当前证据支持的解法

r7.90/r7.91 表明纯视觉 event mean 和风险轮廓/生命周期分别只有 `.3182/.5227` balanced accuracy；r7.94–r7.96 表明未对齐 IMU 对未来意图和当前路径转向也不可分。

r7.97a 保持相同 16 个 actionability 事件和 11 个完整来源，不训练模型，只把冻结 future-route teacher 当作“外部已知路线”的 oracle 代理。它勘误 r7.97 的证据校验：除 12 份 feature report SHA 外，还重新计算并核对 13 个被加载本地视频的文件 SHA。固定连续两秒直接交叠规则得到：

- intervention recall `1.0`（4/4）；
- context recall `.8333`（10/12）；
- balanced accuracy `.9167`；
- context false-upgrade `.1667`；
- 所有事件有效帧覆盖率 `1.0`；
- 报告 SHA256 `ae019fa9f9046dae578713480c391e660281881ef1a869c95af5881042eb1e85`。

四个既有干预事件中，Edmonton/Ulm c01 的 oracle open 与因果 open 同时；Bangkok 两段分别提前 1 秒和 4 秒。两个 disagreement 是 London 与 Ulm c03：它们在 current/past-only 合同下是 context，但显式计划显示连续未来路线交叠，因此属于信息状态改变，不能用 oracle 反写旧标签，也不能直接记为生产假阳性。

## 模型与训练边界

推荐输入输出：

```text
视觉 backbone -> 像素/patch 风险轮廓 --------------------┐
                                                       ├-> 确定性交叠 -> lifecycle
外部 route provider -> 1/2/3s route field + valid mask ┘
```

- route field 是输入，不是视觉 head 要预测的 target。
- 像素分割、free-space 和距离场用于改善风险轮廓定位，仍是辅助监督。
- actionability 不再作为脱离路线的全局二分类 head。
- train-only 可以用未来视频教师生成路线代理，帮助验证结构和训练风险轮廓；eval/runtime 严禁未来视频教师。
- eval 必须使用真实外部 route provider，或明确 route unknown 并只评估 context fallback。
- route provider 与风险模型必须解耦；不得把风险模型自身输出循环包装成“显式意图”。

## 接口

机器可读模板为 `configs/explicit_route_intent_input_template_v1.json`，fail-closed 校验器为 `scripts/validate_explicit_route_intent_episode.py`。每个样本必须包含：

- 单调时间戳、有效期和置信度；
- `normalized_current_camera_frame_xy` 下的 1/2/3 秒 waypoint；
- provider 身份与投影回执；
- 若原始输入是世界坐标 waypoint，必须有 device-to-world 对齐回执。

校验器还会拒绝风险模型自生成意图、future-video runtime provider、非单调时间戳、超过一秒的有效期、越界坐标以及携带 waypoint 的 invalid-route 样本。

确定性生命周期原型位于 `scripts/explicit_route_intent_fusion.py`。unknown route 不会开事件，也不会伪造 clear：已经打开的事件在 route 丢失时保持未确认状态，等待有效路线或其他安全清除证据。

端侧等价原型位于 `device-benchmark/.../ExplicitRouteIntentFusion.kt`，仅参与 benchmark instrumentation；没有被 `app`、`core:assist` 或默认风险状态机引用。

r7.98 使用 repo-local JDK 17 完成 `:device-benchmark:assembleDebug`，APK SHA256 `5c44bcab...284c`；随后在 SM-S9280/API 36 上直接运行目标 instrumentation，`OK (3 tests)`。设备回执位于 `artifacts.local/evidence/public-video-r798-explicit-route-intent-device-20260719/device_benchmark_receipt_r798.json`。这只证明确定性状态机可在端侧等价执行，不代表真实 route provider 或生产门已经闭合。

## 晋级条件

1. 先实现隔离 route provider，不接默认 App；验证时间戳、投影误差、过期和 unknown fallback。
2. 至少两个独立真实来源提供非未来帧生成的 route intent，并重放 r7.97a 同一规则。
3. 比较 `visual risk only` 与 `visual risk + explicit route` 的 event recall、context false-upgrade、open lead、clear latency。
4. 通过后再训练 route-conditioned risk-profile head；五组 prototype/bootstrap 只评估风险轮廓稳定性，不再要求 head 猜路线。
5. 离线、INT8、同设备事件门全部闭合前，Android 默认和生产替换保持关闭；SAM/ASAM 仍是第三顺位。

## r7.99–r8.07：三态选择验证仍未闭合

r7.99 用冻结 future-route teacher 的均值 x 代理 `LEFT/STRAIGHT/RIGHT`，再完全替换为三条固定相机坐标模板，16 个既有事件得到表面上的 balanced accuracy `1.0`。r7.99a 随即发现四个 intervention 全是 `STRAIGHT`，LEFT/RIGHT 各只有 context，故只支持 straight-choice 诊断，不支持完整三态 provider。

r8.00 从 753 个无事件标签的 r7.61 行中找到 3 个 LEFT、2 个 RIGHT 候选。r8.01/r8.02 连续视觉复核全部拒绝，并定位出语义错误：r7.61 的点是“未来下中点经 DIS flow 映回当前帧的对应位置”，不是转向类别。卡车等独立运动、相机平移和低置信 detector 假框都能改变均值 x。因此 r7.99 的满分不能作为三态 provider 证据；r7.97a 直接使用完整路线点的接口上限不受影响。

r8.03 改为只看上半画面稳健背景水平流，在 753 行上产生 13 个 LEFT（4 来源）和 17 个 RIGHT（6 来源）复核候选。r8.05 再与冻结 r7.99 模板做连续两秒相交，只剩 Kampala LEFT、Kampala RIGHT 和 Bramwell RIGHT 三段。r8.07 大模型连续复核确认：两段 Kampala 是真实转向和真实施工物，但命中的是平行边界而非持续阻断所选路线；Bramwell 是墙体/挡土块误检。LEFT/RIGHT intervention 新增均为 0。

当前可以实现并测试 `STRAIGHT` 诊断和通用 route-field 接口，但不能把三态用户选择包装成已验证 provider。下一数据目标必须同时具备：真实 LEFT/RIGHT 路线选择、同一匹配临时障碍连续进入该分支、以及可观察的停步或改道；不能再用 future-flow x、扩大检测框或阈值搜索代替这一因果组合。

## r8.08–r8.09：许可优先的定向公开视频发现

r8.08 在看候选前冻结 Wikimedia Commons 与 Vimeo 各三条查询、每条最多一页/25 项，以及“真实 LEFT/RIGHT 选择 + 同一临时障碍持续阻断所选分支 + 连续两次相交 + 停步或改道”的准入条件。Commons 首次 Python TLS 在握手阶段失败且没有产生 HTTP 响应；r8.08a 只更换为 Windows TLS，并对完全相同的三条查询各请求一次，不分页、不改词。

结果没有新增事件：Commons 唯一候选是警方执法视频，元数据还命中 `dashcam` 负项，未下载即拒绝；Vimeo 两条查询返回维护页，唯一成功查询的 8 项中只有 Burwell 隧道改造标题相关。该视频条目级许可确认为 CC BY 3.0，但 5 秒间隔全片复核显示它是带标题卡、地图、效果图、采访和硬切的公共工程新闻片，不是连续行人第一视角，也没有所选 LEFT/RIGHT 分支被同一障碍持续阻断的因果 episode，因此拒绝准入。

r8.09 后 LEFT/RIGHT intervention 仍为 `0/0`。许可通过不能替代连续性和事件语义；三态 provider、训练、Android 默认接线、校准、blind 与生产门均保持关闭。

r8.10–r8.11 又以同样的先冻结方式查询 Internet Archive：三条查询共得到 3 项带许可字段的候选，但全是 1930–1945 年道路/测绘/军事史料。唯一高分项只是长描述跨段包含 walking、construction、detour，不能构成连续行人 POV。三项均在元数据阶段拒绝，未下载，覆盖与门禁不变。

## r8.12–r8.18：路线条件风险场的 train-only 根因闭环

r8.12/r8.12a 从三个已审阅真实父来源构造严格 train-only 路线条件反事实，同一障碍图分别绑定 LEFT/STRAIGHT/RIGHT 固定 waypoint，标签只由精确合成 bbox 与路线点交叠产生。首版因 `static_obstacle` 类名可能把父画面原有栏杆、长椅等变成漏标目标而拒绝；r8.12a 收窄为 `inserted_temporary_obstacle` 后，36 张图、108 个路线样本、精确 mask、YOLO/COCO 和全图复核通过。

r8.13a 的同一 LOSO 线性 head 在精确风险场上 balanced `1.0`，但冻结 DINO 二值 mask teacher 的全局/路线 readout 只有 `.5000/.6249`，证明 head 可解而风险表示不可解。r8.14a 随后让 barricade 与 sand pile 两个障碍族都出现在三个父来源；原 r8.14 因合同内未来时间戳被拒并全量重执行。双障碍族二值 teacher 仍只有 `.6332`，排除“只缺一个砂堆训练来源”这一解释。

r8.16 保持 DINO、LOSO、route pooling、ridge 和门槛完全不变，只把二值 patch target 换成冻结 bbox 距离场。路线条件 balanced 升至 `.9156`，clear/block recall `.8696/.9615`；LEFT/STRAIGHT/RIGHT 为 `.98/.89/.88`，最差父来源 `.89`，全局 readout 仍只有 `.5446`。这支持“连续局部风险场 + 显式路线交互”，不支持全局分类 head。

r8.17 固定连续两帧 open 生命周期得到 balanced `.9429`、clear/open recall `.9524/.9333`，最差父来源 `.90`。当前合成序列没有障碍离开后的连续片段，因此只验证 open，不宣称 route-clear 已验证。r8.18 五组 prototype/bootstrap 80-step 短跑均稳定（BA std `.0090`），但预注册门失败：mean `.8774`、worst seed `.8682`、worst clear recall `.7971`。不事后调参，不启动 SAM/ASAM；确定性 ridge 仍优于短跑 head。

这些结果只授权把距离场作为风险轮廓辅助监督继续带到真实 provisional 事件诊断。合成数据不填 LEFT/RIGHT 真实 provider/eval 覆盖，也不授权 Android、校准、blind 或生产替换。

## r8.26–r8.26a：端侧几何闭环与纵横比勘误

r8.26 首次把 `route waypoints + detector bbox -> intersectionFraction` 移入 benchmark-only Kotlin，并与既有两帧 open/clear 生命周期串联。6 个几何测试与 3 个生命周期回归在真机通过，但随后的全量逐帧审计拒绝了该实现：r7.97a 的 654 个锚点中有 59 个 hit 不一致，218 帧中有 30 个分数不一致。

根因是坐标单位错误。r7.97a 在像素空间用目标像素高度同时扩张 x/y；r8.26 把归一化目标高度同时当成 x/y margin，只有正方形画面才等价。r8.26a 固定为：

```text
height_px = max(1, (bottom_norm - top_norm) * frame_height_px)
margin_px = 0.5 * height_px
margin_x_norm = margin_px / frame_width_px
margin_y_norm = margin_px / frame_height_px
```

修复后，4 个离线单元测试通过；11 个来源、16 个事件、218 帧、654 个锚点与 r7.97a 达到 `0` hit mismatch、`0` frame-score mismatch。SM-S9280/API 36 上新几何 7 项加既有生命周期 3 项为 `OK (10 tests)`，APK SHA256 `b543dd9d...62fe`。收据位于 `artifacts.local/evidence/public-video-r826a-explicit-route-geometry-device-20260720/device_benchmark_receipt_r826a.json`。

这证明端侧确定性几何与生命周期实现可精确复现冻结 oracle 语义，也证明全量 conformance 不能被少量手工测试替代。它仍不验证真实 route provider、LEFT/RIGHT 真实 intervention、App 接线或生产可用性；默认运行时保持不变。

## r8.27–r8.27a：Android 外部路线 payload 边界

r8.27 新增 benchmark-only Android `Intent` payload parser，要求显式 action、provider ID、投影回执、签发/失效时间、置信度和严格 1/2/3 秒相机归一化 waypoint。风险模型自产路线、future-video 路线、未来签发、过期、超过一秒有效期、低置信、缺回执或畸形数组全部 fail-closed。首轮仅因测试夹具错误地把 `removeExtra()` 的 `Unit` 返回值当成 `Intent` 而编译失败；r8.27a 只修测试写法，合同规则与 provider 逻辑不变。

修复后离线编译成功；SM-S9280/API 36 上 provider 6 项、r8.26a 几何回归 7 项、生命周期回归 3 项合计 `OK (16 tests)`。这关闭的是“Android 能否安全接收外部非未来路线并接上确定性风险链”的接口风险，不是“某个真实导航 App 的路线投影准确”或“LEFT/RIGHT 真实阻断事件已验证”。App/core/default runtime 仍未接线。

## r8.28–r8.28a：世界路线到相机 waypoint 投影

r8.28 增加 benchmark-only `local ENU route + world-to-camera pose + pinhole intrinsics -> normalized camera waypoint`。姿态矩阵必须正交、右手且为 3×3；回执必须在一秒时效内且置信度至少 `.5`；相机后方/过近、画面外、畸形 horizon 或无效内参全部 fail-closed。首轮三项“有效路线”测试被正确拒绝为 `invalid_pose`，原因是测试 identity fixture 少一个零、只有八个元素；r8.28a 只修夹具，不改算法与阈值。

r8.28a 在 SM-S9280/API 36 上通过投影 6 项、外部 payload 6 项、几何 7 项、生命周期 3 项，共 `OK (22 tests)`。因此坐标变换数学、接口和端侧风险链已经连通；仍未验证真实手机姿态提供者、真实相机标定误差、真实导航路线或 LEFT/RIGHT 障碍事件，App/default runtime 保持隔离。

## r8.29–r8.29d：真实设备投影输入能力

在 SM-S9280/API 36 上读取真实 `CameraCharacteristics` 并采集 2.5 秒 `TYPE_ROTATION_VECTOR`。r8.29 的 nullable `FloatArray` 写法先在编译期拒绝；r8.29a–c 的相机/传感器能力断言均已通过，但收据先后因 test APK external/internal data 目录不可写而在最终导出失败；r8.29d 只把同一 JSON 改由 instrumentation status 返回，正式得到 `OK (1 test)`。

设备有两个后置 camera ID。r8.29d 在无 `CAMERA` 权限的 instrumentation context 中只能由焦距、物理 sensor size 和 pixel array 推导内参；当时读到的 exact intrinsic/distortion null 后经 Android 官方权限合同证明是权限过滤，不能解释为设备没有这些字段。rotation vector 取得 119 个样本/2453.9ms，中位间隔 20.80ms（约 48Hz），最大四元数范数误差 `4.16e-8`；本次静置最大相对角变化 `.0255°`，只作观测、不作精度门。

### r8.30 权限化精确标定与时间基准审计

r8.30 在冻结合同后使用已获 `CAMERA` 权限的 target App context 真机执行并通过 `OK (1 test)`。两颗后摄均公开 `LENS_INTRINSIC_CALIBRATION`、`LENS_DISTORTION`、`LENS_POSE_ROTATION/TRANSLATION`；主摄精确 `fx=2766.12px, fy=2771.18px`，两颗相机 `SENSOR_INFO_TIMESTAMP_SOURCE=REALTIME`。pose reference 均为 `PRIMARY_CAMERA`，但 pose rotation 的官方定义仍是 Android sensor 坐标到 camera-aligned 坐标的旋转；因此可与 rotation-vector 的 device-to-world 旋转组合为候选 `R_camera_from_world = R_lens_pose * transpose(R_device_to_world)`。

这消除了“必须先独立求 device-to-camera 旋转外参”的前置缺口，但没有验证真实投影精度。仍须完成矩阵组合确定性 conform、实际分析流 crop/scale 内参换算、真实画面的轴方向和重投影误差上界，以及真实导航路线/LEFT-RIGHT 事件；在此之前 App wiring 与生产授权继续关闭。

### r8.31 Android 姿态矩阵到相机坐标的确定性组合

r8.31 冻结后实现 benchmark-only `AndroidCameraPoseComposer`，严格计算 `R_lens_pose * transpose(R_device_to_world)`，验证四元数范数和输入/输出旋转矩阵并 fail closed。在 SM-S9280/API 36 上通过 `OK (5 tests)`：identity、矩阵逆序、r8.30 主摄实测 `xyzw` 四元数、与 r8.28 world projector 的衔接、非法旋转/四元数拒绝。

该轮只关闭 raw camera-aligned sensor 坐标中的旋转组合缺口。`SENSOR_ORIENTATION`、CameraX 实际 analysis stream 的 crop/scale、相机帧与 rotation-vector 时间配对以及真实画面重投影仍是独立门，不能由这五个确定性测试外推。

### r8.32–r8.34 CameraX 实际分析流与精确内参映射

r8.32 使用与生产一致的 DEFAULT_BACK_CAMERA、640×480 4:3 resolution selector、RGBA_8888 和 KEEP_ONLY_LATEST，在 SM-S9280/API 36 捕获 30 帧并通过 `OK (1 test)`。实际绑定 camera 0，所有帧均为 640×480、crop `[0,0,640,480]`、`rotationDegrees=90`、row stride 2560、pixel stride 4；没有隐藏 crop。capture timestamp 严格递增，中位帧间隔 `66.64ms`，回调减 capture timestamp 中位 `77.43ms`。

r8.33 读取 CameraX `ImageInfo.sensorToBufferTransformMatrix` 并通过 `OK (1 test)`。矩阵为 `diag(0.15686275, 0.15686275, 1)`，把 Camera2 active array 4080×3060 精确映射至 640×480 buffer；精确主点从 sensor `(2041.33,1530.07)` 到 buffer `(320.21,240.01)`，再经 90°顺时针变为 480×640 display 的 `(238.99,320.21)`。

r8.34 将权威 sensor-to-buffer 仿射、精确 intrinsic、ImageInfo clockwise rotation 与 camera x/y 轴同步旋转封装为 fail-closed mapper，并接回 r8.28 world projector；SM-S9280/API 36 通过 `OK (4 tests)`。映射后的 display intrinsic 为 `fx=434.6943, fy=433.9006, cx=238.9884, cy=320.2087`。这关闭了确定性坐标公式缺口；仍未关闭逐帧 IMU 时间配对、镜头畸变后的真实像素误差和真实路线事件门。

### r8.35 相机 capture timestamp 与 rotation-vector 样本配对

r8.35 同时运行生产匹配 CameraX analysis 与 `TYPE_ROTATION_VECTOR/SENSOR_DELAY_GAME`，冻结后在 SM-S9280/API 36 通过 `OK (1 test)`。30 个相机帧全部被一前一后 rotation-vector 样本夹住；最近样本差中位 `4.81ms`、最大 `9.67ms`，bracket 跨度中位 `19.756ms`、最大 `19.756ms`。因此可在每个 `ImageInfo.timestamp` 上对相邻四元数做 SLERP；但该轮只证明时间配对覆盖，不证明插值姿态或真实像素投影精度。

### r8.36–r8.36a 自动真实帧重力轴检查

r8.36 在冻结后自动保存 pose-linked 真实帧，并把 ENU world-up 经完整相机链投到显示帧消失点。手机当时几乎正对昏暗天花板；固定 Canny 60/160 得到 0 candidate，因此按合同只能记为 non-informative。r8.36a 在看见低照原因后另冻合同，复用同一帧、不重采，采用 CLAHE + Canny 15/50：得到 89 条候选、10 条 10°内对齐线，最小角误差 `.42°`，但 aligned length fraction `8.782%` 低于冻结 10% 门，正式记为 informative fail。

该失败不能被改阈值抹去，也不能直接归因于坐标链：画面主要是水平天花板/面板边界，而原门把 aligned length 除以所有 Hough 长线，隐含了“场景由世界竖直线主导”的错误假设。下一独立证据改为短振动产生可测小旋转，以 IMU 预测的 `K R K^-1` 单应矩阵直接对比 LK 光流；它不依赖场景线的三维语义。
