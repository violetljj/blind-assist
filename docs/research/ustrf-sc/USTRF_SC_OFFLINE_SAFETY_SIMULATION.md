# USTRF-SC 离线安全仿真与自动验证

状态：2026-07-20，production-isolated。
目的：在不使用人工采集、相机、模型输出或物理标定的条件下，验证 P0 安全内核对**带标签的合成几何与相对运动证据**的确定性处理。

## 证据边界

仿真输入位于 pure Kotlin `UstrfOfflineSafetyScenarioRunner`，输入是明示 `synthetic-fixture-v1` / `synthetic-motion` 的局部米制证据，不是图像、深度图、ARCore pose 或真实用户轨迹。

因此，本仿真能证明：

- metric geometry → risk observation → risk field → corridor → supervisor 的离线数据通路；
- 五个候选 lateral corridor 可按一格半宽的人体包络扫掠，而非只检查中心线；
- 障碍、全宽下坠、全宽头部障碍、动态交汇、中央未知、过期几何和失跟均遵循预期的 SLOW/STOP fail-closed 行为；
- 全套场景和 trace digest 可重复。

它不能证明：

- 摄像头/模型能从真实环境正确产生这些几何或动态证据；
- 真机时延、功耗、热、光照、镜头遮挡、标定或 VIO 世界稳定性；
- 对真实下坠、树枝、玻璃、行人、自行车的召回率或误停止率；
- 可向用户签发 `CONTINUE`、转向或任何生产导航动作。

## 场景目录

| 场景 | 合成输入 | 断言 |
| --- | --- | --- |
| `CLEAR_CORRIDOR` | 7x4 clear traversable grid | 仅 shadow `SLOW_DOWN`，选择目标中心通道 |
| `CENTER_OCCUPANCY` | 中央单格占用 | 中央及相邻窄通道因包络扫掠被拒绝，trace 选择侧向 `-2` |
| `FULL_WIDTH_DROP` | 全宽地面下坠 | `STOP_AND_REASSESS` + `NO_SAFE_CORRIDOR` |
| `FULL_WIDTH_HEAD_OBSTACLE` | 全宽头部高度障碍 | `STOP_AND_REASSESS` + `NO_SAFE_CORRIDOR` |
| `DYNAMIC_CROSSING` | 全宽、1 秒内最近接近的相对运动 | `STOP_AND_REASSESS` + `NO_SAFE_CORRIDOR` |
| `CENTRAL_UNKNOWN` | 中央可通行证据缺失 | `STOP_AND_REASSESS` + `CENTRAL_CORRIDOR_UNKNOWN`；候选 offset 仅作 trace 元数据 |
| `STALE_GEOMETRY` | 几何 TTL 到期 | `STOP_AND_REASSESS` + `GEOMETRY_UNAVAILABLE` |
| `POSE_LOST` | pose health=LOST | `STOP_AND_REASSESS` + `POSE_NOT_TRACKING` |

## 可复现命令

```powershell
$env:JAVA_HOME='E:\linnan\linnan\.jdk\jdk17.0.19_10'
$env:ANDROID_HOME='E:\linnan\linnan\.android-sdk'
$env:GRADLE_USER_HOME='E:\linnan\linnan\.gradle-local'
.\gradlew.bat :core:ustrf:test --offline --no-daemon --console=plain
```

验收要求：`UstrfOfflineSafetySimulationTest` 全部通过，且全模块零 failure/error。该 gate 仅对应文档中的 G0 离线仿真/回放；不能替代 G1 受控室内或后续真机、人工场景验证。
