# 工具与命令导航

从仓库根目录运行命令。Android 构建、设备回归、发布和仓库检查见
[scripts/README.md](../scripts/README.md)；研究优先级见
[当前决定](../docs/CURRENT_DECISION.md)。本页按用途定位入口。

## 统一入口 `ba.ps1`

[ba.ps1](ba.ps1) 按命令和 profile 分发。常用检查：

```powershell
pwsh -NoProfile -File tools/ba.ps1 doctor base
pwsh -NoProfile -File tools/ba.ps1 doctor android
```

| 命令 | 已实现的 profile | 行为 |
| --- | --- | --- |
| `doctor` | `base`、`android`、`device`、`export`、`research-dtr-r0`、`research-l10-r0` | 检查对应环境；研究 profile 也探测运行后端 |
| `setup` | 同 `doctor` | `export` 创建并安装导出环境，其余 profile 执行环境检查 |
| `run` | `android`、`device` | 转交 Gradle 参数，`device` 同时要求设备可用 |
| `run` | `export` | 在导出环境执行 `--` 后的脚本与参数 |
| `run` | `research-dtr-r0`、`research-l10-r0` | 调用既有 DTR 事件评估或 L10 基准，并登记输入和输出 |
| `run` | `research-ue` | 运行 UE 复用任务；可用 `-RunSpec` 指定运行规格 |
| `smoke` | `research-dtr-r0`、`research-l10-r0` | 分别运行路线测试或 25 episode 基准 |
| `materialize` | `research-dtr-r0` | 用 `-CanaryManifest` 和 `-CanaryOutput` 生成真实输入适配结果 |
| `assets` | `base` | 增量同步资产目录；`-- --full` 执行全量发现 |
| `clean` | `export` | 删除 profile 所属的导出环境 |

研究运行的输入、参数和证据边界分别见
[DTR 当前页](../research/active/dtr-r0/CURRENT.md)、
[L10 当前页](../research/active/l10-r0/CURRENT.md)和
[UE 复用说明](../docs/asset-management/UE_REUSE.md)。这些 profile 对应既有评估入口；
RGB/CNH 融合与实机工具从[硬件路线](../research/active/hardware-bringup/CURRENT.md)进入。
本机配置键见 [local.example.toml](../config/local.example.toml)。

## 按用途找代码

| 用途 | 代码入口 | 使用说明 |
| --- | --- | --- |
| 查找已有方法、结果和继承关系 | [knowledge.py](knowledge.py) | [研究知识库](../research/knowledge/README.md) |
| 研究提案的阶段与证据字段 | [research_workflow.py](research_workflow.py) | [研究工作流](../research/WORKFLOW.md) |
| 计算后端与远程 worker | [research_backend.py](research_backend.py)、[remote_worker.ps1](remote_worker.ps1) | [主机计算说明](../docs/HOST_RESEARCH_COMPUTE.md) |
| 资产查询、登记与运行记录 | [data/asset_catalog.py](data/asset_catalog.py)、[data/asset_runtime.py](data/asset_runtime.py) | [资产管理](../docs/asset-management/README.md) |
| 不可变资源、共享缓存与薄实验 | [data/resource_fabric.py](data/resource_fabric.py) | [资源管理说明](../docs/RESOURCE_FABRIC.md) |
| UE 场景采集与批任务 | [ue_native_capture.py](ue_native_capture.py)、[ue_capture_batch.py](ue_capture_batch.py) | [批量采集](../docs/operations/UE_CAPTURE_BATCH.md)、[UE worker](../docs/operations/WORKER_UE_DEVELOPMENT.md) |
| CARLA 与 City 场景实验 | [carla_assist.ps1](carla_assist.ps1)及本目录 `run_dtr_carla_*`、`city_*` 工具 | [DTR 详细入口](../research/active/dtr-r0/README.md) |
| SkyDiscover 接入 | [skydiscover_assist.ps1](skydiscover_assist.ps1) | [SkyDiscover 使用说明](../docs/SKYDISCOVER_PLAYBOOK.md) |

缺少已知基线或方法背景时，可定向查询：

```powershell
python tools/knowledge.py context --route dtr-r0 --limit 4 --query "nearfield"
```

生成的数据、模型、日志和临时文件统一放在 `artifacts.local/`；
目录归属见[本地资产说明](../docs/LOCAL_ARTIFACTS.md)。
