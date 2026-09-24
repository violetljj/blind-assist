# 冻结 CNH 164 布局 test：采集前离线准备度

状态：**`BLOCKED_REALISTIC_164_NOT_AUTHORED`**。用户选择的是完整 164 布局 test，不以三张巷道作者地图或旧单元测试布局缩减配额。本次只读取协议、配置、作者地图文件身份和旧源清单元数据；未读取 test 帧、标签、预测或模型结果，也未启动 UE。

机器可读的确定性空槽清单为 `artifacts.local/evidence/cnh-full-test-readiness-20260925-v3/plan.json`（SHA-256 `4149d48c6c87973eac5392e3445d1a8b1c6cd5083a5ca556e4d61e835f100ca1`），生成器为 `cnh_full_test_readiness.py`。它以 seed 20260924 固定顺序，四类室外环境各 41 槽；每类 16 HEAD、16 杆、3 BODY、3 墙、3 低矮。每槽固定四段×40 名义时刻，合计 26,240 时间样本。所有槽的物理场地、源 mesh 族、实际几何和采集 spec 均为 `null`；此清单**不是**可发往 worker 的采集 spec。

| 现有输入 | 实际准备度 |
| --- | --- |
| 当前 `cnh_route_spec.py` | 声明四类室外环境及 384/164 数量，但只生成 Engine primitive 的传感器单元测试几何；不具备真实素材来源和场地。 |
| 已物化的 `plan/layouts.json` 与 `source-manifest.json` | 是更早的 corridor/room/sidewalk/plaza primitive 草案；与冻结的 sidewalk/intersection/plaza/alley 环境配额不符，不能改名复用。 |
| R3a 巷道作者地图 | test 预分区有 straight/L/T 三张不同物理场地的候选地图；不是 41 个已审核布局。train/dev 仅有每图一个 Development worker 布局，test 没有正式采集 spec。 |
| Street200V7、BrickServiceA_V3 | 前者是一处共享街区，不能用换机位伪装跨分区独立场地；后者是一处作者预览，未获正式 test 准入。sidewalk/intersection/plaza 的 test 作者地图候选数目前为零。 |
| test 受隔离素材 | R3a 清单仅声明塑料桶、自行车架两个插入物源 kit，另有雨篷、消防接口、长椅三个杂物源 kit；五类障碍的完整 test 来源覆盖及实际渲染依赖隔离未建立。 |
| 正式采集入口 | 现有七路 `cnh_route_source_capture.py`/launcher 明确是 `benchmark_eligible=false` 的工程/Development 路径；`cnh_route_pilot.require_benchmark_source` 对 realistic 模式仍抛 `NotImplementedError`。受保护 test 的真实素材采集与 evaluator-only 标签封存入口尚未实现。 |

**剩余工作量**：164 个正式布局仍需实际场地与物理身份分配、四段轨迹和原生净空/几何核验，其中各室外类 41 个；还需五类目标与同类干扰物的 train/dev/test 原始族及最终渲染可见依赖审计。三张候选巷道地图只提供三个场地背景，不自动提供 41 个通过布局，也不能推出尚需恰好 38 张新地图。完整 train/dev 220 布局及 20 个 train 布局的七路成本试采也未由当前 Development 工程样本替代。

预算仅能粗估：26,240 test 时间样本按协议的**待测** 1–2 样本/秒为净采集约 3.64–7.29 小时，按待测 2–3 MB/样本为约 52.48–78.72 GB 十进制原始负载；不含启动、多 pass、几何、传输、验收与工程余量。须用真实 20 个 train 布局的七路 P50/P95 和 1.3 倍余量核对完整 61,440 样本是否落在 250 GiB 与 24 GPU 墙钟小时预算内，不得据此估算自动放行。

实施次序：先完成最终素材/场地分区与材质隔离、真实 384 布局清单和标签几何；打通受管七路真实素材采集与分离的 evaluator-only 标签存放；用 20 个 train 布局完成完整格式、几何和容量试采；满足准入后按冻结 164 槽**一次逻辑采集**进入封存存储。模型与阈值选择期间不得接触 test 标签和结果；预测、模型和阈值封存后才解封评估。质量检查可检查格式和几何失败，但不能以模型结果替换难例、改配额或追加采集。
