# 电脑端研究计算调度

状态：`current`

适用范围：`E:\linnan\linnan` 的电脑端算法开发、数据处理、训练、离线评测与复算。

本策略不适用于 Android、手机、眼镜、边缘设备或 App 实时流水线。设备端延迟、功耗和 delegate 选择不得反向限制电脑端研究吞吐。

## 本机能力基线

2026-07-29 现场复核的电脑为 Lenovo Legion Y7000X IAX11（83VK），
BIOS `UFCN26WW`（2026-04-09），Windows 11 build 26200：

- Intel Core Ultra 7 251HX：6 P-core + 12 E-core，18 核 18 线程，无超线程，最高 5.1 GHz；
- NVIDIA GeForce RTX 5060 Laptop GPU：8,151 MiB GDDR7、Blackwell、
  compute capability 12.0，驱动 610.62；本机 OEM 最大 TGP 115 W；
- Intel AI Boost NPU：设备与驱动已启动，纸面 INT8 峰值 13 TOPS；
- 单条 Samsung `M435R2GA3PB1-CCPSG` 16 GB DDR5-6400；
- 两块 1 TB NVMe：UMIS `UPJYJ1TBMNV1QWY` 承载 `C:` / `D:`，
  ZHITAI `TiPlus7100` 承载 `E:` / `F:`；
- 当前为 AC 供电、Windows“平衡”方案；CPU Boost 为 Aggressive，异构调度
  由 Windows / Intel Thread Director 自动管理；
- Secure Boot、Hypervisor、VBS 与 HVCI 均开启；为 Android 模拟器和系统安全
  保留，不为了不可复现的小幅跑分关闭。

CPU 和 GPU 均属于高性能移动研究平台。当前主要限制是单条 16 GB 内存：容量容易限制多进程、数据解码和 GPU feeder，并且没有利用第二个 SODIMM 形成双通道。最有价值的硬件升级是补一条匹配的 16 GB 内存，形成 32 GB 双通道；升级后必须重新标定 worker 和数据加载并发。

本轮 `nvidia-smi` 观察到 GPU 动态功耗 ceiling 约 102–110 W，硬件上限
115 W；它会随 Lenovo 热模式、CPU/GPU 共享功耗和温度变化，不能把 115 W 当作
持续固定功耗。当前枚举状态为 NVIDIA 直接承担显示，约 0.9–1.1 GiB 显存被桌面
与虚拟显示占用；GPU 任务的可用显存必须现场查询。

纸面规格来源：

- [Intel Core Ultra 7 251HX](https://www.intel.com/content/www/us/en/products/sku/245959/intel-core-ultra-7-processor-251hx-30m-cache-up-to-5-10-ghz/specifications.html)
- [Lenovo Legion 5 15IAX11 / Y7000X IAX11 PSREF](https://psref.lenovo.com/syspool/Sys/PDF/Legion/Legion_5_15IAX11/Legion_5_15IAX11_Spec.pdf)
- [NVIDIA RTX 50 Laptop GPU](https://www.nvidia.com/en-me/geforce/laptops/compare/)
- [Windows NPU / Windows ML](https://learn.microsoft.com/windows/ai/npu-devices/)

本机易漂移的详细硬件、驱动和短测结果只保存在忽略目录：
`artifacts.local/evidence/host_compute_profile_v2/profile.json`。旧 R1 基线仍在
`artifacts.local/evidence/host_compute_profile_v1/profile.json`，只用于同机趋势
比较。

## CPU 实测结论

2026-07-29 在 AC / Windows 平衡方案下的短测：

| Workload | Result | Boundary |
| --- | ---: | --- |
| WinSAT AES, 1 thread | 1,531.9 MB/s | 约 3.13 秒的短测 |
| WinSAT AES, 18 threads | 25,148.5–25,173.4 MB/s | 相对单线程约 16.4× |
| WinSAT memory copy | 60,850–63,105 MB/s | 短时复制带宽，不是应用端到端吞吐 |

18 线程 AES 未达到理想 18×，符合 6P+12E 混合核、调度和并行开销。不要给
worker 手工绑定固定 P/E 核：当前 Windows 异构调度和 Thread Director 已启用，
项目型 RCLE 实测比纸面核数更能决定并发。

RCLE Phase A R1 固定 trial 短测使用“多进程 trial + 每 worker 单 OpenCV 线程”。第一轮相对单进程：

| Workers | Wall time / 36 trials | Throughput | Speedup |
| ---: | ---: | ---: | ---: |
| 1 | 28.455 s | 1.265 trial/s | 1.00× |
| 4 | 8.102 s | 4.444 trial/s | 3.51× |
| 8 | 5.485 s | 6.563 trial/s | 5.19× |
| 12 | 5.104 s | 7.054 trial/s | 5.58× |

72-trial 高并发复测中，8/12/16 worker 分别为 5.619/6.388/6.560 trial/s；18 worker 降到 5.615 trial/s。16 比 12 只快约 2.7%，但会占满 CPU；18 worker 因混合核竞争、调度和进程开销发生退化。

因此采用三个明确档位：

| Profile | Workers | 用途 |
| --- | ---: | --- |
| `interactive` | 8 | 同时使用 Codex、IDE、浏览器；默认短实验与开发循环 |
| `balanced` | 12 | 用户不密集操作电脑时的正式离线研究；项目默认 |
| `throughput` | 16 | 独占电脑的长矩阵或复算；保留 2 核给系统 |

不得默认使用 18 worker。内存密集任务必须按单 worker RSS 下调；可用内存不足时由启动器自动降低并发。当前 16 GB 主机的普通启动器默认保留
4 GiB 系统内存；正式/长任务应按真实 pilot 使用 4–6 GiB 或更高 reserve，而不是
机械复用旧收据中的 2.5 GiB。

进程池任务必须在 NumPy/OpenCV 导入前把 `OMP_NUM_THREADS`、`OPENBLAS_NUM_THREADS`、`MKL_NUM_THREADS`、`NUMEXPR_NUM_THREADS` 等设为 1，避免外层多进程与内层数值线程相乘。单进程大矩阵任务不自动套用本规则，应单独标定 BLAS 线程数。

## 效率优先的执行闭环

电脑端研究任务不得把“代码能运行”当作“执行方案合格”。预计超过 3 分钟、会
消费正式 claim、或需要大数据/训练的任务，必须依次完成：

1. **分类**：先判断主负载属于 Python 串行、可分 pair/sample 的 CPU、原生数值
   线程、GPU 张量、解码/I/O、内存容量或混合流水线。
2. **短测**：用真实数据格式和相同访问方式跑有上限的代表性 pilot；fixture
   正确性测试不能替代真实性能短测。
3. **标定**：至少比较单 worker 与一个合理并发档，记录 wall time、CPU
   core-equivalent、读写吞吐、RAM/VRAM 和输出 hash/摘要一致性。
4. **选择**：Python 逐项循环优先向量化或进程池；释放 GIL 的解码/I/O 可测试
   有界线程；批量张量优先 CUDA；随机压缩读取先物化缓存。不得为了“用了 GPU”
   而搬运不适合 GPU 的工作。
5. **观测**：正式长任务启动时必须同时启动机器可读进度；发现持续单核、GPU
   饥饿、反复解压或内存换页时，应立即给出瓶颈诊断，不能只重复“仍在运行”。
6. **调整**：尚未进入不可逆协议时，主动停止低效 pilot、修正数据路径或并发后
   重测；claim/lock 已生效时保持证据边界，只监控并把修复放入下一实现版本。

并行不是目的。只有任务单元独立、结果顺序可恢复、内存和 I/O 有余量且短测显示
吞吐提升时才增加 worker。GPU 也不是默认答案：归档解压、Python 控制流和小粒度
几何通常先修数据路径与 CPU 并行；可批量矩阵、模型训练和稠密张量才优先 GPU。

正式执行的性能准入至少要求：

- 能说明为何选择 CPU、GPU 或混合路径；
- 能给出预计 wall-time 区间和超过区间后的诊断动作；
- 无法从短测外推时，结论必须是 `PERFORMANCE_NOT_QUALIFIED`，不得先消费一次性
  claim 再观察；
- 单核实现若不是经实测证明最快，必须显式记录理由；
- 独立 validator 可以复用不可变、hash 验证的输入缓存，不能把重复下载或重复
  解压当作“独立性”。

现有冻结 RCLE runner 的 source manifest 不得为性能方便直接改写。通过稳定启动器显式传入 `--workers`：

```powershell
pwsh -NoProfile -File scripts/run_host_research.ps1 `
  -Profile balanced `
  -Script path/to/process_pool_runner.py
```

启动器只解析资源并设置线程环境，不改变 seed、batch、阈值、数据、科学协议或输出路径。

## 长任务统一准入

`scripts/run_host_research.ps1` 适合可逆开发循环中的 CPU 进程池任务。以下新任务
必须改用 guarded launcher：正式 one-shot 或不可逆 claim、预计超过 15 分钟、
高 I/O/内存/设备风险，或轻量 pilot 无法给出运行上界：

```powershell
pwsh -NoProfile -File scripts/run_guarded_host_research.ps1 `
  -PreflightReceipt artifacts.local/evidence/<task>/preflight.json `
  -Script scripts/<stable-runner>.py `
  -- <runner arguments>
```

若 formal runner 冻结了解释器前置参数，必须通过 `-PythonArguments` 放在 script
之前，而不是混入 runner arguments，例如：

```powershell
pwsh -NoProfile -File scripts/run_guarded_host_research.ps1 `
  -PreflightReceipt artifacts.local/evidence/<task>/preflight.json `
  -Script scripts/<stable-runner>.py `
  -Python <frozen-python.exe> `
  -PythonArguments @("-I", "-B") `
  -RunnerArguments @("produce", "--activation", "<activation.json>")
```

guarded launcher 在创建 runner 进程前调用
`scripts/validate_host_research_preflight.py`。以下任一条件不满足时固定返回
`PERFORMANCE_NOT_QUALIFIED`，runner 不会启动：

- receipt 绑定当前 runner 相对路径与 SHA-256；
- pilot 使用与正式输入相同的数据格式和访问机制；
- 至少有 2 个实际进度样本，且 projected/max wall time 有界；
- 已比较调度方案并确认科学输出等价；
- progress 合同包含 phase、completed/total、throughput、ETA、最后推进时间和状态；
- success/failure/progress 路径均位于 `artifacts.local/`；
- 当前可用 RAM 满足 `reserve + workers × estimated GiB/worker`；
- CUDA/mixed 任务的当前空闲显存满足 receipt 下限；
- formal 任务声明 one-shot、runner-only claim、activation authority 和互异的
  claim/output/failure 路径。

预计 3–15 分钟且可逆的 `CANARY_LITE/DEVELOPMENT_STANDARD` 工作不需要完整 preflight
receipt，只需在启动参数或轻量 run note 中给出 timeout、可观察进度和 scoped output；
短小可逆 Canary 可直接运行。无论时长，只要出现反复解压、交换、明显资源闲置或
进度不透明，就升级为性能诊断；这项豁免不能用于正式 one-shot。

收据的最小结构为：

```json
{
  "schema_version": "blindassist.host_research_preflight.v1",
  "task_id": "DOMAIN-LONG-R0",
  "execution_class": "long",
  "implementation": {
    "script": "scripts/<stable-runner>.py",
    "sha256": "<64 lowercase hex>"
  },
  "workload": {
    "class": "cpu_data_parallel",
    "real_data_mechanics_match": true,
    "input_identity": "manifest-sha256:<64 lowercase hex>"
  },
  "pilot": {
    "representative_units": 20,
    "wall_seconds": 4.0,
    "projected_full_units": 1000,
    "projected_full_wall_seconds": 200.0,
    "maximum_expected_wall_seconds": 300.0,
    "same_access_mechanics": true,
    "output_equivalence": "PASS",
    "progress_samples": 2
  },
  "scheduler": {
    "backend": "cpu_process_pool",
    "workers": 12,
    "reason": "Measured throughput choice",
    "comparison_performed": true,
    "scientific_parameters_unchanged": true,
    "estimated_gib_per_worker": 0.3,
    "reserve_memory_gib": 2.5,
    "requires_ac_power": true,
    "inject_workers": true
  },
  "progress": {
    "path": "artifacts.local/evidence/task/progress.json",
    "fields": [
      "phase",
      "completed_units",
      "total_units",
      "throughput",
      "eta_seconds",
      "last_progress_at",
      "status"
    ],
    "update_interval_seconds": 30,
    "verified_in_pilot": true
  },
  "terminal": {
    "success_path": "artifacts.local/evidence/task/result.json",
    "failure_path": "artifacts.local/evidence/task/failure.json"
  }
}
```

`formal` 在此基础上增加 `formal.one_shot`、
`formal.claim_created_by_runner_only`、`formal.claim_path`、
`formal.output_path`、`formal.failure_receipt_path` 和
`formal.activation_authority`。门禁验证性能准备度，不自行授予科学或正式执行
权限；runner 仍必须验证自己的 activation lock 与协议。

## GPU 实测与调度

2026-07-29 的 PyTorch 2.11.0+cu128、TF32 关闭、6144² 方阵短测：

| Dtype | Measured throughput | Mean time |
| --- | ---: | ---: |
| FP32 | 9.71 TFLOPS | 47.77 ms |
| FP16 | 34.57 TFLOPS | 13.416 ms |

这是纯 device matrix 短测，不含图像解码、PCIe 传输、CPU feeder 或持续温控，
不能替代真实模型吞吐。

旧 R1 在当时约 90 W ceiling 档位下，MobileNetV3Small + LR-ASPP、mixed FP16、256×256 合成 device tensor 的短测为：

| Batch | Throughput | Peak CUDA allocation |
| ---: | ---: | ---: |
| 16 | 160 images/s | 0.71 GiB |
| 32 | 294 images/s | 1.36 GiB |
| 64 | 439 images/s | 2.68 GiB |
| 96 | 509 images/s | 4.03 GiB |

该结果只证明 CUDA 路径和批处理扩展有效，不授权把训练 batch 改为 96，也不代表 RCLE 适合 GPU。科学 batch、AMP、TF32、determinism 和输入分辨率继续由各实验协议决定。

电脑端调度规则：

1. 深度学习训练、大批张量、可批量稠密视觉计算默认优先 CUDA；禁止静默回退 CPU。
2. 8 GB VRAM 任务通常保留至少 1–1.5 GB 余量；同时计入桌面和浏览器已占显存。
3. GPU 训练先用 4–6 个 CPU feeder；只有 GPU 利用率长期不足且 RAM 有余量时才增加到 8，不同时占满 18 个 CPU 核。
4. CPU 和 GPU 同时重载时优先保证 GPU feeder 和数据 I/O，CPU side task 使用 `interactive` 档。
5. 不全局开启 TF32、CuDNN benchmark 或非确定性算法；数值模式属于具体科学协议。
6. 历史上出过问题的 8,580 帧、batch 64、320 px、FP16 detector 长跑组合继续禁止复用。新型长 GPU 任务先做短、可停止 pilot，并记录温度、功耗、显存和输出。

## NPU、iGPU 与专用媒体单元

- Intel AI Boost NPU 当前驱动正常，但 BlindAssist 的电脑端 RCLE 是
  CPU/OpenCV 路径，SANPO 是 PyTorch CUDA 路径；项目尚无 Windows ML、
  OpenVINO 或 ONNX Runtime NPU runner。因此 NPU 当前是
  `AVAILABLE_BUT_NOT_QUALIFIED`，不能自动计入项目吞吐。
- 只有代表性 ONNX/INT8 图能被 NPU execution provider 完整接收，并通过输出
  等价、加载、端到端延迟、功耗和算子 fallback 审计后，才可新增 `npu` backend。
  Android 手机上的 Qualcomm QNN HTP 证据不能迁移成电脑 Intel NPU 证据。
- 251HX 的 Intel Graphics 提供 Quick Sync 和 H.264/H.265/AV1 编解码能力；
  可在视频物化/转码任务中另做 canary，但当前 iGPU 枚举为 phantom、显示由
  NVIDIA 承担，项目不得假设 Quick Sync 已可用。
- Lenovo LA1、Gaming AI 与 Dynamic Tuning 是整机热/功耗管理，不是可供
  Python/Gradle 直接调度的通用 NPU。

## Gradle / Android 构建调度

当前项目固定使用 Gradle 8.10.2 与 Temurin JDK 17；系统默认 JDK 26 不兼容本
仓库。16 GB 内存下 `gradle.properties` 的 `-Xmx2048m` 先保持不变。2026-07-29
未得到可比较的 Gradle 4/6/8 worker 冷/热矩阵，因此全局最优 worker 数为
`NOT_EVALUABLE`，不得把未经测量的值写入 `gradle.properties`。

执行规则：

1. 正常构建显式使用项目 JDK 17 与 canonical Gradle state。
2. CPU/GPU 长任务同时运行时，Gradle 显式加 `--max-workers=4`，避免与 feeder、
   进程池和 16 GB RAM 争用；这是一条共存上限，不是 Gradle 单独最快的声明。
3. 不全局开启 parallel 或 configuration cache；先按同一 compile/test/lint 场景
   比较 4/6/8 worker、首次/复用配置缓存和峰值 RAM。
4. 不用 `clean` 制造“可复现”跑分；分别报告 no-change、warm incremental 和
   必要时的 cold 场景。

## 存储和等待时间

- 代码、小型元数据和工具保留在仓库路径。
- 数据集、解压缓存、中间结果和 benchmark 输出进入 `artifacts.local/`。
- `E:` 与 `F:` 是同一块 ZHITAI TiPlus7100 的两个分区，不是两条物理 I/O
  通道；仓库、`E:\codex-tools` 和 `artifacts.local/` 也都在这块盘上。不得通过
  E/F 跨分区复制或同时解压来假装 I/O 并行。
- 系统盘 UMIS 与项目盘 ZHITAI 才是两块物理 NVMe；除非另有明确授权和空间/
  生命周期策略，不把大型研究 payload 迁回系统盘。
- 对重复解码或训练数据，优先建立可复算缓存、manifest 和 hash，而不是每轮重新下载或解压。
- 并行下载属于 I/O workload，worker 数与 CPU profile 分开标定。

2026-07-29 项目盘短测为：64 KiB 顺序读约 5,186 MB/s；仅 62 MB 临时文件的
顺序写约 5,814 MB/s；16 KiB 随机读约 746.6–760.2 MB/s（约 45.6–46.4k
IOPS）。顺序写受缓存影响，不能外推到数据集持续写入或 SLC 用尽后性能。
两盘健康状态为 `Healthy/OK`，但 SMART 温度、磨损和持续降速因权限/工具缺失为
`NOT_EVALUABLE`。

## 长任务可观测性

预计超过 3 分钟的 host research 不得只在结尾打印结果。新 runner 在正式
claim 前必须完成有上限的小样本预检，并在实现锁中包含不影响科学结果的进度
协议。进度 sidecar 至少应原子发布：

- `phase`、`completed_units`、`total_units`；
- 最近窗口吞吐、基于同阶段吞吐计算的 ETA；
- PID、输入 hash、实现 lock hash、最后推进时间；
- `running`、`possible_stall`、`complete` 或 `failed` 状态。

进度 sidecar 不得包含被协议禁止提前读取的算法 outcome。producer 与 validator
应分别报告阶段。连续三个观测窗口既无 CPU 推进也无 I/O 推进时才标记
`possible_stall`，不能仅凭运行时长判定挂死。

对已经 claim、不能修改或重启的冻结 runner，使用独立外部监控：

```powershell
pwsh -NoProfile -File scripts/monitor_host_research_process.ps1 `
  -ProcessId 22344 `
  -EvidenceDirectory artifacts.local/evidence/<attempt>
```

监控记录位于忽略目录
`artifacts.local/evidence/host_process_monitors/`，不得写入正式结果目录。外部
监控只能根据 claim、临时目录、output/failure receipt 和 OS 计数器报告阶段与
活性；runner 未发布完成单元数时，百分比和 ETA 必须留空，不能用 CPU 时间冒充
算法进度。监控器同时给出 `bottleneck_hint` 与 `action_hint`；它们是资源诊断，
不是算法结果，也不能授权改变已经锁定的正式 attempt。

由 guarded launcher 启动时，监控器还会读取 runner 的 progress sidecar，将
completed/total、throughput、ETA 和 last-progress 合并到外部 OS 遥测中。runner
退出但缺少 progress 或 success/failure 契约时，guarded summary 明确返回合同
违规，不把 exit code 0 单独当作成功。

压缩 tar（尤其 `.tgz`）不得作为逐样本随机访问的数据层。正式 claim 前应把所需
成员顺序解压到 hash/manifest 绑定的本地缓存，或单次顺序扫描并建立不可变缓存。
producer 与独立 validator 可以复用同一份 hash 验证输入；独立验证不等于重复
解压。修正数据路径前，不得通过增加 worker 放大随机解压和 I/O 竞争。

## 重新标定条件

出现下列任一变化时，旧 profile 只作参考：

- 增加或更换内存；
- BIOS、Windows、Intel/NVIDIA 驱动或联想性能模式变化；
- 电源模式、适配器或散热条件变化；
- runner 的图像尺寸、trial 粒度、解码路径或模型图发生实质变化；
- 连续两轮吞吐比本页下降超过 15%。

重新标定仍以输出 hash/科学摘要一致为前提；资源优化不能改变研究结论。
