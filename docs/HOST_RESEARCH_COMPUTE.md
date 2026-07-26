# 电脑端研究计算调度

状态：`current`

适用范围：`E:\linnan\linnan` 的电脑端算法开发、数据处理、训练、离线评测与复算。

本策略不适用于 Android、手机、眼镜、边缘设备或 App 实时流水线。设备端延迟、功耗和 delegate 选择不得反向限制电脑端研究吞吐。

## 本机能力基线

当前电脑为 Lenovo Legion Y7000X IAX11（83VK）：

- Intel Core Ultra 7 251HX：6 P-core + 12 E-core，18 核 18 线程，无超线程，最高 5.1 GHz；
- NVIDIA GeForce RTX 5060 Laptop GPU：8 GB GDDR7，本机 OEM 最大 TGP 115 W；
- 单条 16 GB DDR5-6400；
- 两块 1 TB NVMe；仓库和研究产物位于第二块盘的 `E:` / `F:`；
- 当前 Windows 电源方案为“平衡”，短测时 GPU ceiling power limit 为 90 W。

CPU 和 GPU 均属于高性能移动研究平台。当前主要限制是单条 16 GB 内存：容量容易限制多进程、数据解码和 GPU feeder，并且没有利用第二个 SODIMM 形成双通道。最有价值的硬件升级是补一条匹配的 16 GB 内存，形成 32 GB 双通道；升级后必须重新标定 worker 和数据加载并发。

纸面规格来源：

- [Intel Core Ultra 7 251HX](https://www.intel.com/content/www/us/en/products/sku/245959/intel-core-ultra-7-processor-251hx-30m-cache-up-to-5-10-ghz/specifications.html)
- [Lenovo Legion 5 15IAX11 / Y7000X IAX11 PSREF](https://psref.lenovo.com/syspool/Sys/PDF/Legion/Legion_5_15IAX11/Legion_5_15IAX11_Spec.pdf)
- [NVIDIA RTX 50 Laptop GPU](https://www.nvidia.com/en-me/geforce/laptops/compare/)

本机易漂移的详细硬件、驱动和短测结果只保存在忽略目录：
`artifacts.local/evidence/host_compute_profile_v1/profile.json`。

## CPU 实测结论

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

不得默认使用 18 worker。内存密集任务必须按单 worker RSS 下调；可用内存不足时由启动器自动降低并发。

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

## GPU 实测与调度

当前 90 W ceiling 档位下，MobileNetV3Small + LR-ASPP、mixed FP16、256×256 合成 device tensor 的短测为：

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

## 存储和等待时间

- 代码、小型元数据和工具保留在仓库路径。
- 数据集、解压缓存、中间结果和 benchmark 输出进入 `artifacts.local/`。
- 两块 NVMe 可把系统/工具 I/O 与研究 payload 分开；不要把大数据重新复制到系统盘。
- 对重复解码或训练数据，优先建立可复算缓存、manifest 和 hash，而不是每轮重新下载或解压。
- 并行下载属于 I/O workload，worker 数与 CPU profile 分开标定。

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
