# 全量数据普查与充分训练基线

2026-10-15，进行中，Development。用户已授权按完整研究方案开展全量数据普查、强基线训练与剩余错误分析，暂不定题。

本轮先回答：原生标注是否支持同实例功能区域配对；全量监督训练后仍存哪些定位/标签混淆；哪些错误有足够样本支持进一步研究。定位命中不是物理操作成功，grasp/cut共现不是完整任务角色真值。

## 执行范围

- 全部IIT官方训练/验证图片及原生mask普查，核查train_and_val/val/test集合；按原生对象框统计配对与关联歧义，保留来源/序列信息。
- PACO-LVIS完整annotation与InstructPart完整metadata统计，不把部件数量当独立图片数，不将缺标自动变负例。
- 优先完整训练SegFormer-B0及第二个预训练视觉密集模型；先使用官方train/val可复现划分并检查实例/近重复泄漏，最终以实际manifest为准。全图原生功能类别预测作为强诊断基线，不能称自然语言或指定实例任务已解决。
- 官方AGD/PLSP全量训练资源并行准备，若下载或兼容阻塞，保留具体证据，不用小样本替代完整复现。

## 资源、选择与完成条件

数据、模型和运行证据位于 `artifacts.local/work/functional-parts-full-20261015/`；复用已有模型、Python和原生索引。使用现有本地GPU/已授权secondary worker，未获付费资源额度，不新租算力。外部GPU只读连接检查不构成拥有可用计算资源。

两种密集基线实际队列固定为80epoch学习率计划，40epoch为中期检查点；结束后结合完整曲线判断稳定性，不能仅因跑满轮数称收敛。40k updates为前期研究预算情景，不是本轮强制epoch换算。采用全4,417训练行；原1,767验证行排除10个train完全相同SHA后，主模型选择使用1,757行。训练4,400个唯一RGB、原val1,764个，行数与独立底图并列。只用val选checkpoint，不用test选择模型/阈值；旧消费样本保留Development身份。

长期作业须有独占输出、进度、断点和终态，恢复前核配置/输入与无第二写者。本轮执行到数据普查、完整基线及可解释的剩余错误报告；若需要跨交互窗口继续训练，应保留可恢复状态并明确未完成项目，不能把启动成功写成充分训练完成。

## 状态

2026-10-15 19:24起，按用户“主机卡顿，优化或迁副机”的最新要求转移执行位置。本机可用物理内存一度仅约185MiB；已通过既有STOP_REQUESTED主动停止main/post两队列，约19:24确认训练及loader退出，可用内存回升至约7.6GiB。旧FAILED终态是operator stop，保留原日志、锁和断点，不计为算法失败，不自动重启本机GPU。最后完整断点为epoch17，第18轮未完成部分将重放。

迁移权威为payload的`migration/STATE.json`。注册副机RTX3060 Laptop 6GB，检查时约9GiB空闲RAM、5.1GiB空闲显存且没有Python作业。传输原始数据、冻结源码、模型及完整optimizer/scaler/RNG断点；原manifest与trainer raw SHA保持不变，独立`full_worker_entry.py`仅重定位主副机路径。副机Torch2.9.1/cu128与主机2.11.0/cu130不同，须实际断点加载/GPU更新验收，不声称逐元素复现。训练轮数、种子、有效批量、模型选择及数据划分保持原计划；恢复前核无第二写者。主机只保留轻量下载与状态同步。迁移成功前，下述原18点启动PID仅作历史，不代表当前仍运行。

19:41副机main/post持久runner已接手，19:42确认真实第18轮推进至4,890 updates（原断点4,709），不是只凭启动回执。恢复先导实际完成2次GPU optimizer更新，峰值CUDA allocation 3,597,675,008B；只是兼容与容量验收。主queue PID28148、trainer实际PID26572，post PID27384只读等待；runner ID见`migration/STATE.json`。副机payload在`G:/DevWorkspace/BlindAssist/artifacts/work/functional-parts-full-20261015`，保留原始主机终止记录与完整log，工作曲线只移除已归档的未完成epoch18片段。耗时按主机/副机分段累计，不把恢复后重置的wall_seconds当全程成本。

一次性同步入口为[full_worker_sync.py](full_worker_sync.py)，按用户最新要求每1小时由既有automation执行；没有新增常驻监控进程。它只读采集副机状态，在本地UMD supervisor COMPLETE后才镜像全量作者手标文件，逐文件SHA校验后最后发布supervisor成功证据；不解码GT、不启动GPU。副机主/后队列及其owner是当前执行权威，原local FAILED不得触发本机重启。主副机路径适配入口[full_worker_entry.py](full_worker_entry.py)与mapping SHA均已封入remote request；Windows asyncio继承Popen兼容故障及修复验证保留在migration/canary_attempts。第二批PLSP完整训练/测试资产正在迁入，验收以副机`migration/plsp_asset_receipt.json`为准。

19:55迁移验收完成：副机已逐文件核验第二批PLSP的38,330份payload，source bytes不变、未解码test像素；单线程BelowNormal接收任务终态success并结束。收据已取回本地`migration/worker_acceptance/`。主训练实际进入epoch24，副机可用RAM约8.6GiB；这些是迁移/运行证据，不是完整模型性能结论。后续按既有主/后队列继续，不能恢复本机GPU，也不以新的运行时计时覆盖主机已消耗成本。

20:04 UMD数据链终态COMPLETE且副机镜像已通过：29,711原生帧ID中有9,939个作者手标配对，来自105工具组/3杂乱场景组，逐文件SHA确认9,939个不同RGB、与IIT train/cleanval完全同RGB交叉为0。镜像19,898文件验证后才最后发布supervisor证据，未解码GT像素，尚无跨源模型成绩。完整计数及相关性边界见[全量普查](FULL_DATA_CENSUS.md)。本机下载/supervisor/一次性同步进程已结束；保留归档、失败/请求记录和镜像包用于复核。副机main仍训练、post仍等待，不启动额外GPU任务。

22:05只读快照确认SegFormer-B0终态COMPLETE：80epoch、22,160次更新；训练、全cleanval导出、独立评分、错误图四个队列步骤均SUCCESS，原trainer已退出且owner锁已释放。DINO-S/16于22:03自动接续，实际trainer PID15272（父launcher13172、主queue28148），第1轮已推进210次更新；post仍只读等待main。副机当时可用RAM7,301MiB/16,236MiB、显存使用2,449MiB/6,144MiB；均为瞬时资源快照，不代表持续均值。同步耗时1.58秒，UMD返回ALREADY_MIRRORED且未传输数据。收据见payload `monitor/20261015T140506Z.json`；主队列尚未完成，暂不交付最终性能或稳定性结论，耗时仍按主副机分段累计。

2026-10-15 18:13（UTC+8）：主本地队列已启动。实际进度以`queue/progress.json`、`runs/*/progress.jsonl`及各terminal为准；launcher PID49392对应实际queue Python PID45932，首个训练PID52016。没有把launcher回执当训练完成；已确认首个模型在全部train4417清单上持续更新。

顺序为SegFormer-B0训练80epoch→全val导出/独立评分/错误图→DINO-S/16同样流程→PLSP Seen完整40epoch→曲线与汇总。两密集模型448输入、物理/有效batch16、2个loader worker、同CE+Dice；SegFormer全部3,716,714参数训练，DINO后4块/norm/decoder共7,690,506参数训练。DINO权重为原DINO-S/16，明确不是DINOv2。

已做真实数据资源测量：SegFormer batch16峰值3,598,445,056B，workers0/1/2短窗口稳定更新中位约0.614/0.476/0.171秒；短窗口有预取和初始化，不能据此保证全程ETA。DINO batch16峰值1,744,150,016B；PLSP原微批10有效20峰值5,351,804,416B，FP32。均为CUDA训练实际分配峰值，不是系统总显存或纯kernel时间。先导只验证运行/资源，不提供模型效果结论。

本轮全量数据发现见[普查报告](FULL_DATA_CENSUS.md)：IIT/PACO可支持结构/区域监督，尚不支持把角色推理作为既定贡献；InstructPart当前发布不完整且跨split重复；AGD训练行有大量同底图多动作复用。

UMD原下载supervisor（`data/umd/supervisor_owner.json`）现已COMPLETE并释放；停止文件`data/umd/STOP.flag`仅为历史控制入口，勿据此重复启动。当前主/后queue的STOP_REQUESTED须放在副机对应queue目录；原主机queue已经停止，不能恢复。断点最多丢失当前未完成epoch；恢复先核进程身份/锁/配置，使用last.pt，禁止重复GPU写者。失败记录保留，退出码0但terminal不是COMPLETE也不算训练完成。

已建立本对话自动跟进`automation-2`，按用户最新要求每1小时核终态/必要恢复，正常进度保持安静，训练与完整评估完成或有故障时汇报，本轮完成后停用。不租付费算力，不自行扩至新结构搜索。

18:22已启动独立`post_queue`，实际queue PID53956、等待子进程48852，当前只读等待主队列SUCCESS，不占GPU。随后自动执行PLSP固定epoch40完整Seen test→等待UMD supervisor COMPLETE→两密集基线全量作者手标帧跨源导出/共享6类评分→`final/REPORT.md`。它使用独立冻结request，不能手动重复启动同一GPU推理。停止可创建`post_queue/STOP_REQUESTED`。UMD/source未就绪或失败不会静默变成已评分。

Seen完整测试输入已按CRC/SHA准备：1,710动作—图像行、730精确唯一RGB、1,675 GT，35缺GT保持NOT_EVAL；历史消费传播后fresh仅217行/113来源组，另18行与官方train同SHA，全部单列。该基准不能解释为1,710独立新图。所有test像素首次解码在模型和主训练队列终态成功后，不用test选择模型。UMD仅用有作者代码支持的手标索引；scoop无共享IIT标签置ignore255，不当背景，不按GT屏蔽模型预测。

已实际完成首个模型前2epoch与全部1,757图验证通路，继续完整80epoch；这些中期值仅在原始进度中保留，不作为充分训练效果或研究结论。

尚无充分训练后的新效果或创新结论。完整统计、训练曲线、错误分析和实际成本将在同一记录更新。当前仅保留副机训练与只读等待的后续队列，owner与停止/恢复方式如上；主机没有本任务的GPU作业或常驻下载/同步进程。
