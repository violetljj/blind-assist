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

2026-10-16 01:02，DINO-S/16亦已终态COMPLETE：80epoch、22,160次更新，训练进程壁钟10,546.12秒；全cleanval导出SUCCESS，独立评分尚在运行，不能据训练终态声称完整比较已完成。原trainer15272已退出且owner锁释放，主queue28148当前接续CPU评分，post27384仍等待main。两模型各有40次全val记录，第62–80轮最近10次验证的mIoU范围分别为SegFormer0.523个百分点、DINO0.155个百分点；最近首末变化分别+0.343/+0.135个百分点。只说明该划分下后期曲线波动较小，不等于泛化或全局收敛证明。最佳val checkpoint分别来自epoch74/32，仍按原先val规则选取，不改轮数或据后续test选模型。已取回只读曲线/已完成评分收据 `migration/iit_completed_receipt_20261016T170231Z.json`；PLSP待评分及错误图结束后自动接续，没有手动启动第二GPU任务。

01:05交接复核：DINO独立评分和错误图均SUCCESS，两套IIT基线已完成全部原定train/val步骤。原生分辨率1,757条验证记录/1,754个不同RGB上，SegFormer/DINO前景mIoU为66.266%/71.436%，点命中3,352/3,821（87.726%）与3,279/3,821（85.815%），小区域点命中1,371/1,635（83.853%）与1,286/1,635（78.654%）。DINO相对SegFormer的1,000次RGB组配对bootstrap95%区间：mIoU差+3.287至+7.015个百分点，点命中差−3.150至−0.790个百分点。范围只覆盖该验证划分，不是物理实例分组、跨源或创新确认；后续不能用更高分割mIoU代替点定位指标。完整评分收据已取回 `migration/iit_completed_receipt_20261016T1705Z.json`，不手动重复评分。

同次复核确认PLSP于01:03已自动接续，实际trainer26208、launcher21364、主queue28148，原micro10/effective20未变且已完成30次更新；GPU实际分配峰值5,339,471,360B，系统显存瞬时5,903/6,144MiB，可用RAM8,294MiB。没有OOM或修改训练源码；保留单GPU写者。早期30步的剩余时间估计38,411秒（约10.7小时）尚不含完整epoch实测和后续评估，不据此保证完工时间；24小时训练上限不变。UMD镜像仍ALREADY_MIRRORED、无重传，post继续等待main，最终研究报告仍待PLSP与跨源评估。

02:11用户要求的[全量点定位差异诊断](DINO_POINT_GAP_ANALYSIS.md)已COMPLETE：全部1,757条验证记录、3,821正类查询的原坐标/命中/混淆矩阵严格复现。两模型使用相同稳定log-odds读点后，DINO净差由−73缩至−21；DINO最大预测连通域内部点3,384/3,821，小区域1,368/1,635。数值/解码修正不是模型创新；同稳定解码差及DINO内部点对Seg原始差的RGB组区间均跨零。ImageNet小区域的固定三点oracle仅209/392，优先保留局部细节与新候选生成联合定位质量的待证机制，未启动结构训练。独立CPU单线程BelowNormal诊断壁钟1,013.49秒、峰值RSS726.3MiB，runner终态success、进程已释放，thin evidence逐SHA取回；没有第二GPU写者。

02:20轻量同步确认PLSP完成5/40轮并进入第6轮、1,740次更新；近期step2.71秒、剩余训练估计32,674秒（约9.1小时），不含后续评估。副机瞬时GPU100%、显存5,882/6,144MiB、可用RAM9,737MiB。main/post仍RUNNING，UMD仍ALREADY_MIRRORED且无重传，主机GPU禁用政策不变。本次IIT诊断可独立交付，不能据此称整轮最终报告完成。

11:42核查确认main队列SUCCESS，PLSP终态COMPLETE：40epoch/13,880次更新、进程壁钟35,042.45秒（9.734小时）、CPU133,679.20秒、峰值CUDA allocated5,339,471,360B/reserved5,519,704,064B；原micro10/effective20保持，全Seen测试1,710行亦COMPLETE，1,674可评/36NOT_EVAL。原post在SegFormer的9,939图导出后，评分因原生−1标签失败；这不是训练或模型推理失败。完整编码普查及作者代码确认−1是void，具体像素/分母见[普查](FULL_DATA_CENSUS.md)。

11:49已保留原post terminal/request/日志和失败evaluation至副机`migration/recovery_void_a1/`，在确认原PID/子树/锁释放后恢复同一post_queue。新runner `functional-parts-full-20261015-post-void-a1`，新封存request `migration/worker_post_void_request_a1.json` SHA256 `9ebbbb92d26b5c12dab70464c46bf229b30d19066fc4eb395d9e015814665935`；queue实际PID17336/launcher27536，Seg评分实际26416/launcher14084。已完成步骤按原命令和成功证据跳过；Seg预测manifest SHA未变。只有评分器增加−1→ignore255与分账，部署源SHA `78e62d2855d670d0682080a450fc4ca9b43e015b6d99fb054d1e903a06607d37`，定向selftest和真实clutter/tool两图通路通过；两图仅作故障验收。DINO后续导出仍使用原冻结源，所有原生分割/点规则和bootstrap种子/次数不变，不混入此前诊断的稳定解码。当前顺序为Seg评分→DINO导出/评分→最终汇总，整轮仍待post SUCCESS。

2026-10-15 18:13（UTC+8）：主本地队列已启动。实际进度以`queue/progress.json`、`runs/*/progress.jsonl`及各terminal为准；launcher PID49392对应实际queue Python PID45932，首个训练PID52016。没有把launcher回执当训练完成；已确认首个模型在全部train4417清单上持续更新。

顺序为SegFormer-B0训练80epoch→全val导出/独立评分/错误图→DINO-S/16同样流程→PLSP Seen完整40epoch→曲线与汇总。两密集模型448输入、物理/有效batch16、2个loader worker、同CE+Dice；SegFormer全部3,716,714参数训练，DINO后4块/norm/decoder共7,690,506参数训练。DINO权重为原DINO-S/16，明确不是DINOv2。

已做真实数据资源测量：SegFormer batch16峰值3,598,445,056B，workers0/1/2短窗口稳定更新中位约0.614/0.476/0.171秒；短窗口有预取和初始化，不能据此保证全程ETA。DINO batch16峰值1,744,150,016B；PLSP原微批10有效20峰值5,351,804,416B，FP32。均为CUDA训练实际分配峰值，不是系统总显存或纯kernel时间。先导只验证运行/资源，不提供模型效果结论。

本轮全量数据发现见[普查报告](FULL_DATA_CENSUS.md)：IIT/PACO可支持结构/区域监督，尚不支持把角色推理作为既定贡献；InstructPart当前发布不完整且跨split重复；AGD训练行有大量同底图多动作复用。

UMD原下载supervisor（`data/umd/supervisor_owner.json`）现已COMPLETE并释放；停止文件`data/umd/STOP.flag`仅为历史控制入口，勿据此重复启动。当前主/后queue的STOP_REQUESTED须放在副机对应queue目录；原主机queue已经停止，不能恢复。断点最多丢失当前未完成epoch；恢复先核进程身份/锁/配置，使用last.pt，禁止重复GPU写者。失败记录保留，退出码0但terminal不是COMPLETE也不算训练完成。

已建立本对话自动跟进`automation-2`，按用户最新要求每1小时核终态/必要恢复，正常进度保持安静，训练与完整评估完成或有故障时汇报，本轮完成后停用。不租付费算力，不自行扩至新结构搜索。

18:22已启动独立`post_queue`，实际queue PID53956、等待子进程48852，当前只读等待主队列SUCCESS，不占GPU。随后自动执行PLSP固定epoch40完整Seen test→等待UMD supervisor COMPLETE→两密集基线全量作者手标帧跨源导出/共享6类评分→`final/REPORT.md`。它使用独立冻结request，不能手动重复启动同一GPU推理。停止可创建`post_queue/STOP_REQUESTED`。UMD/source未就绪或失败不会静默变成已评分。

Seen完整测试输入已按CRC/SHA准备：1,710动作—图像行、730精确唯一RGB、1,675 GT，35缺GT保持NOT_EVAL；历史消费传播后fresh仅217行/113来源组，另18行与官方train同SHA，全部单列。该基准不能解释为1,710独立新图。所有test像素首次解码在模型和主训练队列终态成功后，不用test选择模型。UMD仅用有作者代码支持的手标索引；scoop无共享IIT标签置ignore255，不当背景，不按GT屏蔽模型预测。

已实际完成首个模型前2epoch与全部1,757图验证通路，继续完整80epoch；这些中期值仅在原始进度中保留，不作为充分训练效果或研究结论。

本轮完整研究及创新结论尚未交付。两套IIT各80epoch及PLSP40epoch训练、PLSP完整Seen测试均已完成，UMD跨源评分/最终汇总仍待修复后队列完成；完整统计、训练曲线、错误分析和实际成本将在同一记录更新。当前仅保留副机修复后的post队列，owner与停止/恢复方式如上；主机没有本任务的GPU作业或常驻下载/同步进程。
