# MZ63 execution：一次固定推理与一次CPU评分

按[MZ63协议](MZ63_GEOMETRY_TRANSFER_20260911.md)完成4096帧×4profiles×8decisions，0fit、0新cutoff、0旧cohort replay。五个原cutoff及冻结权重来自[输入绑定](../../../../artifacts.local/work/mz63-geometry-transfer-20260911/inputs.json)，所有角色都只用于描述性评价。运行前MZ61完整源封定，primary UE已释放；未重复采集或增加推理预算。

| 实际计时范围 | 秒 |
| --- | ---: |
| CUDA runner receipt interval | 79.0874544 |
| PowerShell inference launcher wall | 82.2701327 |
| CPU score receipt interval（写receipt前记录） | 5.4308841 |
| CPU score CLI最终elapsed（含封存，保存于stdout） | 5.5054417 |
| PowerShell score wrapper wall | 5.8632228 |

这些范围不同，不能混作同一字段。推理后端为CUDA frozen inference / CPU compact I/O，设备NVIDIA GeForce RTX5060 Laptop GPU。4096次原始RGB读取共1,704,484,561B；批16，共global256×144 BOX、crop224、full640×360三种训练时视图各4096次编码（12,288 frame-views），四packet profiles及兼容heads共享。ALL_INVALID增加readout passes而不增加encoder。记录的visual_views25.2661834s、fixed_readouts22.3949439s是runner内子区间，不涵盖全部I/O/封存开销。没有永久dense cache。

独立CPU audit PASS：524,288已知标量决策、327,680候选重构、五个原cutoff exact、各union OR exact、四packet变换（ALL_INVALID无残留ranges）及2048source pairs。源query UNKNOWN0，但fullframe UNKNOWN cells仍12,188,903；原生counts/known只在评分阶段读取。没有RGB或raw-native解码，没有checkpoint再推理。赢家坐标从保存argmax查native标签，**没有保存全部dense logits以独立重算argmax**；native赢家不证明网络因果依赖该表面。所有TP/FP交换及局部UNKNOWN保留，详情见[结果](MZ63_GEOMETRY_TRANSFER_RESULTS_20260911.md)。

资源：推理handle-release报告compact handles关闭、dense files0；[推理execution](../../../../artifacts.local/work/mz63-geometry-transfer-20260911/execution-v1.json)记录exit0且无任务Python残留。[评分execution](../../../../artifacts.local/work/mz63-geometry-transfer-20260911/score-execution-v1.json)同样exit0且无任务Python残留；没有停止其他进程。仅进程退出可释放所有CUDA引用，handle关闭不被单独当作进程退出证据。Android延迟/内存未测量，CPU组合时间不等于部署成本免费。

封定绑定：

- [MZ61 source-index](../../../../artifacts.local/work/mz61-geometry-source-20260911/source-index.json) SHA `6f9e4be717b7fe23d66af58a9002a667d7acff304ec0c0af58497dfc502c72c7`。
- [run receipt](../../../../artifacts.local/work/mz63-geometry-transfer-20260911/run-v1/receipt.json) SHA `42023213ba290cc98cf80d8f7ccc0db5146754ee3c01c037d490bc23a1c5b3da`。
- [score receipt](../../../../artifacts.local/work/mz63-geometry-transfer-20260911/score-v1/receipt.json) SHA `1135e68db083f89688fb5d93d3f021cf09b0b63f9fc71501abb23fee5fa55305`。
- [score result](../../../../artifacts.local/work/mz63-geometry-transfer-20260911/score-v1/result.json) SHA `ac3d9a6cbbc6da861d9bf80a2c2972f21bc0765119ea86ab7e55dbfc342edc81`。
- [score stdout](../../../../artifacts.local/work/mz63-geometry-transfer-20260911/score-stdout.log)保留最终elapsed；原score report与所有封定输出未改。

本文件是对实际执行收据的说明；不提升硬件、自然场景、fresh confirmation或安全声明。MZ61源生产与MZ63固定模型迁移是不同证据。
