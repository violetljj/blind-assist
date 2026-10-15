# 熟路定位与地图外拒识：小规模筛查结果

**建议保留“何时不该相信路线记忆”作为备选研究机制；不把普通地点检索或熟路提醒本身当创新。** 真实夏/冬双遍历上，因果序列将已知路段正确定位从 18/140 提到 135/140，但其置信门仍会把图库未覆盖的真实路段错认成已知地点，50/78 次。收益与缺口都可复现；是否适合毕业设计仍取决于头戴重复步行验证。

## 实际运行与证据身份

- 源为官方 OpenSeqSLAM2 的公开 Nordland trimmed sample，夏/冬各 3576 张图片；配套 MAT/CSV 只标注每10张取1张后的 **358 对**。按源码 `numbers(1:subsample_factor:end)` 取样，每个选中源文件编号增加100。读取716张图，不是相邻同视频帧交叉切分。
- 原计划600对在源合同检查处失败，run01未提特征/计分；修订样本数与切分后才执行run02。旧 L10 保持暂停，未读取其冻结/保护集；现存 SEVN 元数据不能替代重复遍历图像。
- 夏季前240个地点作为图库；冬季q10–69校准60次，q90–229已知评价140次，q280–357地图未覆盖评价78次（索引从0开始）。无匹配负例来自真实图像中的未建图路段，**不等于真实岔路偏离**。
- 样本都是同一铁路线路的两次遍历；评价窗口是时间相关的，不是140/78个独立场景。地面轨道视角和季节变化不能替代低成本头戴视角、转头、倒走或行人遮挡。
- 图像统一64×32灰度、8×8 patch标准化；同一L1距离分别做单帧、局部对比、因果5/10帧序列搜索。序列速度固定0.8/1/1.2，局部对比窗口41；这是SeqSLAM启发的轻量重实现，未声称完整复现官方代码。
- 正确容差为±5个**取样后的参考索引**，无米级/时间真值；不会换算成米或预警秒数。主10帧已知查询绝对索引误差中位0、P95为0，但尾部5次仍越过容差。

## 结果

每臂门槛只由正确校准匹配的margin第5百分位设定；地图外查询不参与校准。它保留校准正确覆盖，**未承诺控制开放集误接受率**。

| 对照 | 已知路段原始正确 / 140 | 已知路段接受正确 / 错误 / UNKNOWN | 地图外错误接受 / 78 |
| --- | ---: | ---: | ---: |
| 单帧原始L1 | 18 | 18 / 104 / 18 | 59 |
| 单帧局部对比 | 37 | 37 / 99 / 4 | 78 |
| 因果5帧 | 106 | 103 / 27 / 10 | 75 |
| 因果10帧 | 135 | 133 / 2 / 5 | 50 |

10帧已知接受精度133/135=98.52%，地图外错误接受50/78=64.10%。这说明序列排序很有用，也说明“比第二候选明显好”不能证明地图确实包含当前位置。

观测成本：5/10帧分别需要当前加过去4/9个取样步；算法无未来帧，冷启动至少积累4/9步。已知首个q90使用q81–90，未使用校准q10–69作为状态；地图外首个q280使用q271–280，全部已在图库终点239之后，没有借用已知路段暖启动。图片源配置未给出可验证的时间轴，故只报取样步，不报毫秒/秒。脚本0.743秒是整个CPU离线计算时间，不是设备端到端延迟。

## 事后消融

看到run02后提出“相对唯一性＋绝对外观残差”双条件。在10帧既有预测上增加当前帧L1残差≤正确校准匹配残差第95百分位，未改原baseline，也未用地图外样本选点。

| 门控 | 已知接受正确 / 错误 / UNKNOWN（140） | 地图外错误接受（78） |
| --- | ---: | ---: |
| 原margin | 133 / 2 / 5 | 50 |
| 仅绝对残差 | 120 / 3 / 17 | 51 |
| margin＋绝对残差 | 118 / 1 / 21 | 38 |

双门在已消费数据上少12次地图外误接受，代价是丢15次正确接受；仍有38/78=48.72%误接受。**没有得到可用的拒识方案，不能凭该消融宣称创新有效。** 它支持把地图存在性和地点唯一性分开建模，也表明单帧绝对残差不是充分证据。

## 创新重合与下一步建议

[NaviNote（CHI 2026）](https://dl.acm.org/doi/full/10.1145/3772318.3790589)已经把视觉定位、语音空间注释、末端导航与安全注释触发结合，并有24人形成性研究和18名BLV参与者评价；“记住危险点，走近提醒”不足以构成新贡献。[OpenSeqSLAM2](https://github.com/qcr/openseqslam2)已经系统研究序列检索、速度变化和匹配门槛，所以这里的10帧收益也不是新算法。

后续必须与官方可获取的强视觉地点识别表征比较：[CosPlace](https://github.com/gmberton/CosPlace)、[DINOv2 SALAD](https://github.com/serizba/salad)、[MegaLoc](https://github.com/gmberton/MegaLoc)。本轮只核查其官方机制与用法，**没有运行这些模型，不能声称超过它们**。

最值得继续的机制是 **“承认路线记忆可能不适用，并用动作选择补证”**：同时维护多个地点假设和地图外状态，要求新观察对预测的下一地点提供独立证据；证据弱时建议短时扫视/停步重定位，再决定是否恢复地标提示。排序置信度只用于分辨地图内候选，地图存在性必须独立评价。应把无动作的序列与同等观察预算的主动观察相比，而不是只比单帧。

潜在收益：减少错段锁定和错误地点提示，同时控制中断/额外观察成本。最弱假设：廉价头戴摄像头在跨日/转头后仍有足够可重复的局部证据；当前铁路数据并未证明它。比例合适的下一检验是固定机位安装、两条相似步行路线各重复两次，预先留一条地图外路线，记录wrong-segment lock、正确覆盖、拒识与恢复所需观察数。无需先做公制VIO或安全报警；如果强表征的地图内检索已足够好，贡献应落在开放集失败控制和观察决策，而非再训地点编码器。

## 可复现输出与核验

- [基线机器结果](../../../../artifacts.local/work/assistive-thesis-screen-20261015/route_memory/run02/result.json)、[事后消融](../../../../artifacts.local/work/assistive-thesis-screen-20261015/route_memory/run03/result.json)、[核验](../../../../artifacts.local/work/assistive-thesis-screen-20261015/route_memory/verification.json)。
- Payload：`artifacts.local/work/assistive-thesis-screen-20261015/route_memory/`。其中`run02/sample_manifest.jsonl`有716张图的源member/索引/hash；`run02/predictions.csv`有1112条逐query/逐臂记录；`run03/predictions.csv`保留消融；`run01/data_audit.json`保留源合同失败。
- 核验复算4臂计数、4个校准门槛及消融门槛/计数，验证5/10帧前缀不变性、跨季图像hash不重合、MAT358×358对角真值和源取样步。抽查4对图片的夏/冬视角；包含真实隧道黑暗帧，未事后删掉。
- 下载archive **69,220,182 bytes**，SHA256 `f7449bcfd3d814608ead1eb8bf57f8f416496230d39ea6591e37fb9345efeebe`，官方源revision `02c6ed1e623b94c450cdce86271c9cfe5e00f2a5`。少量官方文本/metadata额外不到1MB；低于300MB预算。下载写盘跨度115.209秒；所有CPU命令含等待保守记账300秒/600秒，GPU0。基线/消融脚本分别0.743/0.072秒，未运行模型训练。
- 无任务后台进程或端口。样本包、hash与诊断保留为本次Development证据。

在仓库根目录执行（需NumPy/Pillow/SciPy）：

```powershell
# $py points to an existing Python with NumPy, Pillow and SciPy.
$payload = 'artifacts.local/work/assistive-thesis-screen-20261015/route_memory'
& $py research/active/assistive-thesis-screen/route_memory/pilot.py --archive "$payload/samples.zip" --out "$payload/reproduction"
& $py research/active/assistive-thesis-screen/route_memory/absolute_residual_gate.py --baseline "$payload/run02" --out "$payload/ablation_reproduction"
& $py research/active/assistive-thesis-screen/route_memory/verify.py --payload $payload
```
