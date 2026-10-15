# 全量原生标注普查

2026-10-15，Development。与[研究方案](RESEARCH_PLAN.md)对应的实际数据审计；数据和逐项清单保存在 `artifacts.local/work/functional-parts-full-20261015/`。以下是实际读取官方annotation/metadata得到的计数，不是复制论文宣称规模。模型训练状态另见[执行记录](FULL_BASELINE_PROGRESS.md)。

## 当前可用性判断

PACO支持规模化同实例语义部位监督，但没有动作/交互角色真值。InstructPart当前官方发布内容不足以兑现原先拟定的完整1,800/600训练评估；必须修复发布依赖或重新设计语言监督来源。IIT全量原生mask可用于先做充分训练的功能类别基线，但mask属于全图语义标注，框内共现不自动等于同实例双角色。

## PACO-LVIS：完整train/val

官方 `paco_lvis_v1.zip` 已下载134,514,081B，与官方SHA256 `02ac4edb22c251e07853e6231d69aec3fad0a180f03de2f8c880650322debc80` 一致。完整train/val解析，不取RGB或test mask。

| 实际计数 | train | val |
| --- | ---: | ---: |
| 图片 | 45,790 | 2,410 |
| 对象标注 | 217,117 | 10,974 |
| 部件标注 | 395,071 | 20,945 |
| 至少两个不同部件的对象 | 97,176 | 5,224 |
| 明确工具/容器/家电/书写类筛选的图片 | 22,805 | 1,193 |
| 上述筛选中的对象 / 部件 | 82,302 / 180,946 | 4,339 / 9,558 |

416,016个部件parent关联均有效，无跨图或孤立parent。val对象实数10,974与官方说明表10,794不一致，采用实际JSON计数并保留源文件。

| 原生同实例语义部件配对 | train对象数 | val对象数 |
| --- | ---: | ---: |
| knife: blade + handle | 2,125 | 101 |
| scissors: blade + handle | 853 | 44 |
| spoon: bowl + handle | 1,226 | 77 |
| hammer: head + handle | 24 | 1 |

这些是真正的parent-child语义部件关系，足以支撑部件表示/配对区分的研究材料；没有手接触、任务目的或角色字段。类别不均衡很明显，不能用180,946部件总数掩盖hammer等稀少类别。train18,023图包含non-exhaustive类别标记；原生negative、unknown和非穷尽信息需保留，不能把未标注改成不存在。

完整类别筛选表、逐类计数、parent-child清单和字段统计见payload的 `metadata/paco_*`。类别筛选是本研究定义，不是官方功能标签。

来源：[PACO官方数据](https://github.com/facebookresearch/paco/blob/main/docs/PACO_DATASET.md)。

## InstructPart：当前发布完整性与源图重叠

官方HF commit `671f10493cda83335deeaba36a4fe27df4705c70`。递归tree与repository siblings两套API文件集合一致：1,638文件，**没有annotation JSON**。作者Git dataset分支及Drive相应目录也没有找到README宣称的任务JSON。本轮不编造缺失动作/角色信息。

| HF实际目录 | RGB路径 | 唯一RGB SHA256 |
| --- | ---: | ---: |
| all/images | 146 | 84 |
| train1200/images | 200 | 192 |
| train1800/images | 375 | 345 |
| test/images | 250 | 236 |

train1800/masks有550路径；按相同stem与375张RGB匹配仅125对。train1800与test存在**24组相同RGB SHA256**。SHA来自完整LFS元数据，本轮未读取RGB或test mask像素。该事实说明当前发布目录有重叠和不完整，**不据此推断论文原实验必然泄漏**。

当前不能把声明1,800/600当作已经获得的完整可训练/独立评估划分；同图多部件副本也不增加独立图像数。后续若取得更完整版本，保留本次固定commit证据并重新核验，不覆盖本次审计。

证据：`metadata/instructpart_metadata_seal.json`、两套原始API记录、`instructpart_hf_census.json`及24组重叠路径。来源：[官方HF](https://huggingface.co/datasets/zifuwan/InstructPart)、[作者Git](https://github.com/zifuwan/InstructPart/tree/dataset)。

## IIT-AFF：官方划分与相关性

原生split文件实数：train_and_val6,184，val1,767，test2,651；train取train_and_val减val后为4,417，集合互斥。所需RGB、原生mask、对象框三元组6,184/6,184完整；test mask尚不解码。

对象框metadata中train1,636/4,417、val667/1,767图有多个对象。完整原生mask转换、同框类别共现、RGB SHA交叉检查与像素计数以 `data/iit/native_census.json` 等封存文件为准。

全6,184张原生mask转换已完成；全量文件SHA复核通过，8张原生TXT以独立解析器重读后与PNG逐像素一致。train实际4,400个唯一RGB SHA，val1,764个；训练/验证之间有10组完全相同RGB（一一对应10张val）。主模型选择因此使用**4,417训练行／1,757验证行**，保留全部原始记录与10张重复val的排除原因。训练17组、验证3组内部RGB重复也已记录；这些图片对应的mask SHA不同，不能把它们按行数当完全独立图片，具体像素分歧未在本统计中量化。

按非重叠原生框统计，train中knife的cut+grasp共现256框、hammer的pound+grasp377框、pan的contain+grasp224框。框内至少1像素的原生类别共现不等于真实任务双角色；共7,352个训练框有2,988个与其他框重叠，重叠框内语义mask归属尤其不唯一。训练5张、原始验证2张mask全背景，保留原生分母，不解释为现实没有功能区域。

**对象与动作标签强绑定：**只按训练集物体类别众数预测标签集合，就能匹配原始val单物体图994/1,100（90.36%）。这是全量metadata诊断，不是图像模型准确率，也没有定位任何像素。它说明仅靠IIT的固定功能类别，很难证明自由任务角色推理或未见对象—动作组合泛化；普通完整语义分割后按查询取通道，必须作为强基线。详情见payload `metadata/ROLE_FEASIBILITY.md`。

**官方split存在采集相关性：**77个数字/img采集前缀同时出现在train/val，400张val在相同前缀下存在编号差不超过10的train图。这只是命名启发式，不证明同物理实例，但足以限制“未见场景泛化”表述。ImageNet synset未当作scene分组。模型成绩需分别报告相关前缀和其他来源，不能只给一个总体数字。

旧曝光清单精确合并：IIT219唯一ID（193有旧标签清单/审计记录，26仅RGB检查/缓存）；AGD620唯一ID（594有标签记录，26仅RGB/缓存）。保留每ID来源，后续未消费测试选择按此排除，不能只排除正式评分的190张。

来源：[IIT官方数据](https://sites.google.com/site/iitaffdataset/)。

## AGD/PLSP：完整训练资源与独立图片计数

已下载全部Seen训练RGB和官方PLSP训练资产，CRC/长度与内容SHA普查完成，未取test GT进行训练或选模。

| 训练来源 | 动作—图片文件行 | 精确唯一RGB SHA |
| --- | ---: | ---: |
| egocentric | 6,929 | 2,987 |
| exocentric | 20,061 | 14,092 |
| 合计 | 26,990 | 17,078 |

有6,732组相同RGB跨动作复用，1组跨ego/exo；因此未来训练内验证必须按底图分组。官方refined路径对815行存在，剩余6,114行按官方逻辑fallback到initial伪标签，6,929行均有覆盖，不是6,929行全有精炼标签。文本特征、检测框和相似度组覆盖均通过全量核查。

采用官方训练代码固定commit `71a12b2cc941c8eda85176b98c220c2cb3d55f14`，原vendor保持不变，适配Windows路径、Torch读取和累积尾批，预先固定epoch40最终模型，避免官方脚本每轮用test GT挑best。伪标签来自作者发布，不作为新增人工部位真值。实际GPU启动两步已通过，微批10、有效batch20、全FP32、可训练视觉编码器；两步只是运行验收，不是算法效果或收敛证据。

来源：[PLSP官方实现](https://github.com/woyut/WSAG-PLSP)。

## UMD：原生索引与未决条件

已实际获取官方README、gzip前缀和完整split文本。105工具实例、17类；10个fold各17测试工具，工具有跨fold重复，不能称每工具恰好测试一次。novel_split原生值1/2分别76/29个工具，方向需依据作者定义核定。

tools压缩包2,986,470,339B、clutter186,182,704B，是真实gzip而非可尾部索引ZIP。前缀不足以统计全部帧。本轮已安排完整归档获取与索引普查，准确手标帧数待完成后统计；不用自动标帧冒充手标外测。完整进度在 `data/umd/`，此前小前缀证据保留于 `metadata/umd_*`。

后续已取得作者S-HMP源代码证据：`config_compare_feats.m`及`config_clutter.m`均选择末8位帧号`mod3==1`，并核查了filename token调用链。该规则来自原作者代码和README，不是猜测；完整帧数仍待归档完成后统计。独立supervisor已接管下载→完整索引→规则解析→共享标签records，不在下载完成前伪报手标分母。

来源：[UMD官方README](https://users.umiacs.umd.edu/~fermulcm/affordance/part-affordance-dataset/README.txt)。

## 对研究设计的实际影响

1. 优先用完整IIT原生监督建立现代密集模型基线，分析已有类别辨别和小区域错误；不把这直接称为语言任务或指定实例能力。
2. PACO可提供数万图结构监督及数千对象的部位配对，适合后续预适配，但动作/角色映射仍是推断，不能自称原生真值。
3. 角色研究暂时只能从已标区域共现与语义配对中评估可行性；真实任务角色、同实例绑定和未见组合都需明确相应标签与评价分母。
4. InstructPart语言扩展暂停作为必需依赖；当前发布完整性问题不阻碍IIT/PACO研究，也不能用一百多可配对样本替代完整语言基准后宣称结论。
