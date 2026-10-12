# RGB＋ToF 同步评价源可行性（Development，2026-10-10）

**结论：部分可行。** 已用独立 FARO 激光扫描的官方渲染深度、真实 RGB 和真实轨迹，生成第一个可复算的同相机 query 合成 CNH 试点。两个已消费 old6 capture 中，11 个 RGB 锚帧、297 个固定 query 完成配准、合成与描述性评价；另有真实连续 16 帧约 5 Hz 窗口完成冻结 M3/S 推理。直接用 native LiDAR 同时合成和打标签的逐帧循环已避开，但官方 FARO 深度帧曾按与 native LiDAR 的误差筛选，选择相关性仍在。当前数据不能称无偏同步融合基准，也不能给实际 ToF 能力或融合方案排序。

用户基线为 ToF S 集成 `9e150dd8`、RGB DAV2 raw ≥0.8m / UniDepth 近带候选 `ab084ba7`；本轮代码起点 `2cb47cfd`。仅 EXPLORE，不训练、不更改冻结权重或工作点、不访问保护 480/test。原候选和历史失败不改判。

## 固定协议与产物

[合成前 PLAN](SYNC_RGB_TOF_FEASIBILITY_PLAN_DEV_20261010.md) SHA256 `978ed6f84f34b414f884229ff66b0250e39ad24d0f91d750a2918d13cf2bea30`；实施前补充 `implementation_seal.json` 明确实际 bin 宽、配准与阈值。唯一 payload 根为 `artifacts.local/work/sync-rgb-tof-feasibility-dev-20261010/`，物理位于 `F:/ba-data/blindassist-artifacts-20260805/work/sync-rgb-tof-feasibility-dev-20261010/`。以下路径均相对此根：

| 内容 | 文件 |
|---|---|
| 官方来源、原文许可证、大小/哈希/URL | `sources/sources.md`, `sources/LICENSE`, `sources/download_receipts.json`, `sources/head_receipts.json` |
| FARO 深度、内参、轨迹 | `sources/41069021/`, `sources/41069048/` |
| 合成 CNH、期望回波、噪声、覆盖、深度及配准证据 | `pilot/*_cnh.npz`，共11份 |
| 32锚帧配对/缺测与时间差 | `pilot/pairing.csv` |
| 全部逐query结果与逐zone核验 | `pilot/query_table.csv`（2970行/10描述臂）, `pilot/zones.csv`（704行） |
| 汇总、误差分布、近带缺测、样例图 | `pilot/results.json`, `pilot/summary.csv`, `pilot/sanity.json`, `pilot/diagnostic.png` |
| 真实连续时序与冻结描述 | `frozen_window.npz`, `frozen_window.receipt.json`, `frozen_readout/receipt.json`, `frozen_readout/frozen_description.npz` |
| 独立核验 | `independent_audit.json` |
| 资源与完整性 | 各分支 `*_budget_ledger.json` / `pilot_accounting.json`，整合 `delivery_accounting.json`, `delivery_manifest.json` |

实现为 [pilot](sync_rgb_tof_pilot_dev.py)、[冻结分支](sync_rgb_tof_frozen_readout.py)、[独立审计](audit_sync_rgb_tof_feasibility_dev.py)。图由留存 `plot_pilot.py` 生成。数据不随 Git 发布，报告、代码和协议直推 master。

## 独立几何与许可证

官方 [RAW 文档](https://github.com/apple/ARKitScenes/blob/main/raw/README.md)说明，`highres_depth` 是 FARO mesh 渲染的 1920×1440 深度；这是与 iPad native LiDAR 不同的采集源。其 filtered 10 Hz 发布集合按高、低分辨率深度不一致程度过滤，所以只能说**避免直接重复使用同一份深度值**，不能说已完全消除乐观偏差。采用官方已配准的可见 2.5D 表面；本轮未获得完整独立三角网格，也未使用 native LiDAR 填洞。

| old6 capture | visit | 原始 FARO PLY | FARO highres depth |
|---|---|---|---|
|47333462 / arkit16|467138|可得，三扫描共5,580,381,164 B|不在该子集|
|40777060|369709|未发布|不在该子集|
|40777065|369708|未发布|不在该子集|
|41069021|381658|未发布|83,941,165 B，231帧，本轮下载|
|41069042|381649|未发布|162,307,909 B，只查大小|
|41069048|381644|未发布|57,894,892 B，152帧，本轮下载|

47333462 的单个扫描为1.84–1.89 GB，单文件可装入预算，但完整scene超3 GB，且 `_pose.txt` 仅给 FARO 各扫描之间的对齐，本轮未验证 FARO→ARKit 变换，故不盲下大文件。190170的4096 B Range头确认42,893,185个点，字段为XYZ、RGBA、quality、radius，**无intensity**；未把quality或RGB亮度当反射率。强度代理臂 `NOT_AVAILABLE`。该场景 ARKit mesh约17.61 MB，但来自同iPad LiDAR重建，不作为独立几何证明。

官方 [数据格式](https://github.com/apple/ARKitScenes/blob/main/DATA.md)与保存的官方 `tenFpsDataLoader.py` 表明，PNG为毫米 optical-Z；轨迹axis-angle/t组成world→camera矩阵，取逆直接作用于 `K^-1[uZ,vZ,Z]`，使用camera right/down/forward，不额外加OpenGL翻转。原始许可证全文及SHA已留存，依 [Apple许可证](https://github.com/apple/ARKitScenes/blob/main/LICENSE) 的非商业条款用于本地研究；原文另有商业条款，本轮不依赖它，也未发布原始数据。

## 配准、视场与时钟

ToF与RGB共置同光轴，名义相对外参为单位阵；45°正方视场、8×8 zone，每zone16×16 sub-ray，16个径向bin。冻结模型实际raw bin为0.0375348m，聚合宽为 **0.3002784m**，峰中心 `(b+0.5)×0.3002784m`，range_zero=0；PLAN中的0.30m是名义值，没有改传感器bin。

独立核验old6全部96帧native lowres K四边均覆盖±22.5°，最小边界半角23.3308°；11个试点FARO highres K亦通过。两capture全部highres K最小边界半角23.6746°；VGA K分别5656/1006帧，最小半角23.4802°/23.6120°，覆盖45°，见 `sources/vga_fov.json`。覆盖结论针对已核验内参，不外推所有ARKit设备。

固定最近FARO帧与RGB锚帧 `|Δt|≤0.1s`，41069021为3/16、41069048为8/16，合计11/32；其余21个保留未匹配，不按模型或标签补选。仅1/11精确同timestamp。FARO源深度经源K反投影→真实world pose→RGB锚帧pose→最近深度z-buffer，产生256×192可见表面；孔洞保持UNKNOWN。所用最近轨迹记录相对源/目标帧最大残差分别49.559/33.767ms。因此这是**静态表面配准试点**，不能把重投影说成真实动态场景完全同步。

`pairing.csv` 同时记录以capture首帧为原点的0.2s网格及RGB到网格差值。为复用已缓存真实RGB预测，公共query实际在稀疏RGB锚帧时刻合成，**没有完成连续5Hz共同RGB＋ToF整链评价**。另一独立窗口按首个时间合格规则选41069021的309.742–312.741s，16帧真实间隔0.199–0.200s，验证冻结时序链可运行；该窗口未新跑RGB。这是“部分可行”的明确未完成项，未将稀疏16帧拼成假连续序列。

## 合成与健全性

固定rho=0.3；用重投影depth的逐像素前平行表面近似求sub-ray交点，Z转radial，入射余弦取该表面的ray-z。复用冻结 `quiet_electronics` 与四象限/八raw-bin聚合，再用 `sample` 的 `Poisson(E+8A)-Poisson(8A)` signed Skellam。nominal signal_counts=606.1126、ambient=4、pulse/tail、串扰和邻区泄漏沿用原实现。不是完整mesh法线、材质、多径或实物校准。

逐zone平均有效sub-ray覆盖 **95.995%**；固定coverage≥0.75且peak SNR≥3，接受 **618/704** 个zone。SNR由观测峰及已抽样背景估计，不读native标签/期望信号选峰。native近带可见像素96,308个，重投影几何缺测4,546个（**4.72%**）；native自身在本11帧ToF视域未缺测。近带统计用native optical-Z定义，以下距离带用native **radial** 中位值定义，二者不可混用。

| native radial带 | 全部zone / 接受zone | 全部峰绝对误差中位/P90 (m) | 接受峰绝对误差中位/P90 (m) |
|---|---:|---:|---:|
|0.3–0.8m|175 / 164|0.0503 / 0.1380|0.0528 / 0.1380|
|0.8–1.5m|410 / 378|0.0772 / 0.1550|0.0739 / 0.1532|
|1.5–3m|119 / 76|0.0933 / 0.2779|0.0823 / 0.1502|

全部与接受峰均保留，未删除噪声错峰来美化误差。误差是zone峰radial与同zone native深度转换后中位数之差，含量化、角内混合、几何/轨迹误差；不是硬件精度。nominal 2m前平行面健全性对照与pilot的ambient均4 counts/raw-bin，峰SNR中位分别3.128与7.272，处于同光子参数量级；pilot大量近表面导致更大回波，不是实测光子匹配证明。

## 27-query描述性试点

沿用原native标签和冻结public mask，16 pixel为支持门限。ToF峰距离在zone角域展开为射线3D点，再判query内几何margin≥0；16像素不是16个独立ToF量测。正见证W要求支持像素中至少16个落入原native POS像素域；F是strict sampledFREE query上的误支持，U是UNKNOWN query上的支持，非FREE空间认证。

DAV原cut=0.24403834342956543m，只承担≥0.8m；Uni近带原pooled304 cut=0.09616100788116455m，未采用本轮或ARKcal再调阈值。固定朴素融合的另一输入为“近带Uni / 中远DAV”候选；OR、AND均逐像素后再判16pixel，不对两个不相交的query见证做逻辑AND。它只是预先定义的描述臂，不升级RGB混合候选。两种传感器读出阈值不具备同成本含义，不能据下表声称ToF胜过RGB。

每带99个query。单元格顺序为 **W/POS；F/FREE；U/UNKNOWN**；0/0表示无分母，不能解释为零误报率。

| 方案 | 0.3–0.8m | 0.8–1.5m | 1.5–3m |
|---|---|---|---|
|ToF透明峰读出|15/41；0/51；0/7|63/79；0/6；0/14|20/27；0/0；7/72|
|DAV2-only|不启用|20/79；0/6；11/14|11/27；0/0；35/72|
|UniDepth近带-only|11/41；0/51；3/7|不启用|不启用|
|ToF OR 分带RGB候选|22/41；0/51；3/7|67/79；0/6；11/14|20/27；0/0；38/72|
|ToF AND 分带RGB候选|3/41；0/51；0/7|11/79；0/6；0/14|10/27；0/0；3/72|

未匹配21锚帧（567query）不在这些分母中；此损失与每带UNKNOWN均保留。中带只有6个FREE、远带没有FREE，不能验证融合误支持成本；OR的较高W不能据此选方法。2970行表还列独立ToF OR/AND DAV、ToF OR/AND Uni及POS支持像素，不把不适用带的禁用臂解释为模型失败。

冻结M3/S仅作原两走廊描述：16输入帧产生13输出帧，M3 HEAD/BODY阳性 **13/13、0/13**；原5格强档 **13/13、13/13**；S新增轻档 **0/13、0/13**。保留47维、三seed均值、S tau=1.9179517030715945及权重/依赖哈希。无两走廊接触真值，不报告准确率；手持扫描与步行共向分布不同，不能把全强触发解释成可用提醒策略。

## 正式数据源建设草案（未执行）

建议优先建设 **FARO静态表面＋真实RGB的query同步Development源**，把原始FARO无一致性筛选版本作为后续消偏目标。现有41069021/41069048留作回归，41069042可补开发覆盖；所有old6及其visit永久记为已消费，不进入新eval。完整原始FARO scene阶段另核对变换和预算，不在本次3 GB内继续扩张。

以下6个具体候选均有upsampling资产，已排除已查24个消费capture/visit及六个band-hybrid来源；**尚未完成全局消费登记核对，不能称fresh**，未下载：

|拟用途|capture|visit|
|---|---|---|
|train/开发|41048190|381531|
|train/开发|41048223|381654|
|cal|42444946|421337|
|cal|42444966|421383|
|eval|42445021|421380|
|eval|42445028|421378|

初版规模建议每visit两段16帧真实5Hz窗口，6visit共192帧、5,184 query；窗口按时间完整性预选、不按模型/标签选，缺测按原时钟保留。官方Training/Validation仅是来源分组，本研究用途按visit/物理场景隔离；同visit全部capture、重建、相邻窗口、RGB派生图、CNH噪声副本和反射率臂只能属于一侧。全局admission检查任一候选已消费即移入开发池，按封存官方元数据顺序替补，不看eval分数替换。

预算草案：先HEAD封存各asset总字节，以额外下载≤1.5 GB、CPU≤1800 command-wall s、GPU≤300s建立192帧版；这是尚待授权执行的估算上限，不是已经测出的总成本。超额时减窗口/scene并保留原分区，不借eval调参。需在同一0.2s时间轴取真实RGB最近帧，记录RGB/ToF/trajectory三种时间残差；优先精确同stamp，其他帧经明确静态补偿且另列。该阶段只建设数据，不自动训练融合器。

评价单位仍是(query,RGB timestamp,capture)，同时报告visit级覆盖、有效时长、POS/FREE/UNKNOWN分母、W/F/U与跨帧相关性；置信区间按visit聚类，不能把5,184query当独立接触事件。只有6visit、2eval不足以给稳定泛化结论，应把它定位为首版建设验收。train用来开发读出，cal用来封存规则，eval只运行冻结版；保持重建来源哈希、采集设备、时间/坐标、未知掩码、许可和消费账本。正式确认若要求消除筛选相关性，必须新增未按native误差筛选的独立几何，不用本试点改名充当无偏eval。

## 核验、资源与终态

独立代理从FARO源和轨迹复算11帧重投影与覆盖、704zone峰距、2970逐query分支的支持/见证计数及冻结通知计数，结果PASS，详见审计收据。核验确认实现与描述一致，不消除上述源选择、2.5D几何、rho、时钟与手持轨迹限制。

最终网络下载 **143,756,039 B / 3,000,000,000 B**；无模型下载、无训练。总command-wall按 **450/1800s保守计账**（含失败、交付预留以及GPU命令wall再次计入），GPU按 **45/600s** 计账（科学阶段33.047s），详见最终 `delivery_accounting.json`，包含失败与保守上界，未把工具轮询时长当精确计算时间。GPU任务已释放；原始资产、样例、日志和收据为本run持久证据保留，不遗留任务服务。

**终态：PARTIALLY_FEASIBLE / DESCRIPTIVE_STATIC_REPROJECTED_PILOT。** 推荐推进数据源建设；不推荐从本轮选择OR/AND、改变冻结工作点或宣称同步融合实际收益。完整真实传感器RGB＋ToF、连续5Hz公共query基准与接触事件评价仍未建立。
