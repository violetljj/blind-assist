# 巷道杂物与插入物资产隔离复核（采集 test 前）

本复核只覆盖 `Alley_*_R3a` 九张作者地图和已采集的六张 train/dev Development 地图。它不把预分区升格为正式 benchmark split，也没有读取或采集 test 帧。机器可复查回执在 `artifacts.local/evidence/cnh-alley-asset-isolation-audit-20260924-v1/audit.json`，SHA-256 为 `31718e2fcc82d9930cb8b7827f2bcd97170c22b384d9b4c6b6edd831ac4e9294`。输入是 v3r2 的 `generated-plan.json`、九份实际几何回执、`family-receipt.json`、九份依赖闭包，以及六份已冻结 Development 采集 spec；本机还逐文件复核了 9 张地图和 15 个实际使用的源 mesh 的字节哈希。

## 实际摆放和跨角色资产族

R2 实际九图各 8 件，共 72 件杂物；垃圾桶、配电箱、通风设备、管道、托盘 5 个家族在 train/dev/test 每个分区都出现，任意两区交集均为 5，原隔离判定 **FAIL**。R3a 是同九物理场地的后继地图修订，不能算新增九个独立场地。

R3a 九张地图的实际 `clutter` actor 各 8 件，共 72 件；每区 3 图、24 件。实际 actor 键、家族回执、源 mesh 与计划一致。插入物尚未在作者地图中摆放，它们由采集 spec 插入；六份 train/dev spec 与各地图计划的两个素材逐项一致，固定 `centre/boundary/outside/removed` 四段，各 40 帧，合计每图 160 帧。三个 test 场地没有采集 spec 或帧。

| 预分区 | 实际杂物家族（每区 24 件） | 计划插入：目标 / 通道外干扰物 |
| --- | --- | --- |
| train | 垃圾桶、配电箱、通风设备 | 路桩 / 路牌 |
| dev | 管道、托盘、门梯 | 路锥 / 邮箱 |
| test | 雨篷、消防接口、长椅 | 塑料桶 / 自行车架 |

在这些**已声明且已用的**素材中，语义族、源 kit 根、源 mesh 和物理场地身份的三对分区交集均为 0；杂物与插入物作为一个池检查，跨角色不能重复。共享墙、铺装及材质仍是已披露的背景复用。这里不证明旧 Street200 同街区 Development 帧获得独立 test 资格。

## 可见依赖未结案

原 v3r2 每张地图的依赖闭包都额外纳入另外两个分区的 10 个素材源根（包括六种插入物），因为建图时以全批次加载素材作为闭包根；这使地图级闭包无法直接解释为该地图实际渲染内容。新作者代码已把后续闭包根限定为该图实际摆放的网格、该图计划插入物和该图使用的结构材质；原 v3r2 回执原样保留。

复核从源 mesh 出发重建保守包依赖，train/dev、train/test、dev/test 分别仍有 39、55、38 个包交集；其中非 Engine/Script 包分别为 22、36、22。它们包含通用 shader/材料路径，也包含 Megascans 表面和道具纹理路径。UE 包依赖可能覆盖未启用的材质分支，因此这些交集既不能自动判定为实际可见纹理泄漏，也不能忽略后宣布隔离通过。当前结论是 **`DECLARED_AND_ACTUAL_FAMILIES_DISJOINT_VISUAL_DEPENDENCY_NOT_PROVEN`**，正式 test 采集准入为 false。

### 2026-09-25 原生有效材质补充

只读 UE 会话逐一加载上述 15 个源 mesh，读取实际 LOD0 section→material slot、实例父链、有效纹理参数/静态开关，并调用 UE `MaterialEditingLibrary.GetMaterialUsedTextures`。它没有加载九张 test 作者地图，也没有渲染或采集帧。UE 进程和临时目录均已释放；项目文件、九张地图及 15 个源 mesh（共 25 文件）前后 SHA-256 相同。原生回执为 `artifacts.local/evidence/cnh-alley-effective-material-audit-20260925-v2/ue-run/native-materials.json`（SHA-256 `95342655d4c25264d7714029677281b7442d5ebc7c81776b180dc7c7ab7cfe41`）；机器可读交集为同目录上层的 `visible-dependency-intersection.json`（SHA-256 `73fdbefa814d0fe53c862d64b942eb427b9dffd16c387bf6e3e956254ed9824f`）。完整交集含每个路径在两区的资产/角色/slot，不靠路径前缀白名单筛掉纹理。

实际三个分区的 LOD0 源材质接口本身交集为 0，但六个计划插入物——train 路桩/路牌、dev 路锥/邮箱、test 塑料桶/自行车架——都继承同一个 `MS_DefaultMaterial`，且任意两区均有**相同的 3 个 UE 报告使用纹理**：

| 跨区路径末段 | 原生证据 | 判定 |
| --- | --- | --- |
| `Voronoi_Perturbed_2k_RGBA8_nonVT` | 六个插入物的 `GetMaterialUsedTextures` 均包含；不是命名纹理参数 | 共享的编译材质纹理表达式依赖 |
| `T_RooftopPuddle` | 六者有效参数 `Puddle Height Tex MFPD` 均指向它，且均在 used-texture 列表 | 共享的有效参数及使用依赖 |
| `T_tiled_shoeprints_random` | 六者有效参数 `Detail Tex MFPD` 均指向它，且均在 used-texture 列表 | 共享的有效参数及使用依赖 |

六者 `Enable MFPD=True`。另外七个跨区有效纹理参数值**未**出现在 UE 报告的 used-texture 列表；它们保留在回执供核查，不能误报为已使用纹理。各插入物自身的 Albedo、NormalLOD0 和 Roughness 纹理互不重复。`GetMaterialUsedTextures` 使用当前材质质量的编译材质资源（本机 UE 5.8 `MaterialEditingLibrary.cpp` → `MaterialInstance::GetUsedTextures` → 编译资源的纹理表达式），但未指定单一 shader 平台，也未做逐像素贡献试验；因此这里证明共享的 UE 报告使用依赖，不声称这三张纹理在每个最终像素都有非零作用。六份已采 train/dev 采集原始回执还确认 12 次插入物派生保留纹理/外观图、实例覆盖和原始源文件，派生副本未保存；它们不代表 test 已采。按当前“可见依赖隔离”口径，三张共享的 used texture 已足以阻止无条件准入，结论为 **`FAIL_SHARED_EDITOR_USED_TEXTURE_DEPENDENCIES`**，test 采集仍不启动。共享父材质的 shader 代码可以单列为基础设施；本回执没有把父材质下的三张共享纹理自动归入该豁免。

若改用真正不同来源 kit，必须按实际 LOD0 材质重新核对原始/派生 mesh 族、父链、有效纹理和 used-texture 交集，且各区目标与同类干扰物仍成对覆盖。单独替换 test 的两件插入物不够：train 与 dev 已共享上述三张纹理。机械最小方案是保留其中一个 Development 分区的两件插入物，更换另一个分区的两件以及未采 test 的两件；已经采集的那个 Development 分区至少 3 图×160=480 帧需以新身份重采，并重新完成几何/标签/模型或阈值相关冻结检查。若保留 test 两件，则 train 和 dev 的六图共 960 帧都需重采。实际新 kit 是否够用、目标/干扰物类别与尺寸是否可比、预算和独立布局配额能否维持，尚未验证；旧数据作为已消费 Development 保留，不因重采变成新 test。复制、改名、单纯换色、从同一原始纹理派生副本，或只替换父材质路径，都不能伪造独立来源。本轮没有替换素材、改源项目或追加采集。

### `Enable MFPD` 派生材质修复候选（2026-09-25）

六件插入物的源实例均启用可选 `Enable MFPD`。单次 UE 探针沿当前 `derive` 流程复制各源实例及其父材质，在**未保存的派生实例**上把该开关设为 false；未加载地图、渲染帧或采集 test。机器可读回执为 `artifacts.local/evidence/cnh-alley-mfpd-unsaved-probe-20260925-v2/ue-run/mfpd-materials.json`，SHA-256 `ada457975379759a1dc071861226ed170c3478f365bc10f4a77fba419f0a6f73`。首次 v1 启动时 Python 路径未包含复制的派生模块，未进入资产检查；失败记录保留，修复后新建 v2 而未覆盖。v2 完成后 UE/Zen 进程及任务临时目录释放；项目、九张地图、15 个源 mesh、六个源 MI 与共享父材质共 32 文件的前后哈希一致。

每件插入物在开关前的派生材质报告六张 used texture；关闭后恰好删除上表的三张 MFPD 相关纹理，保留各自不同的 `Albedo`、`Normal`（LOD0）和 `Roughness` 三张。train/dev、train/test、dev/test 的**派生材质** UE used-texture 路径交集均为 0；六个源实例的开关及 used-texture 集合均未变化。共享的 MFPD 有效参数值仍在实例/父材质配置中，但关闭分支后的当前质量编译资源未报告其纹理使用；共享父材质代码按冻结协议作为基础设施单列披露。此证据支持一个真正消除这三项 UE 报告外观依赖的**派生材质配方候选**，同时保留六个不同的原始 mesh/kit 族，未通过复制、改名或重分类制造新来源。包级交集仍照实保留为保守依赖，不能由此声称全部包交集消失，也未做逐像素或目标渲染平台的贡献验证。

`derive(..., disable_mfpd=False)` 已增加显式可选配方：只在源覆盖值复制且父链改写完成后修改未保存的克隆实例；默认派生行为不变。回执记录源/克隆开关、UE used-texture 前后列表、移除/新增纹理、源未变和配方标记。正式采用时须将这一开关写入新的采集 spec/策略及原始回执，对最终实际 LOD0 渲染材质重审所有受隔离资产的跨区可见依赖；不能把本次单独探针直接当作已采数据的通过证明。当前 train/dev 六份原始采集仍使用 MFPD-on 材质，旧隔离结论保持 `FAIL_SHARED_EDITOR_USED_TEXTURE_DEPENDENCIES`。为避免按 split 引入系统性的材质开关差异，若统一采用 MFPD-off，应把三张 train 与三张 dev 图的 RGB 全部重采，至少 6×160=960 帧，并保持配对原始回执；先前 Development 帧保留为已消费证据，不重新标成 test。test 两件只在正式准入后按同一新配方首次采集。

关闭该静态开关不更换源 mesh，既有派生根又显式将 WPO/PDO 归零、关闭 Nanite，并由采集端固定 LOD0；因此预期网格拓扑身份不变。但本次仅测材质纹理依赖，**未测**切换前后的最终深度、法线、遮罩、ToF 射线交点或标签是否完全相同。实施新采集配方时应以冻结轨迹做受控几何/深度一致性核查，再确定 RGB 之外的成对模态是否也需重采；这项影响目前为 `UNKNOWN`，不能凭纹理回执宣称 ToF 无影响。正式 test 门槛仍包括实际采集配方的原生交集、布局和几何审核。

## 一次性 test 采集前的剩余门槛

1. 原生 LOD0/材质审计已定位六个插入物共有的三张 UE 报告使用纹理；采用真正不同来源 kit，或采用上述统一的未保存 MFPD-off 派生配方并重采 Development，均须在最终实际渲染材质上测得受隔离资产的跨区 UE used-texture 交集为 0，才可继续准入。探针本身不替代最终采集回执。不能用改名、缩放、角色重标或任意白名单追认。
2. 以最终采集 spec 复核 test 的目标/干扰物来源、物理场地、实际四段采样、原生净空和资产交集；三个作者地图仅是预分区候选，不能替代方案中的巷道 41 个 test 布局及总体 384 布局配额。
3. 冻结主基准的完整场地、来源、抽样和模型/阈值后再按受保护访问计划进行唯一一次 test 采集。本轮未执行该采集，也不以本审计结果作模型效果结论。

复核命令：`python research/active/dtr-r0/nearfield/cnh_alley_asset_isolation_audit.py --build-run artifacts.local/evidence/cnh-street-alley-generator-20260924-v3r2/build-run --development-capture-root artifacts.local/evidence/cnh-alley-worker-r3a-thin-20260924-v1 --verify-local-bytes --output artifacts.local/evidence/cnh-alley-asset-isolation-audit-20260924-v1/audit.json`。

## 2026-09-25 采集后状态

上文“修复候选”与“剩余门槛”记录的是实际重采前的状态。此后六张 train/dev 作者图已统一使用未保存的 `Enable MFPD=False` 派生插入物材质，只重采 RGB，各图 160 对，合计 960 对/1920 张。`artifacts.local/evidence/cnh-alley-rgb-replay-six-20260925-v2/collection-overlay.json` 逐帧按身份及哈希绑定旧 ToF/深度/标签；所有新图通过固定曝光的中央、近场和高光数值门槛，首/中/末双目抽查见同目录 `contact-sheets.json`。

同目录 `actual-used-texture-audit.json` 从六次**实际采集**的派生材质回执复核 train/dev used-texture 交集为 0；与 test 的交集也为 0，但 test 一侧只有未保存材质探针，没有 test 采集。train-T 一个位姿的受控 MFPD 开关对照中，218,694 个共同有效深度像素最大差 0 m，有效掩码差 0；这是单姿态几何证据，不推及全部轨迹。旧 MFPD-on 数据和上述原生共享三纹理的 FAIL 回执仍保留，不能回溯改判。

因此九张作者图中**已知**的杂物/插入物源族交叉复用和三张 MFPD 使用纹理问题，在六份 Development 实际 RGB 配方及三份未采 test 候选材质探针的范围内已修复。完整 164 布局 test 仍缺真实场地、五类素材完整分配和正式采集入口，见 [准备度记录](CNH_FULL_TEST_READINESS_20260925.md)；不能把九图材料复核当作全 test 隔离放行。
