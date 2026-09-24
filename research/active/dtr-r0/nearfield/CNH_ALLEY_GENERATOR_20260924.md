# 参数化巷道作者工具

入口：`cnh_street_alley_demo.json`；纯几何规划 `cnh_street_alley_plan.py`，
UE 建造/预览/受管启动分别为 `cnh_street_alley_build.py`、`cnh_street_alley_preview.py`、
`cnh_street_alley_launch.py`。资产来源、路径与原始证据见 `cnh_street_alley_assets.json`。

## 参数及使用范围

每批 1–24 个场地；宽 1.5–4 m、主段长 12–48 m、支段长 12–32 m；
支持 straight/L/T，墙面/地面材质键、固定随机种子、6–32 件背景杂物及显式均匀缩放。
示例给 train/dev/test **预分区**各 3 张地图，每区都有三种拓扑，
材料循环分配，避免某种拓扑或材料只出现在一个分区。
这不是已冻结的最终测试集，场景不同也不自动证明统计独立或真实街区泛化。

每张新地图在 `/Game/BAResearchAlley` 下单独保存，禁止覆盖。
每段增加两处侧向服务凹位；原生杂物保持原生材质，
用静态派生副本关闭 Nanite、固定 LOD0，不改源资产。
管道 uniform scale=0.195（约 3.00 m 高），配电箱 0.3（约 0.71 m 高）。
这些是 CitySample 工程现有背景资产，部分屋顶设备改作巷道装饰；
不能声称全部本来就出现在 Street200 地图里。仅墙体、地面和材质作为共享背景；
全部摆放杂物按 **DISTRACTOR** 管理，净空之外也不能豁免资产族隔离。

R2 的72件杂物来自5族，全部跨三个划分复用，隔离判定为FAIL；
与当时两个插入物无直接资产族重叠，不能抵消跨划分重复的问题。
R3a 联合分配如下，语义族及源素材kit同时检查，跨角色也不能重复：

| 预分区 | 杂物族 | 插入物族 |
| --- | --- | --- |
| train | 垃圾桶、配电箱、通风设备 | 路桩、路牌 |
| dev | 管道、托盘、门梯 | 路锥、邮箱 |
| test | 雨篷、消防接口、长椅 | 塑料桶、自行车架 |

原生检查确认所用17资产的18个材质槽均opaque且无编译形变；
R3a九场72件实测最小间隔175.334mm。第一次R3的147.924mm失败记录保留，
只增加规划摆放余量，最终150mm门槛不变。

## 净空和身份检查

规划阶段使用实际加载网格边界，保存前再用 UE 组件实际包围盒检查。
保护体积为每段中心线横向 ±0.6 m、高 0–2 m，并在端点延伸 0.6 m；
所有墙和杂物的完整 AABB 距它至少 0.15 m。支撑地面不参与障碍净空判定。
杂物还需在场地地面内，避开墙和其他杂物；不能以隐藏或少放物体通过。
该检查不代表整个标称巷宽无杂物，不代替后续实际四段采集轨迹和转弯姿态的查询体积检查。

几何签名不包含地图名、材质、随机种子或杂物位置：同一墙体/路径几何不能
靠这些变化伪装成新场地。项目 `Saved/CNHAlley/site-registry.json` 持久登记已保存且
实测通过的地图；跨批次重复几何、重复 ID、重新分区均拒绝。
唯一例外是显式 revision_of_map_asset 引用旧版本，必须同几何、同分区、同 physical_site_id；
追加历史而不覆盖。R1/R2/R3/R3a 是同一批九个场地修订，不能重复计算场地数。

## 复现和后续采集

将示例配置复制到 canonical `artifacts.local/work/cnh-route-comparison-20260924/plan/`，
指定新的场地 ID 及真正不同的路径/墙体几何；沿受管 `research-ue` run-spec 调用 launcher。
一次 UE 会话依次生成整批，每张输出俯视、入口和内部三个 1280×720 原生预览。
保存 `generated-plan.json`、逐场地几何回执、源文件哈希、预览和进程释放记录。
预览曝光修正仅用于作者看图，不改变已冻结的数据采集曝光配置。

数据采集时以登记的物理场地身份约束所有布局及端点，先落实实际采集查询净空，
继续原 30/75 mm、2% 规则；作者预览不是深度验收，也不直接增加现有 1920 帧。

## 巷道 RGB 固定曝光重采

旧 train/dev 六场 960 帧 RGB 严重欠曝；旧 ToF、深度和几何原件及回执保留。
最初的 `ALLEY_SINGLE_PROBE_FIXED_BIAS_V1` 在 13.2 EV100 的几乎全黑 LDR 探测帧上
估算 +6 档偏移；train-straight v4/v5 各生成 160 对 RGB，图像仍近黑，首帧 ROI
亮度 p60 为 1/255、p90 为 7/255，不能因传输成功而认定 RGB 可用。v1–v3 单位姿
曝光诊断表明：v2 自动曝光在开启或关闭 Lumen 时 p60 都约为 0.448；v3 的固定
EV100 −2 窄窗 p60 为 0.230、墙面、地面、细杆可见。因此主要障碍是探测起点处于
LDR 量化黑位，不是缺少间接照明。原图太阳、SkyAtmosphere、SkyLight 和几何不修改。

随后 `ALLEY_SINGLE_PROBE_FIXED_EV_V1` 从 EV100 −2 的非黑探测帧按中央 p60
选取固定 EV。train-straight v6 的 160 对图像数值和目视通过；但 train-T v1 被
高亮墙面驱动到 EV100 −0.5，中央 p60 虽通过，近场地面/细杆仍太暗：首帧
近场 p30=0.048、p60=0.075。该两批及 train-L v1 均保留为 Development 诊断，
不冒充最终重采。train-T 的单位姿 EV100 −2 探针近场 p30=0.150、p60=0.205，
中央 p99=0.942，细杆可辨；开启 Lumen 与关闭时几乎相同，所以没有改动原图的
太阳、SkyAtmosphere、SkyLight、间接照明或几何。

最终六图统一采用 `ALLEY_SINGLE_PROBE_SHADOW_FIXED_EV_V2`：每个布局首个 centre
位姿的已摆放插入物上，等待读取就绪，在固定 EV100 −2、曝光补偿 0 下取一张稳定
左目探测帧。每 4 像素采样近场 ROI（x=42.1875–67.1875%，y=50–95.8333%）
的亮度 p30，计算
`最终 EV100 = −2 − round(4 × log2(0.15 / 近场p30)) / 4`。
探测近场 p30 不在 0.03–0.75、中央 ROI（x=15–85%，y=30–85%）p99 超过 0.98、
或需调整超过 4 档即拒绝。布局所有左右目帧使用同一个 EV100 ±0.01 档窄窗，
回执绑定该布局唯一探测、最终 EV、每帧姿态和材质配方。`ALLEY_RGB_ONLY_REPLAY_V1`
只写新 RGB；原 ToF、深度、标签、法线、反照率、实例 ID 和旧原始回执不重写。

逐帧、逐目最终门槛同时要求中央 ROI p60=0.12–0.85、p90−p10≥0.08、p99≤0.98，
以及近场 ROI p30≥0.10、p60≥0.14。任一图失败标为 `FAIL_UNUSABLE_RGB`，数值全过
仍须首、中、末帧双目目视。六个最终 v2 回执位于
`artifacts.local/evidence/cnh-alley-rgb-replay-{train-straight,train-l,train-t,dev-straight,dev-l,dev-t}-20260925-v2/capture/`；
每图各 160 对、320 张均通过上述冻结门槛：

| Development 布局 | 固定 EV100 | 全图最小近场 p30 | 全图最大中央 p99 |
|---|---:|---:|---:|
| train-straight | −1.5 | 0.132 | 0.919 |
| train-L | −2.0 | 0.102 | 0.916 |
| train-T | −2.0 | 0.146 | 0.952 |
| dev-straight | −2.0 | 0.154 | 0.954 |
| dev-L | −1.25 | 0.135 | 0.883 |
| dev-T | −2.0 | 0.102 | 0.952 |

每图在未保存的两件派生插入物材质上关闭 `Enable MFPD`，其余源 mesh/材质身份保留；
实际逐插入物回执记录 UE 使用纹理由 6 降到各自的 3 张 Albedo/Normal/Roughness。
train-T 同位姿的 MFPD-off 一帧七路深度与旧 MFPD-on 原件比较：相机哈希、
插入顶点/三角面/变换完全一致，218,694 个共同有效像素深度差最大 0 m，
有效掩码差 0 像素；纠正的 `Z` 通道回执是
`artifacts.local/evidence/cnh-alley-mfpd-depth-parity-train-t-20260925-v1/capture/format-receipt-corrected-z.json`。
首次 cv2 误读单 `Z` 通道为全零的失败回执保留。MFPD-off 和 on 的同 EV 图像中
插入物仍可见；此单位姿核查并不证明所有姿态深度逐像素不变。

六份 `rgb-tof-overlay.json` 按帧 ID、frame_key、相机/几何/旧左目深度哈希将新双目
RGB 绑定原 `observations.npz` ToF 与 `targets.npz` 标签的同一数组索引。
`artifacts.local/evidence/cnh-alley-rgb-replay-six-20260925-v2/collection-overlay.json`
汇总 960 帧/1920 张 RGB，`visual-review.json` 列出六图各首、中、末帧双目样本；
`contact-sheets.json` 索引每场一张六格抽查图，生成时重新核对了全部 36 张源图哈希。
`actual-used-texture-audit.json` 从最终实际采集材质回执核对 train/dev 的交集为 0；
与 test 的交集也为 0，但 test 一侧仅是未保存材质探针，**不是 test 采集或准入**。
本地精简旧源缺少右目深度原件，右目旧哈希绑定旧回执但未重新核验。
train-L 与 dev-T 的近场 p30 最低仅约 0.102，接近 0.10 门槛；保留这一边界。
所有证据均为已消费 Development RGB 修复，不构成 benchmark、硬件或部署结论。
