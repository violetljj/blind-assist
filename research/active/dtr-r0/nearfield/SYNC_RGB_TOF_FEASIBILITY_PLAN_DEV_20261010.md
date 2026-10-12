# SYNC_RGB_TOF_FEASIBILITY_DEV_20261010 PLAN

EXPLORE，已消费 old6；只检查同步评价源并作描述性试点，不训练、不选融合方法、不定新工作点、不访问保护480/test。基线用户指定9e150dd8 / ab084ba7，实际起点2cb47cfd38d3575ddb62639529033d4d60df8483。

## 固定设计（合成和评价前）
- 预算总和：CPU command-wall <=1800s，GPU <=600s，网络实际下载 <=3,000,000,000 bytes。并行命令按各自wall累加；失败计入。任务完成或上限停；不用等待时间冒充CPU实测。初始分配source180、pilot300、frozen150 CPU+120 GPU、独立审计180，剩余由主代理统筹。
- 优先独立FARO几何，检查原始点云、官方FARO渲染highres_depth；mesh若同LiDAR则不声称独立。无预算内独立资产时LiDAR合成必须标注循环/乐观上界。官方高低深度一致性筛选单独列为选择相关性。
- old6固定名单：47333462、40777060、40777065、41069021、41069042、41069048。先验证可得41069021，再41069048；不据模型分数选source/frame。正式原始FARO全scene超预算可报告不下载。
- 传感器共置、同光轴、camera right/down/forward，ToF相对RGB单位外参；相对世界随ARKit真实轨迹。45deg正方FOV、8x8zone、16bin、0.30m/bin，核实冻结传感器原始bin与range_zero。逐帧用K核验上下左右边界完全覆盖45deg，不只总视角；不自动裁剪后称完整覆盖。
- ToF时钟为ARKit时间轴0.2s网格，RGB取最近可用帧，记录实际delta。旧16帧稀疏anchor试点与连续5Hz模型窗口分开：不能把稀疏帧拼成连续clip。无同步支持的格保留缺测。
- 复用冻结光子模型期望回波、四象限/径向聚合、Skellam噪声与nominal ambient；sub16每zone256条射线，rho=.3。若发布intensity且预算内可得则独立代理rho臂，否则NOT_AVAILABLE；RGB颜色不冒称intensity。
- 缺测/无交为UNKNOWN；逐zone有效subray比例全报。深度图路线仅可见2.5D面，不能称完整mesh；radial与optical-Z明确换算。记录入射余弦假设、边界插值策略。
- 透明ToF读出固定逐zone有符号hist峰，SNR>=3、coverage>=.75；峰为bin中心，转换到相机射线3D。沿用公共27query与原16pixel支持规则：zone量测投到像素时声明它是角覆盖展开，不是16次独立测量。query内几何margin>=0，原native POS像素域的16pixel见证同口径；无支持不等于FREE。
- DAV raw cut=.24403834342956543m，主职责>=.8m；Uni raw pooled304 cut=.09616100788116455m只近带候选。不使用本轮标签调切点；old native labels/public masks保持，近中远分别报W/POS、POS支持、F/FREE、UNKNOWN支持。固定OR/AND以ToF与分带RGB候选(DAV中远/Uni近带)为双输入；其他分支若列必须明确适用域。像素融合后同16pixel统计，以免AND假造共同见证。
- frozen M3/S保持权重、47维、阈值、原5格强档和S轻档。只能在真实连续时钟窗口跑；其HEAD/BODY两个query不能冒充公共27query，单独描述。手持扫描不满足步行共向运动分布。
- 检查峰radial distance vs native zone median radial，另列Z误差防止混单位；按距离带、近带缺测、coverage、peakSNR、ambient量级报告。保留未通过与无数据分母。
- 决策：可独立几何同步且读出可复现=>可行/受限制部分可行；缺独立、覆盖/配准或同步缺口=>部分可行或不可行，不把描述性能当融合选型。单位query，不是接触事件；FARO扫描和手持RGB也不必同一物理时刻。

交付：代码、PLAN、来源/许可证收据、CNH样例与健全性统计、逐query表、独立核验、报告与正式数据源建设草案；CURRENT/RUNS更新，task-owned文件正常push master。所有payload在本目录，物理F:/ba-data/blindassist-artifacts-20260805/work/sync-rgb-tof-feasibility-dev-20261010，F盘初始可用130435883008B。源证据和失败保留。
