# 原 S24 Ultra 上的深度误差从哪里来

**纯 RGB 联合替换 Softplus 和焦距计算后，16 张开发图的设备深度相对冻结 FP32 的平均误差从 3.522% 降到 0.535%，逐图全部改善。** 归档帧上 HTP 执行从 17.86 降到 16.56 ms。保留同样修正，再把深度改为一字节输出，16 图平均误差为 0.706%。优先使用双修正浮点图作设备导出候选；面向只消费深度码的下游，可使用对应的原生 8 位图。

## 用真实中间值定位偏差

2026-11-24 复测原 SM-S9280、QAIRT 2.47.0、Hexagon v75 / SoC 57。冻结权重，共用 NCHW、原 I/O 类型、16 位浮点转换参数、图名和 HTP 扩展配置，均使用同一台手机。纯 RGB 在同一归档图上，原二进制、匹配配方重编译的原图、添加七个中间输出的图，三份最终深度文件 SHA-256 相同。FP32 主机基准为原始 ONNX 在 ORT CPU 的输出。误差为逐像素绝对相对误差乘 100，跨图时等权平均；每图 288 × 384 像素。

从设备读出的 S2 局部深度头与主机原始全局偏置起步，余下节点先用高精度重算。按计算顺序每次再引入一个实机读数，得到以下归档帧深度误差。

| 已换成实机值的最后一处 | 深度平均相对误差 |
| --- | ---: |
| 局部头 | 0.2693% |
| 全局偏置 | 0.8718% |
| 两者相加的 logit | 0.8913% |
| Softplus 输出 | 4.0893% |
| 焦距因子 | 3.3371% |
| 相乘并裁剪 | 3.3367% |
| 插值并限制最小视差 | 3.3343% |
| 设备最终米制深度 | 3.3296% |

该帧焦距标量从 FP32 的 1.510974 变成设备的 1.542969，方向与部分 Softplus 误差抵消。按指定顺序计算的有符号差可逐像素相加；绝对值取平均后无法分摊成唯一的“主干占比”。比较设备实际倒数前视差和最终深度，可进一步单独检验倒数。[分段重建与匹配图](silicon_probe.py) · [原设备分段读数](../../../../../artifacts.local/work/egogeom-omni-v0/math_theory/m5_info_geom_quant_reparam/silicon_followup/probe/matched_archive_device_comparison.json)

我们把原倒数、乘一、深度裁剪节点单独编译到原机，用 561 张输入穷举冻结图裁剪端点间全部 **61,963,852 个 FP32 视差位型**。逐点对照高精度参考，最坏 **0.108398%**，达到 0.15% 的为零。最坏点的设备/参考深度分别为 16.859377 / 16.877672 m。再把上述真实整网的反演前视差输入该子图，110,592 个像素逐位复现整网深度，该帧末端最大误差 0.082761%。因此规格中的 0.15% 门槛在所测独立反演子图上已达到，0.535% 属于整张网络在开发图上的平均误差，两者对应不同的验收对象。[全量程硅片穷举与整网桥接](TERMINAL_SWEEP.md)

## 两处修改各自起了什么作用

第一处以 $s=e^{-|x|}/(2+e^{-|x|})$ 计算稳定的 Softplus，用 $2(s+s^3/3+s^5/5+s^7/7)$ 近似修正项 $2\operatorname{atanh}s$。实数算术下其相对截断误差至多 $1/52488$。第二处给模型增加 FP32 标量输入 `host_focal`，用原公式从当前帧的 FP32 射线计算。设备模型其余部分仍照常接收重力对齐射线。16 张图上，双修正与原图在 CPU 的最大逐像素相对差为 0.001812%，设备上三种辅助输出也逐文件相同。[稳定近似及误差界](SILICON_CODEC.md) · [四组干预](silicon_intervention.py)

| 输入 | 原图 | 仅 Softplus | 仅焦距 | 双修正 |
| --- | ---: | ---: | ---: | ---: |
| 归档单帧 | 3.330% | 1.210% | 4.081% | 0.897% |
| Aria，4 张 | 3.304% | 1.160% | 3.348% | 0.703% |
| CODa，4 张 | 4.210% | 1.450% | 4.027% | 0.734% |
| ARKit 日，4 张 | 3.452% | 2.998% | 2.707% | 0.427% |
| 同一 ARKit 帧模拟暗光，4 张 | 3.124% | 3.093% | 2.516% | 0.277% |
| 16 张等权 | **3.522%** | **2.175%** | **3.149%** | **0.535%** |

四种条件各按文件名排序等距取四段，使用每段中间一帧。开发数据共 12 个不同源片段，配对日/暗光共用原始画面。构造输入只使用 JPEG、内参和当前姿态。双修正逐图误差 0.114%–0.880%。单帧计时采用原图、Softplus、焦距、双修正、双修正、焦距、Softplus、原图的次序，每块 50 次推理；各臂两块的平均 HTP 时间分别为 17.861、19.172、15.984、16.557 ms。纯 RGB 双修正的 NetRun 均值 17.77 ms，原图 19.14 ms。计时排除生成射线和标量、传输、业务后处理及告警；工程开发数据未用于验证真实距离或现场行走。[逐图和 CPU 对照](../../../../../artifacts.local/work/egogeom-omni-v0/math_theory/m5_info_geom_quant_reparam/silicon_followup/factorial_panel/factorial_fidelity.json)

## 同一修正能否转到 Omni

只把两处相同改动转入冻结 Omni。归档帧、三类输入、每类原/改/改/原各 50 次，未重新选阈值。

| 输入 | 原图 → 双修正，设备相对 FP32 | 原图 → 双修正，HTP 平均 |
| --- | ---: | ---: |
| 无 ToF | 3.330% → 0.897% | 19.65 → 18.36 ms |
| 有 CNH | 1.782% → 0.519% | 19.63 → 18.31 ms |
| 有深度 | 1.697% → 0.496% | 19.62 → 18.31 ms |

无 ToF 的双修正 Omni 深度与纯 RGB 双修正逐像素相同。三个辅助输出在每种模式中也都与各自的原图相同。两种非零提示均只测了这一个场景。[Omni 构建](build_omni_intervention.py) · [三模式实测](../../../../../artifacts.local/work/egogeom-omni-v0/math_theory/m5_info_geom_quant_reparam/silicon_followup/omni_transfer/transfer.json)

## 原生一字节深度出口

原 Padé 压扩图在 ORT CPU 的全量程最坏量化误差为 1.00857%，其早期设备版本在 16 张图却从原图的 3.522% 升至 4.056%，16 张全退化。归档图 RGB 的 HTP 从约 17.9 升到 28.7 ms，Omni 加 CNH 从约 19.5 升到 37.4 ms。匹配原转换配方后，缩短 Padé 平方链改善精度，但 DLC 的内部节点仍为附有量化编码信息的 Float16，实机同帧出现 506 个不同浮点深度。[匹配对照、DLC 层表与反例](SILICON_CODEC.md)

独立导出真实的 `depth_code`，对裁剪后的视差计算 $j=\mathrm{clip}_{0,255}(\mathrm{round}(\log(25d)/h))$，$h=\log(25/0.15)/255$。图内使用乘法、Log 与 QuantizeLinear。消费端用固定 256 项 FP32 表 $25e^{-jh}$ 查回米值。检查了 HTP 元数据的 `UFIXED_POINT_8`、scale 0.02006272785、offset 0，以及实际读回的每个 `depth_code_native.raw` 为 110,592 字节。原浮点深度出口为 442,368 字节；辅助出口与主干仍为原精度。

| RGB 导出 | 归档帧相对 FP32 | 16 张相对 FP32 | 归档帧 HTP 两个 50 次块 |
| --- | ---: | ---: | ---: |
| 原图 → 一字节 | 3.330% → 3.360% | 3.522% → 3.560% | 一字节 18.607 / 18.656 ms |
| 双修正 → 一字节 | 0.897% → 0.913% | 0.535% → 0.706% | 一字节 16.806 / 16.748 ms |

在双修正相同输入上，打包相对未打包设备输出的平均像素误差为 0.503%（16 图）。编码后的设备误差包含共同的主干偏差，不能将两个百分数直接相加。查表耗时未计入 HTP 时间。[物理端口及查表程序](silicon_byte.py) · [原生码面板结果](../../../../../artifacts.local/work/egogeom-omni-v0/math_theory/m5_info_geom_quant_reparam/silicon_followup/corrected_byte_panel/byte_fidelity.json)

## 怎样从内参得到焦距标量

对当前针孔模型，固定内参与裁切，刚体旋转保持相邻单位射线的弦长。可将焦距标量按标定缓存；也可直接计算离散格点上的精确最大弦长。设像素中心为 $(u+1/2,v+1/2)$，令相邻两列的归一化横坐标为 $a,b=a+1/f_x$，$r=\sqrt{1+((v+1/2-c_y)/f_y)^2}$。两射线的夹角为 $\theta=\arctan(b/r)-\arctan(a/r)$。固定行时，夹角是偶对称、随横向距离递减的核 $r/(r^2+x^2)$ 在 $[a,b]$ 上的积分，所以最优列对的中点最靠近主点。将 $c_x-1$ 四舍五入并裁到合法左列即得。固定该对后，$\partial\theta/\partial r$ 的符号为 $ab-r^2$，因此最优行只可能在目标 $v=c_y-1/2\pm f_y\sqrt{\max(ab-1,0)}$ 各自相邻的整数行或相应边界。至多检查四行，取最大弦长 $c=2\sin(\theta/2)$，返回 $\mathrm{clip}(350\sqrt{c^2+10^{-12}},0.2,4)$。假设与冻结图的针孔射线、无镜头畸变和有效重力旋转一致。

1,200 组随机针孔网格（含偏出画面的主点）上，与逐对枚举比较的最大弦长绝对差为 $3.34\times10^{-16}$ 以下；有 74 组最优点落在偏离主点的内部行。16 张实际输入按内参计算相对全图 FP32 射线枚举的最大标量差为 0.000094%，16 组转成 IEEE FP16 后均相同。本轮设备测量实际传入全图 FP32 枚举的标量。[离散解析式和复验](focal_intrinsics.py)

## 复现与产物

本机可直接取用 [纯 RGB 双修正 v75 context](../../../../../artifacts.local/work/egogeom-omni-v0/math_theory/m5_info_geom_quant_reparam/silicon_followup/factorial/both/v4_pure_rgb_final_v75.bin)、[Omni 双修正 v75 context](../../../../../artifacts.local/work/egogeom-omni-v0/math_theory/m5_info_geom_quant_reparam/silicon_followup/factorial/omni_both/v4_omni_final_v75.bin)、[一字节纯 RGB 双修正 v75 context](../../../../../artifacts.local/work/egogeom-omni-v0/math_theory/m5_info_geom_quant_reparam/silicon_followup/byte_both/head_byte_both_v75.bin) 与[对应的 256 项小端 FP32 解码表](../../../../../artifacts.local/work/egogeom-omni-v0/math_theory/m5_info_geom_quant_reparam/silicon_followup/byte_both/decode_f32.raw)。三个图都新增 `host_focal` 输入，布局 FP32 NCHW $1\times1\times1\times1$。一字节图用 `--use_native_output_files` 取回原始码值，米制图沿用 `depth_m`。

在本目录使用研究 Python。先运行 `silicon_panel.py prepare` 生成无标签面板，`silicon_probe.py build` 生成切点参考，再以 `build-matched --variant` 构建 `control`、`taps`、`softplus`（需要 Padé 对照时加 `codec`、`codec_r4`）。匹配构建和 `build_omni_intervention.py` 接收 `--qairt-sdk` 与 `--qairt-python`。运行 `silicon_intervention.py prepare` 后，以 `build --arm focal` 和 `build --arm both` 构建纯 RGB 干预；该脚本及 `silicon_byte.py build [--corrected]` 接收 `--sdk` 与 `--python`。先建 RGB 联合图，再建其一字节版本。

手机入口为 `silicon_run.py --task TASK --run NEW_NAME --serial SERIAL --adb PATH`。任务 `probe`、`factor`、`factor_panel`、`transfer`、`corrected_byte` 和 `corrected_byte_panel` 分别对应切点、归档四臂、面板三臂（基线在 `panel`）、Omni 三模式、归档与面板的原生压缩。原始配置、context 哈希、档案输出与计时见每个运行的 `receipt.json`。不同拍摄条件等权的完整均值可从逐帧 JSON 重算。默认启用项目原 HTP 扩展；另存的 `archive_a` 未启用该扩展，仅用于原输出逐位回放，跨配置时间不可混用。[手机运行入口](silicon_run.py) · [本机派生模型、输出和收据目录](../../../../../artifacts.local/work/egogeom-omni-v0/math_theory/m5_info_geom_quant_reparam/silicon_followup/)
