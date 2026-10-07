# 覆盖与三态章节图清单（2026-10-07 编排同步）

配合[章节草稿](CNH_READOUT_COVERAGE_TRISTATE_CHAPTER_20261006.md)。正文采用静默—无法判断主图、预算10模式柱图和方法示意图；完整模式曲线放附录B，转头提示图随探索性结果移入附录C。本次只同步图号与位置，图片及其统计未改变。所有文件提供300 dpi PNG和文字转字形路径的SVG。

## 图1：静默—无法判断主图

[PNG](../../../../artifacts.local/work/cnh-tristate-thesis-revision-20261006/figures/fig1_tristate_A_silent_unknown.png) · [SVG](../../../../artifacts.local/work/cnh-tristate-thesis-revision-20261006/figures/fig1_tristate_A_silent_unknown.svg)

横轴为静默漏报计数，分母960个接触事件；纵轴为3473条对照序列、6946模拟秒的无法判断时间。单路与双路使用A变体的冻结20点网格，标记沿用报告选点。删除原图内总标题和底部说明，相关内容由正文图注承担；零预算标签改为“静默=0”，87.70%与62.14%标签移到曲线外，使用引线关联。

|预算 /960|单路索引：静默，无法判断时间|双路索引：静默，无法判断时间|
|---|---|---|
|0|5：0，87.70%|0：0，97.05%|
|2|13：2，69.00%|7：2，62.14%|
|10|19：10，60.59%|11：10，39.77%|

## 图2：预算10的模式漏报构成及时间负担

[PNG](../../../../artifacts.local/work/cnh-tristate-thesis-revision-20261006/figures/fig2_tristate_A_mode_summary_budget10.png) · [SVG](../../../../artifacts.local/work/cnh-tristate-thesis-revision-20261006/figures/fig2_tristate_A_mode_summary_budget10.svg)

选中点为全体静默预算≤10的单路索引19、双路索引11。左侧为mode分组的水平堆叠柱，显示无法判断漏报和静默漏报；右侧为同模式对照序列的无法判断时间。各模式沿用全体阈值，未重新选点。接触事件分母写入行标签，时间分母在正文图注中给出。

|模式|接触事件 / 对照序列|单路：无法判断漏报，静默，时间|双路：无法判断漏报，静默，时间|
|---|---|---|---|
|mode0 恒偏15°|294 / 1444|33，0，97.06%|2，0，27.56%|
|mode1 ±20°扫视|353 / 1342|8，0，52.28%|1，2，39.17%|
|mode2 转弯|313 / 687|0，10，0.16%|18，8，66.64%|

柱段和时间只读取既存CSV字段unknown_miss、silent、unknown_time；零计数不绘有面积的柱段，不推算新的置信区间。

## 方法示意图：单路与双路的覆盖检查

[PNG](../../../../artifacts.local/work/cnh-tristate-thesis-revision-20261006/figures/fig3_coverage_method_topview_schematic.png) · [SVG](../../../../artifacts.local/work/cnh-tristate-thesis-revision-20261006/figures/fig3_coverage_method_topview_schematic.svg)

两面板俯视示意，包含行人、行进方向、头部方向、单路视场、双路视场并集、收缩检查核心、0.9 m截止线及未检查侧带。头部偏转15°为示意选择；实际视场沿头部方向，参考核心沿行进方向，不将两者画成同一方向。参数来自冻结配置而非实验结果：

|参数|值|只读来源|
|---|---|---|
|标称视场 / 核心收缩|45° / 水平垂直各3°|r3 PLAN及事件PLAN|
|双路相对头部偏航|−15°、+15°|[ARMS定义](../nearfield/cnh_tristate_dev_r3_truth.py)，8f9dbebe|
|覆盖检查前向 / 横向|0.9–2.1 m / ±0.29 m|r3 PLAN|
|标签横向 / 未检查窄侧带|±0.30 m / 0.29<横向绝对值≤0.30 m|r3 PLAN|
|径向限制 / 历史窗|2.303063 m / 0.6 s|r3 PLAN|
|标称俯仰|−10°|r3标定轨迹配置|

图是单姿态、传感器高度的水平截面，按标称俯仰绘制水平投影；实际门还使用最近历史姿态并集与两个高度片。图不表示某个接触事件，也不以阴影面积推出实验覆盖率。源代码[cnh_tristate_dev_r2.py](../nearfield/cnh_tristate_dev_r2.py) 5ed0c72a及[r3核心定义](../nearfield/cnh_tristate_dev_r3_geometry.py) 8f9dbebe用于核对方向和核心语义，未导入运行。

## 附录图C1：头部朝向与转头提示

[PNG，300 dpi](../../../../artifacts.local/work/cnh-active-scan-dev-20261006/figures/fig4_active_scan_summary.png) · [SVG](../../../../artifacts.local/work/cnh-active-scan-dev-20261006/figures/fig4_active_scan_summary.svg)

左：被动单路、被动双路、单路+提示（0.6 s反应）与单路头部正对上限的未及时事件构成（灰为无法判断漏报，彩色为静默），分母369；右：1255条畅通对照的无法判断时间，菱形为响应后窗口frame10–14。只读[主动扫视result.json](../../../../artifacts.local/work/cnh-active-scan-dev-20261006/result.json)最大τ点，代码[cnh_active_scan_figure.py](../nearfield/cnh_active_scan_figure.py)，不重算统计。用户响应为假设模型。

## 附录图A1：完整模式曲线

[PNG](../../../../artifacts.local/work/cnh-tristate-thesis-revision-20261006/figures/figA1_tristate_A_mode_curves.png) · [SVG](../../../../artifacts.local/work/cnh-tristate-thesis-revision-20261006/figures/figA1_tristate_A_mode_curves.svg)

原mode0/1/2曲线移到章节附录。横轴为各模式保存的静默率；标记继承全体预算0/2/10的索引，空心圆标最大低分阈值。曲线补充全网格，正文柱图承担模式间比较。

## 输入、代码与复现

绘图代码：[cnh_tristate_thesis_figures.py](../nearfield/cnh_tristate_thesis_figures.py)。统计图唯一数值输入为[既有curves.csv](../../../../artifacts.local/work/cnh-tristate-event-dev-20261006/curves.csv)，来源报告fd55b74d，SHA256为2f69d24d3a2108c9a4b66bbd117fd18b50f4c95375d448243552bb0a6946919f。代码提取8条A曲线、160行；百分比仅为保存比例的显示单位转换，没有模型推理、几何实验或bootstrap重算。

方法图另读取[r3 PLAN](../../../../artifacts.local/work/cnh-tristate-dev-r3-20261006/PLAN.json)、[事件PLAN](../../../../artifacts.local/work/cnh-tristate-event-dev-20261006/PLAN.json)与上述三个源码的配置文本，使用AST及配置读取，不运行来源代码。[提取值](../../../../artifacts.local/work/cnh-tristate-thesis-revision-20261006/figures/extracted_plot_data.json)保留CSV原始行和方法图配置；[哈希记录](../../../../artifacts.local/work/cnh-tristate-thesis-revision-20261006/figures/hash_manifest.json)记录输入、脚本、四套图及提取文件。

使用已有含matplotlib的Python环境，在checkout执行脚本即可；默认输出为ignored artifacts.local/work/cnh-tristate-thesis-revision-20261006/figures/。旧写作图目录和冻结结果均保留。本轮已检查四图及输入/输出身份。

可选转弯示例时间线跳过：事件ledger.npz含分数和最终事件状态，但没有直接保存完整逐帧覆盖门与三态输出；本轮没有为了补图重新计算。
