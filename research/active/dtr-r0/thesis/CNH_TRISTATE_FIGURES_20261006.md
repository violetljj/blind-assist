# 覆盖与三态章节图清单（2026-10-06）

主线配合[中文章节](CNH_READOUT_COVERAGE_TRISTATE_CHAPTER_20261006.md)：先解释报警器已有的及时提醒，再展示覆盖检查如何改变未及时事件的输出和畅通控制负担。图为已消费模拟Development描述性结果，只有A无保持变体；不将unknown当作安全证明。

## 图1：静默事件与无法判断负担（主图）

[PNG，300 dpi](../../../../artifacts.local/work/cnh-tristate-thesis-writing-20261006/figures/fig1_tristate_A_silent_unknown.png) · [SVG](../../../../artifacts.local/work/cnh-tristate-thesis-writing-20261006/figures/fig1_tristate_A_silent_unknown.svg)

横轴为截止时静默漏报计数，共同事件分母960；纵轴为3473条采样畅通control的unknown时间百分比，时间分母6946模拟秒。single/dual各画冻结20点τ网格，用圆/方/菱形标记报告中预算0/2/10的选中点。曲线连接离散保存点，不对阈值插值，不画未保存的统计区间。

|全体静默预算 /960|single：索引，静默，unknown|dual：索引，静默，unknown|
|---|---|---|
|0|5，0，87.70%|0，0，97.05%|
|2|13，2，69.00%|7，2，62.14%|
|10|19，10，60.59%|11，10，39.77%|

图注可用：**冻结M3与r3门的A变体经验权衡。每臂在共同20点τ网格中，沿用已报告的满足静默上限且unknown负担最小的点。预算2/10的配对重选区间跨零，负担差仅作描述；经验零静默端排序反转。模拟control时间不代表真实用户提示负担。** 区间由[精确事件报告](../nearfield/CNH_TRISTATE_EVENT_DEV_20261006.md) `fd55b74d`提供，在正文表中呈现，未从CSV重造。

## 图2：mode分层（副图）

[PNG，300 dpi](../../../../artifacts.local/work/cnh-tristate-thesis-writing-20261006/figures/fig2_tristate_A_by_mode.png) · [SVG](../../../../artifacts.local/work/cnh-tristate-thesis-writing-20261006/figures/fig2_tristate_A_by_mode.svg)

mode0恒偏15°、mode1±20°扫视、mode2转弯分别成面板。横轴读取各mode保存的静默率，事件分母为294/353/313；纵轴读取该mode controls的unknown时间，control序列分母为1444/1342/687，每条仍按原定义贡献2.0模拟秒。沿用图1全体索引，不在各mode重选。数字0/2/10表示**全体预算**，不表示各mode的静默上限；重合坐标合并标签，空心圆标最大τ端点。

图注可用：**全体选中阈值在不同头部运动模式中的表现。single最大τ静默10个均在转弯，dual为30个中的26个；副图保留阈值继承关系，不将全体负担下降解释为转弯优势。模式来自既有模拟配方，分层不是纯运动或转向因果消融。**

## 输入、代码与复现范围

唯一绘图数值输入是[已保存curves.csv](../../../../artifacts.local/work/cnh-tristate-event-dev-20261006/curves.csv)，SHA256 `2f69d24d3a2108c9a4b66bbd117fd18b50f4c95375d448243552bb0a6946919f`；来源报告提交`fd55b74d`。代码为[cnh_tristate_thesis_figures.py](../nearfield/cnh_tristate_thesis_figures.py)，样式参考既有[cnh_thesis_figures.py](../nearfield/cnh_thesis_figures.py)。新脚本不修改旧图脚本或任何冻结载荷。

代码读取`all/mode0/mode1/mode2`的`single/A/dual/A`，合计8条曲线、160行。只将已保存比例转成百分比，并取报告已定索引；不重新选阈值、不重算bootstrap/模型/几何。CSV无配对重选区间，图未添加误差带。[提取行](../../../../artifacts.local/work/cnh-tristate-thesis-writing-20261006/figures/extracted_plot_data.json)保留原CSV文本值；[哈希manifest](../../../../artifacts.local/work/cnh-tristate-thesis-writing-20261006/figures/hash_manifest.json)记录输入、脚本及输出身份，输入绘图前后不变。

CPU绘图命令（使用已有含matplotlib的Python环境，当前目录为checkout）：

```powershell
$env:PYTHONDONTWRITEBYTECODE = '1'
python research/active/dtr-r0/nearfield/cnh_tristate_thesis_figures.py
```

输出默认在新的ignored `artifacts.local/work/cnh-tristate-thesis-writing-20261006/figures/`，不写入冻结事件目录。PNG为300 dpi，SVG文字转字形路径；本轮已检查两图可读性、标记与标签，修正副图标签重叠。没有重新运行事件分析或场景渲染。
