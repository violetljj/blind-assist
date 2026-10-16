# Assistive object finding

所属：**毕设线**，不是避障线（见[两条工作线](../../../docs/WORK_LINES.md)）。

2026-10-16，Development。上一轮全量普查、强基线及跨源评估COMPLETE。
用户已授权继续[UMD固定读点与结构消融](functional_parts/JOINT_LEARNING_PROGRESS.md)：
先补齐全9939图解码强对照，再以相同真实RGB细节、原DINO预训练和数据开展
六组融合／新候选／点质量消融，实测配平成本；总训练壁钟24小时上限，
副机持久队列自动推进。所有重型计算均在副机，主机GPU不恢复；暂不定题。

[完整结果与结构研究判断](functional_parts/FULL_BASELINE_RESULTS.md) ·
[数据与角色监督普查](functional_parts/FULL_DATA_CENSUS.md) ·
[运行／迁移历史](functional_parts/FULL_BASELINE_PROGRESS.md) · [记录](RUNS.md)

SegFormer／DINO各80轮，IIT train4417行、cleanval1757行；原生9类mIoU
66.266%／71.436%，点3352／3279 of3821。IIT→UMD全9939作者手标帧、
108工具／场景组，共享6类mIoU22.816%／24.573%；DINO−Seg区间跨零。
保守点命中12499／10647 of19370，DINO差−9.56个百分点，组区间明确负；
95.73%净损失集中于小区域。主／后队列SUCCESS，原生−1读取故障已修复，
原失败和预测保留。主机GPU未恢复；本任务进程释放，三份checkpoint取回。

PLSP固定40轮、13880更新，Seen1674可评／1710行，KLD .962095、
SIM .515668、NSS1.538031；此前未消费且无训练同RGB仅213行／111组。
它不是PLSP跨源成绩，不能与语义mIoU直接排榜。完整6929训练目标审计发现
knife cut_with／stick的135/135对空间伪标签相同，限制纯结构归因。

优先待证机制是真实局部细节、新候选生成与最终点质量联合学习。
[全量读点诊断](functional_parts/DINO_POINT_GAP_ANALYSIS.md)表明稳定读点
已将IIT原始DINO差73缩至21；内部点等固定规则及普通高分辨率头是必要强对照。
UMD未跑稳定／内部点对照，不能将9.56个百分点原始差当可恢复的结构收益。
PLSP动作绑定须先区分监督缺口与结构瓶颈；当前续作不再训练PLSP或增加数据。

IIT为语义并集；PACO有同实例语义部位关系但无任务角色；InstructPart当前
发布不完整且跨split重叠。正类查询不支持存在性／拒答或视障操作成功结论。
旧小样本及跨源试验见[功能部位记录](functional_parts/README.md)、[跨源结果](functional_parts/CROSS_SOURCE_RESULTS.md)；
寻物、音频和RefEgo访问状态见[研究记录](RUNS.md)。旧当前页保存在Git revision
216e76162a42390b69dffc4e28e652f631288f14，结果与payload未移除。

本轮payload：`artifacts.local/work/functional-parts-full-20261015/`；
回收证据在`worker_completion/`、`diagnostics/full_final/`，完整原始数据和恢复
状态保留注册副机；旧完成状态见`migration/STATE.json`。
续作payload为`artifacts.local/work/functional-parts-joint-20261016/`，封存request、
实时收据及自动跟进归此新run，不能与旧已完成队列混用。
