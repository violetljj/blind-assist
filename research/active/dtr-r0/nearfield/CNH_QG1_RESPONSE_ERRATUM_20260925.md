# QG-1 Part B 数值检查机械勘误

2026-09-25，原冻结协议及科学臂、所有模拟参数、随机 seed、划分、评分、阈值规则保持不变。原协议提交 `bcd4d78a`，协议 SHA256 `1ab0e8d27482247fdd632271412d25db7fe23bb138bec46115a6cd48fa09a7e6`。

第一次确定性执行在 960 帧完成后，于正缩放检查的 AP/AUROC 严格浮点相等断言失败。`artifacts.local/evidence/cnh-qg1-response-decomposition-20260925-v1/failure.json` 保留；完整循环已通过两端逐位检查，但旧代码尚未持久化 scores/metrics。失败前未输出新指标，也未据此选择科学参数。该断言失败本身不证明同分合并、排序改变或任何具体机制。

经 root 授权，仅修复检查和保存顺序：在缩放诊断前持久化固定 scores 和指标；原 gain=7 与 rho=.5 检查继续计算并报告原/缩放 AP、AUROC 及差值，逐查询原始分数变化、除回倍率后变化、稳定排序位置变化、原排序相邻同分变化、逆序数量及唯一值数量，并补充 pooled 排序/同分检查。`numerical_invariance_verified` 记录是否完全数值不变，不再以该报告项中止已经完成的全表。数学上正比例缩放保持排序；有限精度检查所得差异按实际结果解释，不预设原因。

修复必须先提交，再按原输入/参数/seed确定性重放至新目录 `artifacts.local/evidence/cnh-qg1-response-decomposition-20260925-v2`，不覆盖 v1，不增加科学臂、不改科学预算、不挑较好结果、不改原协议（避免其他已启动 Gate 的身份变化）。原两端逐位完整性硬门槛保持不变；本勘误哈希写入重放结果。
