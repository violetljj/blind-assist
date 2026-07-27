# Motion-diverse RGB-D source search R1

## 稳定 Interface

从仓库根目录通过 `scripts/run_research_tool.py
egomotion-compensated-looming <tool.py>` 调用。稳定 wrapper 覆盖：

- ETH3D/TartanAir pose-only 排序；
- 单窗 depth+pose 提取、8-worker geometry 与 implementation lock；
- 四窗 cohort freeze 和最小 RGB 输入准备；
- 8-worker 四窗 RGB development canary；
- 不导入 producer 的 ledger/aggregate/identity 独立验证。

## 输出

pose-only 排序队列和后续 geometry/RGB receipts 写入 `artifacts.local/`。
trajectory proxy 只冻结消费顺序，所有角色保持 unknown，直到 depth geometry。

## 安全边界

同一新搜索轮次允许在预登记候选间顺序切换，但不得改算法、降低角色门、
使用 `floor3_3`、根据 geometry outcome 重排队列或提前下载 RGB。

## 停止条件

找到首个满足 `2 positive + 2 below-reference` 的四窗 cohort 后立即停止
geometry 搜索、冻结四窗；所有预登记队列耗尽仍不足时才扩展新一轮候选。
本轮冻结 cohort 已完成 RGB development canary；它仍不是 all-real cross-source
holdout、性能/Android 资格或产品证据。
