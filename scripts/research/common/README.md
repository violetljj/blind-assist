# Research common

状态：active_shared

本 Module 只收纳已经被至少两个研究域复用的稳定 Implementation；它不拥有实验结论，也不构成生产授权。

## 稳定 Interface

调用方使用 `research.common.<module>`；历史 CLI 通过各自领域内的薄 Adapter 保持兼容。当前共享 Interface 是 `public_rgb_redaction.py`。

## 输出

共享 Implementation 不定义第二个输出根；所有下载、脱敏草稿和证据仍写入 `artifacts.local/`。

## 安全边界

这里只共享机械能力，不共享标签权威、训练许可或生产晋级。机器脱敏仍不等于隐私审核通过。

## 停止条件

若 Implementation 只剩一个调用域，或不同调用域需要互相冲突的规则，停止扩张本 Module，并把领域规则退回各自 Module。
