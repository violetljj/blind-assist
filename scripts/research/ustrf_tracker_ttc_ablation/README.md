# USTRF tracker TTC ablation

状态：frozen / T0 detector coverage failed / T1-T3 not run / R3 evaluator disabled / production-isolated

## 稳定 Interface

本 Module 读取 hash-bound 的 LILocBench prepared `frames.jsonl`、冻结 R3 candidate route prediction、独立 detector 模型和本 Module 预注册，先生成 candidate-hidden 的逐帧 person detector ledger，再在同一输入上运行 T0/T1/T2/T3 association、route-conditioned event trace 和 source-native depth TTC proxy。离线 PIL 预处理复现 Android 的黑底等比 letterbox 几何，但不声明逐像素 Android Canvas parity；ledger 同时记录阈值前 person 最高分。输入 hash、模型 hash、letterbox、confidence/NMS、窗口边界和臂参数不匹配时 fail closed。

T0 是当前 label + IoU/center greedy 规则；T1 只增加类别门控和 alpha-beta 预测；T2 只替换为 high/low confidence 二阶段关联；T3 只替换为 observation-centric 最近两帧重更新。四臂共享 detector 输出、路线、事件生命周期和指标脚本。TTC 仅为因果 RGB-D range-rate proxy，不是 physical assistive TTC。

## 输出

只写 `artifacts.local/evidence/ustrf-tracker-ttc-ablation-v1/` 和 `artifacts.local/tmp/ustrf-tracker-ttc-ablation-v1/`。Detector ledger 不包含事件 truth、review 输出或 R3 candidate alert；truth/route 只在独立评估阶段按 hash 读取。原始 RGB-D、R3 prepared bundle 和 source authority 不被改写。

## 安全边界

本实验只使用两条已准入 LILocBench 来源做独立研究，不能把它们升格为 R3 三源合集。Bonn 仅可作为单独负样本压力分区，不能贡献第三来源、事件召回或 critical miss。没有对象级 ground-truth track IDs 时，identity switch/id precision/id recall 必须 `not_evaluable`，不能把缺失记为 0。模型生成或模型审核的轨迹证据必须显式标注，不冒充真实用户或生产传感器事实。

## 停止条件

任一输入或 hash 不一致、逐帧 detector ledger 缺失、路线/事件 truth 泄漏、或 T0 不能完整复现时停止 T1-T3。若 T1-T3 只改善轨迹观感而不改善事件主指标，或增加 false alerts/repeats，停止候选；不联动调 detector、route、NMS、confidence 或事件阈值回救。
