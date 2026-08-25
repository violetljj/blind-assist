# GRAIL research module

状态：`active / LAST-METER_ALGORITHM_MAINLINE_REOPENED / M0_RUNNING`

GRAIL（Goal-Relative Affordance and Interaction Localization）把最后十米重新定义为：给定用户目标，在未见场景中预测一组可到达、目标一致、适合完成交互的 `站立位置 + 朝向`，或显式 `NONE`。

核心分解固定为：

```text
referent != affordance != reachability != visibility != arrival
```

M0 不训练网络。它在 fresh、split-disjoint 的程序化 metric 2.5D 建筑中自动生成目标实例、同类替身、障碍、功能侧和 set-valued interaction pose truth；以 oracle referent + oracle geometry 检查 task/teacher 上界、简单闭环、几何扰动稳定性和四类结构化反事实。

运行：

```powershell
E:\codex-tools\bin\blindassist-python.cmd scripts/research/grail/run_grail_m0.py --output-dir artifacts.local/evidence/grail-m0
E:\codex-tools\bin\blindassist-python.cmd -m unittest discover -s scripts/research/grail -p "test_*.py"
```

结果写入 `artifacts.local/evidence/grail-m0/`。M0 只可能建立程序化任务与 teacher mechanics 上界；不能声称 RGB、自然 3D 场景、学习、真实相机、Android、用户、产品或安全能力。只有 M0 全部门通过，唯一 successor 才是 M1：在 building-disjoint 3D-derived Development 数据上冻结视觉编码器，比较 B0/B1/B2/GRAIL；否则停止训练 student，先修任务或 teacher。

旧四边界 V1-C/D/E/F 与 passive exact-instance 主线保持关闭。V2-MARKER-POSE 仅作隐藏的 `DEBUG / CALIBRATION / CONTROLLER CANARY`，二维码不进入论文核心或主 Demo。动态风险降为行进过程辅助能力。
