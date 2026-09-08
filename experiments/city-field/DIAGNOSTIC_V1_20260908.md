# City field v1 fixed-method diagnostic

2026-09-08. EXPLORE / curated Development. User authorized a first diagnostic of
the completed collection field. Input is the immutable 108-frame collection-v1;
the 36-frame replay is acquisition evidence and is not duplicated into scoring.

Question: do the existing frozen G10/G13 methods alert on and spatially cover the
four independently verified controlled obstacle families across the nine routes?
Hypothesis: the new scene/context variety will expose alert and localization
failures that were not resolved by validating capture and target geometry.

Keep the existing three-seed G10 expanded_region and G13 decoupled ensembles,
their hashed checkpoints, historical common-VAL alert cutoffs and support 0.5.
Use tools/evaluate_city_native_route.py unchanged for RGB-only predictions,
cached before evaluator scoring. No training, threshold search, route changes
or capture. Budget: one inference pass for each frozen ensemble on all108 frames.
Stop the model experiment after this comparison; finish reporting and delivery.

Report four-family target alert misses, support overlap and joint hits, separated
by BODY/HEAD, route and original spatial partition. Query-level alerts are not
proof of target attribution; diffuse overlap and native competing obstacles are
reported. Only the seven fixture-clear groups whose measured scene labels are
BODY=HEAD=0 form the fully clear control denominator. The other two clear-fixture
groups retain native obstacles. Keep all108 scene rows, plus a sensitivity view
excluding the five floor-REVIEW frames; never turn UNKNOWN into a negative.

G10/G13 are direct RGB-to-risk/support methods, not monocular-depth pipelines.
Their failures alone cannot identify monocular-depth error as a cause. Failure
cards must distinguish alert miss, localization miss, native occlusion/context,
and unreliable labels without inventing causal attribution. Any additional
depth-specific diagnostic requires its own clearly bounded role.

Retain reproducible field diagnosis and useful failure cases regardless of model
outcome; no model promotion or fresh-confirmation claim. All regions have been
inspected for scene engineering and share City Sample assets. `test` remains a
spatial partition, not a sealed blind test. No continuous-time warning claim.

Mechanical failures may be repaired with receipts and identical input identity;
do not rerun successful inference or expand fit/capture budgets. GPU processes
must exit after the bounded run. Results and failure gallery are recorded
under artifacts.local/nearfield/city-field-diagnostic-20260908/.

## 实测结果与决定

两种冻结模型完成一次108帧推理，零训练步、零阈值调整。G10/G13加载及推理分别
用时0.501/7.859秒，实际后端CUDA，RTX5060 Laptop；这不是手机实时延迟。
保留实验场和失败案例，模型不作升级或推广。

36个已验证实例提供45个目标—高度区域机会：细杆同时覆盖BODY与HEAD，其他
三类各覆盖一个区域。下表是**漏报数/9个可靠机会**，不是独立城市泛化概率。

| 目标条件 | G10 漏报 | G13 漏报 | G10 报警且空间响应覆盖目标 |
| --- | --- | --- | --- |
| 细杆 BODY | 5/9 | 9/9 | 4/9 |
| 细杆 HEAD | 7/9 | 9/9 | 0/9 |
| 侧向突出物 BODY | 4/9 | 9/9 | 5/9 |
| 头部横杆 HEAD | 7/9 | 9/9 | 2/9 |
| 悬挂标牌 HEAD | 7/9 | 9/9 | 0/9 |

G10在45个机会中漏报30个，报警且响应覆盖目标11个，最大响应点命中目标仅1个。
G13漏报45个，0.5阈值下没有目标响应重叠。它在7个真正无遮挡对照中零误报，
同时也不报警，不能据此认为效果好；G10在这些对照中有3帧误报。
另外两组仅移除了人工目标，原生设施仍有风险，未放入无遮挡负例分母。

| 全场景参考 | TP | FN | FP | TN |
| --- | ---: | ---: | ---: | ---: |
| G10 BODY | 16 | 17 | 9 | 66 |
| G10 HEAD | 6 | 27 | 20 | 55 |
| G13 BODY | 0 | 33 | 0 | 75 |
| G13 HEAD | 0 | 33 | 0 | 75 |

去掉5帧地面REVIEW后，正例数和漏报数均不变。G10 HEAD误报由20变16，
其他正例统计不变；没有把这些地面问题当作漏检主因。全场景参考基于可见原生
几何，逐实例独立校验的范围仍是36个受控目标。

失败图显示G10经常对建筑、地面等背景产生分散响应。悬挂标牌的两次HEAD报警
均未覆盖标牌本身，故对应目标的联合命中为0/9。G13风险值较低，固定支持阈值
下响应未覆盖目标。这些是观察到的输出形态，不能单凭此实验断定网络内部原因，
也不能断言只改阈值或只增加训练量就能修好。

当前数据足以作为回归失败集。下一项场景建设应优先围绕这些已验证目标补充
接近距离与观察角度，检验何时开始出现响应；当前静态结果不能说明提前预警时间。
本次没有据结果改路线、追加采集、重训或试阈值。

验证：3项针对UNKNOWN排除、无遮挡对照和报警/定位区分的测试通过；90条
方法—目标—高度区域统计与未修改的原有目标评估器逐项一致。源数据哈希复核
未发现变化，原有5帧地面标记和不确定像素保留，模型进程已退出。

## 复用入口与证据

```powershell
python tools/diagnose_city_field.py `
  --collection artifacts.local/nearfield/city-collection-field-20260908/collection-v1 `
  --output artifacts.local/nearfield/city-field-diagnostic-20260908/fresh-output
```

原运行已缓存所有预测，后续统计应复用它们；上面的命令是明确复现时的入口。
RGB哈希、六份权重哈希、历史阈值、模型源码身份均保存在协议中。
绿块是经过20×20池化的已验证目标查询支持，红块是预测支持≥0.5；绿块不表示
整个物体轮廓，重叠也不表示精准定位。

- [固定输入与模型协议](../../artifacts.local/nearfield/city-field-diagnostic-20260908/fixed-v1/protocol.json)
- [完整统计及空间划分结果](../../artifacts.local/nearfield/city-field-diagnostic-20260908/fixed-v1/result.json)
- [逐帧报警](../../artifacts.local/nearfield/city-field-diagnostic-20260908/fixed-v1/per-frame.json)
- [90条方法—目标—高度区域记录](../../artifacts.local/nearfield/city-field-diagnostic-20260908/fixed-v1/target-cases.json)
- [运行回执](../../artifacts.local/nearfield/city-field-diagnostic-20260908/fixed-v1/receipt.json)
- [与原评估器的交叉核对](../../artifacts.local/nearfield/city-field-diagnostic-20260908/fixed-v1/scoring-check.json)

![G10 悬挂标牌的九个场景](../../artifacts.local/nearfield/city-field-diagnostic-20260908/fixed-v1/gallery/g10-suspended_sign-HEAD.jpg)

![G13 头部横杆的九个场景](../../artifacts.local/nearfield/city-field-diagnostic-20260908/fixed-v1/gallery/g13-head_bar-HEAD.jpg)
