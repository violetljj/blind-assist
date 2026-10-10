# RGB 查询级负校准参考补充

2026-10-10，EXPLORE／Development。新增 15 个官方 train 缓存环境、240 帧、6,480 个固定子盒参考，得到 30 个严格 sampled FREE，来自 25 帧、10 环境。原 cal16 的 2 个负查询保留，合法负校准候选共 32 个、12 环境；原 train56 的 2 个负查询只列作训练内诊断，不混入校准。没有计算预测分数或选新阈值。

| 来源角色 | 环境／帧 | 全部查询 | POS | UNKNOWN | 严格 sampled FREE |
| --- | ---: | ---: | ---: | ---: | ---: |
| 原 train（诊断） | 7／56 | 1512 | 764 | 746 | 2（1 环境） |
| 原 cal | 2／16 | 432 | 185 | 245 | 2（2 环境） |
| 新 additional_cal | 15／240 | 6480 | 3125 | 3325 | 30（10 环境） |

新增负查询按距离带为 0.3–0.8m **6/2160，只有 1 环境**；0.8–1.5m 19/2160、9 环境；1.5–3m 5/2160、3 环境。总体负参考已从原 2 增到 32，但近场条件负校准仍受单环境覆盖限制，不能据此认定距离条件校准已可靠。

## 来源与冻结选择

官方 train 本地缓存有 35 环境。排除原训练／校准／验证 12 环境及已消费迁移验证 8 环境的整组身份（包括全部 rescans）；其余 15 环境全部纳入，不将既有 eval 改为 cal。每环境使用参考优先的一个缓存 scan、16 个均匀 RGB／depth 配对帧。组、scan、帧及全部 27 格在读取新 depth 和标签数组前冻结；不按预测、正例或是否产出 FREE 替换来源。

继承固定 x[−.9,−.3,.3,.9]、y[−.55,−.18,.18,.55]、光轴 Z[.3,.8,1.5,3] 米的 27 个闭子盒。原生首表面超过盒远端才是 FREE ray；盒前首表面、缺失 depth、非 observed ray 为 UNKNOWN。query FREE 必须≥16 个 FREE ray、0 POS、0 UNKNOWN；未知未转负例。原 train/cal 全 72 帧一并保存，完整工件共 312 帧、24 环境、8,424 查询。

## 核验、工件与限制

8,424 查询独立 XYZ／slab 标签与保存数组一致，严格状态和射线分母全部复算。312 帧 native 几何、RGB／参考身份核验通过；冻结复制后的 312 个参考 SHA 再核验通过；34 个负查询保存索引重读通过（包含 train 的 2 个），public observations 只含公开输入字段。执行后发现合并 manifest 的 frames/groups/state_counts/role 文本继承了原 72 帧元数据，已局部更正为 312 帧并记录前后 SHA；没有重写标签。原执行代码快照和 metadata repair receipt 均保留。

首次两次执行在读取 payload 前遇到括号语法错误，第三次在默认 Python 缺 cv2 时退出。修正后复用现有配置包装器指向的同一 Python 解释器，CPU 完成 50.121s；元数据修复／核验 0.275s；前三次失败保守计 3s，合计保守 **54/600 CPU wall s**。本轮训练／模型推理／GPU／下载均为 0。没有保留任务进程或服务。保存的新数据仍是 Development，既有 eval 身份与旧停止规则保留。

工件：[冻结 manifest](../../../../artifacts.local/work/rgb-body-query-negative-reference-dev-20261010/dataset_manifest.json)、[逐查询计数](../../../../artifacts.local/work/rgb-body-query-negative-reference-dev-20261010/query_reference_counts.csv)、[负查询索引](../../../../artifacts.local/work/rgb-body-query-negative-reference-dev-20261010/negative_query_index.json)、[环境覆盖](../../../../artifacts.local/work/rgb-body-query-negative-reference-dev-20261010/environment_coverage.csv)、[距离带覆盖](../../../../artifacts.local/work/rgb-body-query-negative-reference-dev-20261010/distance_band_coverage.csv)、[覆盖与来源](../../../../artifacts.local/work/rgb-body-query-negative-reference-dev-20261010/coverage.json)、[计划](../../../../artifacts.local/work/rgb-body-query-negative-reference-dev-20261010/plan.json)、[terminal](../../../../artifacts.local/work/rgb-body-query-negative-reference-dev-20261010/terminal.json)、[元数据修复记录](../../../../artifacts.local/work/rgb-body-query-negative-reference-dev-20261010/metadata_repair_receipt.json)。owner 为此 run。实现：[参考准备脚本](rgb_body_query_negative_reference_prepare.py)。

严格 sampled FREE 只描述有限首返回射线，不证明物理整盒或身体空间空闲。帧／query／ray 相关；新增参考仍来自同一 3RScan 相机／数据家族，不能冒充新相机、独立确认或真实步行收益。本支线已交付参考，未训练残差，也未执行冻结模型的新参考评分或按距离条件校准；后续可消费本包，但须报告实际离散工作点和环境／带覆盖。
