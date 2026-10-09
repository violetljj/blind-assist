# CNH完整目标—query支持—输入证据—既有响应机制链（Development）

**+3°使正侧完整目标离开literal接触query，但这不等于输入证据全部消失或弱pass监督无价值。** 完整目标六面裁剪与独立实体交集检查显示：正侧HEAD/BODY各16个场景的f3–13均不与±0.3 m接触query相交，负侧各16个场景均仍相交。原身体真值保持接触；因此正侧原已检出事件丢失仍是身体任务损失。当前证据定位了带误差公共query与身体任务之间的几何错位，尚不能把响应差异归因于网络、rho或某个单一机制。

同一像素/bin观察在ideal与+3°分支没有重采，但公共query改变会影响membership、gated证据及compact几何通道。不能从完整目标离当前query推断全部历史/FOV/光子无信息，也不能把“query支持”当成完整模型输入。实际保存raw、last5平滑和各模型自己的冻结阈值仍是阈值跨越依据；重放只验证来源，不重新评价收益。

## 冻结抽样、事件与帧数

固定seed956、独立读出30%cal pass点，按weak_pass理想→+3°的rescue/loss/both_miss×高度×side，每组取按scene/replica排序的前两个不同场景。实际19个事件、12组、209个event-frame；两分支418个输入行、两模型836条保存响应。不是24个独立确认事件，也不是把418行当418个事件。seed956来自此前已消费最大HEAD损失，是目的性诊断抽样，不代表全seed或新确认。

| weak事件组 | 候选事件 | 不同场景 | 本轮选中 | 缺少 | 状态 |
| --- | --- | --- | --- | --- | --- |
| rescue/HEAD/negative_x | 4 | 2 | 2 | 0 | COMPLETE |
| rescue/HEAD/positive_x | 5 | 2 | 2 | 0 | COMPLETE |
| rescue/BODY/negative_x | 17 | 8 | 2 | 0 | COMPLETE |
| rescue/BODY/positive_x | 0 | 0 | 0 | 2 | EMPTY |
| loss/HEAD/negative_x | 0 | 0 | 0 | 2 | EMPTY |
| loss/HEAD/positive_x | 25 | 8 | 2 | 0 | COMPLETE |
| loss/BODY/negative_x | 1 | 1 | 1 | 1 | SHORTAGE |
| loss/BODY/positive_x | 21 | 9 | 2 | 0 | COMPLETE |
| both_miss/HEAD/negative_x | 13 | 4 | 2 | 0 | COMPLETE |
| both_miss/HEAD/positive_x | 27 | 11 | 2 | 0 | COMPLETE |
| both_miss/BODY/negative_x | 22 | 8 | 2 | 0 | COMPLETE |
| both_miss/BODY/positive_x | 20 | 6 | 2 | 0 | COMPLETE |

两个空组（negative HEAD loss、positive BODY rescue）及negative BODY loss仅一个不同场景全部保留，不补例子。整体strata另外覆盖全部1 cm HEAD/BODY各128事件，抽样机制图不替代这些分母。

## 逐帧科学图与所有选例链表

[逐帧机制图](../../../../artifacts.local/work/cnh-pass-mechanism-chain-dev-20261009/mechanism_timeline.png)包含：positive HEAD loss、negative HEAD rescue、negative HEAD both_miss三个预定种类的首例，f3–13每帧；ideal/+3°对照。同事件control与weak_pass的实际saved raw／last5及各自阈值同时显示。

第一行是原目标六个面的总面积被query裁剪后的份额，实线±0.3 m接触核、虚线±0.4 m旁带；不是可见表面、实体交集体积或交集新边界面积。独立SAT实体检查另存，包含query在目标内部时“目标面面积0但实体交集非空”的fixture。第二行是当前所有zone/bin软membership之和，表示公共query假设的全域几何量；全域量近似不变不能说明目标所在zone的支持不变。第三行是当前与有效历史每帧平均的gated正signed-log统计，含背景/噪声，不是完整signed输入、目标专属光子量或检测概率。第四行是已保存两模型响应，不能跨模型logit尺度作因果比较。

[保留16-bin身份的实际compact变化图](../../../../artifacts.local/work/cnh-pass-mechanism-chain-dev-20261009/bin_delta.png)另列同三个例子f3–13，读取实际FP16 channel2（membership-gated history mean），转float64后各bin对zone求和，画+3−ideal。它避免只用全域总量掩盖bin变化，但zone求和仍会丢失zone身份；完整NPZ保留zone/bin，不把这些变化单独裁决为因果机制。

| ID | scene/K | 高度 | side | rho | weak跨越 | 面交集帧ideal/+3 /11 | control跨越 | weak峰值margin ideal/+3 | 全域当前membership总量区间ideal /+3 | 当前gated正证据区间ideal /+3 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 1/0 | HEAD | negative_x | 0.25 | rescue | 11/11 | both_miss | -2.608/+1.232 | 261.927…261.927 / 266.502…266.502 | 63.683…77.390 / 67.744…82.056 |
| 1 | 192/1 | HEAD | negative_x | 0.25 | rescue | 11/11 | keep | -0.276/+4.857 | 261.927…261.927 / 266.502…266.502 | 64.235…84.046 / 69.816…90.650 |
| 2 | 25/2 | HEAD | positive_x | 0.65 | rescue | 11/0 | rescue | -1.271/+1.498 | 261.927…261.927 / 266.502…266.502 | 62.992…75.822 / 64.433…77.949 |
| 3 | 121/1 | HEAD | positive_x | 0.25 | rescue | 11/0 | rescue | -2.428/+0.704 | 261.927…261.927 / 266.502…266.502 | 60.216…80.544 / 60.995…80.675 |
| 4 | 49/0 | BODY | negative_x | 0.65 | rescue | 11/11 | rescue | -1.055/+1.210 | 80.520…80.520 / 83.198…83.198 | 17.039…26.315 / 19.236…28.707 |
| 5 | 50/1 | BODY | negative_x | 0.65 | rescue | 11/11 | both_miss | -0.430/+0.215 | 80.520…80.520 / 83.198…83.198 | 17.032…26.760 / 16.977…28.458 |
| 6 | 122/1 | HEAD | positive_x | 0.25 | loss | 11/0 | loss | +2.222/-2.175 | 261.927…261.927 / 266.502…266.502 | 67.876…84.102 / 66.027…81.055 |
| 7 | 123/2 | HEAD | positive_x | 0.25 | loss | 11/0 | both_miss | +0.505/-3.284 | 261.927…261.927 / 266.502…266.502 | 64.358…79.298 / 63.800…80.504 |
| 8 | 336/1 | BODY | negative_x | 0.25 | loss | 11/11 | both_miss | +0.350/-0.047 | 80.520…80.520 / 83.198…83.198 | 19.547…25.237 / 19.431…27.593 |
| 9 | 169/3 | BODY | positive_x | 0.65 | loss | 11/0 | loss | +3.953/-0.005 | 80.520…80.520 / 83.198…83.198 | 17.861…27.706 / 17.726…27.700 |
| 10 | 170/1 | BODY | positive_x | 0.65 | loss | 11/0 | loss | +3.205/-0.569 | 80.520…80.520 / 83.198…83.198 | 16.545…26.105 / 17.386…26.526 |
| 11 | 0/0 | HEAD | negative_x | 0.25 | both_miss | 11/11 | both_miss | -3.221/-2.594 | 261.927…261.927 / 266.502…266.502 | 62.235…78.751 / 64.085…79.501 |
| 12 | 1/1 | HEAD | negative_x | 0.25 | both_miss | 11/11 | both_miss | -3.916/-0.754 | 261.927…261.927 / 266.502…266.502 | 60.428…89.084 / 62.566…89.987 |
| 13 | 24/0 | HEAD | positive_x | 0.65 | both_miss | 11/0 | both_miss | -0.672/-2.825 | 261.927…261.927 / 266.502…266.502 | 65.230…82.370 / 67.194…81.026 |
| 14 | 25/0 | HEAD | positive_x | 0.65 | both_miss | 11/0 | both_miss | -3.575/-0.538 | 261.927…261.927 / 266.502…266.502 | 59.717…78.121 / 62.224…83.603 |
| 15 | 48/1 | BODY | negative_x | 0.65 | both_miss | 11/11 | both_miss | -2.901/-1.978 | 80.520…80.520 / 83.198…83.198 | 17.922…24.559 / 19.111…24.856 |
| 16 | 50/0 | BODY | negative_x | 0.65 | both_miss | 11/11 | both_miss | -1.405/-0.620 | 80.520…80.520 / 83.198…83.198 | 17.575…25.950 / 18.077…27.898 |
| 17 | 72/0 | BODY | positive_x | 0.25 | both_miss | 11/0 | both_miss | -2.689/-2.496 | 80.520…80.520 / 83.198…83.198 | 13.572…25.167 / 16.587…24.594 |
| 18 | 73/0 | BODY | positive_x | 0.25 | both_miss | 11/0 | both_miss | -3.329/-0.123 | 80.520…80.520 / 83.198…83.198 | 15.683…25.441 / 18.198…25.928 |

反例也保留：唯一negative BODY loss为ID8、scene336/K1。两分支11/11帧仍有contact面/实体交集，weak保存峰值margin却由+0.349697变为-0.047145；实际FP16 gated-history channel2逐bin/zone的+3−ideal绝对差L1在f3–13为2.551812…4.092519。完整box几何仍相交并不保证跨阈值；这是一个混杂选例，不能据此归因为单个通道或宣称高度机制。

全选例中17/19个事件的全域history_sum gated-positive在f3–13的均值增加，包含全部6个positive HEAD选例（含loss ID6/7）。这不是目标专属增益，signed正负抵消及bin/zone分配也可能改变；不能建立‘gated正量多就检出’的单调机制。以下两个固定f10反例从原CSV读取，历史signed差按相同有效历史长度取每帧平均。

| ID | 选例 | 当前gated signed Δ(+3−ideal) | 历史每帧gated signed Δ | weak同阈值margin ideal→+3 |
| --- | --- | --- | --- | --- |
| 1 | HEAD negative_x rescue | -0.626238 | -0.885098 | -1.480019→+0.255379 |
| 18 | BODY positive_x both_miss | +4.425850 | +1.716458 | -4.106813→-2.829411 |

## 完整目标几何与输入来源

全1 cm cohort使用64个物理场景×11帧×两分支=1408个几何行，不按K噪声重复计算目标几何。HEAD与BODY共享横向query边界±0.3 m及公共变换，本静态fixture的Ry(+3)横向变化不依赖高度y；两者y范围仍不同。这不是实物朝向/时间合同认证，也未建立height-specific横向transform故障。

+3°正侧在f10完整目标最小query x约0.36549 m，已经越过接触核0.3 m；扩大到0.4 m旁带后正侧每场景8/11帧有交集、负侧11/11帧。旁带是潜在context证据，不自动成为接触标签、检出收益或低成本策略。裁剪保留完整八角点、六面与裁剪多边形，避免用旋转后world AABB或单个近面代替完整目标。

原past8历史、valid length/padding与每帧输运重建实际FP16 compact：shape=[418, 2, 16, 9, 8, 8]。保留9通道，包括当前signed-log、历史平均、membership-gated历史平均、当前membership mean/min/max、outside-distance min/max及radial center；六个signed-face通道恢复为精确0。冻结checkpoint重放1672个query分数，max abs差0，容差1e-05；原saved分数保持权威。这只能核对本次输入来源，不证明某个通道造成损失。

## side×rho×shape×背景的可比性

完整Cartesian strata表256行：128个observed、128个empty。固定高度／rho／shape／背景实例后，64个side可比性cell中0个同时有两侧。不是全局看到rho=.25/.65就已控制反射率；例如HEAD horizontal负侧.25、正侧.65，而HEAD vertical反向，BODY又随group翻转。该条件混杂同时阻止简单side或height因果解释；也不能据此直接认定rho是原因。

原始strata保留每cell事件暴露、不同场景、两分支检出、阈值margin、救／损／均漏和空cell；两侧的全局rho集合重合与条件cell无重合必须分开报告。当前同rho控制缺失是可比性证据缺口，不是停止可逆机制探索的门槛。

## 推荐下一步（仅proposal）

**先固定query／pose方向不确定性合同，用现有冻结模型和校准预算评价明确的身体走廊服务方向。** 让原身体接触truth、估计query方向与其误差范围分别记录，明确literal query与要服务的身体走廊的关系；核外但可能因方向误差相关的context不应直接被删除或改成身体负例。潜在收益是识别读出在什么已声明方向条件下能保住正侧边界事件，并同时量化pass／clear代价；原弱pass候选保持，其监督仍可用于权衡。

最薄弱假设是方向误差范围有实际依据、额外背景／pass成本可接受；ideal方向只能作仿真参照，模拟±3°界不认证实机。最小下一零训练检查可在同一固定hist和事件身份上，明确以身体走廊为目标，在声明的方向误差范围内作多query读出／区间汇总；固定现有读出与cal成本预算，并列报告1 cm救／损以及clear／pass clip代价。保留完整bin/zone，再以少量局部响应辅助定位；不从全域三标量造门控，不把错位query核外目标改成身体负标签。该方向／不确定性检查和所需新分支回放为NOT_RUN提议，本轮没有修复方向、扩大模型输入、训练、换阈值、增加−3°或重启旧扰动配方，不设零损失门槛。

## 核验与范围

M3、原5格融合、L2、旧阈值/checkpoint/480及旧停止结果保留。本轮没有新光子、观察、训练、硬件或保护480访问；CUDA仅重建既有compact及冻结重放来源。有限AABB、人工+3°、相关K/帧与目的性选例均为已消费模拟Development；没有−3°、实机、导航或安全结论。

- `selection.json`／`selected_events.csv`：19事件与完整空/短组。
- `geometry/per_frame_geometry.csv`／`geometry_summary.json`：全cohort完整表面、实体、角点与裁剪多边形。
- `projection/support_sums.csv`／`compact_inputs.npz`／`saved_responses.csv`：逐时间/bin输入身份、当前/历史统计与保存响应。
- `strata.csv`／`strata_events.csv`／`side_comparability.csv`：全cohort、空cell和可比性暴露。

`audit_receipt.json`：{"status": "PASS", "failures": ["AssertionError()", "AssertionError()"], "selected_events": 19}。

独立NumPy compact复算429/7704576个标量非逐bit相同，最大差0.000244140625，在2ULP(plus2e-6 absolute floor), independent NumPy FP32 math vs CUDA记录容差内；这与冻结CUDA重放1672个分数差精确0是不同核验。原20输入hash不变。audit两次字段/计数断言失败及geometry fixture修复记录保留，不覆盖失败；最终PASS见收据。


## 预算、复算入口与保留资源

冻结PLAN基线为66dc38a2；文档集成时HEAD为183358af，同期RGB当前段原样保留。记录CPU阶段累计22.961/300 s（含选择解释修复、几何fixture失败、两次audit断言失败及报告语法失败），CUDA命令阶段14.047/180 s；各阶段与单位见[budget_receipt.json](../../../../artifacts.local/work/cnh-pass-mechanism-chain-dev-20261009/budget_receipt.json)。这不是完整任务耗时，编辑、交互查看与Git/文档交付开销另属管理时间；GPU阶段含启动/I/O/CPU，不重复计入CPU。

三份可执行来源：[选择与分层](cnh_pass_mechanism_selection_dev.py)、[完整目标几何](cnh_pass_mechanism_geometry_dev.py)、[投影准备与同输入冻结重放](cnh_pass_mechanism_projection_dev.py)。对应执行SHA已核对并统一留在source_snapshot/manifest.json，旧输入20项SHA未变。独立audit与report构建来源、失败/修正收据也留在同一payload目录。阶段按prepare/run区分，原scores仍为报警依据，不把来源重放作为新性能实验。

[独立审计](../../../../artifacts.local/work/cnh-pass-mechanism-chain-dev-20261009/audit_receipt.json)与两张图均留存。CUDA进程和张量已释放，无常驻task worker；约8.32 MB必要NPZ作为本轮持久诊断证据保留在canonical artifacts.local junction中，未重建全量旧feature缓存。
