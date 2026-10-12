# RGB真实细结构补标与有效负例（2026-10-09）

状态：EXPLORE / SPARSE_SUPPLEMENT_COMPLETE_FULL_QUERY_LABELS_PENDING。用户“补齐”承接[真实基线评价](RGB_BODY_QUERY_EVAL_DEV_20261009.md)，本轮补漏标可见细结构及非空角域负例，不重跑已停止Depth Pro预算。

后续：[跨环境实测参考与query训练](RGB_BODY_QUERY_CROSS_SESSION_DEV_20261009.md)已建立96帧/12环境监督/校准/验证接口，并保留两query模型弱于无RGB控制的结果。本文首批补标身份、停止规则与未补齐缺口不改写。

目标：现有公开真实train图像固定抽样，建立可复核的补充细结构空间标签、双目区间支持/UNKNOWN与非空负query；接入既有几何评价，保存补标前后和模型复核来源。模型/agent视觉复核不能写成人类人工真值，也不能把双目一致说成物理距离验证。

本轮预算独立于前阶段/CNH：CPU准备、图像/几何处理、评价与核验累计1800秒；GPU0秒，新增下载0字节，不新采集、不访问保护test、不训练。仅复用原50人工anchor/10session及45帧固定连续段。并行盘点/几何机会子项各240秒，失败计入总额；主线核验与实现使用剩余额度。调整范围为新增补充空间mask、可见核心/边界排除及显式object-conditioned评价单位；原6query、agreement、16像素门槛、旧结果保持。交付补标来源与覆盖表、有效正/负/未知格、现存预测复算、代码/可视化及下一决定。

固定每session首个人工anchor先核验场景覆盖，连续段七个人工anchor用于补充mask。图像核验不使用模型预测选样或画mask。细结构边界混合/遮挡/无双目可靠距离保留UNKNOWN。目标集合负与全场景安全结论分开；若只能得到物体条件负例，明确任务域，不把它当完整避障误报率。原停止run及CNH预算不变，payload经canonical `artifacts.local/work/rgb-body-query-label-completion-dev-20261009/`。

## 补出了什么

固定10个session首anchor实际看图；其中8帧补14条可见核心线，连续段396/402/408/414/420/426/432七帧补27条（20栏杆/7车架），合计15帧/9session/41条。第10场景相机朝脚下，不标前向clear。每条包含原生2208×1242坐标、1–5px采样核心宽度、RGB SHA、结构名、选择方式与来源；宽度是取样线宽，不是物体实际宽度。原图/叠图及坐标中间稿保留，叶遮挡处裁短，426不清楚横杆未采用。主agent查看固定contact sheet、七张最终core crop与八个anchor叠图/局部复核图后接入。

**新增标签全部为agent_visual_review，human_verified=false。** 原图恰为官方人工anchor，不能据此把新增线说成人工标注。它们是部分可见内部采样线，不是完整杆轮廓、物体身份或场景穷尽分割。源RGB不因模型错误被重新挑选，stereo与预测在坐标确定后才读。

reference720格中40条有空间支持，1条远标牌杆的1px原生线在nearest对齐后消失，保留UNKNOWN。逐trace累计5988格像素，旧类池覆盖1336；34条非空core完全不在旧类池。铁栏杆核心多为semantic0，施工栏杆为class8（旧池遗漏），U架部分被class26覆盖；数据集漏标、类别池遗漏与原图无目标分开记录。#3早先车架孔洞probe作废，精确核心探点的语义与旧粗probe不同，均保留坐标，不覆盖旧报告。

## 距离标签与有效负例

继承原6query、光学Z、half-pixel K、双目一致性及16正像素门槛。每条core独立求与query相交的射线域；≥16正像素为POSITIVE_CORE_INTRUSION，域≥16、没有正点或未解射线才为NEGATIVE_CORE_INTRUSION。空角域、少于16或仍有未解射线均UNKNOWN；整图未标注像素不变成负例。

41×6=246相关core-query格：**11正 / 2非空参考负 / 233UNKNOWN**。UNKNOWN中146为空角域，87为非空角域但参考或支持量不闭合。全部13个已知格来自同一连续session；其他8个session补出了图像结构，但未获得严格完整core-query距离标签。这补齐了首批稀疏标签，**没有补齐完整训练/验证真值集**。

两个非空负格来自420同一横杆core：center_near/right_near各258个双目一致盒外像素；同core的right_far有258正像素。这是同一结构的三种query支持，两个负格不是两个独立负事件。CRES/ZED仍为估计深度，不能把这些参考负升级为独立物理距离真值；细结构处共享背景深度错误的可能性尚未用独立距离测量排除。

把补充core并入旧目标域，在同7帧/42query重算：原8正/2空域负/32UNKNOWN → **14正/1空域负/27UNKNOWN**。新正为396/center_far、396/right_far、402/center_far、408/right_far、414/right_far、420/right_far。414/right_far由原空域负变正，其余5由UNKNOWN变正；这是扩大标注目标域，不改写原限定类池结果。仍无完整NEG→POS时间起点，事件提前量不评价。

## 已有预测复算

零新推理。9个有core的首anchor复用旧Depth Pro烟测；连续段前3人工anchor复用四臂Depth Pro缓存，7人工anchor复用VDA因果缓存。新增评估只消费已有RGB对应预测，mask/core只在评价端使用。烟测输出SHA在本轮重新记录，不宣称核验了跨轮未修改；连续臂沿用原manifest SHA。

| 相同core集/参考域 | 正core-query支持 | 非空负core-query误支持 | 正像素召回 / 负像素FPR |
| --- | ---: | ---: | --- |
| 连续前3帧raw Depth Pro原图 | 5/6 | 无负分母 | .6028 / .04536 |
| 连续前3帧raw128 | 5/6 | 无负分母 | .8276 / .11684 |
| 连续7帧VDA因果 | 5/11 | 0/2（同一结构） | .5558 / .000540 |

前3帧像素分母正725、负1455，raw原图TP437/FN288/FP66/TN1389，raw128为600/125/170/1285；不是同误报优势。VDA七帧正1317、负3704，732/585/2/3702；420横杆三格均正确支持或否定。首anchor9帧只有1个已知正core-query、没有非空负格，烟测native/low都支持该格，像素IoU .1214/.5520，不能以此建立跨场景细障碍优劣。

在增补后的完整旧类池角域，VDA相同7帧正query支持由4/8变8/14，像素IoU .4903→.4922。**模型输出没变，变化来自参考域与分母。** 该值只作为补标前后诊断，定位仍来自评价端掩码，不是独立RGB检出率。中间/原结果及每条数据身份均保留。

## 数据交付与下一决定

[生成/复算脚本](rgb_body_query_thin_labels.py)与[8项聚焦测试](test_rgb_body_query_thin_labels.py)可复用。`anchor_traces.json`与`geometry/reviewed_traces.json`保存坐标、图像身份及来源，SHA分别为`64e8fbd90cff6b3c08159a08aa6caf2122de5f8341942d20f05808ef99b07df7`、`e2a0c8c31ec9bd4db21b3aaf5d7c66058111c411feccc2c438b3931dee670350`。`labels/`提供41个core reference NPZ、15个增补场景reference NPZ及叠图；`labels_evaluation.json`含来源/覆盖/已知与未知/逐臂计数。`augmentation_comparison.json`、`augmented_geometry_evaluation.json`和生成脚本保存旧域对照；`dataset_manifest.json`路由全图RGB与稀疏空间监督，core不作为模型输入或GT选crop。payload不入Git。

下一任务是补**非单session的可靠距离参考与独立视觉核验**，再形成按session隔离的训练/验证。当前可用作EXPLORE稀疏空间监督；完整query二分类和独立误报集尚不具备，不能把233UNKNOWN改成负标签或把同一杆的多个query分到训练/验证两端。新增补标与全部已知距离格出自已消费Development，不作新确认。真实定量硬目标保持；未启动训练，不扩旧DP/CNH预算。

## 成本与核验

核心生成/评价56.882s；增加可复用增补reference文件后62.656s；固定7帧增补复算5.179s，三项均计CPU。图像盘点约10.15s，独立27core参考机会约35s，聚焦fixture含首次只读数组错误约8s；其余终端准备/检查未完整计时，以预留扣额合并保守记300s/1800s，扣额不作性能测量。GPU0s、下载0B，无任务模型/服务常驻。

首次fixture发现PIL转数组只读，改可写副本后8项通过；多扩展名`.float16.gz`、source SHA先核验、invalid预测不被插值填好、非空负16门槛及UNKNOWN继承通过。独立27core脚本复算11正/2负与主实现一致。源文件SHA、预测身份及reference格K核对；旧结果、只读失败和坐标修正均保留。文档/结构检查按本次修改面执行，无Android改动。
