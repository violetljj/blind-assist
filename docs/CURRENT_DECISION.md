# 当前研究决定

更新：2026-10-09。主线为盲杖互补前视感知，M3和A+LOCAL保留；原5格局部融合和L2保留。

RGB独立子线已获用户“推进”授权：[真实身体空间查询](../research/active/dtr-r0/nearfield/RGB_BODY_QUERY_DEV_20261009.md)。真实评价为硬目标，补充ToF/CNH主要模拟证据；50真实帧诊断、10帧20次Depth Pro推理及相机局部query读出完成，45连续真实帧配套已取得。保留图像细节有支持，但效果未评，身体外参/事件/pose参考待核验。继承旧BodyQuery并对照COPILOT，候选增量为可变米制query＋薄结构高分辨率因果证据＋真实事件定量。下一完成真实query标签与VDA因果基线；SANPO重新纳入该子线候选，CNH原预算、停止规则及M3保留。

## 当前：简化bin-token有效，先保留互补候选

[新表示试点](../research/active/dtr-r0/nearfield/CNH_DELAYED_QUERY_DEV_20261009.md)复用训练39936行及492场景×K4，16-bin在学习交互后池化。8833名义参数的去signed-face简化臂HEAD578/BODY510各/688，clear46/4576；相对M3救77/117、损16/13。完整signed-face−简化臂净−26/−14（救9/8、损35/22）；本轮不支持signed-face或extent增量。旧12/64轮负结果完整保留。

优先保留“旧5格融合＋简化臂”互补候选：H562/B481各/688，相对旧融合救19/20、损0；暗4cm横杆H12/B7各/56保留，clear仍46格、25段、23clip。但pass74→101/352；+3°沿用ideal阈值clear旧融合175→191、M3为176格，H514/B422，不能称扰动下同成本。48个训练外同生成器单位macro AUC .839265低于M3 .868327，未建立泛化优势。

同批clear选阈值、结果后选机制/组合，均为已消费Development。v1 pair只对同观测HEAD/BODY query作contrast，不是换背景或移除障碍的完整反事实；角区是空间假设，不是重建真值。本轮不追加当前配方训练；下一优先train-only反事实配对及不同背景、擦边和迁移检查，新机制开放。原192输入编码/池化响应仍[NOT_RUN辅诊](../research/active/dtr-r0/nearfield/CNH_VERTICAL_NATIVE_INPUT_DEV_20261009.md)，不是推进前置。

GPU成功阶段397.609s，加两次pre-GPU启动失败5.7036556s，累计预算403.313/1200s；CPU准备/分析预算600s。11项focused tests及独立全账本/AUC核验PASS。16个衍生feature共20.457GB已保存SHA manifest后清理；raw、模型、输入、ledger及失败留存，无新光子、硬件或保护480访问。

## 停止与保留

[旧扰动训练](../research/active/dtr-r0/nearfield/CNH_QUERY_PERTURB_TRAIN_DEV_20261008.md)、T/T2、旧参照和门控的停止规则仍按各自run适用；不同机制开放。[480确认](../research/active/dtr-r0/nearfield/CNH_REAL_HEAD_CONFIRM_RESULTS_20261007.md)原身份保留，UNKNOWN/三态/query覆盖待验证。方向/Nymeria位移、−19°、设备/City/保护test/新UE及硬件第二阶段暂缓；101/101与53ms属A+LOCAL，M3/CNH实机效果未建立。

模拟AABB、有限背景、人工误差、相关K/帧及探索选择不构成实机、安全或独立确认。更新前当前页全文保留于Git 41d4baf3同路径。

[路线当前页](../research/active/dtr-r0/CURRENT.md) · [RUNS](../research/active/dtr-r0/RUNS.md)。
