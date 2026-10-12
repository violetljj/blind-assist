# signed bias / LOCAL合同修复 PLAN

2026-10-11；EXPLORE。继承193e4a21的已封存新骨干/S及所有切点，不训练，不读保护480/test。CPU科学计算/评价≤1200 command-wall s，GPU≤300 s；失败、启动与独立核验计费，达到上限停止。

## 合同与最小修复

明确两个输入入口，不能把已减ambient的模拟signed差分计数当成非负物理光子count：

1. `decoded_raw`（实机）：旧v2 signed模板B拆为Bphys=max(B,0)与Bstat=min(B,0)。Bphys仅为旧128已覆盖cell的非负背景代理，不是新测得的纯串扰；未覆盖保持0/NOT_EVALUABLE，不插补。扣除R=H−min(Bphys,max(H,0))：扣除不能超过已有非负decoded值，H=0与H=Bphys都得到0，负decoded保留。Bstat不加回任何分子；沿用v2拟合方差与数值floor作分母，只保留其统计含义。没有场景减法、拟合新参数或删新bin/zone。
2. `training_signed_z`（模拟）：沿用原(H−Bsim)/sqrt(16A+max(Bsim,0))及FP32→FP16运算，不做全局ReLU。Bsim本来非负，H为Poisson(E+8A)−Poisson(8A)，ambient在variance中，不在mean中。零raw或等于估计背景的整帧显式编码为零；其他帧保持原signed噪声数据。逐cell负signed残差是无正回波的统计表示，不是物理负光子数。
3. 所有入口在既有候选bin mask之后，若一帧没有任何正证据，将其native证据张量置零。几何、coverage及缺失标志保留为可用性/无证据表示，不算表面回波。若同一sequence/time-reset内实际最多8帧历史全部无正证据，则M3/原5格/S输出静默，同时保留未门控分数供诊断。精确`>0`存在性规则，不设幅度、比例、zone或报警分数门槛；历史内有任何正证据就按原链运行。无效/非有限输入不当clear，拒绝并标UNKNOWN。
4. 主臂只继承已授权的bin0 mask，旧候选无新mask。不改变M3/raised/ordinary/S/LOCAL cut。LOCAL仍用原几何投影patch范数与同一signed归一化合同；模拟单位噪声假设是近似（signal-dependent/相关噪声不包含在几何范数中）。v4真实背景扣除有截断，且decoded单位映射未识别，不能宣称真实LOCAL零均值/单位方差已校准。若发现需要新的LOCAL尺度或cut，本轮不从宿舍选点，只能用模拟cal原whole-tie规则，另留前后receipt；默认先保持原合同与cut。

物理扣除截断与完整signed噪声分布并不等价：在正背景很大时，0<H<Bphys的负残差被裁掉，可能抬高真实LOCAL均值。模拟入口不套用这个实机非线性；因此本任务是零证据/符号合同修复，不能称为完整sim→real噪声闭合。

## 事先固定的检查与复测

CPU不变量：零raw、等于Bphys、等于原signed B、纯ambient无噪声差分、纯串扰、负-only decoded、bin1–3负模板构造、真实正表面保留；所有zone/bin逐值检查，既有bin mask不可被重新读取。加入“正回波后零帧”的历史检查，避免把当前零帧错误当整段clear；sequence/time reset原样继承。

GPU完整链：旧冻结候选与193e4a21主候选，共同回放上述确定性构造及墙2bin0背景（各16帧，主分母满8历史9帧）。逐M3/LOCAL/ordinary/S检查零输入证据/无证据表示及强/纯轻计数，保留未门控分数。随机ambient噪声是随机输入而不是确定零证据，只作固定seed描述，不要求每个随机噪声样本严格静默。几何空模拟fixture不含地面/墙等表面；有地面的clear不是零输入。

宿舍十段1507帧/1362满窗：v4下新旧候选逐条件HEAD/BODY强/纯轻，继承v2旧表作历史对照。墙2/3、杆5/6、8HEAD仍提醒的存在性沿193e4a21（每段至少1满窗响应）；1阳台UNKNOWN，7外侧杆只描述；4/4b/9及4–9的尽头幕布约3m，边界场景不评价误报率，不参与任何选点。

同193e4a21的1536物理scene×K2已消费模拟hold：用原raw及同一冻结权重/cut，逐帧核对v4模拟入口与旧入口的表示，重新计算全链grade与gap1成本。可复用已封存scores，但须证明每个被复用输入/分支相同，另用固定首8scene fresh CUDA forward核对；任何输入变化须重算对应分数。该hold不重新选点，不称新确认。

## 判读与交付

共同旧原5格基准，原旧默认S收益G_old=170/1024，193e4a21新S=185；保持原门槛G_new≥0.8G_old=136且far+clear≤旧S168/1536+1pp（最多183/1536）。本轮“保持不变或很小变化”预定为新S及时总数相对193e4a21损失≤1%（最多10/1024），far+clear增加≤1pp（最多15/1536）。点估计工程门槛，不声称统计非劣。

必须主候选所有确定性无表面控制（含墙2bin0与负bias构造）的M3/原5格/S全链0强/0轻、零证据不变量PASS、模拟门槛和小变化门槛PASS，且墙/杆存在性保留，才按用户授权将去bin0+DR+新S提升为**模拟层面默认候选**；真实效果待独立楼道录制。否则保留旧默认并列来源。旧候选对非匹配模板的墙bin0仍可能触发，单列，不回溯改旧模型。

分母/头部空间已饱和，宿舍不要求报警下降；确定性每控制9满窗×2高度，1触发即FAIL，不用百分比掩盖。新hold及时HEAD/BODY各512，K相关副本不当独立人群。该新合同已受消费开发诊断驱动，只是可复现工程修复，无App或安全效果。

交付报告LOCAL_SIGNED_BIAS_CONTRACT_DEV_20261011.md、v4代码/单元测试、构造及真实表、模拟复算/默认manifest、独立核验、CURRENT/RUNS与hardware CURRENT，正常直推master。旧失败与193e4a21全链FAIL保持原样。
