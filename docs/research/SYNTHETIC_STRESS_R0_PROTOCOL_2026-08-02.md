# Controlled synthetic stress R0

状态：`FROZEN / THESIS_DEVELOPMENT / SYNTHETIC_ONLY`

## 目的与边界

本合同用于在进入任何新的 D2 media/mechanics 或 RCLE successor 之前，隔离检查
运动、成像退化、时间采样、坐标方向和字段 transport 的实现边界。它只产生
synthetic mechanism / implementation diagnostic，不打开自然数据，不消费 official
source，不产生人体事件、student utility、Android、默认 App、production 或 safety
权限。

所有 `UNKNOWN` 都保留为 `known=false, clearance_m=null`；不把未知回填成 SAFE。
TTC 只实现框/投影尺寸的 proxy，不能解释为米制 TTC、closest approach 或碰撞概率。

## 设计

使用明确的非笛卡尔设计：

- 完整 motion catalog：yaw/pitch/roll 正负与多角速度、纯前向/横向/竖直平移、
  rotation+translation、正面接近/远离、横向经过、scale 正负、camera shake；
- one-factor sweep：low texture、motion blur、rolling-shutter proxy、shadow、
  occlusion、depth discontinuity、5/10/20/30/60 FPS、timestamp jitter、缺帧、
  compression quality；
- 明确的 pairwise stress packs：成像×时间、遮挡×缺帧、深度断层×阴影等；
- 独立 boundary canary：known `4/5/6`、strict negative `-0/0/+0`、RCLE sign
  band、distance-bin edges、UNKNOWN numeric mutation、正向平移、observed-dt
  normalization。

当前合同为 8 个 deterministic seed replicates，共 `4,448` 个 case：
`704 motion-catalog + 2,592 one-factor + 576 pairwise + 576 stress-pack`。
每个 case 以 parent event/scenario 为单位保留，不按 frame pooling 充当独立样本。

## 自动测试对象

1. RCLE：调用现有 local-affine expansion 与 rotation-compensation；paired point
   identity 真实施加 retention、blur、rolling-shutter proxy、texture/shadow/
   compression/depth-discontinuity degradation。scale 正负做 expansion sign；纯旋转
   做 raw/compensated rotation leakage。front approach 的全场 sign 不被当成可靠 truth，
   因为静态背景与目标混合会制造可解释的混杂。
2. TTC proxy：`1 / d(log projected_height)/dt`，只在正向增长时 evaluable；同时记录
   non-closing `NOT_EVALUABLE`，并固定 `proxy_is_physical_ttc=false`。
3. Field transport/D2：调用当前 G0 signed-clearance primitive 与 D2 predicted basis；
   对 persistence 与 history-only causal advection 比较 future synthetic field，保留
   common-known cell 分母、UNKNOWN 计数和 signed clearance MAE。candidate 计算先于
   truth 计算，future depth/pose 不参与 candidate。
4. 时间/坐标：D2 只把 5/20 FPS、13-frame normalized timeline、exact required indices
   视为 eligible；其他 FPS、jitter 超过 exact mapping、required frame 缺失都保持
   `NOT_EVALUABLE`。正向 translation 的 basis/velocity sign 单独验证。

## 终点规则

本 R0 不定义“所有 stress 必须 PASS”的单一 promotion gate。任何 pooled median 都
不能覆盖最坏 cell、长尾 leakage、低 coverage 或 `NOT_EVALUABLE` 分母。若要改变
实现或阈值，必须另建 protocol version，并先冻结最小区分性实验。

机器可读合同与 runner：

- `scripts/research/synthetic_stress/protocol_r0.json`
- `scripts/research/synthetic_stress/run_stress_r0.py`
- `scripts/research/synthetic_stress/validate_stress_r0.py`
