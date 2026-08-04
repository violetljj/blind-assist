# HFTF 稠密／固定 cell 深度传播族停止决定

日期：2026-08-05

终态：`DENSE_OR_FIXED_CELL_DEPTH_PROPAGATION_FAMILY_STOP`

## 决定

R0 和 R1-A 已形成两级机制负证据：整帧 RGB freshness 以 `21.24%` coverage 换取
零 false-clear；运动补偿局部 cell 在 64,896 个 cell 上只有 macro `28.91%`、
worst-session `19.12%` support coverage。两者共同表明，第一视角移动相机中的像素或
固定二维 cell 不适合作为持续保存米制背景几何的稳定实体。

因此停止以下研究族：

- 用整帧 RGB 变化代理旧深度有效性；
- 稠密像素或固定图像 cell 的低频深度传播；
- 用二维光流 support 维持完整可通行场；
- 在已消费 R1-A 序列上更换网格、点数、光流或 residual 门进行救援。

本停止决定不关闭低频精确深度、异步双环、fresh-frame 米制净空、本项目已经建立的
原创 HFTF foot/body/head 人体扫掠包络、决策层周期米制锚定，或将少量米制状态绑定到经过独立
验证的稳定目标身份。

## 创新权属

foot/body/head、人体包络、方向/距离/高度张量是本项目已经建立的 HFTF 原创基础贡献；
R4 已经在 SANPO-Synthetic challenge cohort 和解析地形上支持其 teacher mechanics。
后继工作可以建立在该原创贡献之上，但不能在同一论文内部把它重复计算为第二项创新。
待证增量是：真实移动端 fresh metric
snapshot 能否让该继承表示在独立物理真值下产生可靠、可拒答的米制侵入证据。

## 后继顺序

1. fresh metric snapshot 的受控实拍三层侵入评价；
2. 只有快照效用成立，才评价决策层 periodic metric anchoring；
3. stable Track 米制锚点必须先补齐独立轨迹真值和身份权威；
4. NPU 主动调度最后进入。

本决定只改变研究路线，不改变默认 App、设备深度 Demo、报警、导航、生产或安全权限。
