# PB-H1 role proxy

状态：`discovery`

## 研究问题与版本

`PB-H1-ROLE-PROXY` / `RCLE-PHASE-B-PROGRESSIVE-DISCOVERY-R0`：raw world
translation speed 是否把横向移动与真实接近错误地等同，而 pose+depth
translation-induced radial expansion 与 time-normalized parallax 能否恢复因果区分。
允许 claim 仅为 `DATA_CHARACTERIZED`。

## 稳定 Interface

`translation_induced_geometry()` 对前一帧点 `X` 计算
`X_r = R X` 与 `X_f = R X + t`。径向扩张为
`log(rho(X_f)/rho(X_r))/dt [s^-1]`，parallax 为两条单位 bearing 的夹角除以
`dt [rad/s]`，raw speed 为 `||t||/dt [m/s]`。径向中心为标定主点；半径小于
`8 px` 的点退出。

可见性固定为：source depth 有效、两个投影均为正深度且在图内、full
translation 投影中每个取整目标像素只保留最小深度。Bonn 用 `8 px` raster；
fixture 用固定全网格。pair 内报告 signed/absolute radial median、positive
fraction 和 parallax Q90；window 对各 pair summary 再取 median。

## 输出

runner 只写
`artifacts.local/evidence/rcle_pb_h1_role_proxy_r0/discovery_r0/` 的
`result.json` 与 `receipt.json`。

## 安全边界

fixture 是 synthetic physical calibration。Bonn 固定使用 B1A denominator 中的
第一个 window `rgbd_bonn_crowd2:0`，不按 PB-H1 输出择窗；它已烧掉且旧 ledger
来自 INVALID execution，只能作 diagnostic。这里不读 RCLE RGB algorithm
outcome，不下载数据，不形成 confirmation、人体、安全或产品权限。

## 停止条件

若纯旋转不归零、前向解析标定失败，或相同 raw speed 的横移/接近不能被 signed
radial coherence + parallax 区分，则停止 PB-H1，不审计 TUM。绝对径向扩张本身
不带方向，必须同时保留 signed median/positive fraction，不能用 absolute
aggregate 冒充 approach 判据。

## 假设、规则质疑与失败资产复用

因果差异是直接投影几何而非 world-speed proxy；信息增益是区分“真实 translation
污染”和“代理错杀”；falsifier 是受控物理标定或因果分离失败；成本限定为一个
fixture 与一个 burned window。失败代码/数据只可作 regression、counterexample
或 source characterization，不得包装为 unseen confirmation。
