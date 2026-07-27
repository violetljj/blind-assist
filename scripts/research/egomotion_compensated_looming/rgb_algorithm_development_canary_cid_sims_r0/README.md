# CID-SIMS RGB algorithm development canary R0

状态：development / frozen

## 研究问题与版本

协议 `RCLE_RGB_ALGORITHM_DEVELOPMENT_CANARY_R0_CID_SIMS_FLOOR3_1` 只检查冻结
RGB algorithm 能否在本地 CID-SIMS `floor3_1` 的相邻弱运动/正向接近窗口产生
有区分度的输出。R0 已执行并保持 `INVALID_R0_EVIDENCE / INVALID`。

## 稳定 Interface

从仓库根目录以模块方式调用 `run.py`，显式提供 repo、contract、implementation
lock、activation、cache、output 和 worker 数。输入身份或独占输出冲突时 fail
closed。

## 输出

只写入 `artifacts.local/evidence/rcle_rgb_algorithm_development_canary_r0_cid_sims_floor3_1/`
与对应本地 cache。

## 安全边界

这是 real-data development canary，不是 confirmation、性能资格、Android 集成、
真人效果或产品安全证据。禁止用本模块调阈值或改写已消费 R0。

## 停止条件与失败复用

身份、锁、coverage 或 validator 失败即关闭该 evidence version。失败 ledger 可作
posthoc audit、diagnostic 和 regression fixture，不得包装为 unseen confirmation。
