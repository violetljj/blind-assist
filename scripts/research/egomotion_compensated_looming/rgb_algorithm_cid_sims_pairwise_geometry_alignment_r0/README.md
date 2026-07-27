# CID-SIMS RGB / pairwise geometry alignment R0

状态：development / frozen

## 研究问题与版本

协议 `RCLE_RGB_ALGORITHM_CID_SIMS_FLOOR3_1_PAIRWISE_GEOMETRY_ALIGNMENT_R0` 对齐
immutable RGB ledger 与 source-native CID-SIMS depth+pose radial geometry，
重点解释 window 0 的延迟连续触发段。

## 稳定 Interface

从仓库根目录以模块方式调用 `run.py`，显式提供 repo、contract、implementation
lock、activation、唯一 output directory 和 worker 数。producer 物化 598-row
ledger；validator 不导入本轮 producer，独立重建 pair、重算 geometry 与 aggregate。

## 输出

只写入
`artifacts.local/evidence/rcle_rgb_algorithm_cid_sims_floor3_1_pairwise_geometry_alignment_r0/`。

## 安全边界

RGB algorithm 不重跑；两个路径共享冻结 CID geometry helper，因此不是独立几何
实现确认。最大权限仅为 outcome-aware posthoc mechanism alignment，禁止阈值调整、
性能资格、Android/真人/产品安全结论。

## 停止条件与失败复用

锁、输入、pair identity、geometry、aggregate 或 authority mismatch 判 INVALID；
每窗 geometry coverage 低于 0.8 判 `NOT_EVALUABLE / VALID`。结果可作 diagnostic、
regression fixture 和后继 holdout 设计依据，不得包装为 confirmation。
