# VI-Task Geometry G0

状态：`PREOUTCOME_PROTOCOL / ATOMS3R_M12_ONBOARD_BMI270_CONFIRMED / SYNCHRONIZED_RGB_IMU_CAPTURE_NOT_MATERIALIZED / REAL_G0_NOT_RUN / NO_TOF / NO_TRAINING / DEFAULT_APP_UNCHANGED`

## 稳定 Interface

- 物理源固定为同一 AtomS3R-M12 刚体上的 OV3660 RGB 与板载 BMI270；两者必须使用同一
  `esp32_boot_monotonic:*` 时钟域；
- camera stream 必须保留连续 `frame_sequence` ledger。传输或写入丢帧必须物化为
  `LOST_BEFORE_WRITER`，不得从分母中删除；
- BMI270 有效采样率必须为 100–400 Hz、最大 gap 20 ms；每个有效 camera frame 必须被 IMU
  样本包围，最近样本差不超过 5 ms；
- capture 不得包含 ToF/range 字段，不得使用手机 IMU替代眼镜刚体 IMU；
- camera intrinsics、IMU-to-camera extrinsics、clock validation、reference instrument 与
  capture/truth writer 全部在 outcome access 前哈希绑定。

## 输出

[`capture_contract.py`](capture_contract.py) 只验证 fresh physical RGB-IMU source 是否可进入
G0；[`evaluation.py`](evaluation.py) 实现 truth-isolated A0–A3 parent-macro/worst-parent evaluator，
并把 degeneracy unsafe-valid 与 observable coverage 分开。真实流、calibration、truth 与 validation
receipt 只能写入 `artifacts.local/evidence/vi-task-geometry-g0/`。当前没有合格 capture，所以
`REAL_G0_NOT_RUN`。

## 安全边界

G0 仅允许 A0 DepthART-only、A1 fixed-height、A2 VIO sparse ground 与 A3 A2-aligned
DepthART 四个冻结 comparator。不得用 opened outcome 调 VIO、换 parent、改门或让 truth 进入
candidate。G0 不训练、不运行主动预算策略、不接 Android 提醒，也不证明 clearance、安全、产品
或论文结果。

## 停止条件

A2/A3 若不能通过 metric-frame observability 绝对门，则关闭 VI metric task-geometry 路线，转向
height-free angular/TTC 表示。只有 G0 PASS 才可另立 task-geometry successor，且不自动授权执行。
