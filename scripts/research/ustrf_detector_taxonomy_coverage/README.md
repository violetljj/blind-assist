# detector_taxonomy_coverage_v1

状态：active

## 稳定 Interface

`validate_manifest.py --config configs/ustrf_detector_taxonomy_coverage_v1.json` 先重算父协议、窗口、模型、labels 与 Android 实现哈希，并拒绝阈值、tensor contract、person index、候选清单或阶段锁漂移。`run_host_coverage.py` 只在冻结 4,594 帧上生成独立 host input/raw-output/decode ledger；它按 `[1,84,2100] = channels-first` 解码，不改写历史 tracker ledger。

## 输出

所有运行输出只写入 `artifacts.local/evidence/ustrf-detector-taxonomy-coverage-v1/`。逐帧账本记录源图、输入 tensor、原始输出的 SHA-256，以及阈值前 winner、正确 COCO 映射和 class-wise NMS 后检测；不得只保存 person count 后声称 taxonomy 归因。

## 安全边界

本 Module 仅为 benchmark-only detector 诊断。旧 0-person 结果已发现 host layout 错位，在 Android/host 全量 parity 闭合前不得称为 taxonomy/domain shift。当前同源负窗口不是 person-absent truth，只能承担冻结 route-event 负窗，不能冒充逐框 detector false-positive truth。训练、App、生产、T0–T3 与 H2 均保持关闭。

## 停止条件

任一输入/实现哈希漂移、帧数不是 4,594、非有限 tensor、Android/host input 不逐像素一致、raw output 超出预注册容差、受控 person canary 的 tensor/index/labels/NMS 失败，均立即关闭 taxonomy 归因与候选比较。只有 G1/G2 通过而冻结 baseline 的两来源覆盖门失败，才允许一次性比较 manifest 中预注册的少量候选；不得通过 `.35`、NMS、route、事件门或 tracker 回救。
