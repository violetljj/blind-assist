# detector_taxonomy_coverage_v1

状态：target attribution R1 complete / baseline hard gate pass / T0-T3 complete / shadow gate fail / H2 closed

## 稳定 Interface

`validate_manifest.py --config configs/ustrf_detector_taxonomy_coverage_v1.json` 先重算父协议、窗口、模型、labels 与 Android 实现哈希。R1 后续用 `run_host_canonical_coverage.py` 直接消费 Android Canvas RGB tensor，`finalize_target_truth.py` 在 baseline target association 隐藏时冻结 target/negative truth，`evaluate_target_attribution.py` 解封同一 baseline，`run_association_only_r1.py` 固定 detection/route/event kernel 重跑 T0–T3。

## 输出

历史 taxonomy 输出保留在 `artifacts.local/evidence/ustrf-detector-taxonomy-coverage-v1/`；target attribution R1 输出写入 `artifacts.local/evidence/ustrf-detector-target-attribution-r1/`。逐帧账本绑定 canonical tensor/raw stream、冻结 truth、target attribution 与 association-only 结果。

## 安全边界

本 Module 仅为 benchmark-only。R1 已补齐逐帧负窗 all-person/confirmed-absent truth，旧 first-fit 窗口不再冒充 FP truth。baseline target coverage 硬门通过后 detector 候选已停止；T0–T3 虽重开并完成，但 shadow gate 失败，所以 App、production shadow、训练与 H2 均未授权。

## 停止条件

任一绑定哈希漂移、帧数不是 4,594、canonical input 或 detection identity 语义 parity 失败、truth lifecycle/negative frame 不完整，均 fail closed。baseline 两来源 coverage 已通过，禁止再开 detector 候选。association-only 的 `14/15`、高负例误提醒与 repeat 失败不得用 `.35`、NMS、route、事件门、TTC 或深度回救；H1 稳定前 H2 保持关闭。
