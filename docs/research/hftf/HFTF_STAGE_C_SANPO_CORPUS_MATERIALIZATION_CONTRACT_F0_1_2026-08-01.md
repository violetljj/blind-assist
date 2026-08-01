# HFTF Stage C SANPO F0.1 corpus materialization contract

日期：2026-08-01

状态：`FROZEN_BEFORE_FIRST_F0_1_CELL_CORPUS_MATERIALIZATION`

## 1. 本步只物化什么

- 6 个 train sources：90 个 candidate-view anchor records；
- 3 个 dev sources：39 个 reference-view anchor records；
- official-test heldout：0 records。

student-visible `student_samples.jsonl` 每个 record 只含 5 张冻结历史 RGB 的 path/hash
和 current/future body/head nullable targets。UNKNOWN 的 risk 必须写为 `null`，
不能写数值 0。

SF arms 的 anchor×5 由 loader 从最后一张历史 RGB 派生，不在 corpus 中复制文件。

## 2. teacher receipt 与 student 输入隔离

future depth/mask/pose、anchor/future teacher identity、causal origin 与 teacher view
只能写入独立 `teacher_receipts.jsonl`。训练/评价 dataloader 禁止打开该文件。

student samples 不得出现 future RGB/depth/mask/pose path 或值，不得把 history/anchor
pose、known score 或 semantic class 当 student feature。

## 3. 完整验证

materializer 必须把 129 个 records 重聚合为 opportunity report 的 exact
source×horizon×height known/positive/negative counts，并验证：

- train/dev source、role、view 与 anchor count 精确；
- student/teacher receipt ID 一一对应；
- 645 个 history RGB references 的本地 bytes 与 SHA 精确；
- UNKNOWN 数值 SAFE target 为 0；
- heldout records 为 0；
- 第二个输出目录的所有 payload files 与第一个目录 byte exact。

只有 `F0_1_SANPO_TRAIN_DEV_CORPUS_READY` 才进入独立 corpus validator。即便成功，
student training 仍未授权；heldout target 与 heldout student output 继续封闭。
