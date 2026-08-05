# DA V2 选择性 W8A16 A5S R1

日期：2026-08-05

R0 在读取模型前因 Windows converter 不接受 `SM8650` SoC 字符串而结束，未生成 DLC。
R1 仅删除该 host 参数，恢复项目已经验证过的部署顺序：host 生成 generic HTP DLC，质量通过后
才在 SM8650/V75 真机生成 cached context。

48 个静态 INT8 权重、所有 FP16 精度边界、无 activation calibration、P1 R1 14/14 门、
`1.15x` full-chain speedup 和 `8 Hz` 门全部不变。generic DLC 必须先通过 precision inspection；
它本身不证明真机支持。
