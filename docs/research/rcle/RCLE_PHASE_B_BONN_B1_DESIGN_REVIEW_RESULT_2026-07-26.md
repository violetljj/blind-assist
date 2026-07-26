# RCLE Phase B Bonn B1 设计审查结果

状态：`PASS_B1A_IMPLEMENTATION_ONLY`

日期：2026-07-26

## 结论

B1 R4 预注册与 machine design lock 完成独立只读复审，允许开始
`B1A_SOURCE_NATIVE_GEOMETRY_ADMISSION` 的实现与 fixture-only 测试。

本结果不授权 B1A canonical execution，不授权读取 ZIP、pose/depth payload，
不授权 B1B implementation/execution，也不授权 Phase C。

## 被审查对象

- preregistration SHA-256：
  `f3974b2c0096dae2334b1d6c8cd563d892b09288df4f2085604b8fee88d4cfd0`
- design lock SHA-256：
  `c53c9edaf7012df481b2ba286902af87f1716e3a5d4f57f27398303c4f74420e`
- design revision：`R5`

## 审查轨迹

1. 第一轮 `FAIL`：range truth、association、pose/source、claim/statistics
   等边界不唯一。
2. 第二轮 `FAIL`：B1A/B1B firewall、direct paired delta、decoder/median、
   support-manager 和 terminal 契约仍有分叉。
3. 第三轮 `FAIL`：active branch、manager orchestration、M2 enum 与
   angular trace 契约仍未完全机器化。
4. 第四轮 `FAIL`：只剩 B1A claim 永久保留/绝对路径、angular
   radians-to-threshold 转换、timing 零样本表示三个精确缺口。
5. R4 最终复审 `PASS`：上述缺口与相邻 machine/text parity 全部闭合。
6. 实现审查发现 Windows/Python 不能以 `os.open(directory)` 执行
   directory fsync；R5 只修 atomic publish 平台契约。Windows 固定为同目录
   temp file fsync 后 `MoveFileExW(REPLACE_EXISTING | WRITE_THROUGH)`，
   POSIX 保持 `os.replace` 后 directory fsync。R5 独立复审 `PASS`。

## 实现门

B1A implementation 必须：

- 逐字段消费 R5 machine lock；
- 只用 synthetic/fixture 测试覆盖 association、pose、geometry、truth、
  role、claim-first、firewall 与独立 validator；
- 在单独 implementation review `PASS` 前保持
  `canonical_execution_authorized=false`；
- implementation review 前不得读取 canonical ZIP 或 payload。
