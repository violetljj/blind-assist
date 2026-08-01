# HFTF Stage C D5-S0A.1：opaque-suffix catalog repair 设计

## 结论

S0A 的 `INVALID` 是控制面语法失配，不是目录容量或科学结果。它没有生成
`catalog.json/result.json`，没有得到 parent/environment count，也没有打开数据 ZIP、
scene、pose、structural authority、opportunity 或 effect。因此允许保留同一个
TartanGround `Data_diff` source population，但只能另立 S0A.1：新版本合同、新
canonical root、新 attempt/preflight、重新独立 fetch 精确提交；绝不称为 S0A resume
或 retry。

本设计本身仍不授权 fetch、manifest read 或任何执行。下一步必须先实现并哈希绑定
S0A.1 planner/tests，再完成独立科学与工程终审并提交推送。

## 唯一 parser 修订

S0A.1 对每个非空 UTF-8 行只按 ASCII whitespace 取第一个 token 作为 path identity。
其余 token 全部直接丢弃：

- 不解释 size；
- 不要求 suffix 存在；
- 不验证数字、单位、正负或格式；
- 不保留 suffix，也不生成 suffix-derived metric；
- 不得针对失败的第 978 行或任何已见 suffix 写特殊规则。

path 自身仍须相对、安全、无重复。父体只能由 manifest 实际列出的
`environment/Data_diff/P1xxx/<archive>.zip` 产生；正则不能生成未列出的身份。

## 不变边界

目录完整仍要求 front image/depth/seg 与 metadata 四个 ZIP；至少 64 个轨迹父体、
8 个环境仍只表示 capacity/coverage。同环境轨迹仍是 cluster。成功也只能要求另冻
S0B structural-authority 合同，不能宣称 source feasibility。

旧 S0A root 保持不可变，不复制、不复用、不读取其中 toolkit 或 manifest。S0A.1
必须在新 root 重新单次 fetch 同一精确提交，并显式禁止 tags 与 submodule recursion。
数据托管端、ZIP、payload、S0B、ecology、effect、student、主线/App/Android、生产
和 safety 权限全部关闭。
