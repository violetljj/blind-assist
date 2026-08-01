# HFTF Stage C D5-S0A.1 path-token catalog execution contract

本合同只授权在提交、推送和双独立终审后执行一次新的控制面目录盘点。它使用新
canonical root，重新 fetch 官方工具仓库精确提交，不读取、复制或复用失败的 S0A
toolkit/manifest，也不把本次执行称为 S0A resume 或 retry。

每个非空 UTF-8 清单行只取首个 ASCII-whitespace token 作为 path identity。其余
suffix token 无条件丢弃：不要求存在、不验证、不保留、不生成指标、不参与任何门，
也不在 catalog 中保留完整 manifest 的 byte count/hash；不得对第 978 行特判。
目录身份仍只来自 manifest 实际列出的
`environment/Data_diff/P1xxx/<archive>.zip`。

正式执行必须在首个新 Git 网络请求前 durable 写入 attempt/preflight，只进行一次带
`--no-tags --depth=1 --recurse-submodules=no` 的 exact-commit fetch；只读
`.gitmodules`、download manifest 两个 blob 与三个精确 gitlink tree entries。
数据托管端、ZIP、submodule 内容、pose、scene payload、structural authority、
opportunity/effect/student 全部禁止。

四 archive、64 trajectories / 8 environments 仍只是 capacity/coverage 门，同环境
轨迹仍是 cluster。成功终点也只能要求另冻 S0B 合同；不会自动授权 S0B、payload、
source feasibility、主线/App/Android、生产或 safety。
