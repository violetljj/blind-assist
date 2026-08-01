# HFTF Stage C D5-S0A.1 TartanGround catalog locked result

## 结论

S0A.1 已按冻结合同一次性关闭为
`D5_S0A1_TARTANGROUND_DIFF_CATALOG_LOCKED_REQUIRES_S0B_STRUCTURAL_AUTHORITY`。
精确提交清单中有 198 个 `Data_diff/P1xxx` 轨迹父体，覆盖 42 个环境；198 个父体
全部同时列出 `lcam_front` image/depth/seg 与 metadata 四个 ZIP path，超过预冻的
64 trajectories / 8 environments 目录门。

这只建立 catalog capacity 和 environment coverage。它没有打开任何数据 ZIP，也没有
检验动态 pose、robot height、robot-camera extrinsic 或 image/depth/seg/pose 共同
25-frame timeline，因此尚未建立 structural authority 或 source feasibility。

## 可复核证据

- 执行提交：`de088fb6be115769aaaaabeb1aed73d7ebc19002`
- 官方 toolkit commit：`158a6844d782942110967325ca3082f50ab2bfc7`
- manifest 非空行 / unique path token：`34671 / 34671`
- target diff archive paths：`7722`
- target / 四归档完整 parents：`198 / 198`
- 四归档完整 environments：`42`
- suffix token 未验证、未保留、未用于门；catalog 也不保留完整 manifest byte/hash
- canonical terminal validator：通过

收据 SHA-256：

- attempt: `5f6b2fe547b43df54e87da4c675df7bc3e02c0177f79b657cbbcfd94f33daf0c`
- preflight: `4a2d5fb59021df43f82ab71ab965db7febee603ffaf6520c435b9faf4186126d`
- catalog: `a8a4c33aa4f57cc6ffdf882f030cac3374e6b381c4aea2d36fd32bfba92c46f4`
- result: `10ab1e74d44753296c5dee58a3bd4bcdaa0c9f4e27cbe96ef59d59200f76cd73`

## 下一权限

当前只允许冻结新的 D5-S0B structural-authority execution contract。S0B 必须在执行前
绑定 provider URL/对象身份、ZIP central-directory bounded reads、exact metadata JSON、
以及只做 SHA/line-count 的 front pose member stream。若 height、extrinsic 或共同时间线
不能由该受限权威读取观察，终点必须是 `SOURCE_AUTHORITY_NOT_EVALUABLE`，不能伪装成
pool insufficient。

S0B 执行、数据 payload、ecology、effect、student、主线/App/Android、生产与 safety
均未自动授权。

机器结果 SHA-256：
`8b2aeb086dcdfd18a675d281a887dbea3cc63a23b2f3b7cac1bd375e613a4a2f`。
[机器结果 JSON](HFTF_STAGE_C_D5_S0A1_TARTANGROUND_CATALOG_LOCKED_RESULT_2026-08-02.json)。
