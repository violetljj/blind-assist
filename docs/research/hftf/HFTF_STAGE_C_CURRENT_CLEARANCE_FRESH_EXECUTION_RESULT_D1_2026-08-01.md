# HFTF Stage C：current signed-clearance D1 fresh 执行结果

## 结论

D1 one-shot fresh 评估终态为：

`G0_D1_FRESH_EVALUATION_NOT_EVALUABLE_NO_SOURCE_REPLACEMENT`

这不是模型负结果，也不支持 signed-clearance。第一次固定 source acquisition 在 frame 0 depth 临时文件创建时因 Windows 路径长度触发传输失败；在此之前，source metadata、首帧 RGB 与首帧 mask 已经打开。因此根据预先推送的合同，不能把输出根改短后重跑，也不能换源或继续打开剩余两条 fresh source。

## 发生了什么

- pre-open 合同提交：`ab9a6cc5257bf20477a097d5aec6fe9cf2703874`，执行前与 `origin/master` 一致；
- 合同 SHA-256：`b13b27d0fd882ec7a9904c6e2dd595629e0b3ca093f9e238549e32fc3f655ae2`；
- 第一个固定 session：`15bc9dde…e02bf`；
- metadata、frame-0 RGB 与 mask 成功写入；
- frame-0 depth 的 `.float16.gz.tmp` 目标路径长度为 263 字符，三次内部下载尝试均返回同一 `FileNotFoundError`；
- acquisition 进程最终返回 `{"ok": false, ...}`；
- stdout SHA-256：`4b738c7cfd9e81058d7021210a49d1ad7a69db1099522182140f3eb9564cc7ee`。

263 字符失败路径与同根较短 sibling 文件成功写入共同构成“Windows 路径长度传输失败”的强推断。它没有产生完整 manifest、authority、teacher opportunity、student prediction 或 truth join。

## 证据边界

本次终态只说明正式 D1 fresh 执行不可评价：

- 不说明 signed-clearance 有效；
- 不说明 signed-clearance 无效；
- 不产生模型、风险、安全或生产证据；
- 不改变传统主线、默认 App 或 Android；
- 不允许继续补全 partial root、重试同一 D1、替换 source 或打开 reserved official-test。

partial 文件和 stdout/stderr 作为 consumed failure evidence 保留在 ignored `artifacts.local/`，不会删除，也不得恢复为 fresh D1。

## 后续边界

若继续 HFTF，必须先承认 D1 已关闭，再提出独立 successor：新问题、新合同、新数据角色边界，并在任何新 fresh source 打开前用 synthetic path-length canary 验证 transport。不能把 successor 写成对这次 D1 的路径修补或同 cohort 救援。
