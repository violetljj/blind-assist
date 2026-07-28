# BlindAssist 全项目审查与优化（2026-07-28）

状态：审查与授权范围内优化已收口；RCLE formal terminal、独立复核与 current 已于 2026-07-28 原子化对齐
范围：仓库治理、Android 架构/构建、研究主支线、文档与指令、脚本入口、本地产物和可恢复清理。
边界：RCLE worker 运行期间未读取 protected outcome 或改写 current；只有在 worker 退出、终态 validator 和最终独立复核均 PASS 后才收口 current。未改写组会稿、`DEVELOPMENT_LOG.md`、已消费 claim/lock 或 raw evidence/payload。

## 结论

项目没有出现需要推倒重来的整体失控：`app -> feature -> core` 的 Android 依赖方向仍清楚，核心模块无循环，当前声明的主要变体可构建。真正的维护风险来自“边界漂移”：

1. RCLE 动态状态曾被复制到多个外围文档且彼此过期，原 current 又混合多代阶段与冲突权限。
2. 已关闭的 USTRF shadow 仍默认进入正式 BASELINE 每帧热路径。
3. 仓库卫生门有 10 项真实失败，历史 consumed 引用与新增稳定接口没有分型。
4. 模型导出默认可覆盖正式 App asset，根目录还隐藏了约 44.8 MiB 模型转换 payload。
5. 根目录混放 current、archive 和实验流水；`idea.md` 已增长到约 170 KiB/920 行。
6. `artifacts.local/` 的实际 23 个顶层目录已超出六类 canonical 契约，不能再按目录名或扩展名盲清。
7. CI、Gradle wrapper、bundletool、Python 检查环境、依赖解析与 QNN 版本原有可复现性和覆盖缺口。

本轮优先修复会继续放大混乱的入口、门禁和默认行为；冻结研究证据保持原字节，未用“整理”破坏可追溯性。

## 已实施优化

### 1. 主线、支线与 authority

- [AGENTS.md](../AGENTS.md) 和 [文档治理](DOCUMENT_GOVERNANCE.md) 现在明确：RCLE 的阶段、终态、权限与下一步只由 [RCLE current](research/rcle/README.md) 持有；根 README、SANPO current、docs/scripts 索引只做 pointer。
- canonical current 若自相矛盾、把未跟踪材料作为 authority，或活动 handoff 正在治理复核，则禁止新 formal execution、claim 消费、protected-outcome access 与 successor authority，直到 current owner 原子收口。
- OpenLORIS/DLR worker 自然终态后，结果文档、`37/37` terminal validator 与最终 independent review 的 SHA 全部匹配；current 现唯一声明 `RGB_SEGMENT_CONFIRMATION_R1_NOT_EVALUABLE / VALID_FAIL_CLOSED_TERMINAL`。原 30 KiB 多阶段叙事保存在 [2026-07-27 历史快照](research/rcle/RCLE_CURRENT_SNAPSHOT_2026-07-27.md)，不再拥有动态 authority。
- 两个 claim 都已消费且禁止重试；0 eligible frame、0 decode、0 RGB call、0 alignment denominator 与全 null metrics 不得解释为算法成功或失败，也不授权 successor、performance、Android、product 或 safety。
- 同一主任务内允许主代理协调“路径完全不重叠”的 subagent；独立任务若需要同一文件，必须合并或使用独立 worktree，解决了此前“鼓励并行”与“同 worktree 禁止并行”的指令冲突。
- 根 [idea.md](../idea.md) 只保留四类待决方向；完整旧流水移入 [idea archive](history/idea/README.md)。已完成事项不再长期占据待决入口。

### 2. Android 运行时与构建边界

- 正式 `BASELINE` 不再构造或执行 USTRF shadow/experimental adapter；`USTRF_EXPERIMENT` 仍显式构造原 fail-closed adapter，并有模式构造回归测试。
- QNN `2.47.0` 集中到 version catalog，`app`、`npu-candidate`、`device-benchmark` 不再各自硬编码。
- Kotlin/Compose plugin 从 `2.0.21` 升至 `2.1.21`；10 个 Kotlin module 全部从已弃用 `kotlinOptions` 迁到 typed `compilerOptions`，Hilt module 启用 `kapt.correctErrorTypes`。K2 kapt 定向重编译通过，不再出现 language 2.0 回退到 1.9 的旧 warning。
- AGP 保持 `8.7.3`：实测降到矩阵表中的 `8.7.2` 虽可构建，却触发官方在 8.7.3 修复的 lint Analysis API 警告；最终组合完成全矩阵和离线复核。该 patch-level 例外被显式记录，不据此推断未来版本也兼容。
- Compose 的 `ArrowBack` / `VolumeUp` 改用 AutoMirrored icons，消除 RTL 不正确与弃用 warning。
- 移除已弃用 `android.targetVariant`；CI 增加 `ustrfExperiment`、NPU candidate 与 shadow benchmark 的构建/编译覆盖。
- Gradle 8.10.2 wrapper 增加官方 SHA256；bundletool 1.17.1 下载后校验 SHA256；Python LiteRT 检查环境固定版本、transitive wheels 与 hashes。
- 三个 GitHub-hosted Actions 都固定到已核对 tag 的 40 位 commit SHA；仓库卫生门会拒绝任何新增的非 SHA external Action。
- 所有 configuration 启用 `failOnNonReproducibleResolution()`；仓库卫生门同时拒绝 `+`、`latest.*`、`SNAPSHOT`、版本区间和 changing module，既有坐标审计均为精确版本。
- 真机全套 Compose instrumentation 首次运行暴露出两个“测试方法已完成、规则 teardown 不收敛”的问题：权限说明弹窗和大字体调试区都在测试结束时保留额外窗口/展开动画树。测试现模拟真实父状态关闭弹窗，并在结束前折叠调试区；生产 UI 未为测试而改写。
- 未引入大型 convention-plugin 或全仓 Gradle 重写；当前模块数量和变更收益不足以支撑那类迁移风险。

### 3. 结构与卫生门禁

- 8 个 USTRF consumed config 与 1 个 RCLE source-transport lock 保持原内容，改用 `path + SHA256 + reason` 的 immutable exception；缺路径、重复、空原因、非法 hash 或任何字节漂移均 fail closed。
- 这些 exception 及当前 RCLE R3/R4/segment 文件族配置定向 `text eol=lf`，避免 Windows clean checkout 改变 hash；未对全仓 JSON/Python 做危险 normalize。
- `generate_qnn_preprocess_candidate.py` 已登记为稳定候选生成器并进入脚本索引，输出只允许在 `artifacts.local/experiments/`，不产生发布或生产权限。
- 两个 REveL detector 历史诊断及其测试从 scripts 根下沉到已关闭的 `research/ustrf_detector_taxonomy_coverage/`；同目录 README 明确不重开 detector、RCLE、shadow、H2 或生产 authority。根 allowlist 从 185 降到 181。
- 仓库卫生门新增“即使被 `.gitignore` 忽略，模型/生成二进制也不得出现在仓库根”的检查和反例测试。
- 文档门从“只检查顶层 Markdown”扩展为同时要求每个 `docs/research/<domain>/` 具备 README 并由总索引链接；补齐 `assets` 与 `frontier-upgrade-2026-07` 入口。

### 4. 模型、工具与本地产物

- `export_yolo11n_tflite.py` 默认输入/输出改为 `artifacts.local/models/`；直接把 `--output` 指向 App assets 会被拒绝。
- 只有显式 `--promote-to-app` 才会在 tensor shape/dtype 静态验证后复制；staging 与 App destination 都必须和已验证 export 的 SHA256 相等。
- 根目录 `yolo11n.pt`、ONNX、calibration NPY 和 SavedModel 已移动到 `artifacts.local/models/`；逐文件大小/hash 保存在本地 `YOLO11N_LEGACY_SOURCE_MANIFEST_2026-07-28.md`。已提交的正式 TFLite asset 未改变。
- 两个设备 benchmark wrapper 已从不存在的 `.venv-export312` 和 `.gradle-local` 迁到 canonical Python/Gradle state，并把新输出改为 `artifacts.local/evidence/`。
- [本地产物契约](LOCAL_ARTIFACTS.md) 新增 canonical/legacy 分类、禁止盲清 `tmp/`、禁止按扩展名清 payload、以及大 payload 清理必须保留 URL/hash/manifest/receipt/ledger/script/result/cleanup record 的规则。
- 初次 `gradlew clean` 释放约 4.67 GiB；完整验证重新生成产物后再次移除 5,737,148,946 bytes（约 5.34 GiB）；真机回归重新生成的 777,063,097 bytes（741.07 MiB）又在保留 XML/runner 收据后清理。构建证据位于忽略目录，不依赖保留 build payload。
- `.gradle-local` 347.74 MiB 的最后一个非历史命令引用已改到 canonical Gradle state；`artifacts.local/tools` 719.72 MiB 的 CLIP/yt-dlp 重复副本也确认无 caller、无进程占用。精确路径、文件数、字节数、reparse point、进程引用和恢复收据复核后，两处共 23,152 个文件 / 1,119,313,061 bytes 已用 Windows 审核 fallback 删除，其他 artifact root 未触碰。

### 5. 文档分层

- 当前演示指南迁入 [docs/DEMO_GUIDE.md](DEMO_GUIDE.md)，版本更新为 `v10.9.0 / code 37`，不再复用旧“测试已通过”结论，并指向真实的两层 APK 归档。
- 旧更新计划、早期阶段回顾和 2026-05-19 真机报告移入 [project-materials archive](history/project-materials/README.md)。
- 当前 SANPO/评测文档中的 Python 命令改为 canonical `E:\codex-tools\bin\blindassist-python.cmd`；历史证据路径保持原样。
- 日志预算文档与机器规则统一为 `6000 行 / 1,200,000 bytes / 28 天`。当前 `DEVELOPMENT_LOG.md` 在预算内，本轮因其属于另一活动任务未追加或重排。

## 验证证据

| 检查 | 结果 |
| --- | --- |
| `AssistFrameProcessorTest` | 12/12 通过；baseline adapters 为空，experiment adapter 存在 |
| exporter stdlib tests | 5/5 通过；覆盖默认隔离、直接 App 输出拒绝、静态验证顺序和 SHA mismatch |
| Kotlin 2.1.21 K2 kapt 定向重编译 | `BUILD SUCCESSFUL`，91 tasks，耗时 5m09s；无 2.0 -> 1.9 fallback |
| Compose UI 定向回归 | `BUILD SUCCESSFUL`，17 tasks；AutoMirrored icon 迁移无 warning |
| App 真机 instrumentation | `SM-S9280 / Android 16 / API 36`，`:app:connectedDebugAndroidTest` 13/13 通过，0 failure / 0 error / 0 skipped；Gradle 160 tasks，耗时 1m35s |
| Android 最终 CI 等价矩阵 | `BUILD SUCCESSFUL`，474 actionable tasks（297 executed / 177 up-to-date），耗时 6m15s；单测、5 个 lint、APK/AAB/AndroidTest/实验/NPU/设备/shadow 全通过 |
| 暖缓存离线复核 | `BUILD SUCCESSFUL`，259 tasks（17 executed / 242 up-to-date），耗时 2m06s；App assemble/lint 与 NPU assemble 可离线解析 |
| 最终 build payload 清理 | 两轮最终 `clean` 均 `BUILD SUCCESSFUL`，10/10 tasks；完整矩阵后移除约 5.34 GiB，真机回归后再移除 741.07 MiB |
| 正式 TFLite 静态检查 | input/output shape、dtype assertions 通过；正式 asset 未改变 |
| `test_check_project_structure.ps1` | immutable exception 正例/hash drift/missing/duplicate 等 smoke 通过 |
| `test_repo_hygiene.ps1` | 通过；覆盖 ignored-root-model、unpinned Action、动态/范围/SNAPSHOT/changing dependency 反例 |
| `check_repo_hygiene.ps1` | 通过：181 个 scripts root 文件、9 个 research module、日志预算正常 |
| REveL 历史诊断迁移回归 | 5/5 tests 通过；迁移后仍按同目录加载 Implementation |
| `test_check_docs_index.ps1` / `check_docs_index.ps1` | 通过：46 个顶层文档、4 个 research domain |
| 两个 benchmark PowerShell wrapper | AST parse 通过；canonical Python/Gradle 路径现场存在 |
| `gradlew clean` | 10 个模块 clean task 通过 |

## 未越权吸收的共享工作

以下内容仍属于共享工作树中的其他记录，本轮没有因 RCLE current 收口而一并吸收：

- `DEVELOPMENT_LOG.md`
- `docs/research/GROUP_MEETING_PROGRESS.md` 与未跟踪组会报告
- 与本次 terminal 无关的未跟踪 RCLE R1/R2/R3/R4 discovery/contracts/results
- `rgb_segment_confirmation_r1/` 的 frozen implementation、tests 与 evidence/cache；只核对最终 terminal/review bindings，不改写实现或 raw payload
- 活动 handoff/INDEX 的其他任务行

任何交付都必须保持这些文件的独立所有权，不能因本审查的广范围而一并暂存、提交或删除。

## 已关闭的关键风险

| 原风险 | 收口证据 |
| --- | --- |
| RCLE current 阶段/authority 矛盾与活动 OpenLORIS/DLR worker | worker 已退出；两段 terminal independent review、37/37 validator 与 final independent review PASS；current、result 与 handoff/INDEX 现一致为 `NOT_EVALUABLE / VALID_FAIL_CLOSED_TERMINAL` |
| `rgb_segment_confirmation_r1` 名称可能被误解成算法 confirmation | current/result 显式声明 0 frame/0 decode/0 RGB call、source-role confounding、algorithm failure/success 均不可推断，所有高权限 false |
| `.gradle-local` 与 project-local tool duplicates | 保留 URL/commit/version/hash/recovery 收据后精确删除约 1.04 GiB，目标路径均不存在 |

## 仍需关闭的风险

| 优先级 | 风险 | 当前决定 / 最小下一步 |
| --- | --- | --- |
| P2 | Kotlin 官方表把 2.1 的 AGP 上限列为 8.7.2，而项目保留修复 lint bug 的 8.7.3 | 已用 K2 kapt、五 lint、完整多变体和离线矩阵控制 patch 例外；未来升级必须重新跑同级证据，不能把本次 PASS 泛化 |
| P2 | 尚未启用 Gradle dependency-verification checksums | 当前先拒绝所有非确定版本；后续在受信任的干净 Gradle home 生成 checksum-only metadata，并同时覆盖 Windows 与 Linux artifacts，避免只收录单平台依赖 |
| P2 | `device-benchmark` 大量 instrumentation tests 只在有设备时执行 | CI 已补编译/assemble；把仍活跃、非冻结的纯 Kotlin helper/test 逐步下沉到 JVM 模块，真机测试保留设备门 |
| P2 | `scripts/` root 仍有 181 文件，allowlist 只能防新增，不能表达理想分层 | 已安全下沉第一组 4 文件；后续按 SANPO、设备/NPU、历史 USTRF 分批迁移，任何 hash-bound/consumed runner 不移动 |
| P2 | `artifacts.local/` 约 73.7 GiB，仍有 17 个 legacy 顶层目录 | 先冻结 taxonomy v2 与调用方，再按 manifest/authority 分批迁移或删除；不碰 evidence/datasets 主体 |
| P2 | handoff active index 仍有 72 行，其中包含较多历史研究终态 | 已把 18 个明确“已提交/已推送/远端一致”的 2026-07 行归档并保留 handoff 文件；其余存在未提交、活动、阻塞或语义不明确状态，不做猜测性批量归档 |
| P3 | GitHub Actions 的 Linux hashed-pip 安装尚未在真实 runner 执行 | wheel hashes 来自 Python 3.12 Linux 官方 PyPI artifacts；以首次真实 Actions 为最终环境证据 |

官方兼容性参考：[AGP 8.7 release notes](https://developer.android.com/build/releases/agp-8-7-0-release-notes)、[Kotlin/Gradle/AGP compatibility](https://kotlinlang.org/docs/gradle-configure-project.html)、[typed compilerOptions migration](https://kotlinlang.org/docs/gradle-compiler-options.html)、[Kotlin 2.1.20 K2 kapt](https://kotlinlang.org/docs/whatsnew2120.html)、[Gradle dependency verification](https://docs.gradle.org/current/userguide/dependency_verification.html)。

## 后续维护准则

1. 动态状态只写一个 current；其他入口只链接，不复制数字、terminal 和 next action。
2. 历史证据不为过新门而改写；需要例外时只能使用精确路径、hash、理由和不可扩大范围的机器门。
3. 已关闭研究代码默认不进入产品热路径；实验能力通过独立 mode/variant 显式启用。
4. 新本地产物只进入六类 canonical root；删除必须从“是否唯一、是否可复现、是否仍有 owner”判断，而不是从文件扩展名判断。
5. 完成审查不等于所有研究任务完成；任何活动 current 矛盾都应诚实保持 HOLD，而不是用外围文档制造“看起来已收口”的假象。
