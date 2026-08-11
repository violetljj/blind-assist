# BlindAssist 开源公共价值说明

状态：current

最后核验：2026-08-11

## 项目使命

BlindAssist 是一个面向无障碍技术研究与工程学习的 Android 端侧辅助感知原型。项目探索如何在普通移动设备上组合 CameraX、本地视觉推理、可解释风险规则、语音、震动和界面反馈，并让实现、验证方法、失败证据和适用边界能够被外部审查与复现。

本项目不是安全认证设备，也没有证明可以替代盲杖、导盲犬、人工判断或专业出行训练。公开项目的目标是推动可审查的工程与研究方法，而不是宣称已经取得真实用户安全效果。

## 可公开复用的产出

- **端侧 Android 参考实现**：展示摄像头取流、本地模型推理、风险状态、TTS、震动和 Jetpack Compose 界面如何按稳定模块边界协作。
- **可复现验证入口**：提供单元测试、静态检查、设备回归、同设备 benchmark 和证据收据入口，使贡献者能够区分代码通过、设备可运行、实验结果和产品安全主张。
- **证据边界与失败治理**：记录数据来源、模型身份、协议、未评价项和负结果，避免把合成证据、模型共识、单设备结果或候选实验升级成部署和安全证明。
- **端侧与隐私友好的方向**：正式 App 路径优先在设备本地完成推理；研究数据、下载、模型和机器产物保存在不提交的本地路径，并按来源和许可单独管理。

## 预期受益者

- 学习 Android、CameraX、Jetpack Compose 和端侧机器学习的学生与开发者；
- 研究辅助技术、移动感知、可复现评测或人机交互的研究者；
- 希望复用测试、证据治理和失败关闭方法的开源维护者；
- 关注无障碍体验、误报漏报边界和负责任表述的审查者与贡献者。

## 当前限制

- 项目处于原型与研究阶段，没有临床、法规、真实用户安全或规模化部署认证。
- 公开仓库的外部采用度仍有限；公共价值目前主要来自可复用的实现、协议和透明边界，而不是已证明的广泛影响。
- 第三方模型、依赖、数据集、媒体和硬件资料不由 BlindAssist 重新授权；使用者必须遵守其各自条款。
- 日期化实验和 benchmark 只支持其绑定设备、数据、版本和协议下的结论。

## 开源维护与 Codex 使用计划

开源支持将优先用于 issue 分类、pull request 审查、回归测试、依赖与许可证核验、文档同步、安全检查和可复现实验维护。Codex 与其他自动化工具产生的修改仍需要通过仓库门禁和证据边界；模型输出本身不构成许可证、用户同意、设备测量或安全事实。

## Application summary (English)

BlindAssist is an open-source Android prototype for on-device assistive perception and evidence-bounded accessibility research. It publishes a runnable mobile architecture, reproducible test and benchmark entry points, and governance methods that preserve provenance, negative results, and the distinction between engineering evidence and safety claims. The project is not a certified mobility aid and does not claim to replace a cane, guide dog, human judgment, or professional mobility training. Open-source support would be used for issue triage, code review, regression automation, dependency and license review, documentation maintenance, and security analysis.
