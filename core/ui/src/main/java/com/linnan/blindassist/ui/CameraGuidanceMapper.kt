package com.linnan.blindassist.ui

import com.linnan.blindassist.alert.AlertProfile
import com.linnan.blindassist.alert.AssistScenario
import com.linnan.blindassist.feedback.FeedbackDecision
import com.linnan.blindassist.localization.AppLanguage
import com.linnan.blindassist.localization.LocalizedText
import com.linnan.blindassist.risk.ProximityBand
import com.linnan.blindassist.risk.RiskDirection
import com.linnan.blindassist.risk.RiskEvidenceState
import com.linnan.blindassist.risk.RiskLevel
import com.linnan.blindassist.risk.RiskResult
import com.linnan.blindassist.session.AssistDisplayFormatter
import com.linnan.blindassist.session.AssistFrameResult
import com.linnan.blindassist.session.RiskExplanation
import com.linnan.blindassist.ui.compose.CameraGuidanceUiState

object CameraGuidanceMapper {
    fun initial(modelStatus: String, scenario: AssistScenario, language: AppLanguage = AppLanguage.ZH): CameraGuidanceUiState {
        return CameraGuidanceUiState.initial(modelStatus, scenario.displayName(language), language)
    }

    fun waiting(modelStatus: String, scenario: AssistScenario, language: AppLanguage = AppLanguage.ZH): CameraGuidanceUiState {
        val scenarioName = scenario.displayName(language)
        val english = language == AppLanguage.EN
        return CameraGuidanceUiState(
            title = if (english) "Detection on" else "检测已开启",
            detail = if (english) "Waiting for live frames and stable risk results" else "等待实时画面和稳定风险结果",
            targetLine = if (english) "Model status: $modelStatus" else "模型状态：$modelStatus",
            careTitle = if (english) "Observing" else "正在观察",
            careDetail = if (english) "Walk naturally. The system will remind you when risk appears ahead." else "请自然前进，系统会在前方有风险时提醒",
            careTargetLine = if (english) "Keep both speech and vibration reminders on if possible" else "建议同时保留语音和震动提醒",
            debugText = if (english) "Model status: $modelStatus" else "模型状态：$modelStatus",
            scenarioName = scenarioName,
            explanationHeadline = if (english) "Keep observing: waiting for stable risk" else "继续观察：等待稳定风险",
            explanationDetail = if (english) "$scenarioName scenario is active. Reminder confirmation, cooldown, and vibration plan follow this scenario." else "${scenarioName}场景已启用，系统会按场景调整提醒确认、冷却和震动计划。",
            careExplanation = if (english) "Observing with $scenarioName scenario" else "正在按${scenarioName}场景观察",
            titleColor = WHITE,
            statusBadge = if (english) "Observing" else "观察中",
            badgeColor = rgb(160, 255, 215),
            badgeTextColor = rgb(6, 24, 18),
            careAccessibilitySummary = if (english) "Observing. Walk naturally; the system will remind you when risk appears ahead." else "正在观察，请自然前进，系统会在前方有风险时提醒",
            accessibilitySummary = if (english) "Detection on, waiting for live frames and stable risk results" else "检测已开启，等待实时画面和稳定风险结果",
            accessibilityKey = "waiting"
        )
    }

    fun starting(modelStatus: String, scenario: AssistScenario, language: AppLanguage = AppLanguage.ZH): CameraGuidanceUiState {
        val scenarioName = scenario.displayName(language)
        val english = language == AppLanguage.EN
        return CameraGuidanceUiState(
            title = if (english) "Camera starting" else "相机启动中",
            detail = if (english) "Preparing live preview and local detection" else "正在准备实时预览和本地识别",
            targetLine = if (english) "Model status: $modelStatus" else "模型状态：$modelStatus",
            careTitle = if (english) "Starting camera" else "正在启动相机",
            careDetail = if (english) "Please wait while the preview opens" else "请稍等，正在打开预览画面",
            careTargetLine = if (english) "Reminders start after frames arrive" else "画面到达后会开始提醒",
            debugText = if (english) "Runtime state: starting\nModel status: $modelStatus" else "运行状态：相机启动中\n模型状态：$modelStatus",
            scenarioName = scenarioName,
            explanationHeadline = if (english) "Camera is starting" else "相机正在启动",
            explanationDetail = if (english) "$scenarioName scenario is ready; waiting for CameraX to report live frames." else "${scenarioName}场景已准备，正在等待 CameraX 返回实时画面。",
            careExplanation = if (english) "Camera is starting" else "相机正在启动",
            titleColor = WHITE,
            statusBadge = if (english) "Starting" else "启动中",
            badgeColor = rgb(206, 221, 235),
            badgeTextColor = rgb(10, 22, 32),
            careAccessibilitySummary = if (english) "Camera is starting. Please wait while preview opens." else "相机正在启动，请稍等，正在打开预览画面",
            accessibilitySummary = if (english) "Camera starting, preparing live preview and local detection" else "相机启动中，正在准备实时预览和本地识别",
            accessibilityKey = "camera-starting"
        )
    }

    fun modelUnavailable(modelStatus: String, scenario: AssistScenario, language: AppLanguage = AppLanguage.ZH): CameraGuidanceUiState {
        val scenarioName = scenario.displayName(language)
        val english = language == AppLanguage.EN
        return CameraGuidanceUiState(
            title = if (english) "Model unavailable" else "模型不可用",
            detail = if (english) "The camera preview can open, but local detection is not ready" else "相机预览可以打开，但本地识别尚未就绪",
            targetLine = if (english) "Model status: $modelStatus" else "模型状态：$modelStatus",
            careTitle = if (english) "Detection not ready" else "识别未就绪",
            careDetail = if (english) "Do not rely on reminders until the model is loaded" else "模型加载前不要依赖提醒",
            careTargetLine = if (english) "Go back and check the model asset if this persists" else "若持续出现，请返回检查模型文件",
            debugText = if (english) "Runtime state: model unavailable\nModel status: $modelStatus" else "运行状态：模型不可用\n模型状态：$modelStatus",
            scenarioName = scenarioName,
            explanationHeadline = if (english) "Model unavailable" else "模型不可用",
            explanationDetail = if (english) "$scenarioName scenario is selected, but detection cannot run until the model is ready." else "当前选择${scenarioName}场景，但模型就绪前无法执行识别。",
            careExplanation = if (english) "Model unavailable" else "模型不可用",
            titleColor = rgb(255, 149, 0),
            statusBadge = if (english) "Model" else "模型",
            badgeColor = rgb(255, 210, 125),
            badgeTextColor = rgb(44, 25, 0),
            careAccessibilitySummary = if (english) "Model unavailable. Do not rely on reminders until the model is loaded." else "模型不可用，模型加载前不要依赖提醒",
            accessibilitySummary = if (english) "Model unavailable, local detection is not ready" else "模型不可用，本地识别尚未就绪",
            accessibilityKey = "model-unavailable-$modelStatus"
        )
    }

    fun paused(language: AppLanguage = AppLanguage.ZH): CameraGuidanceUiState {
        val english = language == AppLanguage.EN
        return CameraGuidanceUiState(
            title = if (english) "Detection paused" else "检测已暂停",
            detail = if (english) "Preview remains; boxes and risk reminders are cleared" else "画面保留预览，目标框和风险提醒已清空",
            targetLine = if (english) "Detection can be turned on again anytime" else "可随时重新开启检测",
            careTitle = if (english) "Paused" else "已暂停",
            careDetail = if (english) "Objects are not detected and reminders are off" else "当前不会识别目标，也不会发出提醒",
            careTargetLine = if (english) "Turn detection on to resume observing" else "打开检测后恢复观察",
            debugText = if (english) "Detection off: no object detection, no speech or vibration reminders" else "检测关闭：不运行目标检测，不触发语音或震动提醒",
            scenarioName = AssistScenario.GENERAL.displayName(language),
            explanationHeadline = if (english) "Detection paused" else "检测已暂停",
            explanationDetail = if (english) "No risk explanation is generated while paused." else "当前不会生成风险解释。",
            careExplanation = if (english) "Detection paused" else "检测已暂停",
            titleColor = rgb(214, 224, 235),
            statusBadge = if (english) "Paused" else "已暂停",
            badgeColor = rgb(198, 210, 222),
            badgeTextColor = rgb(12, 22, 30),
            careAccessibilitySummary = if (english) "Detection paused. Objects are not detected and reminders are off." else "检测已暂停，当前不会识别目标，也不会发出提醒",
            accessibilitySummary = if (english) "Detection paused, boxes and risk reminders are cleared" else "检测已暂停，目标框和风险提醒已清空",
            accessibilityKey = "paused"
        )
    }

    fun permissionDenied(language: AppLanguage = AppLanguage.ZH): CameraGuidanceUiState {
        val english = language == AppLanguage.EN
        return CameraGuidanceUiState(
            title = if (english) "Camera permission needed" else "需要相机权限",
            detail = if (english) "Allow camera permission before using real-time obstacle reminders" else "请授予相机权限后再使用实时避障提醒",
            targetLine = if (english) "CameraX preview and detection cannot start now" else "当前无法启动 CameraX 预览和检测",
            careTitle = if (english) "Permission needed" else "需要权限",
            careDetail = if (english) "Allow camera permission so the system can observe ahead" else "请允许相机权限，系统才能观察前方",
            careTargetLine = if (english) "Live preview starts after permission is granted" else "授权后会自动启动实时预览",
            debugText = "CAMERA denied",
            scenarioName = AssistScenario.GENERAL.displayName(language),
            explanationHeadline = if (english) "Camera permission needed" else "需要相机权限",
            explanationDetail = if (english) "Local recognition and risk explanations need camera permission." else "未取得相机权限前无法进行本地识别或风险解释。",
            careExplanation = if (english) "Camera permission needed" else "需要相机权限",
            titleColor = rgb(255, 149, 0),
            statusBadge = if (english) "Action" else "需处理",
            badgeColor = rgb(255, 210, 125),
            badgeTextColor = rgb(44, 25, 0),
            careAccessibilitySummary = if (english) "Camera permission needed. Allow permission so the system can observe ahead." else "需要相机权限，请允许相机权限，系统才能观察前方",
            accessibilitySummary = if (english) "Camera permission needed before real-time obstacle reminders can start" else "需要相机权限，请授予相机权限后再使用实时避障提醒",
            accessibilityKey = "permission"
        )
    }

    fun cameraError(message: String, language: AppLanguage = AppLanguage.ZH): CameraGuidanceUiState {
        val english = language == AppLanguage.EN
        return CameraGuidanceUiState(
            title = if (english) "Camera failed to start" else "相机启动失败",
            detail = if (english) "CameraX cannot open the rear camera right now" else "CameraX 暂时无法打开后置摄像头",
            targetLine = message,
            careTitle = if (english) "Camera not started" else "相机未启动",
            careDetail = if (english) "Cannot observe ahead now. Go back and reopen the phone camera." else "当前无法观察前方，请返回后重新进入手机摄像头",
            careTargetLine = message,
            debugText = "CameraX error：$message",
            scenarioName = AssistScenario.GENERAL.displayName(language),
            explanationHeadline = if (english) "Camera failed to start" else "相机启动失败",
            explanationDetail = if (english) "The camera is not running, so real-time risk explanations cannot be generated." else "相机未启动，无法生成实时风险解释。",
            careExplanation = if (english) "Camera not started" else "相机未启动",
            titleColor = rgb(255, 149, 0),
            statusBadge = if (english) "Error" else "异常",
            badgeColor = rgb(255, 210, 125),
            badgeTextColor = rgb(44, 25, 0),
            careAccessibilitySummary = if (english) "Camera failed to start. Cannot observe ahead now." else "相机启动失败，当前无法观察前方",
            accessibilitySummary = if (english) "Camera failed to start. CameraX cannot open the rear camera right now." else "相机启动失败，CameraX 暂时无法打开后置摄像头",
            accessibilityKey = "camera-error-$message"
        )
    }

    fun fromFrameResult(frameResult: AssistFrameResult, language: AppLanguage = AppLanguage.ZH): CameraGuidanceUiState {
        val evaluation = frameResult.evaluation
        return fromRisk(
            rawRisk = evaluation.rawRisk,
            stableRisk = evaluation.stableRisk,
            count = evaluation.detectionCount,
            profile = evaluation.profile,
            scenario = evaluation.scenario,
            feedbackDecision = frameResult.feedbackDecision,
            explanation = frameResult.explanation,
            metricsText = "total ${evaluation.metrics.totalMs}ms / pre ${evaluation.metrics.preprocessMs}ms / " +
                "infer ${evaluation.metrics.inferenceMs}ms / post ${evaluation.metrics.postprocessMs}ms / " +
                "drop ${"%.1f".format(evaluation.metrics.droppedFrameRate * 100f)}% / " +
                "p50 ${evaluation.metrics.inferenceP50Ms}ms / p95 ${evaluation.metrics.inferenceP95Ms}ms",
            fps = evaluation.metrics.fps,
            modelStatus = evaluation.metrics.modelStatus,
            language = language
        )
    }

    private fun fromRisk(
        rawRisk: RiskResult,
        stableRisk: RiskResult,
        count: Int,
        profile: AlertProfile,
        scenario: AssistScenario,
        feedbackDecision: FeedbackDecision,
        explanation: RiskExplanation,
        metricsText: String,
        fps: Float,
        modelStatus: String,
        language: AppLanguage
    ): CameraGuidanceUiState {
        val risk = stableRisk
        val levelText = levelText(risk.level, language)
        val directionText = directionText(risk.direction, language)
        val title = when (risk.level) {
            RiskLevel.HIGH -> "$levelText：$directionText"
            RiskLevel.MEDIUM -> "$levelText：$directionText"
            RiskLevel.LOW -> levelText
            RiskLevel.NONE -> if (language == AppLanguage.EN) "Monitoring" else "持续检测中"
        }
        val localizedExplanation = localizedExplanation(explanation, feedbackDecision, stableRisk, rawRisk, count, scenario, language)
        val detail = AssistDisplayFormatter.detailFor(risk, language)
        val targetLine = AssistDisplayFormatter.targetLine(rawRisk, risk, count, language)
        val careTitle = careTitle(risk.level, risk.direction, risk.proximity, language)
        val careDetail = AssistDisplayFormatter.careDetailFor(risk, language)
        val careTargetLine = AssistDisplayFormatter.careTargetLine(rawRisk, risk, count, language)
        val targetAccessibility = AssistDisplayFormatter.accessibilityTargetSummary(rawRisk, risk, count, language)
        val debug = if (language == AppLanguage.EN) {
            "FPS: ${"%.1f".format(fps)}\n" +
                "Timing: $metricsText\n" +
                "Model: $modelStatus\n" +
                "Recent risk: raw ${riskSummaryText(rawRisk, language)} / stable ${riskSummaryText(stableRisk, language)}\n" +
                AssistDisplayFormatter.urgencyLine(rawRisk, stableRisk, language) + "\n" +
                "Profile: ${profile.displayName(language)} / Scenario: ${scenario.displayName(language)} / Feedback: ${feedbackDecision.reason.displayText(language)}\n" +
                "Explanation: ${localizedExplanation.headline}, ${localizedExplanation.detail}"
        } else {
            "FPS：${"%.1f".format(fps)}\n" +
                "耗时：$metricsText\n" +
                "模型：$modelStatus\n" +
                "最近风险判定：原始 ${riskSummaryText(rawRisk, language)} / 稳定 ${riskSummaryText(stableRisk, language)}\n" +
                AssistDisplayFormatter.urgencyLine(rawRisk, stableRisk, language) + "\n" +
                "提醒模式：${profile.displayName(language)} / 场景：${scenario.displayName(language)} / 反馈：${feedbackDecision.reason.displayText(language)}\n" +
                "解释：${localizedExplanation.headline}，${localizedExplanation.detail}"
        }
        return CameraGuidanceUiState(
            title = title,
            detail = detail,
            targetLine = targetLine,
            careTitle = careTitle,
            careDetail = careDetail,
            careTargetLine = careTargetLine,
            debugText = debug,
            scenarioName = scenario.displayName(language),
            explanationHeadline = localizedExplanation.headline,
            explanationDetail = localizedExplanation.detail,
            careExplanation = localizedExplanation.headline,
            titleColor = colorForLevel(risk.level, risk.proximity),
            statusBadge = statusBadge(risk.level, risk.proximity, language),
            badgeColor = badgeColor(risk.level, risk.proximity),
            badgeTextColor = badgeTextColor(risk.level, risk.proximity),
            careAccessibilitySummary = if (language == AppLanguage.EN) "$careTitle, $careDetail, $targetAccessibility, ${localizedExplanation.accessibilityText}" else "$careTitle，$careDetail，$targetAccessibility，${localizedExplanation.accessibilityText}",
            accessibilitySummary = if (language == AppLanguage.EN) "$title, $detail, $targetAccessibility, ${localizedExplanation.accessibilityText}" else "$title，$detail，$targetAccessibility，${localizedExplanation.accessibilityText}",
            accessibilityKey = "${risk.level}-${risk.evidenceState}-${risk.direction}-${risk.proximity}-${rawRisk.level}-${scenario.name}-$count"
        )
    }

    private fun riskSummaryText(risk: RiskResult, language: AppLanguage): String {
        return "${levelText(risk.level, language)} ${directionText(risk.direction, language)} ${proximityText(risk.proximity, language)}"
    }

    private fun levelText(level: RiskLevel, language: AppLanguage): String {
        return LocalizedText.level(level, language)
    }

    private fun directionText(direction: RiskDirection, language: AppLanguage): String {
        return if (direction == RiskDirection.NONE && language == AppLanguage.ZH) "无方向" else LocalizedText.direction(direction, language, short = true)
    }

    private fun proximityText(proximity: ProximityBand, language: AppLanguage): String {
        return LocalizedText.proximity(proximity, language)
    }

    private fun careTitle(level: RiskLevel, direction: RiskDirection, proximity: ProximityBand, language: AppLanguage): String {
        val directionText = directionText(direction, language)
        return if (language == AppLanguage.EN) {
            when {
                proximity == ProximityBand.CRITICAL -> "Immediate attention: $directionText"
                level == RiskLevel.HIGH -> "Risk ahead: $directionText"
                level == RiskLevel.MEDIUM -> "Please watch: $directionText"
                level == RiskLevel.LOW -> "Keep observing"
                else -> "Monitoring"
            }
        } else {
            when {
                proximity == ProximityBand.CRITICAL -> "立刻注意：$directionText"
                level == RiskLevel.HIGH -> "前方有风险：$directionText"
                level == RiskLevel.MEDIUM -> "请留意：$directionText"
                level == RiskLevel.LOW -> "保持观察"
                else -> "持续检测中"
            }
        }
    }

    private fun localizedExplanation(
        explanation: RiskExplanation,
        feedbackDecision: FeedbackDecision,
        stableRisk: RiskResult,
        rawRisk: RiskResult,
        count: Int,
        scenario: AssistScenario,
        language: AppLanguage
    ): RiskExplanation {
        if (language == AppLanguage.ZH) return explanation
        val scenarioName = scenario.displayName(language)
        val stableText = riskSummaryText(stableRisk, language)
        val rawText = riskSummaryText(rawRisk, language)
        return when (feedbackDecision.reason) {
            com.linnan.blindassist.feedback.FeedbackReason.TRIGGERED -> RiskExplanation(
                headline = "Reminder triggered with $scenarioName strategy",
                detail = "Stable risk is $stableText. $scenarioName uses its own cooldown and vibration plan.",
                accessibilityText = "Reminder triggered. Stable risk is $stableText. Current scenario is $scenarioName."
            )
            com.linnan.blindassist.feedback.FeedbackReason.UNSTABLE_RISK -> RiskExplanation(
                headline = "No reminder: risk is not stable",
                detail = "$scenarioName requires confirmation for medium risk. Current raw risk is $rawText.",
                accessibilityText = "No reminder because risk is not stable. Current scenario is $scenarioName."
            )
            com.linnan.blindassist.feedback.FeedbackReason.DISTANCE_TOO_FAR -> RiskExplanation(
                headline = "No reminder: object is farther away",
                detail = "$count objects were detected, but stable risk is $stableText.",
                accessibilityText = "No reminder because the object is farther away. Current scenario is $scenarioName."
            )
            com.linnan.blindassist.feedback.FeedbackReason.COOLDOWN -> RiskExplanation(
                headline = "No repeat: reminder is cooling down",
                detail = "$scenarioName is limiting repeated reminders to reduce interruption.",
                accessibilityText = "No repeat reminder because cooldown is active. Current scenario is $scenarioName."
            )
            com.linnan.blindassist.feedback.FeedbackReason.EVENT_ALREADY_ALERTED -> RiskExplanation(
                headline = "No repeat: same risk was already alerted",
                detail = "This centre-path risk has already received feedback. It will be evaluated again after passing, receding, or clearing.",
                accessibilityText = "The same risk was already alerted. Waiting for it to pass or clear before evaluating another reminder."
            )
            com.linnan.blindassist.feedback.FeedbackReason.HELD_ALERT -> RiskExplanation(
                headline = "Holding the previous reminder briefly",
                detail = "The current frame did not confirm the object again, but the stabilizer is briefly keeping $stableText.",
                accessibilityText = "Previous reminder is briefly held. Current scenario is $scenarioName."
            )
            com.linnan.blindassist.feedback.FeedbackReason.SPEECH_DISABLED,
            com.linnan.blindassist.feedback.FeedbackReason.VIBRATION_DISABLED -> RiskExplanation(
                headline = "Risk exists, but feedback is off",
                detail = "Stable risk is $stableText. Check whether speech or vibration reminders are enabled.",
                accessibilityText = "Risk exists, but feedback switches are off. Current scenario is $scenarioName."
            )
            com.linnan.blindassist.feedback.FeedbackReason.FEEDBACK_UNAVAILABLE -> RiskExplanation(
                headline = "Risk exists, but feedback is unavailable",
                detail = "Stable risk is $stableText. The device did not accept speech or vibration feedback.",
                accessibilityText = "Risk exists, but speech or vibration feedback is unavailable. Current scenario is $scenarioName."
            )
            com.linnan.blindassist.feedback.FeedbackReason.NO_FEEDBACK_RISK -> RiskExplanation(
                headline = "Monitoring",
                detail = if (stableRisk.evidenceState == RiskEvidenceState.SUPPORTED_TARGET_EVIDENCE) {
                    "A supported target is detected, but it does not currently meet the reminder threshold."
                } else {
                    "No supported target currently meets the reminder threshold. Keep checking your surroundings."
                },
                accessibilityText = "Monitoring continues. Keep checking your surroundings. Current scenario is $scenarioName."
            )
        }
    }

    private fun statusBadge(level: RiskLevel, proximity: ProximityBand, language: AppLanguage): String {
        return if (language == AppLanguage.EN) {
            when {
                proximity == ProximityBand.CRITICAL -> "Critical"
                level == RiskLevel.HIGH -> "High risk"
                level == RiskLevel.MEDIUM -> "Watch"
                level == RiskLevel.LOW -> "Observe"
                else -> "Monitoring"
            }
        } else {
            when {
                proximity == ProximityBand.CRITICAL -> "迫近提醒"
                level == RiskLevel.HIGH -> "高风险"
                level == RiskLevel.MEDIUM -> "需留意"
                level == RiskLevel.LOW -> "观察"
                else -> "检测中"
            }
        }
    }

    private fun badgeColor(level: RiskLevel, proximity: ProximityBand): Int {
        return when {
            proximity == ProximityBand.CRITICAL -> rgb(255, 99, 119)
            level == RiskLevel.HIGH -> rgb(255, 132, 105)
            level == RiskLevel.MEDIUM -> rgb(255, 205, 112)
            level == RiskLevel.LOW -> rgb(239, 226, 133)
            else -> rgb(160, 255, 215)
        }
    }

    private fun badgeTextColor(level: RiskLevel, proximity: ProximityBand): Int {
        return if (proximity == ProximityBand.CRITICAL || level == RiskLevel.HIGH) {
            WHITE
        } else {
            rgb(15, 24, 18)
        }
    }

    private fun colorForLevel(level: RiskLevel, proximity: ProximityBand): Int {
        return when {
            proximity == ProximityBand.CRITICAL -> rgb(255, 99, 119)
            level == RiskLevel.HIGH -> rgb(255, 112, 97)
            level == RiskLevel.MEDIUM -> rgb(255, 183, 77)
            level == RiskLevel.LOW -> rgb(255, 224, 102)
            else -> rgb(99, 230, 166)
        }
    }

    private fun rgb(red: Int, green: Int, blue: Int): Int {
        return (0xFF shl 24) or (red shl 16) or (green shl 8) or blue
    }

    private const val WHITE = -1
}
