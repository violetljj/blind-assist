package com.linnan.blindassist.ui

import com.linnan.blindassist.alert.AlertProfile
import com.linnan.blindassist.alert.AssistScenario
import com.linnan.blindassist.feedback.FeedbackDecision
import com.linnan.blindassist.risk.ProximityBand
import com.linnan.blindassist.risk.RiskDirection
import com.linnan.blindassist.risk.RiskLevel
import com.linnan.blindassist.risk.RiskResult
import com.linnan.blindassist.session.AssistDisplayFormatter
import com.linnan.blindassist.session.AssistFrameResult
import com.linnan.blindassist.session.RiskExplanation
import com.linnan.blindassist.ui.compose.CameraGuidanceUiState

object CameraGuidanceMapper {
    fun initial(modelStatus: String, scenario: AssistScenario): CameraGuidanceUiState {
        return CameraGuidanceUiState.initial(modelStatus, scenario.displayName)
    }

    fun waiting(modelStatus: String, scenario: AssistScenario): CameraGuidanceUiState {
        return CameraGuidanceUiState(
            title = "检测已开启",
            detail = "等待实时画面和稳定风险结果",
            targetLine = "模型状态：$modelStatus",
            careTitle = "正在观察",
            careDetail = "请自然前进，系统会在前方有风险时提醒",
            careTargetLine = "建议同时保留语音和震动提醒",
            debugText = "模型状态：$modelStatus",
            scenarioName = scenario.displayName,
            explanationHeadline = "继续观察：等待稳定风险",
            explanationDetail = "${scenario.displayName}场景已启用，系统会按场景调整提醒确认、冷却和震动计划。",
            careExplanation = "正在按${scenario.displayName}场景观察",
            titleColor = WHITE,
            statusBadge = "观察中",
            badgeColor = rgb(160, 255, 215),
            badgeTextColor = rgb(6, 24, 18),
            careAccessibilitySummary = "正在观察，请自然前进，系统会在前方有风险时提醒",
            accessibilitySummary = "检测已开启，等待实时画面和稳定风险结果",
            accessibilityKey = "waiting"
        )
    }

    fun paused(): CameraGuidanceUiState {
        return CameraGuidanceUiState(
            title = "检测已暂停",
            detail = "画面保留预览，目标框和风险提醒已清空",
            targetLine = "可随时重新开启检测",
            careTitle = "已暂停",
            careDetail = "当前不会识别目标，也不会发出提醒",
            careTargetLine = "打开检测后恢复观察",
            debugText = "检测关闭：不运行目标检测，不触发语音或震动提醒",
            scenarioName = AssistScenario.GENERAL.displayName,
            explanationHeadline = "检测已暂停",
            explanationDetail = "当前不会生成风险解释。",
            careExplanation = "检测已暂停",
            titleColor = rgb(214, 224, 235),
            statusBadge = "已暂停",
            badgeColor = rgb(198, 210, 222),
            badgeTextColor = rgb(12, 22, 30),
            careAccessibilitySummary = "检测已暂停，当前不会识别目标，也不会发出提醒",
            accessibilitySummary = "检测已暂停，目标框和风险提醒已清空",
            accessibilityKey = "paused"
        )
    }

    fun permissionDenied(): CameraGuidanceUiState {
        return CameraGuidanceUiState(
            title = "需要相机权限",
            detail = "请授予相机权限后再使用实时避障提醒",
            targetLine = "当前无法启动 CameraX 预览和检测",
            careTitle = "需要权限",
            careDetail = "请允许相机权限，系统才能观察前方",
            careTargetLine = "授权后会自动启动实时预览",
            debugText = "权限状态：CAMERA denied",
            scenarioName = AssistScenario.GENERAL.displayName,
            explanationHeadline = "需要相机权限",
            explanationDetail = "未取得相机权限前无法进行本地识别或风险解释。",
            careExplanation = "需要相机权限",
            titleColor = rgb(255, 149, 0),
            statusBadge = "需处理",
            badgeColor = rgb(255, 210, 125),
            badgeTextColor = rgb(44, 25, 0),
            careAccessibilitySummary = "需要相机权限，请允许相机权限，系统才能观察前方",
            accessibilitySummary = "需要相机权限，请授予相机权限后再使用实时避障提醒",
            accessibilityKey = "permission"
        )
    }

    fun cameraError(message: String): CameraGuidanceUiState {
        return CameraGuidanceUiState(
            title = "相机启动失败",
            detail = "CameraX 暂时无法打开后置摄像头",
            targetLine = message,
            careTitle = "相机未启动",
            careDetail = "当前无法观察前方，请返回后重新进入手机摄像头",
            careTargetLine = message,
            debugText = "CameraX error：$message",
            scenarioName = AssistScenario.GENERAL.displayName,
            explanationHeadline = "相机启动失败",
            explanationDetail = "相机未启动，无法生成实时风险解释。",
            careExplanation = "相机未启动",
            titleColor = rgb(255, 149, 0),
            statusBadge = "异常",
            badgeColor = rgb(255, 210, 125),
            badgeTextColor = rgb(44, 25, 0),
            careAccessibilitySummary = "相机启动失败，当前无法观察前方",
            accessibilitySummary = "相机启动失败，CameraX 暂时无法打开后置摄像头",
            accessibilityKey = "camera-error-$message"
        )
    }

    fun fromFrameResult(frameResult: AssistFrameResult): CameraGuidanceUiState {
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
                "infer ${evaluation.metrics.inferenceMs}ms / post ${evaluation.metrics.postprocessMs}ms",
            fps = evaluation.metrics.fps,
            modelStatus = evaluation.metrics.modelStatus
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
        modelStatus: String
    ): CameraGuidanceUiState {
        val risk = stableRisk
        val levelText = levelText(risk.level)
        val directionText = directionText(risk.direction)
        val title = when (risk.level) {
            RiskLevel.HIGH -> "$levelText：$directionText"
            RiskLevel.MEDIUM -> "$levelText：$directionText"
            RiskLevel.LOW -> levelText
            RiskLevel.NONE -> "安全观察中"
        }
        val detail = AssistDisplayFormatter.detailFor(risk)
        val targetLine = AssistDisplayFormatter.targetLine(rawRisk, risk, count)
        val careTitle = careTitle(risk.level, risk.direction, risk.proximity)
        val careDetail = AssistDisplayFormatter.careDetailFor(risk)
        val careTargetLine = AssistDisplayFormatter.careTargetLine(rawRisk, risk, count)
        val targetAccessibility = AssistDisplayFormatter.accessibilityTargetSummary(rawRisk, risk, count)
        val debug = "FPS：${"%.1f".format(fps)}\n" +
            "耗时：$metricsText\n" +
            "模型：$modelStatus\n" +
            "最近风险判定：原始 ${riskSummaryText(rawRisk)} / 稳定 ${riskSummaryText(stableRisk)}\n" +
            AssistDisplayFormatter.urgencyLine(rawRisk, stableRisk) + "\n" +
            "提醒模式：${profile.displayName} / 场景：${scenario.displayName} / 反馈：${feedbackDecision.reason.displayText}\n" +
            "解释：${explanation.headline}，${explanation.detail}"
        return CameraGuidanceUiState(
            title = title,
            detail = detail,
            targetLine = targetLine,
            careTitle = careTitle,
            careDetail = careDetail,
            careTargetLine = careTargetLine,
            debugText = debug,
            scenarioName = scenario.displayName,
            explanationHeadline = explanation.headline,
            explanationDetail = explanation.detail,
            careExplanation = explanation.headline,
            titleColor = colorForLevel(risk.level, risk.proximity),
            statusBadge = statusBadge(risk.level, risk.proximity),
            badgeColor = badgeColor(risk.level, risk.proximity),
            badgeTextColor = badgeTextColor(risk.level, risk.proximity),
            careAccessibilitySummary = "$careTitle，$careDetail，$targetAccessibility，${explanation.accessibilityText}",
            accessibilitySummary = "$title，$detail，$targetAccessibility，${explanation.accessibilityText}",
            accessibilityKey = "${risk.level}-${risk.direction}-${risk.proximity}-${rawRisk.level}-${scenario.name}-$count"
        )
    }

    private fun riskSummaryText(risk: RiskResult): String {
        return "${levelText(risk.level)} ${directionText(risk.direction)} ${proximityText(risk.proximity)}"
    }

    private fun levelText(level: RiskLevel): String {
        return when (level) {
            RiskLevel.HIGH -> "高风险"
            RiskLevel.MEDIUM -> "中风险"
            RiskLevel.LOW -> "低风险"
            RiskLevel.NONE -> "安全"
        }
    }

    private fun directionText(direction: RiskDirection): String {
        return when (direction) {
            RiskDirection.LEFT -> "左前"
            RiskDirection.CENTER -> "正前"
            RiskDirection.RIGHT -> "右前"
            RiskDirection.NONE -> "无方向"
        }
    }

    private fun proximityText(proximity: ProximityBand): String {
        return when (proximity) {
            ProximityBand.CRITICAL -> "迫近"
            ProximityBand.NEAR -> "近处"
            ProximityBand.MID -> "中距"
            ProximityBand.FAR -> "远处"
        }
    }

    private fun careTitle(level: RiskLevel, direction: RiskDirection, proximity: ProximityBand): String {
        return when {
            proximity == ProximityBand.CRITICAL -> "立刻注意：${directionText(direction)}"
            level == RiskLevel.HIGH -> "前方有风险：${directionText(direction)}"
            level == RiskLevel.MEDIUM -> "请留意：${directionText(direction)}"
            level == RiskLevel.LOW -> "保持观察"
            else -> "前方平稳"
        }
    }

    private fun statusBadge(level: RiskLevel, proximity: ProximityBand): String {
        return when {
            proximity == ProximityBand.CRITICAL -> "迫近提醒"
            level == RiskLevel.HIGH -> "高风险"
            level == RiskLevel.MEDIUM -> "需留意"
            level == RiskLevel.LOW -> "观察"
            else -> "平稳"
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
