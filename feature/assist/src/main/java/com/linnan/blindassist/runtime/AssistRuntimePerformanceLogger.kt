package com.linnan.blindassist.runtime

import android.util.Log
import com.linnan.blindassist.risk.RiskResult
import com.linnan.blindassist.session.AssistFrameResult

internal class AssistRuntimePerformanceLogger(
    private val clockMs: () -> Long = System::currentTimeMillis
) {
    private var lastPerfLogAtMs = 0L

    fun logIfNeeded(frameResult: AssistFrameResult, runtimeConfig: AssistRuntimeConfig) {
        val now = clockMs()
        if (now - lastPerfLogAtMs < PERF_LOG_INTERVAL_MS) return
        lastPerfLogAtMs = now
        val evaluation = frameResult.evaluation
        val metrics = evaluation.metrics
        Log.i(
            PERF_TAG,
            "frame=${evaluation.frameSize.width}x${evaluation.frameSize.height}, " +
                "count=${evaluation.detectionCount}, " +
                "total=${metrics.totalMs}ms, pre=${metrics.preprocessMs}ms, " +
                "infer=${metrics.inferenceMs}ms, post=${metrics.postprocessMs}ms, " +
                "dropRate=${"%.1f".format(metrics.droppedFrameRate * 100f)}%, " +
                "p50=${metrics.inferenceP50Ms}ms, p95=${metrics.inferenceP95Ms}ms, " +
                "fps=${"%.1f".format(metrics.fps)}, profile=${evaluation.profile.storageValue}, " +
                "scenario=${evaluation.scenario.storageValue}, " +
                "rawRisk=${riskSummary(evaluation.rawRisk)}, stableRisk=${riskSummary(evaluation.stableRisk)}, " +
                "feedbackReason=${frameResult.feedbackDecision.reason.displayText(runtimeConfig.appLanguage)}, " +
                "explanation=${frameResult.explanation.headline}, " +
                "session=${frameResult.sessionSummary.displayText(runtimeConfig.appLanguage)}, status=${metrics.modelStatus}"
        )
    }

    private fun riskSummary(risk: RiskResult): String {
        return "${risk.level}/${risk.direction}/${risk.proximity}"
    }

    companion object {
        const val PERF_LOG_INTERVAL_MS = 1000L
        const val PERF_TAG = "BlindAssistPerf"
    }
}
