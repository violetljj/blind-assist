package com.linnan.blindassist.runtime

import com.linnan.blindassist.session.AssistFrameResult
import com.linnan.blindassist.ui.BlindAssistViewModel
import com.linnan.blindassist.ui.CameraGuidanceMapper
import com.linnan.blindassist.ui.DetectionOverlayView
import com.linnan.blindassist.ui.compose.CameraGuidanceUiState
import com.linnan.blindassist.vision.DetectorFrameResult
import com.linnan.blindassist.vision.ObjectDetector

internal class AssistRuntimeRenderer(
    private val appViewModel: BlindAssistViewModel,
    private val detector: ObjectDetector,
    private val guidanceFactory: AssistRuntimeGuidanceFactory,
    private val fieldTestSummaryProvider: FieldTestSummaryProvider,
    private val mode: AssistRuntimeMode = AssistRuntimeMode.BASELINE,
    private val performanceLogger: AssistRuntimePerformanceLogger = AssistRuntimePerformanceLogger(),
    private val diagnosticsClockMs: () -> Long = { System.nanoTime() / NANOS_PER_MILLISECOND }
) {
    private var overlayView: DetectionOverlayView? = null
    private var debugVisibleOnLastFrame = false
    private var lastDiagnosticsRefreshMs: Long? = null

    fun attachOverlay(overlay: DetectionOverlayView) {
        overlayView = overlay
    }

    fun detachOverlay() {
        overlayView = null
    }

    fun currentOverlay(): DetectionOverlayView? = overlayView

    fun renderInitial() {
        renderUi(guidanceFactory.initial())
    }

    fun activateCamera(runtimeConfig: AssistRuntimeConfig) {
        resetFrameDiagnostics()
        val guidance = if (detector.isReady) {
            guidanceFactory.starting()
        } else {
            guidanceFactory.modelUnavailable()
        }
        appViewModel.activateCamera(
            fieldTestSummary = currentFieldTestSummary(active = true, runtimeConfig),
            guidance = guidance,
            modelStatus = detector.statusMessage
        )
    }

    fun closeCamera(runtimeConfig: AssistRuntimeConfig) {
        resetFrameDiagnostics()
        appViewModel.closeCamera(
            fieldTestSummary = currentFieldTestSummary(active = false, runtimeConfig),
            guidance = guidanceFactory.initial(),
            modelStatus = detector.statusMessage
        )
    }

    fun showPermissionDenied() {
        appViewModel.onCameraPermissionDenied(guidanceFactory.permissionDenied())
    }

    fun clearOverlay() {
        overlayView?.update(emptyList(), null, null)
    }

    fun renderTarget(target: AssistRuntimeRenderTarget, message: String?) {
        renderUi(guidanceFactory.forTarget(target, message))
    }

    fun renderState(state: AssistRuntimeState) {
        renderUi(guidanceFactory.forState(state))
    }

    fun renderModelUnavailable() {
        renderUi(guidanceFactory.modelUnavailable())
    }

    fun updateFieldTestSummary(active: Boolean, runtimeConfig: AssistRuntimeConfig) {
        appViewModel.updateFieldTestSummary(currentFieldTestSummary(active, runtimeConfig))
    }

    fun renderFrame(
        detectorFrame: DetectorFrameResult,
        frameResult: AssistFrameResult,
        runtimeConfig: AssistRuntimeConfig
    ) {
        overlayView?.update(
            detectorFrame.detections,
            detectorFrame.frameSize,
            frameResult.evaluation.stableRisk
        )
        val currentUiState = appViewModel.uiState.value
        val refreshDiagnostics = shouldRefreshDiagnostics(currentUiState.controls.debugVisible)
        val mappedGuidance = CameraGuidanceMapper.fromFrameResult(
            frameResult = frameResult,
            language = runtimeConfig.appLanguage,
            includeDebugDetails = refreshDiagnostics
        )
        val guidanceWithCachedDiagnostics = if (refreshDiagnostics) {
            mappedGuidance
        } else {
            mappedGuidance.copy(debugText = currentUiState.cameraGuidance.debugText)
        }
        val guidance = if (mode == AssistRuntimeMode.USTRF_EXPERIMENT) {
            guidanceWithCachedDiagnostics.withUstrfExperimentalMessage(frameResult.evaluation.rawRisk.message)
        } else {
            guidanceWithCachedDiagnostics
        }
        val fieldTestSummary = if (refreshDiagnostics) {
            fieldTestSummaryProvider.fromSummary(
                summary = frameResult.sessionSummary,
                active = true,
                runtimeConfig = runtimeConfig
            )
        } else {
            null
        }
        appViewModel.renderFrame(
            guidance = guidance,
            fieldTestSummary = fieldTestSummary,
            modelStatus = detector.statusMessage
        )
        performanceLogger.logIfNeeded(frameResult, runtimeConfig)
    }

    private fun currentFieldTestSummary(active: Boolean, runtimeConfig: AssistRuntimeConfig) =
        fieldTestSummaryProvider.current(active, runtimeConfig)

    private fun renderUi(snapshot: CameraGuidanceUiState) {
        appViewModel.renderCameraGuidance(snapshot, detector.statusMessage)
    }

    private fun shouldRefreshDiagnostics(debugVisible: Boolean): Boolean {
        if (!debugVisible) {
            debugVisibleOnLastFrame = false
            return false
        }
        val nowMs = diagnosticsClockMs()
        val lastRefreshMs = lastDiagnosticsRefreshMs
        val shouldRefresh = !debugVisibleOnLastFrame ||
            lastRefreshMs == null ||
            nowMs < lastRefreshMs ||
            nowMs - lastRefreshMs >= DIAGNOSTICS_REFRESH_INTERVAL_MS
        debugVisibleOnLastFrame = true
        if (shouldRefresh) {
            lastDiagnosticsRefreshMs = nowMs
        }
        return shouldRefresh
    }

    private fun resetFrameDiagnostics() {
        debugVisibleOnLastFrame = false
        lastDiagnosticsRefreshMs = null
    }

    private fun CameraGuidanceUiState.withUstrfExperimentalMessage(message: String): CameraGuidanceUiState {
        return copy(
            detail = message,
            careDetail = message,
            explanationHeadline = "USTRF 实验代理判断",
            explanationDetail = message,
            careExplanation = message,
            careAccessibilitySummary = "$careTitle，$message",
            accessibilitySummary = "$title，$message",
            accessibilityKey = "$accessibilityKey-ustrf-experiment"
        )
    }

    private companion object {
        const val DIAGNOSTICS_REFRESH_INTERVAL_MS = 500L
        const val NANOS_PER_MILLISECOND = 1_000_000L
    }
}
