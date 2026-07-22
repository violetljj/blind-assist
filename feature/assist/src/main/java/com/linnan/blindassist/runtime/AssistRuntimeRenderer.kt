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
    private val performanceLogger: AssistRuntimePerformanceLogger = AssistRuntimePerformanceLogger()
) {
    private var overlayView: DetectionOverlayView? = null

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
        val guidance = CameraGuidanceMapper.fromFrameResult(frameResult, runtimeConfig.appLanguage)
        renderUi(
            if (mode == AssistRuntimeMode.USTRF_EXPERIMENT) {
                guidance.withUstrfExperimentalMessage(frameResult.evaluation.rawRisk.message)
            } else {
                guidance
            }
        )
        appViewModel.updateFieldTestSummary(
            fieldTestSummaryProvider.fromSummary(
                summary = frameResult.sessionSummary,
                active = true,
                runtimeConfig = runtimeConfig
            )
        )
        performanceLogger.logIfNeeded(frameResult, runtimeConfig)
    }

    private fun currentFieldTestSummary(active: Boolean, runtimeConfig: AssistRuntimeConfig) =
        fieldTestSummaryProvider.current(active, runtimeConfig)

    private fun renderUi(snapshot: CameraGuidanceUiState) {
        appViewModel.renderCameraGuidance(snapshot, detector.statusMessage)
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
}
