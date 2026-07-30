package com.linnan.blindassist.runtime

import android.util.Log
import com.linnan.blindassist.session.AssistSessionCoordinator
import com.linnan.blindassist.session.DualLoopRuntimeMode
import com.linnan.blindassist.session.DualLoopShadowObservation
import com.linnan.blindassist.vision.FrameClockDomain
import com.linnan.blindassist.vision.ObjectDetector
import com.linnan.blindassist.vision.VisionFrame
import java.util.concurrent.atomic.AtomicBoolean

internal class AssistFrameProcessor(
    private val detector: ObjectDetector,
    private val coordinator: AssistSessionCoordinator,
    private val configSnapshot: AssistRuntimeConfigSnapshot,
    private val renderer: AssistRuntimeRenderer,
    private val stats: FramePipelineStats,
    private val lifecycleGate: AssistRuntimeLifecycleGate,
    private val isCameraActive: () -> Boolean,
    private val runOnUiThread: (() -> Unit) -> Unit,
    private val onCameraFailure: (String) -> Unit,
    private val decisionClockNs: () -> Long = System::nanoTime,
    private val onDualLoopShadowObservation: (DualLoopShadowObservation) -> Unit = {},
    private val mode: AssistRuntimeMode = AssistRuntimeMode.BASELINE,
    private val ustrfAdapters: UstrfRuntimeAdapters =
        UstrfRuntimeAdapters.forMode(mode, coordinator)
) {
    private val isProcessing = AtomicBoolean(false)

    fun process(frame: VisionFrame) {
        val lease = lifecycleGate.tryEnterFrame()
        if (lease == null) {
            frame.close()
            return
        }
        stats.onReceived()
        val token = lease.token
        val runtimeConfig = configSnapshot.get()
        if (!isCameraActive() || !runtimeConfig.detectionEnabled) {
            stats.onDroppedInactive()
            frame.close()
            lease.close()
            return
        }
        if (!detector.isReady) {
            stats.onDroppedDetectorUnavailable()
            runOnUiThread {
                if (lifecycleGate.isCurrent(token)) {
                    renderer.renderModelUnavailable()
                }
            }
            frame.close()
            lease.close()
            return
        }
        if (!isProcessing.compareAndSet(false, true)) {
            stats.onDroppedBusy()
            frame.close()
            lease.close()
            return
        }

        try {
            val detectedFrame = detector.detect(frame)
            val detectorFrame = when {
                detectedFrame.sourceFrame != null && frame.frameStamp != null &&
                    detectedFrame.sourceFrame != frame.frameStamp ->
                    error("detector result source frame does not match the input frame")
                detectedFrame.sourceFrame == null && frame.frameStamp != null ->
                    detectedFrame.copy(sourceFrame = frame.frameStamp)
                else -> detectedFrame
            }
            val committedFrame = lifecycleGate.commitIfCurrent(lease) {
                val decisionAtNs = decisionClockNs()
                require(decisionAtNs >= 0L) { "decision clock must be non-negative" }
                val eventTimeMs = detectorFrame.sourceFrame?.capturedAtNs?.div(NANOS_PER_MILLISECOND)
                    ?: decisionAtNs / NANOS_PER_MILLISECOND
                val snapshot = stats.onProcessed(detectorFrame.metrics.inferenceMs)
                val detectorFrameWithPipelineStats = detectorFrame.copy(
                    metrics = detectorFrame.metrics.copy(
                        droppedFrameRate = snapshot.droppedFrameRate,
                        inferenceP50Ms = snapshot.inferenceP50Ms,
                        inferenceP95Ms = snapshot.inferenceP95Ms
                    )
                )
                val frameResult = when (mode) {
                    AssistRuntimeMode.BASELINE,
                    AssistRuntimeMode.DUAL_LOOP_SHADOW -> coordinator.processFrame(
                        detectorFrameWithPipelineStats,
                        runtimeConfig.alertProfile,
                        runtimeConfig.assistScenario,
                        nowMs = eventTimeMs,
                        decisionAtNs = decisionAtNs,
                        dualLoopMode = if (mode == AssistRuntimeMode.DUAL_LOOP_SHADOW) {
                            DualLoopRuntimeMode.SHADOW_ABSTAIN_ONLY
                        } else {
                            DualLoopRuntimeMode.OFF
                        },
                        dualLoopGeometryEvidence = null,
                        dualLoopDecisionClockDomain = FrameClockDomain.ANDROID_ELAPSED_REALTIME
                    )
                    AssistRuntimeMode.USTRF_EXPERIMENT ->
                        requireNotNull(ustrfAdapters.experimental) {
                            "USTRF experiment mode requires its isolated adapter"
                        }.process(
                            frame = detectorFrameWithPipelineStats,
                            profile = runtimeConfig.alertProfile,
                            scenario = runtimeConfig.assistScenario,
                            nowMs = eventTimeMs,
                            decisionAtNs = decisionAtNs
                        )
                }
                if (mode == AssistRuntimeMode.DUAL_LOOP_SHADOW) {
                    try {
                        onDualLoopShadowObservation(frameResult.evaluation.dualLoopShadow)
                    } catch (observerError: RuntimeException) {
                        logError("Dual-loop shadow observer failed", observerError)
                    }
                }
                CommittedFrame(detectorFrameWithPipelineStats, frameResult)
            }
            if (committedFrame != null) {
                runOnUiThread {
                    if (lifecycleGate.isCurrent(token)) {
                        renderer.renderFrame(
                            committedFrame.detectorFrame,
                            committedFrame.frameResult,
                            runtimeConfig
                        )
                    }
                }
            }
        } catch (error: Exception) {
            logError("Frame processing failed", error)
            if (lifecycleGate.isCurrent(token)) {
                runOnUiThread {
                    if (lifecycleGate.isCurrent(token)) {
                        onCameraFailure("Detection failed: ${error.message ?: "unknown error"}")
                    }
                }
            }
        } finally {
            isProcessing.set(false)
            frame.close()
            lease.close()
        }
    }

    fun resetSessionStats() {
        stats.reset()
    }

    private data class CommittedFrame(
        val detectorFrame: com.linnan.blindassist.vision.DetectorFrameResult,
        val frameResult: com.linnan.blindassist.session.AssistFrameResult
    )

    private companion object {
        const val PERF_TAG = AssistRuntimePerformanceLogger.PERF_TAG
        const val NANOS_PER_MILLISECOND = 1_000_000L

        fun logError(message: String, error: Throwable) {
            try {
                Log.e(PERF_TAG, message, error)
            } catch (_: RuntimeException) {
                // Android Log is not available in local JVM tests.
            }
        }
    }
}

internal data class UstrfRuntimeAdapters(
    val experimental: AssistUstrfExperimentalAdapter?
) {
    companion object {
        fun forMode(
            mode: AssistRuntimeMode,
            coordinator: AssistSessionCoordinator
        ): UstrfRuntimeAdapters = when (mode) {
            AssistRuntimeMode.BASELINE,
            AssistRuntimeMode.DUAL_LOOP_SHADOW -> UstrfRuntimeAdapters(
                experimental = null
            )
            AssistRuntimeMode.USTRF_EXPERIMENT -> UstrfRuntimeAdapters(
                experimental = AssistUstrfExperimentalAdapter(coordinator)
            )
        }
    }
}
