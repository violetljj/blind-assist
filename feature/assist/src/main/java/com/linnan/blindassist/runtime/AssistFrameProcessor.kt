package com.linnan.blindassist.runtime

import android.util.Log
import com.linnan.blindassist.session.AssistSessionCoordinator
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
    private val onCameraFailure: (String) -> Unit
) {
    private val isProcessing = AtomicBoolean(false)

    fun process(frame: VisionFrame) {
        stats.onReceived()
        val lease = lifecycleGate.tryEnterFrame()
        if (lease == null) {
            stats.onDroppedInactive()
            frame.close()
            return
        }
        val runtimeConfig = configSnapshot.get()
        if (!isCameraActive() || !runtimeConfig.detectionEnabled) {
            stats.onDroppedInactive()
            frame.close()
            lease.close()
            return
        }
        if (!detector.isReady) {
            stats.onDroppedDetectorUnavailable()
            runOnUiThread { renderer.renderModelUnavailable() }
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
            val detectorFrame = detector.detect(frame)
            val snapshot = stats.onProcessed(detectorFrame.metrics.inferenceMs)
            val detectorFrameWithPipelineStats = detectorFrame.copy(
                metrics = detectorFrame.metrics.copy(
                    droppedFrameRate = snapshot.droppedFrameRate,
                    inferenceP50Ms = snapshot.inferenceP50Ms,
                    inferenceP95Ms = snapshot.inferenceP95Ms
                )
            )
            val frameResult = coordinator.processFrame(
                detectorFrameWithPipelineStats,
                runtimeConfig.alertProfile,
                runtimeConfig.assistScenario
            )
            runOnUiThread {
                renderer.renderFrame(detectorFrameWithPipelineStats, frameResult, runtimeConfig)
            }
        } catch (error: Exception) {
            logError("Frame processing failed", error)
            runOnUiThread {
                onCameraFailure("Detection failed: ${error.message ?: "unknown error"}")
            }
        } finally {
            isProcessing.set(false)
            frame.close()
            lease.close()
        }
    }

    fun reset() {
        isProcessing.set(false)
        stats.reset()
    }

    private companion object {
        const val PERF_TAG = AssistRuntimePerformanceLogger.PERF_TAG

        fun logError(message: String, error: Throwable) {
            try {
                Log.e(PERF_TAG, message, error)
            } catch (_: RuntimeException) {
                // Android Log is not available in local JVM tests.
            }
        }
    }
}
