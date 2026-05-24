package com.linnan.blindassist.runtime

import android.util.Log
import androidx.camera.view.PreviewView
import com.linnan.blindassist.camera.FrameSource
import com.linnan.blindassist.ui.DetectionOverlayView
import com.linnan.blindassist.vision.VisionFrame

internal class AssistCameraLifecycleAdapter(
    private val frameSource: FrameSource,
    private val renderer: AssistRuntimeRenderer,
    private val isCameraActive: () -> Boolean,
    private val hasCameraPermission: () -> Boolean,
    private val onCameraStarted: () -> Unit,
    private val onCameraFailure: (String) -> Unit,
    private val runOnUiThread: (() -> Unit) -> Unit
) {
    private var previewView: PreviewView? = null

    fun onCameraViewsReady(preview: PreviewView, overlay: DetectionOverlayView) {
        previewView = preview
        renderer.attachOverlay(overlay)
    }

    fun startIfReady(onFrame: (VisionFrame) -> Unit) {
        val preview = previewView ?: return
        if (!isCameraActive()) return
        if (!hasCameraPermission()) {
            renderer.renderTarget(AssistRuntimeRenderTarget.PermissionDenied, null)
            return
        }
        frameSource.start(
            previewView = preview,
            onFrame = onFrame,
            onStarted = onCameraStarted,
            onError = ::handleCameraSourceError
        )
    }

    fun stop() {
        frameSource.stop()
    }

    fun shutdown() {
        frameSource.shutdown()
    }

    private fun handleCameraSourceError(error: Throwable) {
        val message = error.message ?: "Unknown camera error"
        Log.e(PERF_TAG, "Camera source failed", error)
        runOnUiThread {
            onCameraFailure(message)
        }
    }

    private companion object {
        const val PERF_TAG = AssistRuntimePerformanceLogger.PERF_TAG
    }
}
