package com.linnan.blindassist.runtime

import android.util.Log
import androidx.camera.view.PreviewView
import com.linnan.blindassist.camera.FrameSource
import com.linnan.blindassist.ui.DetectionOverlayView
import com.linnan.blindassist.vision.VisionFrame

internal class AssistCameraLifecycleAdapter(
    initialFrameSource: FrameSource,
    private val renderer: AssistRuntimeRenderer,
    private val isCameraActive: () -> Boolean,
    private val requiresCameraPermission: () -> Boolean,
    private val hasCameraPermission: () -> Boolean,
    private val onCameraStarted: () -> Unit,
    private val onCameraFailure: (String) -> Unit,
    private val runOnUiThread: (() -> Unit) -> Unit
) {
    private var frameSource: FrameSource = initialFrameSource
    private var previewView: PreviewView? = null

    fun onCameraViewsReady(preview: PreviewView?, overlay: DetectionOverlayView) {
        previewView = preview
        renderer.attachOverlay(overlay)
    }

    fun replaceFrameSource(replacement: FrameSource) {
        frameSource.stop()
        frameSource.shutdown()
        frameSource = replacement
        previewView = null
        renderer.detachOverlay()
    }

    fun startIfReady(onFrame: (VisionFrame) -> Unit) {
        if (!isCameraActive()) return
        if (requiresCameraPermission() && !hasCameraPermission()) {
            renderer.renderTarget(AssistRuntimeRenderTarget.PermissionDenied, null)
            return
        }
        frameSource.start(
            previewView = previewView,
            onFrame = onFrame,
            onStarted = {
                runOnUiThread {
                    if (isCameraActive()) onCameraStarted()
                }
            },
            onError = ::handleCameraSourceError
        )
    }

    fun stop() {
        frameSource.stop()
    }

    fun clearViews() {
        previewView = null
        renderer.detachOverlay()
    }

    fun shutdown() {
        frameSource.shutdown()
        clearViews()
    }

    private fun handleCameraSourceError(error: Throwable) {
        val message = error.message ?: "Unknown camera error"
        Log.e(PERF_TAG, "Camera source failed", error)
        runOnUiThread {
            if (isCameraActive()) onCameraFailure(message)
        }
    }

    private companion object {
        const val PERF_TAG = AssistRuntimePerformanceLogger.PERF_TAG
    }
}
