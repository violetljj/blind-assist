package com.linnan.blindassist.runtime

import android.util.Log
import android.graphics.Bitmap
import android.graphics.drawable.BitmapDrawable
import android.widget.ImageView
import androidx.camera.view.PreviewView
import com.linnan.blindassist.camera.FrameSource
import com.linnan.blindassist.ui.DetectionOverlayView
import com.linnan.blindassist.vision.VisionFrame
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicReference

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
    private var externalPreview: ImageView? = null
    private val latestPreviewBitmap = AtomicReference<Bitmap?>(null)
    private val previewDispatchScheduled = AtomicBoolean(false)

    fun onCameraViewsReady(
        preview: PreviewView?,
        externalPreview: ImageView?,
        overlay: DetectionOverlayView
    ) {
        previewView = preview
        this.externalPreview = externalPreview
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
            onError = ::handleCameraSourceError,
            onPreviewBitmap = ::renderExternalPreview
        )
    }

    fun stop() {
        frameSource.stop()
    }

    fun clearViews() {
        previewView = null
        (externalPreview?.drawable as? BitmapDrawable)?.bitmap?.let { bitmap ->
            if (!bitmap.isRecycled) bitmap.recycle()
        }
        externalPreview?.setImageDrawable(null)
        externalPreview = null
        latestPreviewBitmap.getAndSet(null)?.let { bitmap ->
            if (!bitmap.isRecycled) bitmap.recycle()
        }
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

    private fun renderExternalPreview(bitmap: Bitmap) {
        val ownedCopy = bitmap.copy(Bitmap.Config.ARGB_8888, false)
        latestPreviewBitmap.getAndSet(ownedCopy)?.let { stale ->
            if (!stale.isRecycled) stale.recycle()
        }
        schedulePreviewDrain()
    }

    private fun schedulePreviewDrain() {
        if (!previewDispatchScheduled.compareAndSet(false, true)) return
        runOnUiThread {
            val next = latestPreviewBitmap.getAndSet(null)
            val target = externalPreview
            if (next != null) {
                if (target == null || !isCameraActive()) {
                    if (!next.isRecycled) next.recycle()
                } else {
                    val previous = (target.drawable as? BitmapDrawable)?.bitmap
                    target.setImageBitmap(next)
                    if (previous != null && previous !== next && !previous.isRecycled) previous.recycle()
                }
            }
            previewDispatchScheduled.set(false)
            if (latestPreviewBitmap.get() != null) schedulePreviewDrain()
        }
    }

    private companion object {
        const val PERF_TAG = AssistRuntimePerformanceLogger.PERF_TAG
    }
}
