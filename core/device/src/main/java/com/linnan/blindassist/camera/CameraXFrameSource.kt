package com.linnan.blindassist.camera

import android.content.Context
import android.graphics.Bitmap
import android.hardware.camera2.CameraCharacteristics
import android.hardware.camera2.CameraManager
import android.os.SystemClock
import android.util.Range
import android.util.Size
import androidx.camera.camera2.interop.Camera2CameraInfo
import androidx.camera.camera2.interop.ExperimentalCamera2Interop
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.Preview
import androidx.camera.core.resolutionselector.AspectRatioStrategy
import androidx.camera.core.resolutionselector.ResolutionSelector
import androidx.camera.core.resolutionselector.ResolutionStrategy
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import androidx.lifecycle.LifecycleOwner
import com.linnan.blindassist.util.FatalThrowables
import com.linnan.blindassist.vision.FrameClockDomain
import com.linnan.blindassist.vision.FrameStamp
import com.linnan.blindassist.vision.VisionFrame
import java.util.concurrent.Executor
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicLong

class CameraXFrameSource(
    context: Context,
    private val lifecycleOwner: LifecycleOwner,
    private val analysisExecutor: ExecutorService = Executors.newSingleThreadExecutor()
) : FrameSource {
    private val appContext = context.applicationContext
    private val mainExecutor: Executor = ContextCompat.getMainExecutor(appContext)
    private val lifecycleLock = Any()
    private var cameraProvider: ProcessCameraProvider? = null
    private var starting = false
    private var started = false
    private var shutdownRequested = false
    private var sessionGeneration = 0L
    private val analyzerFailureReported = AtomicBoolean(false)

    override fun start(
        previewView: PreviewView?,
        onFrame: (VisionFrame) -> Unit,
        onStarted: () -> Unit,
        onError: (Throwable) -> Unit,
        onPreviewBitmap: ((Bitmap) -> Unit)?
    ) {
        if (previewView == null) {
            onError(IllegalArgumentException("CameraXFrameSource requires a non-null PreviewView"))
            return
        }
        val generation = synchronized(lifecycleLock) {
            if (shutdownRequested || started || starting) return
            starting = true
            sessionGeneration += 1L
            sessionGeneration
        }

        analyzerFailureReported.set(false)
        val providerFuture = ProcessCameraProvider.getInstance(appContext)
        providerFuture.addListener({
            try {
                val provider = providerFuture.get()
                if (!isCurrentSession(generation)) return@addListener
                val preview = Preview.Builder()
                    .setTargetFrameRate(Range(TARGET_FPS, TARGET_FPS))
                    .build()
                    .also {
                    it.setSurfaceProvider(previewView.surfaceProvider)
                }
                val analysis = ImageAnalysis.Builder()
                    .setResolutionSelector(resolutionSelector())
                    .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                    .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
                    .build()

                synchronized(lifecycleLock) {
                    if (!isCurrentSessionLocked(generation)) return@addListener
                    cameraProvider = provider
                    provider.unbindAll()
                    val camera = provider.bindToLifecycle(
                        lifecycleOwner,
                        CameraSelector.DEFAULT_BACK_CAMERA,
                        preview,
                        analysis
                    )
                    val source = cameraSourceDescriptor(camera.cameraInfo)
                    analysis.setAnalyzer(analysisExecutor) { imageProxy ->
                        val capturedAtNs = imageProxy.imageInfo.timestamp
                        val observedReceivedAtNs = SystemClock.elapsedRealtimeNanos()
                        val stamp = FrameStamp(
                            frameId = NEXT_FRAME_ID.getAndIncrement(),
                            capturedAtNs = capturedAtNs,
                            receivedAtNs = normalizedFrameReceiptTime(
                                capturedAtNs = capturedAtNs,
                                observedReceivedAtNs = observedReceivedAtNs,
                                clockDomain = source.clockDomain
                            ),
                            sourceId = source.sourceId,
                            coordinateFrame = source.coordinateFrame,
                            clockDomain = source.clockDomain
                        )
                        val frame = ImageProxyVisionFrame(imageProxy, stamp)
                        if (!isCurrentSession(generation)) {
                            frame.close()
                            return@setAnalyzer
                        }
                        try {
                            onFrame(frame)
                        } catch (error: Throwable) {
                            FatalThrowables.rethrowIfFatal(error)
                            frame.close()
                            reportAnalyzerError(generation, error, onError)
                        }
                    }
                    started = true
                    onStarted()
                }
            } catch (error: Throwable) {
                FatalThrowables.rethrowIfFatal(error)
                synchronized(lifecycleLock) {
                    if (isCurrentSessionLocked(generation)) {
                        onError(error)
                    }
                }
            } finally {
                synchronized(lifecycleLock) {
                    if (sessionGeneration == generation) {
                        starting = false
                    }
                }
            }
        }, mainExecutor)
    }

    override fun stop() {
        val provider = synchronized(lifecycleLock) {
            sessionGeneration += 1L
            starting = false
            started = false
            cameraProvider.also {
                cameraProvider = null
            }
        }
        provider?.unbindAll()
        analyzerFailureReported.set(false)
    }

    override fun shutdown() {
        val shouldShutdown = synchronized(lifecycleLock) {
            if (shutdownRequested) {
                false
            } else {
                shutdownRequested = true
                true
            }
        }
        stop()
        if (shouldShutdown) {
            analysisExecutor.shutdown()
            try {
                if (!analysisExecutor.awaitTermination(500, TimeUnit.MILLISECONDS)) {
                    analysisExecutor.shutdownNow()
                }
            } catch (error: InterruptedException) {
                analysisExecutor.shutdownNow()
                Thread.currentThread().interrupt()
            }
        }
    }

    private fun resolutionSelector(): ResolutionSelector {
        return ResolutionSelector.Builder()
            .setAspectRatioStrategy(AspectRatioStrategy.RATIO_4_3_FALLBACK_AUTO_STRATEGY)
            .setResolutionStrategy(
                ResolutionStrategy(
                    Size(ANALYSIS_WIDTH, ANALYSIS_HEIGHT),
                    ResolutionStrategy.FALLBACK_RULE_CLOSEST_HIGHER_THEN_LOWER
                )
            )
            .build()
    }

    private fun reportAnalyzerError(generation: Long, error: Throwable, onError: (Throwable) -> Unit) {
        if (isCurrentSession(generation) && analyzerFailureReported.compareAndSet(false, true)) {
            mainExecutor.execute {
                synchronized(lifecycleLock) {
                    if (isCurrentSessionLocked(generation)) {
                        onError(error)
                    }
                }
            }
        }
    }

    @androidx.annotation.OptIn(markerClass = [ExperimentalCamera2Interop::class])
    private fun cameraSourceDescriptor(cameraInfo: androidx.camera.core.CameraInfo): CameraSourceDescriptor {
        val cameraId = Camera2CameraInfo.from(cameraInfo).cameraId
        val manager = appContext.getSystemService(Context.CAMERA_SERVICE) as CameraManager
        val timestampSource = manager.getCameraCharacteristics(cameraId)
            .get(CameraCharacteristics.SENSOR_INFO_TIMESTAMP_SOURCE)
        val clockDomain = if (timestampSource == CameraCharacteristics.SENSOR_INFO_TIMESTAMP_SOURCE_REALTIME) {
            FrameClockDomain.ANDROID_ELAPSED_REALTIME
        } else {
            FrameClockDomain.CAMERA_HARDWARE_UNMAPPED
        }
        return CameraSourceDescriptor(
            sourceId = "camera2:$cameraId",
            coordinateFrame = "camera2:$cameraId:analysis-buffer",
            clockDomain = clockDomain
        )
    }

    private fun isCurrentSession(generation: Long): Boolean {
        return synchronized(lifecycleLock) {
            isCurrentSessionLocked(generation)
        }
    }

    private fun isCurrentSessionLocked(generation: Long): Boolean {
        return !shutdownRequested && sessionGeneration == generation
    }

    private data class CameraSourceDescriptor(
        val sourceId: String,
        val coordinateFrame: String,
        val clockDomain: FrameClockDomain
    )

    companion object {
        private val NEXT_FRAME_ID = AtomicLong(0L)
        private const val ANALYSIS_WIDTH = 640
        private const val ANALYSIS_HEIGHT = 480
        private const val TARGET_FPS = 24
    }
}

internal fun normalizedFrameReceiptTime(
    capturedAtNs: Long,
    observedReceivedAtNs: Long,
    clockDomain: FrameClockDomain
): Long = if (clockDomain == FrameClockDomain.ANDROID_ELAPSED_REALTIME) {
    observedReceivedAtNs.coerceAtLeast(capturedAtNs)
} else {
    observedReceivedAtNs
}
