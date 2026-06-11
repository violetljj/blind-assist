package com.linnan.blindassist.camera

import android.content.Context
import android.util.Size
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
import com.linnan.blindassist.vision.VisionFrame
import java.util.concurrent.Executor
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicBoolean

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
        previewView: PreviewView,
        onFrame: (VisionFrame) -> Unit,
        onStarted: () -> Unit,
        onError: (Throwable) -> Unit
    ) {
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
                val preview = Preview.Builder().build().also {
                    it.setSurfaceProvider(previewView.surfaceProvider)
                }
                val analysis = ImageAnalysis.Builder()
                    .setResolutionSelector(resolutionSelector())
                    .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                    .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
                    .build()
                    .also { analyzer ->
                        analyzer.setAnalyzer(analysisExecutor) { imageProxy ->
                            val frame = ImageProxyVisionFrame(imageProxy)
                            try {
                                onFrame(frame)
                            } catch (error: Throwable) {
                                FatalThrowables.rethrowIfFatal(error)
                                frame.close()
                                reportAnalyzerError(error, onError)
                            }
                        }
                    }

                if (!isCurrentSession(generation)) return@addListener
                synchronized(lifecycleLock) {
                    cameraProvider = provider
                }
                provider.unbindAll()
                if (!isCurrentSession(generation)) return@addListener
                provider.bindToLifecycle(
                    lifecycleOwner,
                    CameraSelector.DEFAULT_BACK_CAMERA,
                    preview,
                    analysis
                )
                if (isCurrentSession(generation)) {
                    synchronized(lifecycleLock) {
                        started = true
                    }
                    onStarted()
                } else {
                    provider.unbindAll()
                }
            } catch (error: Throwable) {
                FatalThrowables.rethrowIfFatal(error)
                if (isCurrentSession(generation)) {
                    onError(error)
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

    private fun reportAnalyzerError(error: Throwable, onError: (Throwable) -> Unit) {
        if (analyzerFailureReported.compareAndSet(false, true)) {
            mainExecutor.execute {
                onError(error)
            }
        }
    }

    private fun isCurrentSession(generation: Long): Boolean {
        return synchronized(lifecycleLock) {
            !shutdownRequested && sessionGeneration == generation
        }
    }

    companion object {
        private const val ANALYSIS_WIDTH = 640
        private const val ANALYSIS_HEIGHT = 480
    }
}
