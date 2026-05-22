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
import com.linnan.blindassist.vision.toArgbBitmap
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

class CameraXFrameSource(
    context: Context,
    private val lifecycleOwner: LifecycleOwner,
    private val analysisExecutor: ExecutorService = Executors.newSingleThreadExecutor()
) : FrameSource {
    private val appContext = context.applicationContext
    private var cameraProvider: ProcessCameraProvider? = null
    private var starting = false
    private var started = false

    override fun start(
        previewView: PreviewView,
        onFrame: (android.graphics.Bitmap) -> Unit,
        onStarted: () -> Unit,
        onError: (Throwable) -> Unit
    ) {
        if (started || starting) return

        starting = true
        val providerFuture = ProcessCameraProvider.getInstance(appContext)
        providerFuture.addListener({
            try {
                val provider = providerFuture.get()
                cameraProvider = provider
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
                            try {
                                onFrame(imageProxy.toArgbBitmap())
                            } finally {
                                imageProxy.close()
                            }
                        }
                    }

                provider.unbindAll()
                provider.bindToLifecycle(
                    lifecycleOwner,
                    CameraSelector.DEFAULT_BACK_CAMERA,
                    preview,
                    analysis
                )
                started = true
                onStarted()
            } catch (error: Exception) {
                onError(error)
            } finally {
                starting = false
            }
        }, ContextCompat.getMainExecutor(appContext))
    }

    override fun stop() {
        cameraProvider?.unbindAll()
        started = false
        starting = false
    }

    override fun shutdown() {
        stop()
        analysisExecutor.shutdown()
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

    companion object {
        private const val ANALYSIS_WIDTH = 640
        private const val ANALYSIS_HEIGHT = 480
    }
}
