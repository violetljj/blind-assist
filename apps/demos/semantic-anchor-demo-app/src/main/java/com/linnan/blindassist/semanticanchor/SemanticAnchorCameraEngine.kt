package com.linnan.blindassist.semanticanchor

import android.content.Context
import android.util.Size
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.core.resolutionselector.ResolutionSelector
import androidx.camera.core.resolutionselector.ResolutionStrategy
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import androidx.lifecycle.LifecycleOwner
import com.google.mlkit.vision.barcode.BarcodeScanner
import com.google.mlkit.vision.barcode.BarcodeScannerOptions
import com.google.mlkit.vision.barcode.BarcodeScanning
import com.google.mlkit.vision.barcode.common.Barcode
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.text.TextRecognizer
import com.google.mlkit.vision.text.TextRecognition
import com.google.mlkit.vision.text.latin.TextRecognizerOptions
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicReference

internal class SemanticAnchorCameraEngine(
    context: Context,
    private val lifecycleOwner: LifecycleOwner,
    private val previewView: PreviewView,
    initialTarget: AnchorTarget,
    private val onObservation: (AnchorObservation) -> Unit,
    private val onStatus: (String) -> Unit,
) {
    private val appContext = context.applicationContext
    private val mainExecutor = ContextCompat.getMainExecutor(appContext)
    private val analyzerExecutor: ExecutorService = Executors.newSingleThreadExecutor()
    private val target = AtomicReference(initialTarget)
    private val processing = AtomicBoolean(false)
    private val active = AtomicBoolean(false)
    private val closed = AtomicBoolean(false)
    private val replayPaused = AtomicBoolean(false)
    private val barcodeScanner: BarcodeScanner = BarcodeScanning.getClient(
        BarcodeScannerOptions.Builder().setBarcodeFormats(Barcode.FORMAT_QR_CODE).build(),
    )
    private val textRecognizer: TextRecognizer =
        TextRecognition.getClient(TextRecognizerOptions.DEFAULT_OPTIONS)
    private var provider: ProcessCameraProvider? = null
    private var analysis: ImageAnalysis? = null

    fun updateTarget(value: AnchorTarget) {
        target.set(value)
    }

    fun setReplayPaused(paused: Boolean) {
        replayPaused.set(paused)
    }

    fun start() {
        check(!closed.get()) { "camera engine is already closed" }
        if (!active.compareAndSet(false, true)) return
        onStatus("正在启动后置相机…")
        val future = ProcessCameraProvider.getInstance(appContext)
        future.addListener({
            try {
                if (!active.get()) return@addListener
                val cameraProvider = future.get(10, TimeUnit.SECONDS)
                val preview = Preview.Builder().build().also { it.surfaceProvider = previewView.surfaceProvider }
                val imageAnalysis = ImageAnalysis.Builder()
                    .setResolutionSelector(
                        ResolutionSelector.Builder().setResolutionStrategy(
                            ResolutionStrategy(
                                Size(1280, 720),
                                ResolutionStrategy.FALLBACK_RULE_CLOSEST_HIGHER_THEN_LOWER,
                            ),
                        ).build(),
                    )
                    .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                    .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_YUV_420_888)
                    .build()
                imageAnalysis.setAnalyzer(analyzerExecutor, ::analyze)
                cameraProvider.unbindAll()
                cameraProvider.bindToLifecycle(
                    lifecycleOwner,
                    CameraSelector.DEFAULT_BACK_CAMERA,
                    preview,
                    imageAnalysis,
                )
                provider = cameraProvider
                analysis = imageAnalysis
                onStatus("LIVE · 等待独立语义证据")
            } catch (error: Throwable) {
                active.set(false)
                onStatus("CameraX 启动失败：${error.message ?: error.javaClass.simpleName}")
            }
        }, mainExecutor)
    }

    fun shutdown() {
        if (!closed.compareAndSet(false, true)) return
        active.set(false)
        analysis?.clearAnalyzer()
        analysis = null
        provider?.unbindAll()
        barcodeScanner.close()
        textRecognizer.close()
        analyzerExecutor.shutdownNow()
    }

    private fun analyze(imageProxy: ImageProxy) {
        if (!active.get() || replayPaused.get() || !processing.compareAndSet(false, true)) {
            imageProxy.close()
            return
        }
        val mediaImage = imageProxy.image
        if (mediaImage == null) {
            processing.set(false)
            imageProxy.close()
            return
        }
        val input = InputImage.fromMediaImage(mediaImage, imageProxy.imageInfo.rotationDegrees)
        val snapshot = target.get()
        when (snapshot.mode) {
            AnchorMode.MARKER -> barcodeScanner.process(input)
                .addOnSuccessListener { barcodes ->
                    emit(barcodes.mapNotNull { it.rawValue }, "LIVE QR")
                }
                .addOnFailureListener(::emitFailure)
                .addOnCompleteListener { finishFrame(imageProxy) }
            AnchorMode.OCR -> textRecognizer.process(input)
                .addOnSuccessListener { text ->
                    emit(text.text.takeIf(String::isNotBlank)?.let(::listOf).orEmpty(), "LIVE OCR")
                }
                .addOnFailureListener(::emitFailure)
                .addOnCompleteListener { finishFrame(imageProxy) }
        }
    }

    private fun emit(candidates: List<String>, source: String) {
        if (active.get() && !replayPaused.get()) {
            mainExecutor.execute { onObservation(AnchorObservation(candidates, source)) }
        }
    }

    private fun emitFailure(error: Exception) {
        if (active.get()) mainExecutor.execute {
            onStatus("识别帧失败：${error.message ?: error.javaClass.simpleName}")
        }
    }

    private fun finishFrame(imageProxy: ImageProxy) {
        imageProxy.close()
        processing.set(false)
    }
}
